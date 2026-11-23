"""Create (or rotate) a personal read-only login on the group database.

Usage (data admin only, GROUP_DB_ADMIN_URL set in .env):
    python scripts/add_member.py alice
    python scripts/add_member.py alice --rotate      # new password for an existing login
    python scripts/add_member.py alice --revoke      # remove the login

Prints the member's ML2_DB_URL once. Send it privately; it is not stored anywhere.
The login is a member of ml2_read: select-only on schema ml2, read-only transactions,
at most 5 connections.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="short lowercase name, e.g. alice")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rotate", action="store_true", help="set a new password")
    group.add_argument("--revoke", action="store_true", help="drop the login")
    parser.add_argument("--host", help="host name teammates should use (default: the admin URL's host)")
    parser.add_argument("--port", type=int, help="port teammates should use (default: the admin URL's port)")
    args = parser.parse_args()

    if not NAME_RE.match(args.name):
        print("Name must be lowercase letters, digits or underscores, starting with a letter.", file=sys.stderr)
        return 2
    role = f"ml2_{args.name}"

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    admin_url = os.environ.get("GROUP_DB_ADMIN_URL")
    if not admin_url:
        print("Set GROUP_DB_ADMIN_URL in .env", file=sys.stderr)
        return 2

    engine = create_engine(admin_url)
    with engine.begin() as conn:
        exists = conn.execute(text("select 1 from pg_roles where rolname = :r"), {"r": role}).scalar() is not None
        if args.revoke:
            if exists:
                conn.execute(text(f'drop role "{role}"'))
            print(f"Revoked {role}")
            return 0
        if exists and not args.rotate:
            print(f"{role} already exists. Use --rotate for a new password.", file=sys.stderr)
            return 1

        password = secrets.token_urlsafe(18)
        # Role names are validated above; the password is URL-safe base64, so quoting is safe.
        if exists:
            conn.execute(text(f"alter role \"{role}\" with password '{password}'"))
        else:
            conn.execute(
                text(f"create role \"{role}\" login password '{password}' connection limit 5 in role ml2_read")
            )
            conn.execute(text(f'alter role "{role}" set default_transaction_read_only = on'))

    url = make_url(admin_url).set(username=role, password=password)
    if args.host:
        url = url.set(host=args.host)
    if args.port:
        url = url.set(port=args.port)
    print("Send this privately to the member (it is shown only once):")
    print(f"ML2_DB_URL={url.render_as_string(hide_password=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
