-- Group database setup. Runs automatically on the first start of the local
-- Postgres container. On a managed host (Neon, Supabase, ...) run it once by hand
-- as the owner role:  psql "<admin url>" -f docker/initdb/01-init.sql

create schema if not exists ml2;

-- Read-only group role. Each teammate gets a personal login that is a member of it
-- (scripts/add_member.py), so access can be revoked per person.
do $$
begin
    if not exists (select from pg_roles where rolname = 'ml2_read') then
        create role ml2_read nologin;
    end if;
end
$$;

do $$
begin
    execute format('grant connect on database %I to ml2_read', current_database());
end
$$;

grant usage on schema ml2 to ml2_read;
grant select on all tables in schema ml2 to ml2_read;
alter default privileges in schema ml2 grant select on tables to ml2_read;

-- Nobody but the owner creates objects in public.
revoke create on schema public from public;
