"""Dagster entry point: dagster dev -m ml2_pipeline.definitions"""

from dagster import AssetSelection, Definitions, EnvVar, define_asset_job, load_assets_from_modules

from ml2_pipeline import assets
from ml2_pipeline.resources import PostgresResource

refresh_all = define_asset_job("refresh_ml2_data", selection=AssetSelection.all())

defs = Definitions(
    assets=load_assets_from_modules([assets]),
    jobs=[refresh_all],
    resources={
        "lake": PostgresResource(url=EnvVar("LAKE_DB_URL"), read_only=True),
        "groupdb": PostgresResource(url=EnvVar("GROUP_DB_URL")),
    },
)
