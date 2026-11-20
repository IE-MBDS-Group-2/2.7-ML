"""ML II group project: corporate bond spread pricing.

Shared code every notebook imports:
    config     read config.yaml
    data       load the modelling panel (group database or parquet)
    schema     the panel's columns, their meaning, and which ones are allowed as features
    features   xy() and one shared preprocessing recipe (impute, scale, one-hot)
    splits     chronological partitions, walk-forward folds, issuer-grouped folds
    evaluate   regression metrics in basis points, metrics by stratum, quality gate
"""
