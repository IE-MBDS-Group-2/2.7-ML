# Dagster loader image: reads Miguel's data lake, builds the panel, publishes to the group database.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DAGSTER_HOME=/opt/dagster/home \
    ML2_REPO_ROOT=/app \
    PYTHONPATH=/app/pipeline

WORKDIR /app

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY pyproject.toml README.md config.yaml ./
COPY src ./src
COPY pipeline ./pipeline
RUN pip install --no-cache-dir --no-deps . && mkdir -p "$DAGSTER_HOME" /app/data/processed

EXPOSE 3000
CMD ["dagster", "dev", "-h", "0.0.0.0", "-p", "3000", "-m", "ml2_pipeline.definitions"]
