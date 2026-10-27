FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ASSISTANT_DATA_DIR=/app/data

# git is needed once, to fetch the pinned docs corpus during `assistant ingest`.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY evals ./evals

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data /app/reports \
    && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["assistant", "serve", "--host", "0.0.0.0", "--port", "8000"]
