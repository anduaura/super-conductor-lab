FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8765 \
    SCL_RUNS_DIR=/data/runs

WORKDIR /app

COPY pyproject.toml README.md ./
COPY scl ./scl

RUN pip install -e ".[web]"

RUN mkdir -p /data/runs

EXPOSE 8765

# `--host 0.0.0.0` so the container is reachable from outside; auth token
# is read from the SCL_AUTH_TOKEN environment variable.
CMD ["sh", "-c", "scl serve --host 0.0.0.0 --port ${PORT} --runs-dir ${SCL_RUNS_DIR}"]
