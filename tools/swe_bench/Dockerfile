FROM docker:27-cli AS dockercli

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# docker CLI is required because run_swe_docker.py shells out to docker commands
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker

# Install runtime dependencies required by OpenCollab and runner
COPY opencollab /app/opencollab
RUN pip install --no-cache-dir -e "/app/opencollab[swebench]"

# Runner scripts
RUN mkdir -p /app/tools/swe_bench
COPY tools/swe_bench/run_swe_docker.py /app/tools/swe_bench/run_swe_docker.py

ENTRYPOINT ["python", "/app/tools/swe_bench/run_swe_docker.py"]
