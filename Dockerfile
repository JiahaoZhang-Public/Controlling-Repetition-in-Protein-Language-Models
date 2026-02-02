FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# System deps: build tools + BLAS for torch wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libopenblas-dev \
        libssl-dev \
        libffi-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy codebase
COPY . /app

# Install python dependencies (dev includes runtime + pytest/ruff/black/mypy)
RUN pip install --upgrade pip && \
    pip install -r requirements/dev.txt && \
    pip install -e .

CMD ["/bin/bash"]
