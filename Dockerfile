# MagicQuant Dockerfile — multi-stage build
# Supports both CUDA and ROCm via build args.

ARG BASE_IMAGE=python:3.12-slim

# ── Builder stage ──────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY magicquant/ magicquant/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /build/dist

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS runtime

ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} magicquant && \
    useradd -u ${UID} -g ${GID} -m magicquant

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

USER magicquant

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import magicquant; print(magicquant.__version__)"

ENTRYPOINT ["magicquant"]
