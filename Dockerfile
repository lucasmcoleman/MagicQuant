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
# python:3.12-slim has no compiler, so llama-cpp-python (a hard dep) must come
# from a prebuilt CPU wheel — pull it from the official index rather than
# compiling from sdist (which would fail without gcc/cmake).
RUN pip install --no-cache-dir \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        /tmp/*.whl \
    && rm -f /tmp/*.whl \
    && python -c "from magicquant.quant.ggml_binding import get_handle; get_handle()"

USER magicquant

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import magicquant; from magicquant.quant.ggml_binding import get_handle; get_handle(); print(magicquant.__version__)"

ENTRYPOINT ["magicquant"]
