# MagicQuant Docker Build
# Multi-stage: compile llama.cpp, then run Python application

# Stage 1: Build llama.cpp
FROM ubuntu:22.04 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/llama \
    && cmake --build build -j$(nproc) \
    && cmake --install build

# Stage 2: Runtime
FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/llama/bin/ /usr/local/bin/
COPY --from=builder /opt/llama/lib/ /usr/local/lib/
RUN ldconfig

# Use the libggml we just built and installed (no llama-cpp-python compile in
# the runtime stage — there is no build toolchain here). ggml_binding
# discovers it via MAGICQUANT_LIBGGML_DIR.
ENV MAGICQUANT_LIBGGML_DIR=/usr/local/lib

# requirements.txt was removed; pyproject is the single source of truth.
COPY pyproject.toml README.md ./
COPY magicquant/ ./magicquant/
# Install MagicQuant. llama-cpp-python (a hard dep) needs no compiler when a
# CPU wheel is available from the official index; we already have system
# libggml so its bundled libs are only a fallback.
RUN pip3 install --no-cache-dir \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        -e .

# Smoke check: binding must load at build time.
RUN python3 -c "from magicquant.quant.ggml_binding import get_handle; get_handle()"

RUN useradd -m -s /bin/bash magicuser \
    && chown -R magicuser:magicuser /app
USER magicuser

ENTRYPOINT ["magicquant"]
CMD ["--help"]
