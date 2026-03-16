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

COPY pyproject.toml requirements.txt ./
COPY magicquant/ ./magicquant/
RUN pip3 install --no-cache-dir -e .

RUN useradd -m -s /bin/bash magicuser \
    && chown -R magicuser:magicuser /app
USER magicuser

ENTRYPOINT ["magicquant"]
CMD ["--help"]
