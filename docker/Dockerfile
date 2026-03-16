# MagicQuant Docker Build
# Multi-stage build: first compiles llama.cpp, second stage runs the Python application

# Stage 1: Build llama.cpp
FROM ubuntu:22.04 AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git \
    && cd llama.cpp \
    && make -j$(nproc) \
    && make install

# Stage 2: Runtime environment with Python + MagicQuant
FROM ubuntu:22.04

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy built llama.cpp binaries from builder stage
COPY --from=builder /build/llama.cpp/bin/ /usr/local/bin/
COPY --from=builder /build/llama.cpp/lib/ /usr/local/lib/

# Install Python dependencies
RUN pip3 install --no-cache-dir numpy

# Add magicquant to Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Copy MagicQuant code
COPY magicquant/ ./magicquant/
COPY requirements.txt ./

# Create non-root user for security
RUN useradd -m -s /bin/bash magicuser && \
    chown -R magicuser:magicuser /app

USER magicuser

# Default command shows help
CMD ["python3", "-m", "magicquant"]