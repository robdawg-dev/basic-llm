# Linux/native-Triton comparison environment for the FineWeb-Edu training run.
# Code and data are provided via volume mount at `docker run` time (see
# RUNBOOK.md), not copied into the image, so rebuilding this image doesn't
# require re-sending checkpoints/data through the Docker build context.
FROM python:3.11-slim

WORKDIR /workspace

# Triton JIT-compiles its launcher stub through a real C compiler at
# runtime, same requirement as cl.exe on Windows — python:3.11-slim doesn't
# ship one.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# cu132 matches this project's driver (CUDA 13.2); adjust if yours differs
# (see pyproject.toml's GPU-wheels comment).
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
RUN pip install --no-cache-dir tokenizers datasets pandas matplotlib
