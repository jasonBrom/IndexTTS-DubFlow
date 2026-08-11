FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg git python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN chmod +x /app/scripts/*.sh /app/docker/entrypoint.sh

ENV DUBBER_RUNTIME_ROOT=/runtime \
    DUBBER_OUTPUT_ROOT=/outputs \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

VOLUME ["/runtime", "/outputs"]
EXPOSE 7860

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["web", "--server-name", "0.0.0.0", "--server-port", "7860"]

