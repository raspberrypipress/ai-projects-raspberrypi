FROM arm64v8/python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --fix-missing -y \
    libgl1 \
    nano \
    wget \
    tree \
    default-jdk \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install -U pip --no-cache-dir && \
    pip install --no-cache-dir \
    tensorflow==2.14.0 \
    --extra-index-url https://pkgs.raspberrypi.com/python && \
    pip install --no-cache-dir \
    importlib_resources \
    tensorflow_datasets==4.9.* \
    model-compression-toolkit==2.2.0 \
    imx500-converter[tf]

# # Create a non-root user with UID/GID 1000
# RUN groupadd -g 1000 appuser && \
#     useradd -m -u 1000 -g 1000 appuser && \
#     mkdir -p /app/models /app/converted && \
#     chown -R appuser:appuser /app

# Switch to non-root user
# USER appuser

COPY custom_mobilenet.py /app/
