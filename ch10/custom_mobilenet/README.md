# Custom MobileNet (RPS) for IMX500

This project trains a `MobileNetV2` classifier on the TensorFlow Datasets
Rock-Paper-Scissors dataset, applies post-training quantization with Sony's
Model Compression Toolkit, and converts the quantized model with `imxconv-tf`
to generate `packerOut.zip` for IMX500 packaging.

## What the script does

Running `custom_mobilenet.py` performs the full pipeline:

1. Loads and preprocesses `rock_paper_scissors` from `tensorflow_datasets`
2. Trains a transfer-learning `MobileNetV2` model
3. Saves the float model to `models/mobilenet-rps`
4. Quantizes and saves a Keras model to `models/mobilenet-quant-rps.keras`
5. Converts the quantized model into `converted/` artifacts with `imxconv-tf`
6. Verifies `converted/packerOut.zip` exists

## Prerequisites

- Docker installed on your system
- ARM64 host (the Docker image is based on `arm64v8/python:3.10-slim`)
- Enough disk/network bandwidth for dataset and model downloads

Raspberry Pi Docker setup reference:
<https://www.raspberrypi.com/documentation/computers/ai.html#step3-llm>

## Build the Docker image

From this repository root:

```bash
docker build -t mobilenet .
```

The `Dockerfile` installs:

- `tensorflow==2.14.0` (Raspberry Pi package index)
- `tensorflow_datasets`
- `model-compression-toolkit==2.2.0`
- `imx500-converter[tf]`
- system dependencies such as `default-jdk`, `ffmpeg`, and OpenCV libs

## Run the container
Start a container and mount outputs so artifacts persist on host:

```bash
docker run -it --name mobilenet \
  -v "$(pwd)/output/models:/app/models" \
  -v "$(pwd)/output/converted:/app/converted" \
  mobilenet bash
```

Inside the container:

```bash
python custom_mobilenet.py
```

## Expected outputs

After completion, you should have:

- `output/models/mobilenet-rps` (trained float model)
- `output/models/mobilenet-quant-rps.keras` (quantized Keras model)
- `output/converted/packerOut.zip` (required for IMX500 packaging)
- additional conversion files in `output/converted/` (such as `.xml`, `.pbtxt`)

## Notes and troubleshooting

- Training runs in a headless container; matplotlib visualization is intentionally
  disabled in `custom_mobilenet.py`.
- First run can take time due to package downloads, dataset download, and model
  training.
- If you see architecture-related image errors, verify Docker is running on an
  ARM64 system (or use emulation explicitly).
- If container name already exists, remove it first:

```bash
docker rm -f mobilenet
```

## Next step

Use `output/converted/packerOut.zip` for the IMX500 packaging step:
<https://www.raspberrypi.com/documentation/accessories/ai-camera.html#packaging>
