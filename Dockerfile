# Runs the upload-a-photo demo on the CPU. Mount trained checkpoints at /app/checkpoints:
#   docker build -t glaucoma-seg .
#   docker run --rm -p 7860:7860 -v "$PWD/checkpoints:/app/checkpoints:ro" glaucoma-seg
FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 GRADIO_SERVER_NAME=0.0.0.0

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[demo]"

COPY app ./app
EXPOSE 7860
CMD ["python", "app/demo.py"]
