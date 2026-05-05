FROM python:3.11-slim

# System libraries required by PaddleOCR / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Prevents a glibc memory arena crash in PaddlePaddle on Linux containers.
ENV MALLOC_ARENA_MAX=1

WORKDIR /app

# Install Python package (separate COPY so pip layer is cached when only src changes)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# AZURE_STORAGE_CONNECTION_STRING, AZURE_INPUT_CONTAINER, AZURE_OUTPUT_CONTAINER
# are injected by Azure Container Apps at runtime — no .env file needed in the image.

ENTRYPOINT ["python", "-m", "pdf_extractor.cli", "--azure", "--split-documents"]

# Default blob to process; override at runtime:
#   docker run ... <image> "Sample-2.pdf"
CMD ["RO-1.pdf"]
