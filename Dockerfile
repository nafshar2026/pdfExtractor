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

# OCR reliability settings validated against large scanned PDFs on constrained Azure SKUs.
# ISOLATED runs PaddleOCR in a subprocess to prevent OOM in the main process.
ENV PDF_EXTRACTOR_OCR_ISOLATED=1
ENV PDF_EXTRACTOR_OCR_RECYCLE_CALLS=6
ENV PDF_EXTRACTOR_OCR_POOL_RETRIES=2
ENV PDF_EXTRACTOR_OCR_MAX_WIDTH=800
ENV PDF_EXTRACTOR_OVERLAP_CHUNK_PAGES=20

# AZURE_STORAGE_CONNECTION_STRING, AZURE_INPUT_CONTAINER, AZURE_OUTPUT_CONTAINER
# are injected by Azure Container Apps at runtime — no .env file needed in the image.

ENTRYPOINT ["sh", "-c", "python -m pdf_extractor.cli --azure --split-documents \"${PDF_INPUT_FILE}\""]
