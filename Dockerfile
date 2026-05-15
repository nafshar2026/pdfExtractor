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
# cffi 2.0 breaks PaddlePaddle 2.6.2 at the C extension level — pin to 1.x.
RUN pip install --no-cache-dir "cffi<2" && pip install --no-cache-dir .

# Pre-download PaddleOCR models so they are baked into the image layer.
# Eliminates the 3-4 minute cold-start model download on every job execution.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=False, lang='en', show_log=False)"

# OCR reliability settings validated against large scanned PDFs on constrained Azure SKUs.
# ISOLATED runs PaddleOCR in a subprocess to prevent OOM in the main process.
ENV PDF_EXTRACTOR_OCR_ISOLATED=1
ENV PDF_EXTRACTOR_OCR_RECYCLE_CALLS=6
ENV PDF_EXTRACTOR_OCR_POOL_RETRIES=2
ENV PDF_EXTRACTOR_OCR_MAX_WIDTH=800
ENV PDF_EXTRACTOR_OVERLAP_CHUNK_PAGES=20

# AZURE_STORAGE_CONNECTION_STRING, AZURE_INPUT_CONTAINER, AZURE_OUTPUT_CONTAINER
# are injected by Azure Container Apps at runtime — no .env file needed in the image.

ENTRYPOINT ["python", "-m", "pdf_extractor.cli", "--azure", "--split-documents"]

# Default blob to process; override at runtime:
#   docker run ... <image> "Sample-2.pdf"
CMD ["RO-1.pdf"]
