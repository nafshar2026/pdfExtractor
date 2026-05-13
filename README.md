# pdf-extractor

Splits a multi-document PDF (e.g. a scanned auto-finance deal jacket) into one PDF
per logical document. Supports both digital text PDFs and scanned image PDFs via
PaddleOCR. Input files can be local or from Azure Blob Storage.

---

## Quick Start (Recommended)

### Any platform (local or Azure):
```powershell
.\run.ps1
```

This unified launcher guides you through:
- **Local mode**: Fast splitting on your machine (minutes)
- **Azure mode**: Cloud-based splitting with persistence to Blob Storage

---

## How it works

Each page is analysed for structural signals — page-numbering footers, embedded
document markers, or a prominent title — and the results are grouped into document
runs before writing output files. The same logic handles both digital text pages
and scanned image pages; only the extraction method differs.

### Large-file hardening (smallest hardware objective)

To reliably process large scanned PDFs on constrained Azure job hardware, OCR now
supports process isolation, worker recycling, bounded retries, and a configurable
render-width cap.

Available controls:
- `PDF_EXTRACTOR_OCR_ISOLATED=1`: run OCR in a dedicated subprocess
- `PDF_EXTRACTOR_OCR_RECYCLE_CALLS=<N>`: recycle OCR worker every N OCR calls
- `PDF_EXTRACTOR_OCR_POOL_RETRIES=<N>`: retry when OCR worker dies (for example OOM)
- `PDF_EXTRACTOR_OCR_MAX_WIDTH=<pixels>`: downscale rendered page width before OCR

Recommended production profile for large files:
- `PDF_EXTRACTOR_OCR_ISOLATED=1`
- `PDF_EXTRACTOR_OCR_RECYCLE_CALLS=6`
- `PDF_EXTRACTOR_OCR_POOL_RETRIES=2`
- `PDF_EXTRACTOR_OCR_MAX_WIDTH=1200`

Validated outcomes:
- 600157742.pdf and 600157748.pdf complete successfully in Azure with the hardened profile.
- Wildcard batch processing (for example `6*`) can process multiple large files in one run and produce one consolidated Excel output.

---

## Running the Tool

### Setup (one-time)

**Local mode only:**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

**Azure mode only:**
1. Download Azure CLI: `https://aka.ms/installazurecliwindows`
2. Ask your project owner to grant you **Contributor** access to `nader-test-rag` resource group
3. Before each session: activate your PIM role (Azure Portal → Privileged Identity Management → My Roles → Activate)

### Run the interactive launcher

```powershell
.\run.ps1
```

**Choose a mode:**
- **1 (Local)**: Split PDFs on your machine using `src/pdf_extractor/Data/*.pdf`
  - Output: `output/split-*/` folders + Excel files
  - Logs: `output/job-logs/*.log`
  
- **2 (Azure)**: Submit jobs to Azure Container Apps
  - Input: Azure Blob Storage (`pdfinput` container)
  - Output: Azure Blob Storage (`pdfoutput` container) + local Excel + logs

Both modes support optional opt-in extraction (generates Excel with credit app data).

### Output files

All modes produce:
- **Split PDFs**: One folder per source file under `output/split-*/`
- **Excel files** (if opt-in enabled): `output/opt_in_results-<filename>.xlsx`
  - Columns: Filename, Form Type, Name, Opt-In Status, Phones, Confidence
- **Job logs**: `output/job-logs/<execution-id>-<timestamp>.log`
  - Timestamped to preserve history; clean up manually as needed

---

## Advanced: Command-line usage

### Local mode (non-interactive)
```powershell
.\run.ps1 -Mode local -File Sample-1.pdf -OptIn
```

### Azure mode (non-interactive)
```powershell
.\run.ps1 -Mode azure
# Follow prompts to start a job
```

### Legacy launcher (still supported)
```powershell
.\run-local.ps1 -FileName Sample-1.pdf -WithOptIn
```

### Direct Python calls
```powershell
# Split a local PDF
.venv\Scripts\python -m pdf_extractor.cli src/pdf_extractor/Data/Sample-1.pdf \
  --split-documents --split-output-dir output/split-sample

# Optional: isolate OCR in a worker process (helps large scanned PDFs)
$env:PDF_EXTRACTOR_OCR_ISOLATED = "1"
# Recycle OCR worker every N OCR calls (two calls per scanned page)
$env:PDF_EXTRACTOR_OCR_RECYCLE_CALLS = "6"
# Retry if the isolated worker crashes
$env:PDF_EXTRACTOR_OCR_POOL_RETRIES = "2"
# Cap OCR render width (lower memory use for very large scans)
$env:PDF_EXTRACTOR_OCR_MAX_WIDTH = "1200"
.venv\Scripts\python -m pdf_extractor.cli src/pdf_extractor/Data/Sample-1.pdf \
  --split-documents --split-output-dir output/split-sample

# Azure wildcard batch (single run, single consolidated Excel)
.venv\Scripts\python -m pdf_extractor.cli "6*" --azure --split-documents

# Extract opt-in data to Excel
.venv\Scripts\python -c \
  "from pdf_extractor.opt_in_extractor import process_folder_to_excel; \
   process_folder_to_excel('output/split-sample', 'output/opt_in_results.xlsx')"

# Run tests
.venv\Scripts\python -m pytest tests/ -v
```

---

## Credit Application Extraction (Opt-In Data)

Supported forms:
- **RouteOne** (digital or scanned): Extracts name, phone, and opt-in status from "Optional Consent" signature line
- **DealerTrack** (digital or scanned): Extracts name, phone, and opt-in status from checkbox

Excel output columns:
- **Filename**: Source PDF name
- **Form Type**: routeone / dealertrack / unknown
- **Last Name / First Name**: Extracted from credit application
- **Opt-In Status**: opted_in / opted_out / unclear / not_found
- **Telemarketing Phones**: List of phone numbers
- **Confidence**: high / medium / low

### Local mode (non-interactive)

---

## Repository layout

```
src/pdf_extractor/
    extractor.py        # Core pypdf utilities and data models
    image_splitter.py   # split_pdf() entry point and all boundary-detection logic
    cli.py              # argparse CLI — local and Azure modes
    opt_in_extractor.py # Credit application opt-in extraction (text + vision)
    azure_storage.py    # Azure Blob Storage download/upload helpers
    genpdf.py           # Utility: generates text-based test PDFs

tests/
    test_extractor.py
    test_image_splitter.py

run.ps1                 # Unified launcher (local + Azure modes) — **START HERE**
run-local.ps1           # Legacy local launcher (delegates to run.ps1)
Dockerfile              # Container image definition
azure-deploy.sh         # One-time infrastructure setup (admin use only)
```

---

## Roadmap / Future Enhancements

### Deduplication by Hash
- Detect and eliminate duplicate pages within a PDF (regardless of position)
- Use cryptographic hash (SHA-256) of page content to identify copies
- Useful when deal jackets contain multiple copies of the same form or signature page
- Targeted for Q2 2026

### Other Potential Improvements
- CLI flag for `--opt-in` extraction directly in `split-documents`
- Batch processing dashboard with progress tracking
- Custom page boundary rules per organization
- Export split decisions as JSON for audit/review workflows
.env.example            # Template for local environment variables
```

---

## Deploying from scratch (admin only)

If the Azure infrastructure needs to be rebuilt from scratch, follow the numbered
steps in `azure-deploy.sh`. Steps 1–2 (resource group and registry) are one-time.
Step 3 (image build) is repeated whenever code changes. Steps 6–7 (environment
and job) are one-time.

After any code change, rebuild the image and the job picks it up on the next run:

```powershell
az acr build --registry NaderContainerRegistry --resource-group nader-test-rag --image pdf-extractor:latest .
```
