# pdf-extractor

Splits a multi-document PDF (e.g. a scanned auto-finance deal jacket) into one PDF
per logical document. Supports both digital text PDFs and scanned image PDFs via
PaddleOCR. Input and output files are stored in Azure Blob Storage.

---

## How it works

Each page is analysed for structural signals — page-numbering footers, embedded
document markers, or a prominent title — and the results are grouped into document
runs before writing output files. The same logic handles both digital text pages
and scanned image pages; only the extraction method differs.

---

## Running on Azure (recommended)

The application runs as an **Azure Container Apps Job**. All infrastructure is
already deployed. You only need:

1. **Contributor access** to the `nader-test-rag` resource group — ask the project
   owner to add you via Azure Portal → nader-test-rag → Access Control (IAM).

2. **Azure CLI** installed — ask your admin to install it from
   `https://aka.ms/installazurecliwindows` if it is not already available.

3. **PIM role activated** — your access expires every 8 hours. Before each session:
   Azure Portal → Privileged Identity Management → My Roles → Activate.

4. **`run.ps1`** — download this single file from the repo. No cloning required.
   Run it with:
   ```powershell
   .\run.ps1
   ```
   An interactive menu guides you through all operations.

### Menu options

| Option | What it does |
|--------|-------------|
| 1 | Run the job on the current target file |
| 2 | Pick a different input file and run |
| 3 | Check the status of the last run |
| 4 | List output files in blob storage |
| 5 | Rebuild the Docker image after a code change |
| 6 | Show logs from the last run (use when a run fails) |

### Azure resources

| Resource | Name |
|----------|------|
| Resource group | `nader-test-rag` |
| Container App Job | `pdf-extractor-job` |
| Container Registry | `NaderContainerRegistry` |
| Storage account | `naderblob02` |
| Input container | `pdfinput` |
| Output container | `pdfoutput` |

Input PDFs are uploaded to the `pdfinput` blob container via the Azure Portal.
Split output PDFs appear under `pdfoutput/<filename>/` after a successful run.

---

## Running locally

### One-time setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in your Azure Storage connection string
(Azure Portal → naderblob02 → Security + networking → Access keys → key1 →
Connection string).

### Split a PDF from Azure Blob Storage

```powershell
.venv\Scripts\python -m pdf_extractor.cli "RO-1.pdf" --azure --split-documents
```

### Split a local PDF file

```powershell
.venv\Scripts\python -m pdf_extractor.cli src/pdf_extractor/Data/Sample-1.pdf --split-documents --split-output-dir output/split-1
```

### Run tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

### Extract opt-in data (DealerTrack & RouteOne credit applications)

After splitting PDFs, you can extract telemarketing opt-in status and contact information from credit application forms:

```powershell
# Split the PDF first
.venv\Scripts\python -m pdf_extractor.cli src/pdf_extractor/Data/RouteOne.pdf --split-documents --split-output-dir output/split-routeone

# Then extract opt-in data to Excel
.venv\Scripts\python -c "from pdf_extractor.opt_in_extractor import process_folder_to_excel; process_folder_to_excel('output/split-routeone', 'output/opt_in_results-routeone.xlsx')"
```

Or use the interactive local menu (recommended):

```powershell
.\run-local.ps1
```

Menu options include splitting with and without opt-in extraction in one command.

**Supported forms:**
- **RouteOne** (digital or scanned): Extracts name, phone numbers, and opt-in status from the "Optional Consent" signature line
- **DealerTrack** (digital or scanned): Extracts name, phone numbers, and opt-in status from the "You opt in / You do not opt in" checkbox

**Output:** Excel file at `output/opt_in_results-<filename>.xlsx` with columns:
- Filename
- Form Type
- Last Name
- First Name
- Opt-In Status (opted_in / opted_out / unclear)
- Telemarketing Phones
- Confidence (high / medium / low)

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

Dockerfile              # Container image definition
run.ps1                 # Interactive Azure operations menu (share this with users)
run-local.ps1           # Interactive local operations menu (split + opt-in extraction)
azure-deploy.sh         # One-time infrastructure setup commands (admin use only)
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
