# ===========================================================================
# pdf-extractor - Unified PDF Splitting Tool (Local + Azure)
# ===========================================================================
#
# QUICK START:
#   .\run.ps1
#   Choose between Local (fast, on your machine) or Azure (cloud-based)
#
# WHAT IT DOES:
#   Splits multi-document PDFs (e.g. auto-finance deal jackets) into one
#   PDF per logical document. Uses AI-based page detection to find boundaries.
#   - Generates Excel files with opt-in extraction (optional)
#   - Saves all logs to output/job-logs/ with timestamps
#   - Works on local machine OR Azure Container Apps
#
# SUPPORTED MODES:
#   LOCAL  - Fast splitting on your machine (minutes)
#            Input:  src/pdf_extractor/Data/*.pdf
#            Output: output/split-*/ and output/opt_in_results-*.xlsx
#
#   AZURE  - Cloud-based splitting (via Container Apps)
#            Input:  Azure Blob Storage (pdfinput container)
#            Output: Azure Blob Storage (pdfoutput container) + local Excel
#            Note:   Azure jobs cannot access your local folders; use blob input/output.
#
# LOCAL MODE MENU OPTIONS:
#   1. Process one PDF           - Select and split a single file by number
#   2. Process all PDFs          - Split all files in src/pdf_extractor/Data/
#   3. Process by pattern        - Selective processing with wildcards
#      Examples:
#        6*           → All files starting with "6" (600156961.pdf, 600157748.pdf)
#        Sample-*     → All files starting with "Sample-" (Sample-1.pdf, etc.)
#        RouteOne*    → All files matching pattern (RouteOne.pdf)
#      (Optionally include opt-in extraction to Excel)
#   4. View outputs              - Show split folders, Excel files, recent logs
#   5. Configure settings        - Customize input/output directories
#   6. Switch to Azure mode      - Change to cloud-based splitting
#
# CONFIGURATION:
#   Settings are saved to: run-config.json (in repo root)
#   Each location can have different input/output paths
#   Paths persist between runs and can be reset to defaults
#
# SETUP (ONE-TIME):
#   Local mode:
#     py -3.11 -m venv .venv
#     .venv\Scripts\pip install -e ".[dev]"
#
#   Azure mode:
#     Download Azure CLI: https://aka.ms/installazurecliwindows
#     Ensure you have Contributor access to nader-test-rag resource group
#     (Ask project owner to add you via Azure Portal IAM)
#
# COMMAND-LINE USAGE (NON-INTERACTIVE):
#   Single file with opt-in extraction:
#     .\run.ps1 -Mode local -File Sample-1.pdf -OptIn
#
#   Single file without extraction:
#     .\run.ps1 -Mode local -File Sample-1.pdf
#
#   Force Azure mode (default via menu selection):
#     .\run.ps1 -Mode azure
#
# ===========================================================================

param(
    [string]$Mode,        # Force mode: "local" or "azure"
    [string]$File,        # Non-interactive: file to process
    [switch]$OptIn,       # Include opt-in extraction
    [switch]$Help,        # Display help and exit
    [string]$Config       # Path to config file (defaults to run-config.json)
)

$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

$REPO_ROOT = $PSScriptRoot
if (-not $REPO_ROOT) { $REPO_ROOT = (Get-Location).Path }
Push-Location $REPO_ROOT

# Config file path
if (-not $Config) {
    $Config = Join-Path $REPO_ROOT "run-config.json"
}

# Paths (defaults)
$VENV = Join-Path $REPO_ROOT ".venv\Scripts\python.exe"
$DATA_DIR_DEFAULT = Join-Path $REPO_ROOT "src\pdf_extractor\Data"
$OUTPUT_DIR_DEFAULT = Join-Path $REPO_ROOT "output"
$LOG_DIR_DEFAULT = Join-Path $OUTPUT_DIR_DEFAULT "job-logs"

# These will be overridden by config
$DATA_DIR = $DATA_DIR_DEFAULT
$OUTPUT_DIR = $OUTPUT_DIR_DEFAULT
$LOG_DIR = $LOG_DIR_DEFAULT

# Azure defaults
$RESOURCE_GROUP = "nader-test-rag"
$JOB_NAME = "pdf-extractor-job"
$ACR_NAME = "NaderContainerRegistry"
$STORAGE_ACCOUNT = "naderblob02"
$INPUT_CONTAINER = "pdfinput"
$OUTPUT_CONTAINER = "pdfoutput"

# Ensure az CLI is on PATH
$azPath = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
if (Test-Path $azPath) { $env:PATH += ";$azPath" }

# ==================== Helper Functions ====================

function Show-Help {
    Write-Host ""
    Write-Host "pdf-extractor - Unified PDF Splitting Tool" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run.ps1                              # Interactive menu (choose local or Azure)" -ForegroundColor White
    Write-Host "  .\run.ps1 -Help                        # Show this help" -ForegroundColor White
    Write-Host ""
    Write-Host "LOCAL MODE (split on your machine):" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 -Mode local                  # Interactive local menu" -ForegroundColor White
    Write-Host "  .\run.ps1 -Mode local -File PDF.pdf    # Split single file" -ForegroundColor White
    Write-Host "  .\run.ps1 -Mode local -File PDF.pdf -OptIn  # Split + extract opt-in to Excel" -ForegroundColor White
    Write-Host ""
    Write-Host "  Local Menu Options:" -ForegroundColor DarkCyan
    Write-Host "    1 = Process one PDF (select by number)" -ForegroundColor DarkCyan
    Write-Host "    2 = Process all PDFs" -ForegroundColor DarkCyan
    Write-Host "    3 = Process by pattern (e.g., 6*, Sample-*)" -ForegroundColor DarkCyan
    Write-Host "    4 = View outputs (split folders, Excel files, logs)" -ForegroundColor DarkCyan
    Write-Host "    5 = Configure settings (input/output paths)" -ForegroundColor DarkCyan
    Write-Host "    6 = Switch to Azure mode" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "AZURE MODE (cloud-based splitting):" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 -Mode azure                  # Interactive Azure menu (requires az CLI)" -ForegroundColor White
    Write-Host ""
    Write-Host "  Azure Menu Options:" -ForegroundColor DarkCyan
    Write-Host "    1 = Run job on current file" -ForegroundColor DarkCyan
    Write-Host "    2 = Pick different file and run" -ForegroundColor DarkCyan
    Write-Host "    3 = Check last run status" -ForegroundColor DarkCyan
    Write-Host "    4 = List output files in blob storage" -ForegroundColor DarkCyan
    Write-Host "    5 = Rebuild Docker image" -ForegroundColor DarkCyan
    Write-Host "    6 = View job logs" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "INPUT/OUTPUT:" -ForegroundColor Yellow
    Write-Host "  Local mode input:        src/pdf_extractor/Data/" -ForegroundColor DarkCyan
    Write-Host "  Local mode split output: output/split-<filename>/" -ForegroundColor DarkCyan
    Write-Host "  Local mode Excel output: output/opt_in_results-<filename>.xlsx" -ForegroundColor DarkCyan
    Write-Host "  Azure mode input:        Azure Blob container 'pdfinput'" -ForegroundColor DarkCyan
    Write-Host "  Azure mode output:       Azure Blob container 'pdfoutput'" -ForegroundColor DarkCyan
    Write-Host "  Azure note:              container jobs cannot access your local folders" -ForegroundColor DarkCyan
    Write-Host "  Job logs (saved local):  output/job-logs/<execution>-<timestamp>.log" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "SETUP (ONE-TIME):" -ForegroundColor Yellow
    Write-Host "  Local mode:" -ForegroundColor DarkCyan
    Write-Host "    py -3.11 -m venv .venv" -ForegroundColor Gray
    Write-Host "    .venv\Scripts\pip install -e `".[dev]`"" -ForegroundColor Gray
    Write-Host "  Azure mode:" -ForegroundColor DarkCyan
    Write-Host "    Download Azure CLI: https://aka.ms/installazurecliwindows" -ForegroundColor Gray
    Write-Host ""
}

function Require-LocalPrereqs {
    if (-not (Test-Path $VENV)) {
        Write-Host "ERROR: Python venv not found at .venv" -ForegroundColor Red
        Write-Host ""
        Write-Host "Set up Python environment:" -ForegroundColor Yellow
        Write-Host "  py -3.11 -m venv .venv" -ForegroundColor Cyan
        Write-Host "  .venv\Scripts\pip install -e `".[dev]`"" -ForegroundColor Cyan
        exit 1
    }
    if (-not (Test-Path $DATA_DIR)) {
        Write-Host "ERROR: Data folder not found: $DATA_DIR" -ForegroundColor Red
        exit 1
    }
}

function Require-AzureCli {
    if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: Azure CLI not found" -ForegroundColor Red
        Write-Host "Download from: https://aka.ms/installazurecliwindows" -ForegroundColor Yellow
        exit 1
    }
}

function Require-AzureAuth {
    $account = az account show --query "user.name" -o tsv 2>$null
    if (-not $account) {
        Write-Host "Logging into Azure..." -ForegroundColor Yellow
        az login | Out-Null
    }
    Write-Host "Logged in as: $account" -ForegroundColor Green
}

function Load-Config {
    if (Test-Path $Config) {
        try {
            return Get-Content $Config -Raw | ConvertFrom-Json
        } catch {
            Write-Host "Warning: Config file exists but is invalid. Using defaults." -ForegroundColor Yellow
            return $null
        }
    }
    return $null
}

function Save-Config {
    $cfg = @{
        data_dir    = $DATA_DIR
        output_dir  = $OUTPUT_DIR
        log_dir     = $LOG_DIR
        last_updated = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    }
    $cfg | ConvertTo-Json | Out-File -FilePath $Config -Encoding utf8
    Write-Host "Config saved to: $Config" -ForegroundColor Green
}

function Show-ConfigMenu {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Configuration Settings" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Current Settings:" -ForegroundColor Yellow
    Write-Host "    Input PDFs:   $DATA_DIR" -ForegroundColor DarkCyan
    Write-Host "    Output Files: $OUTPUT_DIR" -ForegroundColor DarkCyan
    Write-Host "    Job Logs:     $LOG_DIR" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "  1. Change input PDF directory" -ForegroundColor White
    Write-Host "  2. Change output directory" -ForegroundColor White
    Write-Host "  3. Reset to defaults" -ForegroundColor White
    Write-Host "  4. View config file location" -ForegroundColor White
    Write-Host "  0. Back" -ForegroundColor White
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Initialize-Dirs {
    @($OUTPUT_DIR, $LOG_DIR) | ForEach-Object {
        if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
    }
}

function Get-LocalPdfs {
    return @(Get-ChildItem -Path $DATA_DIR -Filter "*.pdf" -File | Sort-Object Name)
}

function Get-OutputSplitDir([string]$pdfName) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($pdfName)
    return Join-Path $OUTPUT_DIR ("split-" + $stem)
}

function Get-OutputExcelFile([string]$pdfName) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($pdfName)
    return Join-Path $OUTPUT_DIR ("opt_in_results-" + $stem + ".xlsx")
}

function Invoke-Split([string]$pdfPath, [string]$outputDir) {
    Write-Host ""
    Write-Host "Splitting: $(Split-Path -Leaf $pdfPath)" -ForegroundColor Cyan
    & $VENV -m pdf_extractor.cli $pdfPath --split-documents --split-output-dir $outputDir
    return $LASTEXITCODE -eq 0
}

function Invoke-OptInExtraction([string]$splitDir, [string]$excelPath) {
    Write-Host "Extracting opt-in data..." -ForegroundColor Cyan
    & $VENV -c "
from dotenv import load_dotenv
load_dotenv(override=True)
from pdf_extractor.opt_in_extractor import process_folder_to_excel
import sys
n = process_folder_to_excel('$splitDir', '$excelPath')
print(f'Processed {n} credit application(s).')
"
    return $LASTEXITCODE -eq 0
}

function Invoke-ImageBuild {
    # Stage only the files Docker needs so the ACR upload is a few MB, not 800+ MB.
    # (az acr build uploads the full directory tree before applying .dockerignore, so
    # running from a minimal staging directory is the only reliable way to keep it small.)
    $staging = Join-Path $env:TEMP "pdf-extractor-stage"
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
    New-Item -ItemType Directory -Path $staging | Out-Null

    Copy-Item -Recurse (Join-Path $REPO_ROOT "src")             $staging\
    Copy-Item           (Join-Path $REPO_ROOT "pyproject.toml") $staging\
    Copy-Item           (Join-Path $REPO_ROOT "Dockerfile")     $staging\

    $sizeMB = [math]::Round((Get-ChildItem $staging -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Host "Staging directory: $staging ($sizeMB MB)" -ForegroundColor DarkCyan

    $loginServer = az acr show --name $ACR_NAME --query "loginServer" -o tsv 2>$null
    $image = "$loginServer/pdf-extractor:latest"

    Write-Host "Submitting build to ACR..." -ForegroundColor Cyan
    # --no-wait avoids a Unicode encoding crash in Azure CLI log streaming on Windows.
    az acr build --registry $ACR_NAME --resource-group $RESOURCE_GROUP --image pdf-extractor:latest --no-wait $staging

    # Poll for completion, then re-pin the Container App job to the new image digest.
    # Without this, the job stays locked to the old image even after a rebuild.
    Write-Host "Waiting for build to complete (polling every 15s)..." -ForegroundColor Cyan
    $maxWait = 600
    $elapsed = 0
    $buildDone = $false
    while ($elapsed -lt $maxWait) {
        Start-Sleep -Seconds 15
        $elapsed += 15
        $status = az acr task list-runs --registry $ACR_NAME --top 1 --query "[0].status" -o tsv 2>$null
        Write-Host "  [$elapsed s] Build: $status" -ForegroundColor DarkCyan
        if ($status -eq "Succeeded") { $buildDone = $true; break }
        if ($status -eq "Failed" -or $status -eq "Canceled") {
            Write-Host "Build $status — job image NOT updated." -ForegroundColor Red
            return
        }
    }

    if (-not $buildDone) {
        Write-Host "Build did not complete within ${maxWait}s." -ForegroundColor Yellow
        Write-Host "Run manually when done:  az containerapp job update --name $JOB_NAME --resource-group $RESOURCE_GROUP --image $image" -ForegroundColor Yellow
        return
    }

    Write-Host "Build succeeded. Re-pinning Container App job to new image..." -ForegroundColor Cyan
    az containerapp job update --name $JOB_NAME --resource-group $RESOURCE_GROUP --image $image --output none 2>&1 | Out-Null
    Write-Host "Done — next job execution will use the new image." -ForegroundColor Green
}

function Invoke-AzureJob([string]$filename) {
    Write-Host ""
    Write-Host "Starting Azure job for: $filename" -ForegroundColor Cyan

    # Set PDF_INPUT_FILE in the job template so the ENTRYPOINT can read it.
    # Using --set-env-vars on update (not --env-vars on start) is the only reliable
    # way to inject env vars into the running container via the shell-form ENTRYPOINT.
    az containerapp job update `
        --name $JOB_NAME `
        --resource-group $RESOURCE_GROUP `
        --set-env-vars "PDF_INPUT_FILE=$filename" `
        --output none 2>&1 | Out-Null

    $execName = az containerapp job start `
        --name $JOB_NAME `
        --resource-group $RESOURCE_GROUP `
        --query name -o tsv
    
    Write-Host "Execution: $execName" -ForegroundColor Yellow
    
    # Wait for completion
    Write-Host "Waiting for job to complete..." -ForegroundColor Cyan
    $maxWait = 600
    $elapsed = 0
    
    while ($elapsed -lt $maxWait) {
        $status = az containerapp job execution show `
            --name $JOB_NAME `
            --resource-group $RESOURCE_GROUP `
            --job-execution-name $execName `
            --query "properties.status" -o tsv 2>$null
        
        if ($status -in "Succeeded", "Failed") {
            Write-Host "Status: $status" -ForegroundColor $(if ($status -eq "Succeeded") { "Green" } else { "Red" })
            break
        }
        
        Write-Host "  Status: $status (${elapsed}s)" -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        $elapsed += 10
    }

    if ($elapsed -ge $maxWait) {
        Write-Host "Job still running after 10 minutes. Check status with menu option 3." -ForegroundColor Yellow
    }

    # Save logs locally
    Write-Host "Saving logs to output/job-logs/..." -ForegroundColor Cyan
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $logFile = Join-Path $LOG_DIR "${execName}-${stamp}.log"
    
    az containerapp job logs show `
        --name $JOB_NAME `
        --resource-group $RESOURCE_GROUP `
        --execution $execName `
        --container $JOB_NAME 2>&1 | Out-File -FilePath $logFile -Encoding utf8
    
    # Append replica diagnostics
    $replicas = az containerapp job replica list `
        --name $JOB_NAME `
        --resource-group $RESOURCE_GROUP `
        --execution $execName -o json 2>$null
    
    Add-Content -Path $logFile -Value "`n`n===== REPLICA DIAGNOSTICS =====`n$replicas"
    
    Write-Host "Logs saved: $logFile" -ForegroundColor Green
    return $status -eq "Succeeded"
}

function Show-LocalMenu {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  pdf-extractor - LOCAL MODE" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Split PDFs on your machine (fast, no Azure needed)" -ForegroundColor White
    Write-Host ""
    Write-Host "  Input:   $DATA_DIR" -ForegroundColor DarkCyan
    Write-Host "  Output:  $OUTPUT_DIR/split-*/" -ForegroundColor DarkCyan
    Write-Host "  Logs:    $LOG_DIR/" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "  1. Process one PDF" -ForegroundColor White
    Write-Host "  2. Process all PDFs" -ForegroundColor White
    Write-Host "  3. Process by pattern (e.g., 6* or Sample-*)" -ForegroundColor White
    Write-Host "  4. View outputs" -ForegroundColor White
    Write-Host "  5. Configure settings (input/output paths)" -ForegroundColor White
    Write-Host "  6. Switch to Azure mode" -ForegroundColor White
    Write-Host "  0. Exit" -ForegroundColor White
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Show-AzureMenu {
    Clear-Host
    $current = az containerapp job show `
        --name $JOB_NAME `
        --resource-group $RESOURCE_GROUP `
        --query "properties.template.containers[0].env[?name=='PDF_INPUT_FILE'].value | [0]" -o tsv 2>$null

    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  pdf-extractor - AZURE MODE" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Submit PDF splitting jobs to Azure Container Apps" -ForegroundColor White
    Write-Host ""
    Write-Host "  Input:   Azure Blob (pdfinput)" -ForegroundColor DarkCyan
    Write-Host "  Output:  Azure Blob (pdfoutput) + Local Excel + Logs" -ForegroundColor DarkCyan
    Write-Host "  Logs:    $LOG_DIR/" -ForegroundColor DarkCyan
    Write-Host "  Current: $current" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  1. Run job on current file" -ForegroundColor White
    Write-Host "  2. Pick different file and run" -ForegroundColor White
    Write-Host "  3. Check last run status" -ForegroundColor White
    Write-Host "  4. List output files in blob storage" -ForegroundColor White
    Write-Host "  5. Rebuild Docker image" -ForegroundColor White
    Write-Host "  6. View job logs" -ForegroundColor White
    Write-Host "  0. Exit" -ForegroundColor White
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Show-LocalOutputs {
    Write-Host ""
    Write-Host "Split folders:" -ForegroundColor Yellow
    Get-ChildItem -Path $OUTPUT_DIR -Directory -Filter "split-*" 2>$null | ForEach-Object {
        Write-Host "  $_"
    }
    
    Write-Host ""
    Write-Host "Excel files:" -ForegroundColor Yellow
    Get-ChildItem -Path $OUTPUT_DIR -Filter "opt_in_results-*.xlsx" 2>$null | ForEach-Object {
        Write-Host "  $_"
    }
    
    Write-Host ""
    Write-Host "Job logs (recent):" -ForegroundColor Yellow
    Get-ChildItem -Path $LOG_DIR -Filter "*.log" 2>$null | Sort-Object LastWriteTime -Descending | Select-Object -First 3 | ForEach-Object {
        Write-Host "  $_"
    }
}

function Process-PatternSelect([string]$pattern, [bool]$withOptIn) {
    $pdfs = Get-LocalPdfs
    # Strip .pdf suffix if the user included it — the regex appends \.pdf$ itself.
    if ($pattern -match '\.pdf$') { $pattern = $pattern -replace '\.pdf$', '' }
    # Match pattern: convert wildcard to regex (6* → 6.*, Sample-* → Sample-.*)
    $regexPattern = "^" + [regex]::Escape($pattern).Replace("\*", ".*") + "\.pdf$"
    $matched = $pdfs | Where-Object { $_.Name -match $regexPattern }
    
    if ($matched.Count -eq 0) {
        Write-Host "No files match pattern '$pattern'" -ForegroundColor Yellow
        return
    }
    
    Write-Host ""
    Write-Host "Matched files:" -ForegroundColor Green
    $matched | ForEach-Object { Write-Host "  - $($_.Name)" }
    Write-Host ""
    
    $confirm = Read-Host "Process these $($matched.Count) file(s)? (y/n)"
    if ($confirm -ne "y") {
        Write-Host "Cancelled." -ForegroundColor Yellow
        return
    }
    
    foreach ($pdf in $matched) {
        $outDir = Get-OutputSplitDir $pdf.Name
        $ok = Invoke-Split $pdf.FullName $outDir
        if ($ok -and $withOptIn) {
            $excelPath = Get-OutputExcelFile $pdf.Name
            Invoke-OptInExtraction $outDir $excelPath | Out-Null
        }
    }
}

# ==================== Main ====================

Initialize-Dirs

# Load configuration
$cfg = Load-Config
if ($cfg) {
    if ($cfg.data_dir -and (Test-Path $cfg.data_dir)) { $DATA_DIR = $cfg.data_dir }
    if ($cfg.output_dir) { $OUTPUT_DIR = $cfg.output_dir; $LOG_DIR = Join-Path $OUTPUT_DIR "job-logs" }
    if ($cfg.log_dir -and (Test-Path (Split-Path $cfg.log_dir))) { $LOG_DIR = $cfg.log_dir }
    Initialize-Dirs  # Re-initialize with new paths
}

# Check for help flag
if ($Help) {
    Show-Help
    Pop-Location
    exit 0
}

# Mode selection
if (-not $Mode) {
    Clear-Host
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  pdf-extractor" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1. Local   - Split on your machine (minutes)" -ForegroundColor White
    Write-Host "  2. Azure   - Cloud-based splitting (slower, durable)" -ForegroundColor White
    Write-Host "  0. Exit" -ForegroundColor White
    Write-Host ""
    $choice = Read-Host "Choose mode"
    
    if ($choice -eq "1") { $Mode = "local" }
    elseif ($choice -eq "2") { $Mode = "azure" }
    else { Write-Host "Goodbye."; Pop-Location; exit 0 }
}

# ===== LOCAL MODE =====
if ($Mode -eq "local") {
    Require-LocalPrereqs
    
    if ($File) {
        # Non-interactive: single file
        $pdf = Get-LocalPdfs | Where-Object { $_.Name -eq $File } | Select-Object -First 1
        if (-not $pdf) {
            Write-Host "ERROR: File '$File' not found in $DATA_DIR" -ForegroundColor Red
            Pop-Location
            exit 1
        }
        $outDir = Get-OutputSplitDir $pdf.Name
        $ok = Invoke-Split $pdf.FullName $outDir
        if ($ok -and $OptIn) {
            $excelPath = Get-OutputExcelFile $pdf.Name
            Invoke-OptInExtraction $outDir $excelPath | Out-Null
        }
        Pop-Location
        exit $(if ($ok) { 0 } else { 1 })
    }
    
    # Interactive menu
    while ($true) {
        Show-LocalMenu
        $choice = Read-Host "Enter choice"
        
        switch ($choice) {
            "1" {
                $pdfs = Get-LocalPdfs
                if ($pdfs.Count -eq 0) {
                    Write-Host "No PDFs found in $DATA_DIR" -ForegroundColor Yellow
                    continue
                }
                
                Write-Host ""
                for ($i = 0; $i -lt $pdfs.Count; $i++) {
                    Write-Host "  $($i+1). $($pdfs[$i].Name)"
                }
                
                $idx = Read-Host "Select PDF (number)"
                if ($idx -match "^\d+$" -and [int]$idx -ge 1 -and [int]$idx -le $pdfs.Count) {
                    $pdf = $pdfs[[int]$idx - 1]
                    $outDir = Get-OutputSplitDir $pdf.Name
                    $ok = Invoke-Split $pdf.FullName $outDir
                    
                    if ($ok) {
                        $resp = Read-Host "Extract opt-in data to Excel? (y/n)"
                        if ($resp -eq "y") {
                            $excelPath = Get-OutputExcelFile $pdf.Name
                            Invoke-OptInExtraction $outDir $excelPath | Out-Null
                        }
                    }
                }
                Read-Host "Press Enter to continue"
            }
            
            "2" {
                $pdfs = Get-LocalPdfs
                if ($pdfs.Count -eq 0) {
                    Write-Host "No PDFs found." -ForegroundColor Yellow
                    continue
                }
                
                $resp = Read-Host "Include opt-in extraction? (y/n)"
                $withOpt = $resp -eq "y"
                
                foreach ($pdf in $pdfs) {
                    $outDir = Get-OutputSplitDir $pdf.Name
                    $ok = Invoke-Split $pdf.FullName $outDir
                    if ($ok -and $withOpt) {
                        $excelPath = Get-OutputExcelFile $pdf.Name
                        Invoke-OptInExtraction $outDir $excelPath | Out-Null
                    }
                }
                Read-Host "Press Enter to continue"
            }
            
            "3" {
                $pattern = Read-Host "Enter file pattern (e.g., 6* or Sample-*)"
                if (-not $pattern) {
                    Write-Host "No pattern entered." -ForegroundColor Yellow
                    Read-Host "Press Enter to continue"
                    continue
                }
                
                $resp = Read-Host "Include opt-in extraction? (y/n)"
                $withOpt = $resp -eq "y"
                
                Process-PatternSelect $pattern $withOpt
                Read-Host "Press Enter to continue"
            }
            
            "4" {
                Show-LocalOutputs
                Read-Host "Press Enter to continue"
            }
            
            "5" {
                # Configuration menu
                while ($true) {
                    Show-ConfigMenu
                    $cfgChoice = Read-Host "Enter choice"
                    
                    switch ($cfgChoice) {
                        "1" {
                            $newPath = Read-Host "Enter full path to input PDF directory"
                            if ($newPath -and (Test-Path $newPath)) {
                                $DATA_DIR = $newPath
                                Write-Host "Input directory updated to: $DATA_DIR" -ForegroundColor Green
                                Save-Config
                            } else {
                                Write-Host "Path not found or invalid." -ForegroundColor Red
                            }
                            Read-Host "Press Enter to continue"
                        }
                        
                        "2" {
                            $newPath = Read-Host "Enter full path to output directory"
                            if ($newPath) {
                                $OUTPUT_DIR = $newPath
                                $LOG_DIR = Join-Path $OUTPUT_DIR "job-logs"
                                Initialize-Dirs
                                Write-Host "Output directory updated to: $OUTPUT_DIR" -ForegroundColor Green
                                Save-Config
                            } else {
                                Write-Host "Invalid path." -ForegroundColor Red
                            }
                            Read-Host "Press Enter to continue"
                        }
                        
                        "3" {
                            $DATA_DIR = $DATA_DIR_DEFAULT
                            $OUTPUT_DIR = $OUTPUT_DIR_DEFAULT
                            $LOG_DIR = $LOG_DIR_DEFAULT
                            Initialize-Dirs
                            Write-Host "Settings reset to defaults." -ForegroundColor Green
                            Save-Config
                            Read-Host "Press Enter to continue"
                        }
                        
                        "4" {
                            Write-Host ""
                            Write-Host "Config file location: $Config" -ForegroundColor Yellow
                            if (Test-Path $Config) {
                                Write-Host "Config file exists." -ForegroundColor Green
                                Write-Host ""
                                Write-Host "Content:" -ForegroundColor Yellow
                                Get-Content $Config | Write-Host
                            } else {
                                Write-Host "Config file not created yet." -ForegroundColor Yellow
                            }
                            Write-Host ""
                            Read-Host "Press Enter to continue"
                        }
                        
                        "0" {
                            break
                        }
                        
                        default {
                            Write-Host "Invalid choice (0-4)." -ForegroundColor Red
                            Read-Host "Press Enter to continue"
                        }
                    }
                }
            }
            
            "6" {
                if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
                    Write-Host ""
                    Write-Host "ERROR: Azure CLI not found. Cannot switch to Azure mode." -ForegroundColor Red
                    Write-Host "Download from: https://aka.ms/installazurecliwindows" -ForegroundColor Yellow
                    Read-Host "Press Enter to continue"
                } else {
                    $Mode = "azure"
                    break
                }
            }
            
            "0" {
                Write-Host "Goodbye." -ForegroundColor Cyan
                Pop-Location
                exit 0
            }
            
            default {
                Write-Host "Invalid choice (0-6)." -ForegroundColor Red
                Read-Host "Press Enter to continue"
            }
        }
    }
}

# ===== AZURE MODE =====
if ($Mode -eq "azure") {
    Require-AzureCli
    Require-AzureAuth
    
    while ($true) {
        Show-AzureMenu
        $choice = Read-Host "Enter choice"
        
        switch ($choice) {
            "1" {
                $current = az containerapp job show `
                    --name $JOB_NAME `
                    --resource-group $RESOURCE_GROUP `
                    --query "properties.template.containers[0].env[?name=='PDF_INPUT_FILE'].value | [0]" -o tsv 2>$null
                Invoke-AzureJob $current | Out-Null
                Read-Host "Press Enter to continue"
            }
            
            "2" {
                Write-Host ""
                az storage blob list `
                    --account-name $STORAGE_ACCOUNT `
                    --container-name $INPUT_CONTAINER `
                    --query "[].name" -o tsv --auth-mode key 2>$null
                
                $newFile = Read-Host "`nEnter blob name to process"
                if ($newFile) {
                    Invoke-AzureJob $newFile | Out-Null
                }
                Read-Host "Press Enter to continue"
            }
            
            "3" {
                Write-Host ""
                $q1 = "[0].name"
                $execName = az containerapp job execution list `
                    --name $JOB_NAME `
                    --resource-group $RESOURCE_GROUP `
                    --query $q1 -o tsv 2>$null
                
                $q2 = "[0].properties.status"
                $status = az containerapp job execution list `
                    --name $JOB_NAME `
                    --resource-group $RESOURCE_GROUP `
                    --query $q2 -o tsv 2>$null
                
                Write-Host "Last execution: $execName" -ForegroundColor Yellow
                Write-Host "Status: $status" -ForegroundColor $(if ($status -eq "Succeeded") { "Green" } else { "Red" })
                Read-Host "Press Enter to continue"
            }
            
            "4" {
                Write-Host ""
                az storage blob list `
                    --account-name $STORAGE_ACCOUNT `
                    --container-name $OUTPUT_CONTAINER `
                    --query "[].name" -o tsv --auth-mode key 2>$null
                Read-Host "Press Enter to continue"
            }
            
            "5" {
                Write-Host ""
                Invoke-ImageBuild
                Read-Host "Press Enter to continue"
            }
            
            "6" {
                Write-Host ""
                $q = "[0].name"
                $execName = az containerapp job execution list `
                    --name $JOB_NAME `
                    --resource-group $RESOURCE_GROUP `
                    --query $q -o tsv 2>$null
                
                if ($execName) {
                    Write-Host "Latest execution: $execName`n" -ForegroundColor Yellow
                    az containerapp job logs show `
                        --name $JOB_NAME `
                        --resource-group $RESOURCE_GROUP `
                        --execution $execName `
                        --container $JOB_NAME `
                        --tail 50 2>$null
                } else {
                    Write-Host "No executions found." -ForegroundColor Yellow
                }
                Read-Host "Press Enter to continue"
            }
            
            "0" {
                Write-Host "Goodbye." -ForegroundColor Cyan
                Pop-Location
                exit 0
            }
            
            default {
                Write-Host "Invalid choice (0-6)." -ForegroundColor Red
                Read-Host "Press Enter to continue"
            }
        }
    }
}

Pop-Location
