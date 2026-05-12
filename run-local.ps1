# ===========================================================================
# pdf-extractor - Local operations menu (no Azure)
# ===========================================================================
#
# WHAT THIS DOES:
#   Splits multi-document PDFs on your local machine and optionally runs
#   opt-in extraction to Excel.
#
# HOW TO RUN:
#   PowerShell -> from repo root:
#     .\run-local.ps1
#
# NOTES:
#   - Uses local files under src/pdf_extractor/Data and output/
#   - Does not call Azure APIs or Azure Container Apps
#   - Uses .venv\Scripts\python.exe
#
# OPTIONAL NON-INTERACTIVE MODE:
#   Run one PDF:
#     .\run-local.ps1 -FileName Sample-1.pdf
#
#   Run all PDFs:
#     .\run-local.ps1 -All
#
#   Include opt-in extraction in either mode:
#     .\run-local.ps1 -FileName Sample-1.pdf -WithOptIn
#     .\run-local.ps1 -All -WithOptIn
# ===========================================================================

param(
    [string]$FileName,
    [switch]$All,
    [switch]$WithOptIn
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$SCRIPT_ROOT = $PSScriptRoot
if (-not $SCRIPT_ROOT) {
    $SCRIPT_ROOT = (Get-Location).Path
}
Push-Location $SCRIPT_ROOT

$PYTHON_EXE = Join-Path $SCRIPT_ROOT ".venv\Scripts\python.exe"
$DATA_DIR = Join-Path $SCRIPT_ROOT "src\pdf_extractor\Data"
$OUTPUT_DIR = Join-Path $SCRIPT_ROOT "output"

function Assert-Prereqs {
    if (-not (Test-Path $PYTHON_EXE)) {
        Write-Host "" 
        Write-Host "ERROR: Python venv not found at .venv\Scripts\python.exe" -ForegroundColor Red
        Write-Host "Create it first:" -ForegroundColor Yellow
        Write-Host "  py -3.11 -m venv .venv" -ForegroundColor Yellow
        Write-Host "  .venv\Scripts\pip install -e \".[dev]\"" -ForegroundColor Yellow
        exit 1
    }

    if (-not (Test-Path $DATA_DIR)) {
        Write-Host "" 
        Write-Host "ERROR: Data folder not found: $DATA_DIR" -ForegroundColor Red
        exit 1
    }

    if (-not (Test-Path $OUTPUT_DIR)) {
        New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
    }
}

function Get-PdfFiles {
    return @(Get-ChildItem -Path $DATA_DIR -Filter "*.pdf" -File | Sort-Object Name)
}

function Get-PdfByName([string]$name) {
    if (-not $name) {
        return $null
    }
    $pdfs = Get-PdfFiles
    return $pdfs | Where-Object { $_.Name -ieq $name } | Select-Object -First 1
}

function Get-DefaultTarget {
    $pdfs = Get-PdfFiles
    if ($pdfs.Count -eq 0) {
        return $null
    }

    $sample1 = $pdfs | Where-Object { $_.Name -ieq "Sample-1.pdf" } | Select-Object -First 1
    if ($sample1) {
        return $sample1
    }
    return $pdfs[0]
}

function Get-SplitDirForPdf([System.IO.FileInfo]$pdf) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($pdf.Name)
    return Join-Path $OUTPUT_DIR ("split-" + $stem)
}

function Invoke-SplitOne([System.IO.FileInfo]$pdf) {
    $splitDir = Get-SplitDirForPdf -pdf $pdf
    Write-Host "" 
    Write-Host "Splitting $($pdf.Name) ..." -ForegroundColor Cyan
    Write-Host "Output folder: $splitDir" -ForegroundColor DarkCyan

    & $PYTHON_EXE -m pdf_extractor.cli $pdf.FullName --split-documents --split-output-dir $splitDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Split failed for $($pdf.Name) (exit code $LASTEXITCODE)." -ForegroundColor Red
        return $false
    }

    Write-Host "Split complete." -ForegroundColor Green
    return $true
}

function Invoke-OptInForSplitDir([string]$splitDir, [string]$xlsxPath) {
    Write-Host "" 
    Write-Host "Running opt-in extraction ..." -ForegroundColor Cyan
    Write-Host "Input folder: $splitDir" -ForegroundColor DarkCyan
    Write-Host "Excel output: $xlsxPath" -ForegroundColor DarkCyan

    & $PYTHON_EXE -c "from dotenv import load_dotenv; load_dotenv(override=True); from pdf_extractor.opt_in_extractor import process_folder_to_excel; import sys; n=process_folder_to_excel(sys.argv[1], sys.argv[2]); print(f'Processed {n} credit application(s).')" $splitDir $xlsxPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Opt-in extraction failed (exit code $LASTEXITCODE)." -ForegroundColor Red
        return $false
    }

    Write-Host "Opt-in extraction complete." -ForegroundColor Green
    return $true
}

function Invoke-AllSplits([switch]$WithOptIn) {
    $pdfs = Get-PdfFiles
    if ($pdfs.Count -eq 0) {
        Write-Host "No PDFs found in $DATA_DIR" -ForegroundColor Yellow
        return
    }

    Write-Host "" 
    Write-Host "Found $($pdfs.Count) PDF(s)." -ForegroundColor Cyan

    foreach ($pdf in $pdfs) {
        $ok = Invoke-SplitOne -pdf $pdf
        if (-not $ok) {
            continue
        }

        if ($WithOptIn) {
            $stem = [System.IO.Path]::GetFileNameWithoutExtension($pdf.Name)
            $splitDir = Get-SplitDirForPdf -pdf $pdf
            $xlsx = Join-Path $OUTPUT_DIR ("opt_in_results-" + $stem + ".xlsx")
            Invoke-OptInForSplitDir -splitDir $splitDir -xlsxPath $xlsx | Out-Null
        }
    }
}

function Show-Outputs {
    Write-Host "" 
    Write-Host "Output folders:" -ForegroundColor Cyan
    Get-ChildItem -Path $OUTPUT_DIR -Directory | Sort-Object Name | ForEach-Object {
        Write-Host ("  " + $_.Name)
    }

    Write-Host "" 
    Write-Host "Excel files:" -ForegroundColor Cyan
    Get-ChildItem -Path $OUTPUT_DIR -Filter "*.xlsx" -File | Sort-Object Name | ForEach-Object {
        Write-Host ("  " + $_.Name)
    }
}

# Start
Assert-Prereqs

if ($All -and $FileName) {
    Write-Host "ERROR: Use either -All or -FileName, not both." -ForegroundColor Red
    Pop-Location
    exit 1
}

if ($All) {
    Invoke-AllSplits -WithOptIn:$WithOptIn
    Pop-Location
    exit 0
}

if ($FileName) {
    $single = Get-PdfByName -name $FileName
    if (-not $single) {
        Write-Host "ERROR: File '$FileName' not found under $DATA_DIR" -ForegroundColor Red
        Write-Host "Available PDFs:" -ForegroundColor Yellow
        Get-PdfFiles | ForEach-Object { Write-Host ("  " + $_.Name) }
        Pop-Location
        exit 1
    }

    $ok = Invoke-SplitOne -pdf $single
    if ($ok -and $WithOptIn) {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($single.Name)
        $splitDir = Get-SplitDirForPdf -pdf $single
        $xlsx = Join-Path $OUTPUT_DIR ("opt_in_results-" + $stem + ".xlsx")
        Invoke-OptInForSplitDir -splitDir $splitDir -xlsxPath $xlsx | Out-Null
    }

    Pop-Location
    if ($ok) {
        exit 0
    }
    exit 1
}

$currentTarget = Get-DefaultTarget

Clear-Host
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  pdf-extractor - Local PDF Operations" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  This script runs locally only (no Azure)." -ForegroundColor White
Write-Host "  Input  : src/pdf_extractor/Data/*.pdf" -ForegroundColor White
Write-Host "  Output : output/split-*/ and output/*.xlsx" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to continue"

while ($true) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  pdf-extractor - Local Operations" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan

    if ($currentTarget) {
        Write-Host "  Current target file: $($currentTarget.Name)" -ForegroundColor Yellow
    } else {
        Write-Host "  Current target file: (none found)" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "  1. Split current target"
    Write-Host "  2. Change target file"
    Write-Host "  3. Split current target + run opt-in extraction"
    Write-Host "  4. Split all PDFs in Data folder"
    Write-Host "  5. Split all PDFs + run opt-in extraction"
    Write-Host "  6. List local output folders/files"
    Write-Host "  0. Exit"
    Write-Host ""

    $choice = Read-Host "Enter choice"

    switch ($choice) {
        "1" {
            if (-not $currentTarget) {
                Write-Host "No target file selected." -ForegroundColor Red
                break
            }
            Invoke-SplitOne -pdf $currentTarget | Out-Null
        }

        "2" {
            $pdfs = Get-PdfFiles
            if ($pdfs.Count -eq 0) {
                Write-Host "No PDFs found in $DATA_DIR" -ForegroundColor Yellow
                break
            }

            Write-Host ""
            Write-Host "Available PDFs:" -ForegroundColor Cyan
            for ($i = 0; $i -lt $pdfs.Count; $i++) {
                Write-Host ("  " + ($i + 1) + ". " + $pdfs[$i].Name)
            }

            $idxRaw = Read-Host "Enter number"
            $idx = 0
            if ([int]::TryParse($idxRaw, [ref]$idx) -and $idx -ge 1 -and $idx -le $pdfs.Count) {
                $currentTarget = $pdfs[$idx - 1]
                Write-Host "Target updated to $($currentTarget.Name)." -ForegroundColor Green
            } else {
                Write-Host "Invalid selection." -ForegroundColor Red
            }
        }

        "3" {
            if (-not $currentTarget) {
                Write-Host "No target file selected." -ForegroundColor Red
                break
            }

            $ok = Invoke-SplitOne -pdf $currentTarget
            if ($ok) {
                $stem = [System.IO.Path]::GetFileNameWithoutExtension($currentTarget.Name)
                $splitDir = Get-SplitDirForPdf -pdf $currentTarget
                $xlsx = Join-Path $OUTPUT_DIR ("opt_in_results-" + $stem + ".xlsx")
                Invoke-OptInForSplitDir -splitDir $splitDir -xlsxPath $xlsx | Out-Null
            }
        }

        "4" {
            Invoke-AllSplits
        }

        "5" {
            Invoke-AllSplits -WithOptIn
        }

        "6" {
            Show-Outputs
        }

        "0" {
            Write-Host "Goodbye." -ForegroundColor Cyan
            Pop-Location
            exit 0
        }

        default {
            Write-Host "Invalid choice. Please enter 0-6." -ForegroundColor Red
        }
    }
}
