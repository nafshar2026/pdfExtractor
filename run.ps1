# ============================================================
# pdf-extractor — Azure operations menu
# Save this file anywhere on your machine and run it with:
#   .\run.ps1
#
# Prerequisites:
#   - Azure CLI installed (ask admin if not available)
#   - PIM role activated in the Azure Portal (expires every 8 hours)
#     Portal → Privileged Identity Management → My Roles → Activate
# ============================================================

# ── Azure resource names (do not change) ────────────────────────────────────
$RESOURCE_GROUP  = "nader-test-rag"
$JOB_NAME        = "pdf-extractor-job"
$ACR_NAME        = "NaderContainerRegistry"
$STORAGE_ACCOUNT = "naderblob02"
$INPUT_CONTAINER = "pdfinput"
$OUTPUT_CONTAINER= "pdfoutput"
$ACR_SERVER      = "nadercontainerregistry-cuhjhrfcgpckd0hh.azurecr.io"

# ── Ensure az CLI is on the PATH ─────────────────────────────────────────────
$azPath = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
if (Test-Path $azPath) {
    $env:PATH += ";$azPath"
}

# ── Helper: check az is available ───────────────────────────────────────────
function Assert-AzCli {
    if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "ERROR: Azure CLI (az) not found." -ForegroundColor Red
        Write-Host "Ask your admin to install it from: https://aka.ms/installazurecliwindows" -ForegroundColor Yellow
        exit 1
    }
}

# ── Helper: ensure logged in ─────────────────────────────────────────────────
function Assert-LoggedIn {
    $account = az account show --query "user.name" -o tsv 2>$null
    if (-not $account) {
        Write-Host ""
        Write-Host "Not logged in. Opening browser for login..." -ForegroundColor Yellow
        az login | Out-Null
    } else {
        Write-Host "Logged in as: $account" -ForegroundColor Green
    }
}

# ── Helper: get current job target file ──────────────────────────────────────
function Get-CurrentTarget {
    $target = az containerapp job show --name $JOB_NAME --resource-group $RESOURCE_GROUP `
        --query "properties.template.containers[0].args[0]" -o tsv 2>$null
    return $target
}

# ── Helper: poll job until done ───────────────────────────────────────────────
function Wait-ForJob {
    Write-Host ""
    Write-Host "Waiting for job to complete (Ctrl+C to stop watching)..." -ForegroundColor Cyan
    while ($true) {
        $result = az containerapp job execution list --name $JOB_NAME --resource-group $RESOURCE_GROUP `
            --query "[0].{name:name, status:properties.status}" -o table 2>$null
        Write-Host $result
        $status = az containerapp job execution list --name $JOB_NAME --resource-group $RESOURCE_GROUP `
            --query "[0].properties.status" -o tsv 2>$null
        if ($status -eq "Succeeded") {
            Write-Host ""
            Write-Host "Job completed successfully." -ForegroundColor Green
            break
        } elseif ($status -eq "Failed") {
            Write-Host ""
            Write-Host "Job failed. Choose option 6 from the menu to view logs." -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 20
    }
}

# ── Menu ─────────────────────────────────────────────────────────────────────
Assert-AzCli
Assert-LoggedIn

while ($true) {
    $current = Get-CurrentTarget
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  pdf-extractor — Azure Operations" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  Current target file: $current" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  1. Run job on current file ($current)"
    Write-Host "  2. Change target file then run"
    Write-Host "  3. Check status of last run"
    Write-Host "  4. List output files in blob storage"
    Write-Host "  5. Rebuild Docker image (after code changes)"
    Write-Host "  6. Show logs from last run"
    Write-Host "  0. Exit"
    Write-Host ""

    $choice = Read-Host "Enter choice"

    switch ($choice) {

        "1" {
            # Start the job on the current target file
            Write-Host ""
            Write-Host "Starting job for '$current'..." -ForegroundColor Cyan
            az containerapp job start --name $JOB_NAME --resource-group $RESOURCE_GROUP | Out-Null
            Write-Host "Job started. First run takes 3-5 min (model download). Subsequent runs ~1-2 min."
            Wait-ForJob
        }

        "2" {
            # Change the target file and run
            Write-Host ""
            Write-Host "Files available in '$INPUT_CONTAINER':" -ForegroundColor Cyan
            az storage blob list --account-name $STORAGE_ACCOUNT --container-name $INPUT_CONTAINER `
                --query "[].name" -o tsv --auth-mode key
            Write-Host ""
            $newFile = Read-Host "Enter blob name to process (e.g. Sample-2.pdf)"
            if ($newFile) {
                az containerapp job update --name $JOB_NAME --resource-group $RESOURCE_GROUP `
                    --args $newFile | Out-Null
                Write-Host "Target updated to '$newFile'. Starting job..." -ForegroundColor Cyan
                az containerapp job start --name $JOB_NAME --resource-group $RESOURCE_GROUP | Out-Null
                Wait-ForJob
            }
        }

        "3" {
            # Show status of the most recent execution
            Write-Host ""
            Write-Host "Last execution status:" -ForegroundColor Cyan
            az containerapp job execution list --name $JOB_NAME --resource-group $RESOURCE_GROUP `
                --query "[0].{name:name, status:properties.status, started:properties.startTime}" -o table
        }

        "4" {
            # List all output files in the output blob container
            Write-Host ""
            Write-Host "Output files in '$OUTPUT_CONTAINER':" -ForegroundColor Cyan
            az storage blob list --account-name $STORAGE_ACCOUNT --container-name $OUTPUT_CONTAINER `
                --query "[].name" -o tsv --auth-mode key
        }

        "5" {
            # Rebuild the Docker image from the latest code in the current directory
            Write-Host ""
            $repoPath = Read-Host "Enter the full path to your local pdfExtractor folder (or press Enter to use current directory)"
            if ($repoPath) { Set-Location $repoPath }
            Write-Host "Building image... this takes ~3 minutes." -ForegroundColor Cyan
            az acr build --registry $ACR_NAME --resource-group $RESOURCE_GROUP `
                --image pdf-extractor:latest .
        }

        "6" {
            # Show logs from the most recent execution
            Write-Host ""
            $execName = az containerapp job execution list --name $JOB_NAME `
                --resource-group $RESOURCE_GROUP --query "[0].name" -o tsv
            Write-Host "Fetching logs for '$execName'..." -ForegroundColor Cyan
            az containerapp job logs show --name $JOB_NAME --resource-group $RESOURCE_GROUP `
                --execution $execName --container $JOB_NAME --tail 100
        }

        "0" {
            Write-Host "Goodbye." -ForegroundColor Cyan
            exit 0
        }

        default {
            Write-Host "Invalid choice. Please enter 0-6." -ForegroundColor Red
        }
    }
}
