# ============================================================
# pdf-extractor — Azure operational commands
# ============================================================
# BEFORE RUNNING ANY COMMAND:
#   1. Activate your PIM role in the Azure Portal (expires every 8 hours)
#   2. Open a new terminal and add az CLI to the PATH:
#        $env:PATH += ";C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
#   3. Log in:
#        az login
# ============================================================


# ── Rebuild & push the Docker image after a code change ─────────────────────
# Run this whenever you change Python code and want the Azure job to pick it up.
# The build runs in the cloud — no Docker Desktop needed locally.
# Takes ~3 minutes.
#
# az acr build --registry NaderContainerRegistry --resource-group nader-test-rag --image pdf-extractor:latest .


# ── Run the job on a single file ─────────────────────────────────────────────
# Pulls the named PDF from the 'pdfinput' blob container, splits it into one
# PDF per logical document, and uploads the results to 'pdfoutput/<filename>/'.
# The default file is RO-1.pdf (set when the job was created).
# To process a different file, change --args in the job first (see below).
# Takes 3-5 minutes on first run (OCR model download), ~1-2 minutes after.
#
# az containerapp job start --name pdf-extractor-job --resource-group nader-test-rag


# ── Poll job status until it finishes ────────────────────────────────────────
# Checks every 20 seconds and prints Running / Succeeded / Failed.
# Press Ctrl+C to stop polling.
#
# while ($true) {
#     az containerapp job execution list --name pdf-extractor-job --resource-group nader-test-rag --query "[0].{name:name, status:properties.status}" -o table
#     Start-Sleep -Seconds 20
# }


# ── List output files after a successful run ─────────────────────────────────
# Shows all split PDFs currently in the output blob container.
#
# az storage blob list --account-name naderblob02 --container-name pdfoutput --query "[].name" -o tsv --auth-mode key


# ── Change which file the job processes ──────────────────────────────────────
# Update the --args value to the blob name you want to process, then start
# the job as normal.  The file must exist in the 'pdfinput' container.
#
# az containerapp job update --name pdf-extractor-job --resource-group nader-test-rag --args "Sample-2.pdf"
# az containerapp job start  --name pdf-extractor-job --resource-group nader-test-rag


# ── Get logs from the last execution (useful when a run fails) ───────────────
# Replace <execution-name> with the name from the poll status output above.
#
# az containerapp job logs show --name pdf-extractor-job --resource-group nader-test-rag --execution <execution-name> --container pdf-extractor-job --tail 100


# ── Azure resource reference ─────────────────────────────────────────────────
# Resource group:       nader-test-rag
# Container registry:   NaderContainerRegistry
# Container App Job:    pdf-extractor-job
# Container App Env:    pdf-extractor-env
# Storage account:      naderblob02
# Input container:      pdfinput
# Output container:     pdfoutput
# Registry login server: nadercontainerregistry-cuhjhrfcgpckd0hh.azurecr.io
