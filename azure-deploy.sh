#!/usr/bin/env bash
# Azure setup for pdf-extractor — run steps manually one at a time.
# Prerequisites: az CLI logged in.

RESOURCE_GROUP="nader-test-rag"
LOCATION="centralus"
ACR_NAME="NaderContainerRegistry"
STORAGE_ACCOUNT="naderblob02"

# ── Step 1: Create resource group (skip if it already exists) ────────────────
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

# ── Step 2: Create Azure Container Registry ──────────────────────────────────
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true

# ── Step 3: Build and push image to ACR (runs the Docker build in the cloud) ─
# This avoids needing Docker Desktop locally for the push step.
az acr build \
    --registry "$ACR_NAME" \
    --image pdf-extractor:latest \
    .

# ── Step 4: Get the ACR login server and admin password ──────────────────────
ACR_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

# ── Step 5: Get the storage connection string ─────────────────────────────────
CONN_STR=$(az storage account show-connection-string \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --query connectionString -o tsv)

# ── Step 6: Create Container Apps environment ─────────────────────────────────
az containerapp env create \
    --name "pdf-extractor-env" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION"

# ── Step 7: Create a Container Apps Job (single-file run) ────────────────────
# Change the --args value to process a different blob.
az containerapp job create \
    --name "pdf-extractor-job" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "pdf-extractor-env" \
    --trigger-type Manual \
    --replica-timeout 3600 \
    --replica-retry-limit 0 \
    --parallelism 1 \
    --replica-completion-count 1 \
    --image "$ACR_SERVER/pdf-extractor:latest" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_NAME" \
    --registry-password "$ACR_PASSWORD" \
    --cpu 2 --memory 4Gi \
    --args "RO-1.pdf" \
    --env-vars \
        "AZURE_STORAGE_CONNECTION_STRING=secretref:conn-str" \
        "AZURE_INPUT_CONTAINER=pdfinput" \
        "AZURE_OUTPUT_CONTAINER=pdfoutput" \
    --secrets "conn-str=$CONN_STR"

# ── Step 8: Start a job execution ────────────────────────────────────────────
az containerapp job start \
    --name "pdf-extractor-job" \
    --resource-group "$RESOURCE_GROUP"

# ── Step 9: Watch the logs ───────────────────────────────────────────────────
# Get the execution name first, then stream its logs.
EXECUTION=$(az containerapp job execution list \
    --name "pdf-extractor-job" \
    --resource-group "$RESOURCE_GROUP" \
    --query "[0].name" -o tsv)

az containerapp logs show \
    --name "pdf-extractor-job" \
    --resource-group "$RESOURCE_GROUP" \
    --execution "$EXECUTION" \
    --follow
