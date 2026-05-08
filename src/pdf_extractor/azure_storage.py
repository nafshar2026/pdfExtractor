"""Azure Blob Storage helpers for downloading input PDFs and uploading split output."""

from __future__ import annotations

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient


def _get_client() -> BlobServiceClient:
    """Return a BlobServiceClient using the connection string from the environment."""
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise EnvironmentError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. "
            "Copy .env.example to .env and fill in your connection string."
        )
    return BlobServiceClient.from_connection_string(conn_str)


def download_blob(blob_name: str, destination: Path) -> None:
    """Download a single blob from the input container to a local path.

    Args:
        blob_name:   Name of the blob in the input container (e.g. "Sample-1.pdf").
        destination: Local file path where the blob will be written.
    """
    container = os.environ.get("AZURE_INPUT_CONTAINER", "pdfinput")
    client = _get_client()
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as fh:
        fh.write(blob_client.download_blob().readall())


def upload_blob(source: Path, blob_name: str) -> None:
    """Upload a local file to the output container.

    Args:
        source:    Local file path to upload.
        blob_name: Destination blob name in the output container.
    """
    container = os.environ.get("AZURE_OUTPUT_CONTAINER", "pdfoutput")
    client = _get_client()
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    with source.open("rb") as fh:
        blob_client.upload_blob(fh, overwrite=True)


def download_output_blob(blob_name: str, destination: Path) -> bool:
    """Download a blob from the output container to a local path.

    Returns True if the blob existed and was downloaded, False if it does not exist.
    """
    from azure.core.exceptions import ResourceNotFoundError

    container = os.environ.get("AZURE_OUTPUT_CONTAINER", "pdfoutput")
    client = _get_client()
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as fh:
            fh.write(blob_client.download_blob().readall())
        return True
    except ResourceNotFoundError:
        return False


def list_input_blobs() -> list[str]:
    """Return blob names in the input container that end with '.pdf'.

    Returns:
        Sorted list of PDF blob names.
    """
    container = os.environ.get("AZURE_INPUT_CONTAINER", "pdfinput")
    client = _get_client()
    container_client = client.get_container_client(container)
    return sorted(
        b.name for b in container_client.list_blobs() if b.name.lower().endswith(".pdf")
    )
