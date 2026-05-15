"""Command-line interface for finding PDFs and writing extracted output files."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import tempfile
from pathlib import Path

from .extractor import (
    ExtractedDocument,
    extract_pdf,
    find_pdf_files,
)
from .image_splitter import split_pdf


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the pdf-extractor CLI.

    Defines the following arguments:

    - ``input_path``         — positional; a PDF file or a directory of PDFs.
                               Pass ``--azure`` to treat this as a blob name instead.
    - ``--output-dir``       — where extracted text/JSON files are written (default: ``output/``).
    - ``--format``           — ``"text"`` (default) or ``"json"`` for extraction output.
    - ``--recursive``        — descend into subdirectories when ``input_path`` is a directory.
    - ``--split-documents``  — split a single PDF into one file per logical document.
    - ``--split-output-dir`` — destination for split PDFs (default: ``output/split/``).
    - ``--azure``            — download ``input_path`` from Azure Blob Storage and upload
                               split output back to the output container.

    Returns:
        Configured ``ArgumentParser`` instance ready for ``parse_args()``.
    """
    parser = argparse.ArgumentParser(
        description="Extract text and metadata from PDF files.",
    )
    parser.add_argument(
        "input_path",
        help=(
            "PDF file, directory of PDFs, or (with --azure) a blob name in the "
            "Azure input container. With --azure you may also pass glob patterns "
            "such as '6*' or 'abc*'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where extracted files will be written",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for extracted content",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for PDFs when the input path is a directory",
    )
    parser.add_argument(
        "--split-documents",
        action="store_true",
        help="Split one PDF into one PDF per logical document (handles text and scanned image pages)",
    )
    parser.add_argument(
        "--split-output-dir",
        default="output/split",
        help="Directory where split PDFs will be written",
    )
    parser.add_argument(
        "--azure",
        action="store_true",
        help=(
            "Read input from Azure Blob Storage and write split output back to the "
            "output container.  Requires AZURE_STORAGE_CONNECTION_STRING in the environment."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a per-page signal table after analysis to help diagnose mis-splits.",
    )
    return parser


def _build_output_path(output_dir: Path, source_path: str, output_format: str) -> Path:
    """Derive the output file path for an extracted document.

    Uses the source PDF's stem (filename without extension) as the output name,
    appending ``.txt`` or ``.json`` depending on the requested format.

    Args:
        output_dir:    Directory where the file will be written.
        source_path:   Absolute path of the source PDF (used for the stem only).
        output_format: ``"text"`` → ``.txt`` extension; ``"json"`` → ``.json``.

    Returns:
        Full Path for the output file (directory is not created here).
    """
    extension = ".txt" if output_format == "text" else ".json"
    return output_dir / f"{Path(source_path).stem}{extension}"


def _write_output(document: ExtractedDocument, output_dir: Path, output_format: str) -> Path:
    """Serialize an extracted document to disk and return the written path.

    Creates ``output_dir`` (including any missing parents) if it does not exist.
    JSON output is written with 2-space indentation and ASCII-safe encoding so
    the file is readable in any locale.  Plain-text output writes the concatenated
    page content directly.

    Args:
        document:      Populated ``ExtractedDocument`` from ``extract_pdf()``.
        output_dir:    Destination directory.
        output_format: ``"text"`` or ``"json"``.

    Returns:
        Path to the written output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = _build_output_path(output_dir, document.source_path, output_format)

    if output_format == "json":
        destination.write_text(
            json.dumps(document.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    else:
        destination.write_text(document.combined_text, encoding="utf-8")

    return destination


def _run_azure_split(blob_name: str, *, verbose: bool = False) -> list[dict]:
    """Download one blob, split it, upload the results, and return opt-in data.

    Requires ``AZURE_STORAGE_CONNECTION_STRING`` in the environment (loaded from
    ``.env`` via python-dotenv before this function is called).

    Args:
        blob_name: Name of the PDF blob in the Azure input container.

    Returns:
        List of opt-in result dicts (one per Credit_Application.pdf found).
        Each dict includes a ``source_file`` key with the blob path.
    """
    from .azure_storage import download_blob, upload_blob

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local_input = tmp_path / blob_name
        split_output_dir = tmp_path / "split"

        print(f"Downloading {blob_name} …")
        download_blob(blob_name, local_input)

        print("Splitting …")
        written_files = split_pdf(str(local_input), str(split_output_dir), verbose=verbose)

        print(f"Uploading {len(written_files)} file(s) …")
        stem = Path(blob_name).stem
        for idx, written_file in enumerate(written_files, start=1):
            dest_blob = f"{stem}/{written_file.name}"
            upload_blob(written_file, dest_blob)
            print(f"  [{idx}] -> {dest_blob}")

        opt_in_results = []
        credit_apps = [f for f in written_files if f.name == "Credit_Application.pdf"]
        if credit_apps:
            from .opt_in_extractor import extract_credit_app_data
            from pypdf import PdfReader
            for ca in credit_apps:
                print(f"Extracting opt-in data from {ca.name} …")
                try:
                    data = extract_credit_app_data(PdfReader(str(ca)))
                except Exception as exc:
                    print(f"  WARNING: extraction failed: {exc}")
                    data = {
                        "last_name": None, "first_name": None,
                        "opt_in_status": "error", "telemarketing_phones": [],
                    }
                data["source_file"] = f"{stem}/{ca.name}"
                opt_in_results.append(data)

        return opt_in_results


def main() -> int:
    """Entry point for the ``pdf-extractor`` CLI command.

    Three modes of operation:

    **Azure split mode** (``--azure --split-documents``):
        Downloads the named PDF blob (or every PDF blob when ``input_path`` is ``*``)
        from the Azure input container, splits it into one PDF per logical document,
        and uploads the results to a sub-folder in the Azure output container.
        Reads credentials from the environment (populate ``.env`` locally;
        set container environment variables when deployed to Azure).

    **Extraction mode** (default):
        Extracts text from each discovered PDF and writes one ``.txt`` or ``.json``
        file per PDF to ``--output-dir``.  Accepts a single file or a directory
        (optionally recursive via ``--recursive``).

    **Local split mode** (``--split-documents``):
        Requires exactly one input PDF.  Splits it into one PDF per logical document
        using ``split_pdf()`` from ``image_splitter``.  Output files are written to
        ``--split-output-dir``.

    Returns:
        0 on success.  Calls ``parser.error()`` (which exits with status 2) on bad
        input — no PDFs found, wrong number of files for split mode, etc.
    """
    # Load .env if present (no-op when running in Azure where env vars are set directly).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = build_parser()
    args = parser.parse_args()

    # ── Azure split mode ──────────────────────────────────────────────────────
    if args.azure:
        if not args.split_documents:
            parser.error("--azure currently requires --split-documents.")
        from .azure_storage import list_input_blobs

        pattern = args.input_path
        has_glob = any(ch in pattern for ch in "*?[]")

        if pattern == "*":
            blob_names = list_input_blobs()
            if not blob_names:
                parser.error("No PDF blobs found in the input container.")
        elif has_glob:
            all_blobs = list_input_blobs()
            blob_names = [
                name for name in all_blobs
                if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(Path(name).name, pattern)
            ]
            if not blob_names:
                parser.error(f"No PDF blobs matched Azure pattern: {pattern}")
        else:
            blob_names = [pattern]
        all_opt_in: list[dict] = []
        for blob_name in blob_names:
            all_opt_in.extend(_run_azure_split(blob_name, verbose=args.verbose))

        if all_opt_in:
            import datetime
            from .azure_storage import upload_blob
            from .opt_in_extractor import write_results_to_excel
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            xlsx_name = f"opt_in_results_{timestamp}.xlsx"
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                excel_path = Path(tmp.name)
            write_results_to_excel(all_opt_in, excel_path)
            upload_blob(excel_path, xlsx_name)
            excel_path.unlink(missing_ok=True)
            print(f"Opt-in results ({len(all_opt_in)} record(s)) -> {xlsx_name}")

        return 0

    # ── Local modes ───────────────────────────────────────────────────────────
    try:
        pdf_files = find_pdf_files(args.input_path, recursive=args.recursive)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if not pdf_files:
        parser.error("No PDF files found in the given input path.")

    if args.split_documents:
        if len(pdf_files) != 1:
            parser.error("--split-documents requires exactly one input PDF.")
        split_pdf(pdf_files[0], args.split_output_dir, verbose=args.verbose)
        return 0

    output_dir = Path(args.output_dir)
    written_files: list[Path] = []

    for pdf_file in pdf_files:
        document = extract_pdf(pdf_file)
        written_files.append(_write_output(document, output_dir, args.format))

    for written_file in written_files:
        print(written_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
