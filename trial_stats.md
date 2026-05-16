# PDF Processing Throughput Trials
Run date: 2026-05-16  
Hardware: Azure Container Apps Consumption — 4 vCPU / 8 GiB  
Rate: $0.44/active hour

---

## Results Table

| File | Pages | Start (UTC) | End (UTC) | Duration (min) | Output Docs | Pages/hr | Docs/hr | Cost ($) | $/page | $/doc |
|---|---|---|---|---|---|---|---|---|---|---|
| 603110820.pdf | 457 | 12:59:48 | TIMEOUT (run 1) | >120 | 0 | <228 | n/a | ~$0.88 | n/a | n/a |
| 603110820.pdf | 457 | PENDING | — | — | — | — | — | — | — | — |
| 601602726.pdf | ? | PENDING | — | — | — | — | — | — | — | — |
| 602954198.pdf | ? | PENDING | — | — | — | — | — | — | — | — |

---

## Job 1a: 603110820.pdf — FAILED (TIMEOUT, run 1)

- **Execution ID:** pdf-extractor-job-anejgqm
- **File size:** ~191 MB, 457 pages (multiple internal copies; many near-white/faint pages)
- **Start:** 2026-05-16T12:59:48 UTC
- **End:** Job timeout — Azure Container Apps 2-hour (7,200-second) limit exceeded
- **Output:** None — tool writes all output at completion; nothing landed in pdfoutput
- **Diagnosis:** 457 pages exceeded the 2-hour job limit. All pages are scanned (OCR required for every page). Faint/near-white pages still consume full OCR time. Throughput for dense all-scanned files is **under 228 pages/hour** on Consumption 4 vCPU / 8 GiB.

---

## Job 1b: 603110820.pdf — PENDING (run 2)

- **Execution ID:** TBD
- **File size:** ~191 MB, 457 pages
- **Timeout:** 86,400 seconds (24 hours)
- **Note:** Two premature starts were stopped before processing: pdf-extractor-job-9eiml70 (stopped immediately) and a second attempt aborted.

---

## Job 2: 601602726.pdf — CANCELLED (never ran)

- **Execution ID:** pdf-extractor-job-4cpub80
- **File size:** ~85 MB
- **Start:** 2026-05-16T15:08:45 UTC
- **Cancelled:** 2026-05-16T16:18 UTC (approx)
- **Reason:** Container stuck in Waiting state ("Unknown on legion") for 70+ minutes — never started processing. Additionally, job template was set to wrong file (602819077.pdf). All jobs cancelled to allow clean restart after code fix.

---

## Job 3: 602954198.pdf — PENDING

- **File size:** ~85 MB
- Will start after job 2 completes.

---

## Infrastructure Changes (2026-05-16)

- **Replica timeout** updated from 7,200 s (2 hr) → 86,400 s (24 hr) at ~15:55 UTC
- **Root cause identified:** `az containerapp job update --args <file>` persists the filename to the job template. Using `az containerapp job start --args` does NOT reliably override it per-execution. Fix: switch container to read filename from `PDF_INPUT_FILE` env var; pass per-execution via `az containerapp job start --env-vars`.
- **Pending code change:** Dockerfile ENTRYPOINT and run.ps1 `Invoke-AzureJob` to use env var pattern before restarting trials.

---

## Notes

- 603110820.pdf is the most challenging file — 457 pages, heavily scanned, multiple internal copies, many faint pages. Exceeded the 2-hour timeout on first run.
- The original Section 6 estimate of "73 docs/hour" appears significantly optimistic for all-scanned files. Real throughput is under 228 pages/hour.
- **All trials on hold** pending Dockerfile/run.ps1 env var fix and image rebuild.
