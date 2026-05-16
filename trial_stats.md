# PDF Processing Throughput Trials
Run date: 2026-05-16  
Hardware: Azure Container Apps Consumption — 4 vCPU / 8 GiB  
Rate: $0.44/active hour

---

## Results Table

| File | Pages | Start (UTC) | End (UTC) | Duration (min) | Output Docs | Pages/hr | Docs/hr | Cost ($) | $/page | $/doc |
|---|---|---|---|---|---|---|---|---|---|---|
| 603110820.pdf | 457 | 12:59:48 | TIMEOUT (run 1) | >120 | 0 | <228 | n/a | ~$0.88 | n/a | n/a |
| 603110820.pdf | 457 | 17:41:57 | 19:37:21 | 115.4 | 36 | 238 | 18.7 | $0.847 | $0.00185 | $0.0235 |
| 601602726.pdf | 35 | 17:35:23 | 17:39:54 | 4.52 | 15 | 465 | 199 | $0.033 | $0.00094 | $0.0022 |
| 602954198.pdf | 152 | 20:19:36 | 20:36:13 | 16.6 | 18 | 549 | 65 | $0.122 | $0.00080 | $0.0068 |

---

## Job 1a: 603110820.pdf — FAILED (TIMEOUT, run 1)

- **Execution ID:** pdf-extractor-job-anejgqm
- **File size:** ~191 MB, 457 pages (multiple internal copies; many near-white/faint pages)
- **Start:** 2026-05-16T12:59:48 UTC
- **End:** Job timeout — Azure Container Apps 2-hour (7,200-second) limit exceeded
- **Output:** None — tool writes all output at completion; nothing landed in pdfoutput
- **Diagnosis:** 457 pages exceeded the 2-hour job limit. All pages are scanned (OCR required for every page). Faint/near-white pages still consume full OCR time. Throughput for dense all-scanned files is **under 228 pages/hour** on Consumption 4 vCPU / 8 GiB.

---

## Job 1b: 603110820.pdf — SUCCEEDED (run 2)

- **Execution ID:** pdf-extractor-job-szigevg
- **File size:** ~191 MB, 457 pages (multiple internal copies; many near-white/faint pages)
- **Start:** 2026-05-16T17:41:57 UTC
- **End:** 2026-05-16T19:37:21 UTC
- **Duration:** 1h 55m 24s (6,924 s / 115.4 min)
- **Output:** 36 documents
- **Chunks:** 23 × 20-page chunks (windowed overlap)
- **Throughput:** 238 pages/hr | 18.7 docs/hr
- **Cost:** 6,924 s × $0.44/hr = **$0.847**
- **$/page:** $0.00185 | **$/doc:** $0.0235
- **Timeout:** 86,400 s (24 hr)
- **Note:** Worst-case file — all pages scanned, many near-white/faint, multiple internal deal-jacket copies. Recurring OCR pool exhaustion on faint pages (fell back to AMBIGUOUS). Output quality expected to be rough; trial purpose was throughput/stability validation, not split quality. 238 pages/hr is the **floor** for all-scanned files.

---

## Job 2: 601602726.pdf — SUCCEEDED

- **Execution ID:** pdf-extractor-job-drhqfyt
- **File size:** ~85 MB, 35 pages
- **Start:** 2026-05-16T17:35:23 UTC
- **End:** 2026-05-16T17:39:54 UTC
- **Duration:** 4 min 31 sec (271 s)
- **Output:** 15 documents
- **Chunks:** 2 × 20-page chunks (windowed overlap)
- **Throughput:** 465 pages/hr | 199 docs/hr
- **Cost:** 271 s × $0.44/hr = **$0.033**
- **$/page:** $0.00094 | **$/doc:** $0.0022
- **Note:** Mix of text pages (1–10, fast) and scanned OCR pages (11–35). First 10 pages processed in ~2 s; 25 OCR pages in ~177 s (~509 pages/hr OCR-only).

### Previous attempts (cancelled before real run)
- pdf-extractor-job-4cpub80: stuck in Waiting state 70+ min, never started
- pdf-extractor-job-5ki77yl: wrong file (602819077.pdf from stale template), stopped
- pdf-extractor-job-sl04mbo: PDF_INPUT_FILE env var empty (--env-vars on start unreliable), failed

---

## Job 3: 602954198.pdf — SUCCEEDED

- **Execution ID:** pdf-extractor-job-tklk05z
- **File size:** ~85 MB, 152 pages
- **Start:** 2026-05-16T20:19:36 UTC
- **End:** 2026-05-16T20:36:13 UTC
- **Duration:** 16 min 37 sec (997 s)
- **Output:** 18 documents
- **Chunks:** 8 × 20-page chunks (windowed overlap)
- **Throughput:** 549 pages/hr | 65 docs/hr
- **Cost:** 997 s × $0.44/hr = **$0.122**
- **$/page:** $0.00080 | **$/doc:** $0.0068
- **Note:** File type unknown (text/scan mix not confirmed from logs). Throughput of 549 pages/hr is between the all-scanned floor (238) and mixed ceiling (465 observed for Job 2). Blob storage shows 26 files in the 602954198/ output folder; 8 are from prior test runs — 18 is the authoritative count per DONE log.
- **Aborted attempts:** pdf-extractor-job-o8da79q stopped immediately — started with wrong file (603110820.pdf) due to race between `--set-env-vars` update and job start.

---

## Infrastructure Changes (2026-05-16)

- **Replica timeout** updated from 7,200 s (2 hr) → 86,400 s (24 hr) at ~15:55 UTC
- **Root cause identified:** `az containerapp job update --args <file>` persists the filename to the job template. Using `az containerapp job start --args` does NOT reliably override it per-execution. Fix: switch container to read filename from `PDF_INPUT_FILE` env var; pass per-execution via `--set-env-vars` on `az containerapp job update` before each start.

---

## Notes

- 603110820.pdf is the most challenging file — 457 pages, heavily scanned, multiple internal copies, many faint pages. Throughput of 238 pages/hr is the **floor** for worst-case all-scanned files.
- The original Section 6 estimate of "73 docs/hour" appears significantly optimistic for all-scanned files.
- 601602726.pdf (mixed text+scan) ran at 465 pages/hr — text pages dramatically accelerate overall throughput.
- **Cost baseline (all-scanned, worst case):** ~$0.00185/source page | ~$0.847/457-page file | 238 pages/hr
- **Cost baseline (mixed/typical):** ~$0.00080–$0.00094/source page | 465–549 pages/hr
- **Section 6 recommendation:** Use $0.00185/page as the conservative upper bound for cost estimates; $0.00094/page as a typical mixed-file figure.
