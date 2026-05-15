# Perceptual Hash Deduplication — Design Spec

**Date:** 2026-05-15
**Status:** Approved

---

## Problem

The existing deduplication pipeline in `image_splitter.py` has two layers:

1. **SHA-256 byte hash** — catches exact duplicates (identical PDF bytes)
2. **Semantic key `title|page_count`** — catches near-duplicates with matching title and page count

The gap: two scanned copies of the same physical document where the scanner introduces compression or angle differences produce different byte hashes, and OCR noise on the title produces a different semantic key. Both layers miss them.

This is observed in large batch files (e.g. `602819077.pdf`, 220 pages) where the same deal jacket is scanned twice at different times.

---

## Goal

Add a third deduplication layer based on perceptual image hashing that:

- **Never suppresses output** — both documents are always written
- **Flags suspected duplicates** for manual review
- Adds no new dependencies
- Does not meaningfully increase memory usage

---

## Algorithm

**Average hash (aHash) of the first page.**

1. Render page `group[0]` at ~64px wide using the existing `_render_page_fitz()` helper
2. Convert the rendered image to 8×8 grayscale (64 pixels total)
3. Compute the mean pixel value across all 64 pixels
4. Build a 64-bit integer: bit `i` = 1 if pixel `i` ≥ mean, else 0
5. Store in `seen_phashes: dict[int, tuple[int, str]]` mapping `hash → (group_index, title_text)`
6. For each new group, compare its hash against all stored hashes using Hamming distance
7. If `hamming_distance(new_hash, stored_hash) ≤ 10`, record the pair as a suspected duplicate

**Threshold rationale:** Same-document rescans typically differ by 2–5 bits out of 64. Completely unrelated documents differ by 25–35 bits. A threshold of 10 (~15% pixel disagreement) provides a comfortable margin.

**Two new helper functions (no new imports):**

```python
def _perceptual_hash(fitz_doc, page_idx: int) -> int | None:
    """Render first page at low resolution and return a 64-bit average hash."""

def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
```

If rendering fails for a group, the hash is skipped silently — no crash, no false negatives.

---

## Integration Points

Both existing dedup paths are extended. The perceptual hash tracking runs **in parallel** with the existing byte-hash and semantic-key logic — it never replaces them.

### Non-streaming path (`split_pdf`, lines ~1550)

All groups are already collected in `flat_groups` before the dedup loop. The perceptual hash comparison runs in the same `for i, group in enumerate(flat_groups)` loop:

- Compute hash for each group
- Compare against all previously seen hashes
- Collect hits in `phash_hits: list[tuple[int, int, int, str, str]]`
  (group_a_idx, group_b_idx, distance, title_a, title_b)

### Streaming path (`split_pdf`, lines ~1418)

Groups are emitted and written one at a time. The perceptual hash runs **incrementally**:

- As each group is written, compute its hash and compare against `seen_phashes`
- If a match is found, record the pair
- Add the hash to `seen_phashes` and continue

Implication: in the streaming path, only later occurrences of a duplicate are detected (the first occurrence has no prior hash to match against). This is acceptable — both copies are always written, and the flagging is for manual review.

`seen_phashes` and `phash_hits` are initialized at the start of each `split_pdf` call — fresh state per file, no cross-file persistence.

---

## Memory Profile

`seen_phashes` holds one entry per unique document group:

```python
seen_phashes: dict[int, tuple[int, str]] = {}
# key:   64-bit int (8 bytes)
# value: (group_index: int, title_text: str)
```

For a 220-page file with ~20 document groups, this is a few hundred bytes. No page data or image data is retained.

---

## Report Format

Reported whenever at least one suspected pair is found. Not gated on `--verbose`.

**Console output:**

```
--- SUSPECTED DUPLICATES (perceptual hash) ---
  Group 3  (page 15)  "RETAIL INSTALLMENT CONTRACT" ~ Group 28 (page 142) "RETAIL INSTALLMENT CONTRACT" [distance: 7]
  Group 5  (page 22)  "CREDIT APPLICATION"          ~ Group 31 (page 155) "CREDIT APPLICATION"          [distance: 4]
```

**File:** `suspected_duplicates.txt` written to the split output directory.

- Header line: source filename + timestamp
- One line per suspected pair (same format as console)
- File is **not created** when there are no hits (no empty files)

---

## What This Does Not Change

- Output files: all documents are always written, no suppression
- Existing byte-hash and semantic-key dedup layers: unchanged
- Dependencies: no new packages
- CLI interface: no new flags
- Test suite: new unit tests for `_perceptual_hash` and `_hamming_distance` added

---

## Out of Scope

- Auto-suppressing duplicates based on perceptual hash alone
- Hashing pages beyond the first
- Cross-file deduplication (comparing across multiple `split_pdf` calls)
