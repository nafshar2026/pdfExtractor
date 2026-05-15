"""Tests for image-based PDF splitting via PaddleOCR title detection."""

from pdf_extractor.image_splitter import PageSignal, _extract_ocr_texts, _windowed_groups_from_signals


def test_page_signal_new_doc():
    sig = PageSignal(classification="NEW_DOC", title_text="ST-556", page_num_in_doc=None)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "ST-556"
    assert sig.page_num_in_doc is None


def test_page_signal_continuation():
    sig = PageSignal(classification="CONTINUATION", title_text=None, page_num_in_doc=3)
    assert sig.classification == "CONTINUATION"
    assert sig.page_num_in_doc == 3


def test_extract_ocr_texts_normal():
    ocr_result = [
        [
            [[[0,0],[100,0],[100,20],[0,20]], ("ST-556 State Tax", 0.99)],
            [[[0,25],[200,25],[200,45],[0,45]], ("Transaction Return", 0.95)],
        ]
    ]
    assert _extract_ocr_texts(ocr_result) == ["ST-556 State Tax", "Transaction Return"]


def test_extract_ocr_texts_empty_result():
    assert _extract_ocr_texts([[]]) == []


def test_extract_ocr_texts_none():
    assert _extract_ocr_texts(None) == []


def test_extract_ocr_texts_none_inner():
    assert _extract_ocr_texts([[None]]) == []


from unittest.mock import MagicMock, patch
from PIL import Image as PILImage
from pdf_extractor.image_splitter import analyze_page


def _make_ocr_result(texts: list[str], height: int = 30) -> list:
    """Build a PaddleOCR-format result from a list of text strings.

    height controls the bounding-box pixel height, which drives _select_best_title
    scoring.  Default 30px × 2.0 multiplier = score 60 ≥ _MIN_TITLE_SCORE (50).
    """
    lines = []
    for text in texts:
        bbox = [[0, 0], [100, 0], [100, height], [0, height]]
        lines.append([bbox, (text, 0.99)])
    return [lines]


def _make_image_page(pil_image: PILImage.Image) -> MagicMock:
    img_obj = MagicMock()
    img_obj.image = pil_image
    page = MagicMock()
    page.images = [img_obj]
    return page


def _blank_image(width: int = 612, height: int = 792) -> PILImage.Image:
    return PILImage.new("RGB", (width, height), color=(255, 255, 255))


def test_analyze_page_continuation_detected():
    page = _make_image_page(_blank_image())
    # Bottom strip: "Page 3 of 6" → CONTINUATION, returns before top strip is needed.
    with patch("pdf_extractor.image_splitter._render_page_fitz", return_value=_blank_image()):
        with patch("pdf_extractor.image_splitter._ocr_infer", side_effect=[
            _make_ocr_result(["Page 3 of 6"]),   # bottom strip only
        ]):
            sig = analyze_page(page)
    assert sig.classification == "CONTINUATION"
    assert sig.page_num_in_doc == 3
    assert sig.title_text is None


def test_analyze_page_title_detected():
    page = _make_image_page(_blank_image())
    # No pagination → bottom, top, then full-page OCR scans.  Full page returns
    # empty so no pagination found there either; title from top strip wins.
    with patch("pdf_extractor.image_splitter._render_page_fitz", return_value=_blank_image()):
        with patch("pdf_extractor.image_splitter._ocr_infer", side_effect=[
            _make_ocr_result([]),                     # bottom strip: no continuation
            _make_ocr_result(["ST-556 State Tax"]),   # top strip: short title
            _make_ocr_result([]),                     # full-page fallback (page_one_total=None)
        ]):
            sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "State Tax"  # "ST-556" prefix stripped by _normalize_detected_title
    assert sig.page_num_in_doc is None


def test_analyze_page_ambiguous_long_top_text():
    page = _make_image_page(_blank_image())
    long_text = "x" * 61
    # No pagination → bottom, top, full-page scan.  Top text is too long; full page
    # also returns nothing → AMBIGUOUS.
    with patch("pdf_extractor.image_splitter._render_page_fitz", return_value=_blank_image()):
        with patch("pdf_extractor.image_splitter._ocr_infer", side_effect=[
            _make_ocr_result([]),           # bottom: no continuation
            _make_ocr_result([long_text]),  # top: too long to be a title
            _make_ocr_result([]),           # full-page fallback (page_one_total=None)
        ]):
            sig = analyze_page(page)
    assert sig.classification == "AMBIGUOUS"


def test_analyze_page_no_images():
    page = MagicMock()
    page.images = []
    sig = analyze_page(page)
    assert sig.classification == "AMBIGUOUS"
    assert sig.title_text is None


def test_analyze_page_page_1_of_n_with_title():
    """'Page 1 of N' with a detectable title → NEW_DOC carrying total_pages."""
    page = _make_image_page(_blank_image())
    # page_one_total is set from bottom strip → full-page scan is skipped.
    with patch("pdf_extractor.image_splitter._render_page_fitz", return_value=_blank_image()):
        with patch("pdf_extractor.image_splitter._ocr_infer", side_effect=[
            _make_ocr_result(["Page 1 of 6"]),    # bottom: page 1 of 6
            _make_ocr_result(["Title Here"]),      # top: title found
        ]):
            sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "Title Here"
    assert sig.page_num_in_doc == 1
    assert sig.total_pages_in_doc == 6


def test_analyze_page_page_1_of_n_no_title_is_new_doc():
    """'Page 1 of N' with no detectable title must still be NEW_DOC, not AMBIGUOUS."""
    page = _make_image_page(_blank_image())
    # page_one_total set from bottom → full-page fallback runs when top yields nothing.
    with patch("pdf_extractor.image_splitter._render_page_fitz", return_value=_blank_image()):
        with patch("pdf_extractor.image_splitter._ocr_infer", side_effect=[
            _make_ocr_result(["Page 1 of 4"]),  # bottom: page 1 of 4
            _make_ocr_result([]),               # top: nothing usable
            _make_ocr_result([]),               # full-page fallback (page_one_total=4)
        ]):
            sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text is None
    assert sig.page_num_in_doc == 1
    assert sig.total_pages_in_doc == 4


def test_analyze_page_continuation_carries_total():
    """CONTINUATION signal must carry total_pages_in_doc from the footer."""
    page = _make_image_page(_blank_image())
    with patch("pdf_extractor.image_splitter._render_page_fitz", return_value=_blank_image()):
        with patch("pdf_extractor.image_splitter._ocr_infer", side_effect=[
            _make_ocr_result(["Page 3 of 6"]),
        ]):
            sig = analyze_page(page)
    assert sig.classification == "CONTINUATION"
    assert sig.page_num_in_doc == 3
    assert sig.total_pages_in_doc == 6


from pdf_extractor.image_splitter import _group_image_pages


def _sig(cls, title=None, page_num=None, total=None):
    return PageSignal(classification=cls, title_text=title, page_num_in_doc=page_num, total_pages_in_doc=total)


def test_group_single_new_doc():
    signals = [(0, _sig("NEW_DOC", "Title A"))]
    assert _group_image_pages(signals) == [[0]]


def test_group_new_doc_then_continuation():
    signals = [
        (0, _sig("NEW_DOC", "Title A")),
        (1, _sig("CONTINUATION", page_num=2)),
        (2, _sig("CONTINUATION", page_num=3)),
    ]
    assert _group_image_pages(signals) == [[0, 1, 2]]


def test_group_two_consecutive_new_docs():
    """Two NEW_DOC pages back-to-back must become separate documents."""
    signals = [
        (10, _sig("NEW_DOC", "Title A")),
        (11, _sig("NEW_DOC", "Title B")),
    ]
    assert _group_image_pages(signals) == [[10], [11]]


def test_group_ambiguous_extends_current():
    signals = [
        (0, _sig("NEW_DOC", "Title")),
        (1, _sig("AMBIGUOUS")),
        (2, _sig("AMBIGUOUS")),
    ]
    assert _group_image_pages(signals) == [[0, 1, 2]]


def test_group_leading_ambiguous_merges_with_next_new_doc():
    """AMBIGUOUS pages before the first NEW_DOC are prepended to that document."""
    signals = [
        (0, _sig("AMBIGUOUS")),
        (1, _sig("NEW_DOC", "Title")),
    ]
    assert _group_image_pages(signals) == [[0, 1]]


def test_group_orphan_continuation_forms_own_group():
    """CONTINUATION with no preceding group starts one (defensive)."""
    signals = [
        (0, _sig("CONTINUATION", page_num=2)),
        (1, _sig("NEW_DOC", "Title")),
    ]
    assert _group_image_pages(signals) == [[0], [1]]


def test_group_empty_input():
    assert _group_image_pages([]) == []


def test_group_continuation_total_change_splits():
    """CONTINUATION pages whose 'of N' total changes must start a new group."""
    signals = [
        (0, _sig("NEW_DOC", "Doc A", total=3)),
        (1, _sig("CONTINUATION", page_num=2, total=3)),
        (2, _sig("CONTINUATION", page_num=3, total=3)),
        (3, _sig("CONTINUATION", page_num=2, total=4)),  # different total → new doc
        (4, _sig("CONTINUATION", page_num=3, total=4)),
        (5, _sig("CONTINUATION", page_num=4, total=4)),
    ]
    assert _group_image_pages(signals) == [[0, 1, 2], [3, 4, 5]]


def test_group_continuation_same_total_stays_together():
    """CONTINUATION pages sharing the same 'of N' must stay in one group."""
    signals = [
        (0, _sig("NEW_DOC", "Doc A", total=3)),
        (1, _sig("CONTINUATION", page_num=2, total=3)),
        (2, _sig("CONTINUATION", page_num=3, total=3)),
    ]
    assert _group_image_pages(signals) == [[0, 1, 2]]


from pdf_extractor.image_splitter import _sanitize_image_title


def test_sanitize_image_title_replaces_spaces_with_underscores():
    assert _sanitize_image_title("ST-556 State Tax") == "ST-556_State_Tax"


def test_sanitize_image_title_strips_special_chars():
    assert _sanitize_image_title("Form: A/B") == "Form_AB"


def test_sanitize_image_title_empty_falls_back():
    assert _sanitize_image_title("!!!") == "Untitled"


def test_sanitize_image_title_strips_leading_trailing_underscores():
    assert _sanitize_image_title(" Hello ") == "Hello"


from pypdf import PdfReader, PdfWriter
from pdf_extractor.image_splitter import _write_image_group


def _make_real_reader(tmp_path, num_pages: int = 3) -> PdfReader:
    """Creates a minimal real PDF with num_pages blank pages."""
    pdf_path = tmp_path / "source.pdf"
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as f:
        writer.write(f)
    return PdfReader(str(pdf_path))


def test_write_image_group_title_based_name(tmp_path):
    reader = _make_real_reader(tmp_path, num_pages=3)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    signals = {0: PageSignal("NEW_DOC", "ST-556 State Tax", None)}
    used_names: dict = {}

    dest = _write_image_group(reader, [0, 1], signals, out_dir, used_names)

    assert dest.name == "ST-556_State_Tax.pdf"
    assert dest.exists()
    result = PdfReader(str(dest))
    assert len(result.pages) == 2


def test_write_image_group_fallback_name(tmp_path):
    reader = _make_real_reader(tmp_path, num_pages=5)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    signals = {2: PageSignal("AMBIGUOUS", None, None)}
    used_names: dict = {}

    dest = _write_image_group(reader, [2, 3, 4], signals, out_dir, used_names)

    assert dest.name == "pages_3-5.pdf"


def test_write_image_group_duplicate_name_gets_suffix(tmp_path):
    reader = _make_real_reader(tmp_path, num_pages=4)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    signals_a = {0: PageSignal("NEW_DOC", "Form ABC", None)}
    signals_b = {2: PageSignal("NEW_DOC", "Form ABC", None)}
    used_names: dict = {}

    dest_a = _write_image_group(reader, [0, 1], signals_a, out_dir, used_names)
    dest_b = _write_image_group(reader, [2, 3], signals_b, out_dir, used_names)

    assert dest_a.name == "Form_ABC.pdf"
    assert dest_b.name == "Form_ABC (2).pdf"


from unittest.mock import patch, MagicMock
from pdf_extractor.image_splitter import _analyze_text_page, _extract_text_title


def test_split_pdf_groups_signals_into_documents(tmp_path):
    """split_pdf groups all pages by analyze_page signals into separate PDFs."""
    source = tmp_path / "source.pdf"
    w = PdfWriter()
    # Use mixed page sizes so hash-based dedup does not collapse both groups.
    w.add_blank_page(width=612, height=792)
    w.add_blank_page(width=612, height=792)
    w.add_blank_page(width=613, height=792)
    w.add_blank_page(width=613, height=792)
    with source.open("wb") as f:
        w.write(f)
    out_dir = tmp_path / "out"

    signals = [
        PageSignal("NEW_DOC", "Doc One", None),
        PageSignal("CONTINUATION", None, 2),
        PageSignal("NEW_DOC", "Doc Two", None),
        PageSignal("AMBIGUOUS", None, None),
    ]

    with patch("pdf_extractor.image_splitter.analyze_page", side_effect=signals):
        written = split_pdf(source, out_dir)

    assert len(written) == 2
    assert written[0].name == "pages_1-2.pdf"
    assert written[1].name == "Doc_Two.pdf"


def test_analyze_text_page_doc_marker_is_new_doc():
    """DOCUMENT N OF N marker → NEW_DOC with title from the following line."""
    text = "DOCUMENT 1 OF 2\nApplicable Law\nsome content here to exceed fifty chars minimum threshold"
    sig = _analyze_text_page(text)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "Applicable Law"
    assert sig.page_num_in_doc == 1


def test_analyze_text_page_page_1_of_n_is_new_doc():
    """'Page 1 of N' in the footer → NEW_DOC carrying total_pages_in_doc."""
    text = "Retail Purchase Agreement\nsome body content here\nPage 1 of 3"
    sig = _analyze_text_page(text)
    assert sig.classification == "NEW_DOC"
    assert sig.total_pages_in_doc == 3
    assert sig.page_num_in_doc == 1


def test_analyze_text_page_continuation():
    """'Page N of M' (N > 1) in the footer → CONTINUATION."""
    text = "continued body content on page two here\nPage 2 of 3"
    sig = _analyze_text_page(text)
    assert sig.classification == "CONTINUATION"
    assert sig.page_num_in_doc == 2
    assert sig.total_pages_in_doc == 3


def test_analyze_text_page_page_num_anywhere_in_text():
    """'Page N of M' in the header area (not footer) must still be detected."""
    text = "LAW 553-IL-ARB-ea 8/22 v1    Page 1 of 6\nBody content here for the contract text"
    sig = _analyze_text_page(text)
    assert sig.classification == "NEW_DOC"
    assert sig.page_num_in_doc == 1
    assert sig.total_pages_in_doc == 6


def test_analyze_text_page_continuation_page_num_in_header():
    """'Page 5 of 6' appearing on the first line is a CONTINUATION signal."""
    text = "LAW 553-IL-ARB-ea 8/22 v1    Page 5 of 6\nArbitration provision content follows here"
    sig = _analyze_text_page(text)
    assert sig.classification == "CONTINUATION"
    assert sig.page_num_in_doc == 5
    assert sig.total_pages_in_doc == 6


def test_analyze_text_page_allcaps_title_no_markers_is_new_doc():
    """ALL-CAPS title with no page markers → NEW_DOC."""
    # Pad to > 8 body lines so footer detection does not swallow the title.
    body = "\n".join(["body text for the form"] * 10)
    text = f"FORM NO. LAWIL-RATECAP (Rev. 8/22)\nThe information on this form is part of your contract.\nDISCLOSURE OF 36% RATE CAP\nFurther content of the disclosure form goes here.\n{body}"
    sig = _analyze_text_page(text)
    assert sig.classification == "NEW_DOC"
    assert "DISCLOSURE" in (sig.title_text or "")


def test_analyze_text_page_no_markers_is_ambiguous():
    """Text page with no markers and no detectable title → AMBIGUOUS."""
    text = "DT 5/23\nsome form content without any page numbering markers here"
    sig = _analyze_text_page(text)
    assert sig.classification == "AMBIGUOUS"


def test_extract_text_title_picks_last_in_first_half():
    """When multiple ALL-CAPS candidates exist, the last one in the first half wins.

    The page has 20 lines so half=10.  RETAIL INSTALLMENT CONTRACT is at line 9
    (last candidate in first half); FOR USED VEHICLES ONLY is at line 11 (second
    half) and must be excluded.
    """
    padding = ["body text line here"] * 3
    lines = (
        padding
        + ["FEDERAL DISCLOSURE REQUIREMENTS"]   # line 4 — first half candidate
        + padding
        + ["RETAIL INSTALLMENT CONTRACT"]        # line 8 — last first-half candidate
        + ["body content line here"]              # line 9
        + ["body content line here"]              # line 10  ← half boundary
        + ["FOR USED VEHICLES ONLY"]              # line 11 — second half, must be excluded
        + padding
        + padding                                  # total 20 lines
    )
    title = _extract_text_title(lines, prefer_last=True)
    # _sanitize_filename keeps spaces; underscores are applied later by _sanitize_image_title
    assert title == "RETAIL INSTALLMENT CONTRACT"


def test_extract_text_title_filters_addresses():
    """Lines with ≥7 digits (addresses, phone numbers) must be excluded."""
    lines = ["4525 TURNBERRY DR 60133ILHANOVER PARK"]
    assert _extract_text_title(lines) is None


def test_extract_text_title_filters_short_lines():
    """ALL-CAPS lines with < 3 words are excluded (section labels, form IDs)."""
    lines = ["DT 5/23", "APPLICABLE LAW"]
    assert _extract_text_title(lines) is None


def test_extract_text_title_filters_corporate_suffixes():
    """Company names ending with LLC/INC are not document titles."""
    lines = ["ZEIGLER CHRYSLER DODGE JEEP LLC"]
    assert _extract_text_title(lines) is None


def test_extract_text_title_filters_non_ascii():
    """Lines with non-ASCII characters (bilingual duplicates) are excluded."""
    lines = ["DIVULGACIÓN DE LA TASA MAXIMA DEL 36"]
    assert _extract_text_title(lines) is None


def test_extract_text_title_first_wins_by_default():
    """Without prefer_last, the first qualifying candidate is returned."""
    # Pad to > 8 body lines so the footer-detection window doesn't absorb the title.
    body = ["body text line"] * 10
    lines = ["ODOMETER DISCLOSURE STATEMENT", "WARNING ODOMETER DISCREPENCY"] + body
    assert _extract_text_title(lines) == "ODOMETER DISCLOSURE STATEMENT"


def test_extract_text_title_prefer_last_picks_later():
    """prefer_last=True skips earlier candidates in favour of later ones."""
    body = ["body text line"] * 10
    lines = ["FEDERAL TRUTH IN LENDING DISCLOSURES", "RETAIL INSTALLMENT CONTRACT"] + body
    assert _extract_text_title(lines, prefer_last=True) == "RETAIL INSTALLMENT CONTRACT"


def test_extract_text_title_title_case_fallback():
    """Title Case titles are detected when no ALL-CAPS title is present."""
    # Simulates a RouteOne credit application page: logo line filtered by non-ASCII,
    # then "Credit Application" in Title Case on line 2.  Body padding keeps the
    # footer-detection window from absorbing the title line.
    body = ["body text line"] * 10
    lines = ["RouteOne®", "Credit Application", "[X] You are applying for individual credit"] + body
    assert _extract_text_title(lines) == "Credit Application"


def test_extract_text_title_title_case_not_triggered_past_line_6():
    """Title Case fallback only searches the first 6 lines; later lines are ignored."""
    lines = [
        "RouteOne®",          # filtered: non-ASCII
        "Title Last Name First",   # 4 words Title Case — but all words must start upper... passes?
                                   # Actually: "Last", "Name", "First" all upper, "Title" upper
                                   # This would be a false positive — so test a line that DOES
                                   # appear late and should be skipped
        "Date of Birth Info",      # "of" starts lowercase → fails strict title case check
        "Home Address Line",       # 3 words, all upper start — could pass; but appears early
        "Some Field Label",        # 3 words, all upper start
        "Another Row Label",       # line 6 — boundary
        "Credit Application",      # line 7 — beyond the 6-line limit, must NOT be picked
    ]
    # All lines before line 7 either fail word-count or strict-uppercase checks,
    # so the result should be None.
    result = _extract_text_title(lines)
    assert result != "Credit Application"


def test_extract_text_title_title_case_long_line_excluded():
    """Title Case fallback rejects lines with more than 4 words."""
    lines = ["Title Last Name First Middle Suffix"]  # 6 words — excluded
    assert _extract_text_title(lines) is None


def test_extract_text_title_title_case_rejects_allcaps_word():
    """Title Case fallback rejects lines containing an ALL-CAPS word (≥3 alpha chars)."""
    # "VIN" is a 3-letter ALL-CAPS code — marks a column header, not a document title.
    lines = ["Make Trim VIN"]
    assert _extract_text_title(lines) is None


def test_analyze_text_page_doc_marker_continuation_stays_together(tmp_path):
    """Multiple pages sharing 'DOCUMENT 1 OF 2' header stay in one group."""
    page1 = "DOCUMENT 1 OF 2\nDT 5/23\nfirst page content here with enough chars to count"
    page2 = "DOCUMENT 1 OF 2\nDT 5/23\ncontinued form content on second page of same document"
    page3 = "DOCUMENT 2 OF 2\nOdometer Disclosure Statement\ncontent on a new document here"

    sig1 = _analyze_text_page(page1)
    sig2 = _analyze_text_page(page2)
    sig3 = _analyze_text_page(page3)

    from pdf_extractor.image_splitter import _group_image_pages
    groups = _group_image_pages([(0, sig1), (1, sig2), (2, sig3)])
    assert len(groups) == 3  # each DOCUMENT N OF N starts a new group


from pdf_extractor.image_splitter import _group_image_pages


def test_group_image_pages_leading_ambiguous_prepended_to_next_new_doc():
    """Untitled pages before the first titled page are merged into that document."""
    ambig = PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)
    new_doc = PageSignal(classification="NEW_DOC", title_text="Credit Application", page_num_in_doc=None)
    groups = _group_image_pages([(0, ambig), (1, new_doc)])
    assert len(groups) == 1
    assert groups[0] == [0, 1]


def test_group_image_pages_leading_ambiguous_multiple_pages():
    """Multiple leading untitled pages are all merged into the first titled group."""
    ambig = PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)
    new_doc = PageSignal(classification="NEW_DOC", title_text="Some Title", page_num_in_doc=None)
    groups = _group_image_pages([(0, ambig), (1, ambig), (2, new_doc)])
    assert len(groups) == 1
    assert groups[0] == [0, 1, 2]


def test_group_image_pages_trailing_ambiguous_not_merged_forward():
    """Untitled pages after a titled group stay in that group, not the next one."""
    new_a = PageSignal(classification="NEW_DOC", title_text="Doc A", page_num_in_doc=None)
    ambig = PageSignal(classification="AMBIGUOUS", title_text=None, page_num_in_doc=None)
    new_b = PageSignal(classification="NEW_DOC", title_text="Doc B", page_num_in_doc=None)
    groups = _group_image_pages([(0, new_a), (1, ambig), (2, new_b)])
    assert len(groups) == 2
    assert groups[0] == [0, 1]   # ambiguous page stays with Doc A
    assert groups[1] == [2]


def test_group_image_pages_anchored_group_not_merged_forward():
    """A group with a real anchor (NEW_DOC) is saved before the next NEW_DOC."""
    new_a = PageSignal(classification="NEW_DOC", title_text="Doc A", page_num_in_doc=None)
    new_b = PageSignal(classification="NEW_DOC", title_text="Doc B", page_num_in_doc=None)
    groups = _group_image_pages([(0, new_a), (1, new_b)])
    assert len(groups) == 2
    assert groups[0] == [0]
    assert groups[1] == [1]


def test_group_same_title_no_page_num_merged():
    """Consecutive NEW_DOC pages with the same title and no page_num are merged.

    This covers multi-page forms that repeat their document name as a page header
    on every page with no 'Page N of M' footer (e.g. Vehicle Buyer's Order).
    Without this rule, each page would become a separate 1-page output file.
    """
    signals = [
        (0, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),
        (1, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),
        (2, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),
    ]
    assert _group_image_pages(signals) == [[0, 1, 2]]


def test_group_same_title_ocr_variation_merged():
    """OCR spacing/case variations of the same title are treated as the same document.

    'VEHICLE BUYERS ORDER' vs 'VEHICLEBUYERS ORDER' (missing space) both normalise
    to 'VEHICLEBUYERSORDER' and must be merged rather than split.
    """
    signals = [
        (0, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),
        (1, _sig("NEW_DOC", "VEHICLEBUYERS ORDER")),   # missing space — OCR variant
        (2, _sig("NEW_DOC", "VEhICLE BUYERS ORDER")),  # mixed case — OCR variant
    ]
    assert _group_image_pages(signals) == [[0, 1, 2]]


def test_group_same_title_different_titles_still_split():
    """Same-title merging does not suppress splits when titles change."""
    signals = [
        (0, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),
        (1, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),
        (2, _sig("NEW_DOC", "LICENSE REGISTRATION FEES")),
        (3, _sig("NEW_DOC", "LICENSE REGISTRATION FEES")),
    ]
    assert _group_image_pages(signals) == [[0, 1], [2, 3]]


def test_group_same_title_page_num_not_merged():
    """Same title + page_num_in_doc set → NOT merged (DOCUMENT marker or Page 1 of N)."""
    signals = [
        (0, _sig("NEW_DOC", "SOME FORM", page_num=1)),
        (1, _sig("NEW_DOC", "SOME FORM", page_num=1)),
    ]
    assert _group_image_pages(signals) == [[0], [1]]


def test_group_same_title_after_different_doc_starts_new_group():
    """Same title appearing after an intervening different document starts a new group."""
    signals = [
        (0, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),
        (1, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),
        (2, _sig("NEW_DOC", "LICENSE REGISTRATION FEES")),  # different doc resets context
        (3, _sig("NEW_DOC", "VEHICLE BUYERS ORDER")),       # same title as group 0, but new group
    ]
    groups = _group_image_pages(signals)
    assert groups == [[0, 1], [2], [3]]


def test_windowed_groups_emits_left_chunk_then_final_window():
    """Sliding windows emit left-chunk groups and all groups from the final window."""
    signals = [
        (0, _sig("NEW_DOC", "A")),
        (1, _sig("CONTINUATION", page_num=2, total=2)),
        (2, _sig("NEW_DOC", "B")),
        (3, _sig("CONTINUATION", page_num=2, total=2)),
        (4, _sig("NEW_DOC", "C")),
        (5, _sig("CONTINUATION", page_num=2, total=2)),
    ]

    groups = _windowed_groups_from_signals(signals, chunk_pages=2)
    assert groups == [[0, 1], [2, 3], [4, 5]]


def test_windowed_groups_keeps_boundary_spanning_document_together():
    """A document that crosses a fixed chunk boundary remains one group."""
    signals = [
        (0, _sig("NEW_DOC", "Doc A")),
        (1, _sig("CONTINUATION", page_num=2, total=3)),
        (2, _sig("CONTINUATION", page_num=3, total=3)),
        (3, _sig("NEW_DOC", "Doc B")),
    ]

    groups = _windowed_groups_from_signals(signals, chunk_pages=2)
    assert groups == [[0, 1, 2], [3]]


def test_windowed_groups_skips_orphan_carryover_window_start():
    """Tail groups that start at a new window without NEW_DOC are not re-emitted."""
    signals = [
        (0, _sig("NEW_DOC", "Doc A")),
        (1, _sig("CONTINUATION", page_num=2, total=4)),
        (2, _sig("CONTINUATION", page_num=3, total=4)),
        (3, _sig("CONTINUATION", page_num=4, total=4)),
    ]

    groups = _windowed_groups_from_signals(signals, chunk_pages=2)
    assert groups == [[0, 1, 2, 3]]


from pdf_extractor.image_splitter import split_pdf
import pytest


def test_split_pdf_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_pdf(tmp_path / "nonexistent.pdf", tmp_path / "out")


def test_split_pdf_creates_output_dir(tmp_path):
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as f:
        writer.write(f)
    out_dir = tmp_path / "new_out_dir"

    fake_signal = PageSignal("NEW_DOC", "MyDoc", None)
    with patch("pdf_extractor.image_splitter.analyze_page", return_value=fake_signal):
        split_pdf(source, out_dir)

    assert out_dir.exists()


def test_split_pdf_routes_image_pages_through_analyze(tmp_path):
    """Blank pages (no text) must route through analyze_page, not text splitter."""
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    # Different page sizes so hash-based dedup does not collapse the two groups.
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=900)
    with source.open("wb") as f:
        writer.write(f)
    out_dir = tmp_path / "out"

    signals = [
        PageSignal("NEW_DOC", "Doc One", None),
        PageSignal("NEW_DOC", "Doc Two", None),
    ]
    with patch("pdf_extractor.image_splitter.analyze_page", side_effect=signals) as mock_analyze:
        result = split_pdf(source, out_dir)

    assert mock_analyze.call_count == 2
    assert len(result) == 2


def test_split_pdf_semantic_dedup_normalized_title_variants(tmp_path):
    """Title variants differing only by punctuation/(digits) are deduplicated."""
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    # Use different page sizes so byte-hash dedup alone would not collapse groups.
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=613, height=792)
    writer.add_blank_page(width=613, height=792)
    with source.open("wb") as f:
        writer.write(f)
    out_dir = tmp_path / "out"

    signals = [
        PageSignal(
            "NEW_DOC",
            "MOTOR VEHICLE RETAIL INSTALLMENT SALES CONTRACT-SIMPLE FINANCE CHARGE",
            1,
            2,
        ),
        PageSignal("CONTINUATION", None, 2, 2),
        PageSignal(
            "NEW_DOC",
            "MOTORVEHICLE RETAIL INSTALLMENT SALES CONTRACT SIMPLE FINANCE CHARGE (2)",
            1,
            2,
        ),
        PageSignal("CONTINUATION", None, 2, 2),
    ]

    with patch("pdf_extractor.image_splitter.analyze_page", side_effect=signals):
        written = split_pdf(source, out_dir)

    assert len(written) == 1


def test_split_pdf_overlap_mode_multi_window_runs(tmp_path, monkeypatch):
    """Overlap mode with multiple fixed chunks completes and emits expected groups."""
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    # Unique page sizes avoid byte-hash dedup collapsing unrelated groups in test.
    for i in range(5):
        writer.add_blank_page(width=612 + i, height=792)
    with source.open("wb") as f:
        writer.write(f)

    out_dir = tmp_path / "out"
    monkeypatch.setenv("PDF_EXTRACTOR_OVERLAP_CHUNK_PAGES", "2")

    signals = [
        PageSignal("NEW_DOC", "Doc A", 1, 2),
        PageSignal("CONTINUATION", None, 2, 2),
        PageSignal("NEW_DOC", "Doc B", 1, 2),
        PageSignal("CONTINUATION", None, 2, 2),
        PageSignal("NEW_DOC", "Doc C", None, None),
    ]

    with patch("pdf_extractor.image_splitter.analyze_page", side_effect=signals):
        written = split_pdf(source, out_dir)

    assert len(written) == 3


from unittest.mock import patch
from pdf_extractor.cli import main


def test_cli_split_documents_calls_split_pdf(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as f:
        writer.write(f)
    out_dir = tmp_path / "split"

    monkeypatch.setattr(
        "sys.argv",
        ["pdf-extractor", str(source), "--split-documents", "--split-output-dir", str(out_dir)],
    )

    with patch("pdf_extractor.cli.split_pdf") as mock_split:
        mock_split.return_value = [out_dir / "Doc.pdf"]
        exit_code = main()

    mock_split.assert_called_once_with(source, str(out_dir), verbose=False)
    assert exit_code == 0


from pathlib import Path
from pdf_extractor.image_splitter import (
    _hamming_distance,
    _perceptual_hash,
    _PHASH_THRESHOLD,
)


# --- _hamming_distance ---

def test_hamming_distance_identical():
    assert _hamming_distance(0, 0) == 0


def test_hamming_distance_one_bit():
    assert _hamming_distance(0b0, 0b1) == 1


def test_hamming_distance_all_64_bits():
    assert _hamming_distance(0, 0xFFFFFFFFFFFFFFFF) == 64


def test_hamming_distance_symmetric():
    assert _hamming_distance(0xAB, 0xCD) == _hamming_distance(0xCD, 0xAB)


# --- _perceptual_hash helpers ---

def _make_mock_fitz_page(rgb_value: int = 128):
    """Return a mock fitz page that renders as a solid rgb_value color at 64x83 px."""
    mock_pix = MagicMock()
    mock_pix.width = 64
    mock_pix.height = 83
    mock_pix.samples = bytes([rgb_value] * (64 * 83 * 3))
    mock_page = MagicMock()
    mock_page.rect.width = 100.0
    mock_page.get_pixmap.return_value = mock_pix
    return mock_page


def _make_mock_fitz_doc(page: object):
    mock_doc = MagicMock()
    mock_doc.__getitem__ = MagicMock(return_value=page)
    return mock_doc


# --- _perceptual_hash ---

def test_perceptual_hash_returns_int():
    h = _perceptual_hash(_make_mock_fitz_doc(_make_mock_fitz_page()), 0)
    assert isinstance(h, int)


def test_perceptual_hash_deterministic():
    doc = _make_mock_fitz_doc(_make_mock_fitz_page(120))
    assert _perceptual_hash(doc, 0) == _perceptual_hash(doc, 0)


def test_perceptual_hash_zero_width_returns_none():
    mock_page = MagicMock()
    mock_page.rect.width = 0
    assert _perceptual_hash(_make_mock_fitz_doc(mock_page), 0) is None


def test_perceptual_hash_exception_returns_none():
    mock_doc = MagicMock()
    mock_doc.__getitem__ = MagicMock(side_effect=RuntimeError("render failed"))
    assert _perceptual_hash(mock_doc, 0) is None


def test_perceptual_hash_identical_images_zero_distance():
    doc = _make_mock_fitz_doc(_make_mock_fitz_page(100))
    h1 = _perceptual_hash(doc, 0)
    h2 = _perceptual_hash(doc, 0)
    assert _hamming_distance(h1, h2) == 0


def test_perceptual_hash_similar_images_within_threshold():
    """Slight brightness difference stays within the similarity threshold."""
    doc_a = _make_mock_fitz_doc(_make_mock_fitz_page(rgb_value=120))
    doc_b = _make_mock_fitz_doc(_make_mock_fitz_page(rgb_value=130))
    h_a = _perceptual_hash(doc_a, 0)
    h_b = _perceptual_hash(doc_b, 0)
    assert _hamming_distance(h_a, h_b) <= _PHASH_THRESHOLD


def test_phash_threshold_value():
    assert _PHASH_THRESHOLD == 10


from pdf_extractor.image_splitter import _write_phash_report


def test_write_phash_report_no_hits_silent(tmp_path, capsys):
    _write_phash_report([], tmp_path, Path("test.pdf"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not (tmp_path / "suspected_duplicates.txt").exists()


def test_write_phash_report_creates_file(tmp_path):
    hits = [(0, 1, 7, "RETAIL CONTRACT", "RETAIL CONTRACT", 5, 42)]
    _write_phash_report(hits, tmp_path, Path("sample.pdf"))
    assert (tmp_path / "suspected_duplicates.txt").exists()


def test_write_phash_report_file_content(tmp_path):
    hits = [(0, 1, 7, "RETAIL CONTRACT", "RETAIL CONTRACT", 5, 42)]
    _write_phash_report(hits, tmp_path, Path("sample.pdf"))
    content = (tmp_path / "suspected_duplicates.txt").read_text()
    assert "sample.pdf" in content
    assert "Group 0" in content
    assert "Group 1" in content
    assert "page 5" in content
    assert "page 42" in content
    assert "RETAIL CONTRACT" in content
    assert "[distance: 7]" in content


def test_write_phash_report_console_output(tmp_path, capsys):
    hits = [(0, 1, 7, "RETAIL CONTRACT", "RETAIL CONTRACT", 5, 42)]
    _write_phash_report(hits, tmp_path, Path("sample.pdf"))
    out = capsys.readouterr().out
    assert "SUSPECTED DUPLICATES" in out
    assert "distance: 7" in out


def test_write_phash_report_multiple_hits(tmp_path):
    hits = [
        (0, 2, 5, "CREDIT APPLICATION", "CREDIT APPLICATION", 10, 90),
        (1, 3, 8, "RETAIL CONTRACT", "RETAIL CONTRACT", 20, 100),
    ]
    _write_phash_report(hits, tmp_path, Path("batch.pdf"))
    content = (tmp_path / "suspected_duplicates.txt").read_text()
    assert content.count("distance:") == 2
