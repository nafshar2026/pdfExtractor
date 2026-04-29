"""Tests for image-based PDF splitting via PaddleOCR title detection."""

from pdf_extractor.image_splitter import PageSignal, _extract_ocr_texts


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
    # Bottom strip: "Page 3 of 6"; top strip: short title text
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result(["Page 3 of 6"]),   # bottom strip
        _make_ocr_result(["Some Title"]),     # top strip (not reached, continuation wins)
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "CONTINUATION"
    assert sig.page_num_in_doc == 3
    assert sig.title_text is None


def test_analyze_page_title_detected():
    page = _make_image_page(_blank_image())
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result([]),                     # bottom strip: no continuation
        _make_ocr_result(["ST-556 State Tax"]),   # top strip: short title
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "ST-556 State Tax"
    assert sig.page_num_in_doc is None


def test_analyze_page_ambiguous_long_top_text():
    page = _make_image_page(_blank_image())
    long_text = "x" * 61
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result([]),           # bottom: no continuation
        _make_ocr_result([long_text]),  # top: too long to be a title
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
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
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result(["Page 1 of 6"]),    # bottom: page 1 of 6
        _make_ocr_result(["Title Here"]),      # top: title found
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "Title Here"
    assert sig.page_num_in_doc == 1
    assert sig.total_pages_in_doc == 6


def test_analyze_page_page_1_of_n_no_title_is_new_doc():
    """'Page 1 of N' with no detectable title must still be NEW_DOC, not AMBIGUOUS."""
    page = _make_image_page(_blank_image())
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result(["Page 1 of 4"]),  # bottom: page 1 of 4
        _make_ocr_result([]),               # top: nothing usable
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text is None
    assert sig.page_num_in_doc == 1
    assert sig.total_pages_in_doc == 4


def test_analyze_page_continuation_carries_total():
    """CONTINUATION signal must carry total_pages_in_doc from the footer."""
    page = _make_image_page(_blank_image())
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result(["Page 3 of 6"]),
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
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


def test_group_leading_ambiguous_forms_own_group():
    """AMBIGUOUS pages before any NEW_DOC form their own group."""
    signals = [
        (0, _sig("AMBIGUOUS")),
        (1, _sig("NEW_DOC", "Title")),
    ]
    assert _group_image_pages(signals) == [[0], [1]]


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
    for _ in range(4):
        w.add_blank_page(width=612, height=792)
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
    assert written[0].name == "Doc_One.pdf"
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
    text = "FORM NO. LAWIL-RATECAP (Rev. 8/22)\nThe information on this form is part of your contract.\nDISCLOSURE OF 36% RATE CAP\nFurther content of the disclosure form goes here."
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
    lines = ["ODOMETER DISCLOSURE STATEMENT", "WARNING ODOMETER DISCREPENCY"]
    assert _extract_text_title(lines) == "ODOMETER DISCLOSURE STATEMENT"


def test_extract_text_title_prefer_last_picks_later():
    """prefer_last=True skips earlier candidates in favour of later ones."""
    lines = ["FEDERAL TRUTH IN LENDING DISCLOSURES", "RETAIL INSTALLMENT CONTRACT"]
    assert _extract_text_title(lines, prefer_last=True) == "RETAIL INSTALLMENT CONTRACT"


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
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
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

    mock_split.assert_called_once_with(source, str(out_dir))
    assert exit_code == 0
