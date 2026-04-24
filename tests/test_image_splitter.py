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


def _make_ocr_result(texts: list[str]) -> list:
    """Build a PaddleOCR-format result from a list of text strings."""
    lines = []
    for text in texts:
        bbox = [[0, 0], [100, 0], [100, 20], [0, 20]]
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


def test_analyze_page_page_1_of_n_is_not_continuation():
    """'Page 1 of N' must NOT trigger CONTINUATION — only X > 1 does."""
    page = _make_image_page(_blank_image())
    mock_ocr = MagicMock()
    mock_ocr.ocr.side_effect = [
        _make_ocr_result(["Page 1 of 6"]),    # bottom: page 1 of 6 → ignored
        _make_ocr_result(["Title Here"]),      # top: short title → NEW_DOC
    ]
    with patch("pdf_extractor.image_splitter._get_ocr", return_value=mock_ocr):
        sig = analyze_page(page)
    assert sig.classification == "NEW_DOC"
    assert sig.title_text == "Title Here"


from pdf_extractor.image_splitter import _group_image_pages


def _sig(cls, title=None, page_num=None):
    return PageSignal(classification=cls, title_text=title, page_num_in_doc=page_num)


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
from pdf_extractor.image_splitter import _split_image_run, _split_text_run


def test_split_image_run_produces_one_file_per_group(tmp_path):
    reader = _make_real_reader(tmp_path, num_pages=4)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    used_names: dict = {}

    signals = [
        PageSignal("NEW_DOC", "Doc One", None),
        PageSignal("CONTINUATION", None, 2),
        PageSignal("NEW_DOC", "Doc Two", None),
        PageSignal("AMBIGUOUS", None, None),
    ]

    with patch("pdf_extractor.image_splitter.analyze_page", side_effect=signals):
        written = _split_image_run(reader, 0, 4, out_dir, used_names)

    assert len(written) == 2
    assert written[0].name == "Doc_One.pdf"
    assert written[1].name == "Doc_Two.pdf"
    result_0 = PdfReader(str(written[0]))
    assert len(result_0.pages) == 2
    result_1 = PdfReader(str(written[1]))
    assert len(result_1.pages) == 2


def test_split_text_run_splits_by_markers(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    pdf_path = tmp_path / "source.pdf"
    with pdf_path.open("wb") as f:
        writer.write(f)
    reader = PdfReader(str(pdf_path))

    page_texts = [
        "DOCUMENT 1 OF 2\nApplicable Law\nsome content here to exceed fifty chars minimum",
        "continued content page two here with enough text to count as substantial",
        "DOCUMENT 2 OF 2\nOdometer Disclosure Statement\ncontent here enough chars",
    ]
    used_names: dict = {}

    written = _split_text_run(reader, page_texts, 0, 3, out_dir, used_names)

    assert len(written) == 2
    r0 = PdfReader(str(written[0]))
    assert len(r0.pages) == 2
    r1 = PdfReader(str(written[1]))
    assert len(r1.pages) == 1


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
