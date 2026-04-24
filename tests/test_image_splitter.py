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
