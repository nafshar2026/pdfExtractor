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
