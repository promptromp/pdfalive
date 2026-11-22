"""Unit tests for OCR processor."""

import atexit
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pymupdf
import pytest

from pdfalive.processors.ocr_detection import NoTextDetectionStrategy
from pdfalive.processors.ocr_processor import OCRProcessor
from pdfalive.tests.scanned_pages import (
    BOOKMARKS,
    CROPBOX,
    OCR_DPI,
    SCAN_DPI,
    TITLE,
    ocr_word_boxes,
    placed_on,
    scan_page,
    shown_word_boxes,
    text_page,
)


@pytest.fixture
def expected_word_boxes() -> dict[str, pymupdf.Rect]:
    """Where an unrotated test page shows each of SHOWN_WORDS."""
    with pymupdf.open() as doc:
        return shown_word_boxes(text_page(doc))


# Text of an annotation added over a scanned page, large enough that OCR would read it.
ANNOTATION_TEXT = "MARGINALIA"


def _sequential(**kwargs) -> OCRProcessor:
    return OCRProcessor(language="eng", dpi=OCR_DPI, num_processes=1, **kwargs)


class TestOCRProcessor:
    """Tests for OCRProcessor class."""

    @pytest.fixture
    def mock_detection_strategy(self):
        """Create a mock detection strategy."""
        strategy = MagicMock(spec=NoTextDetectionStrategy)
        strategy.needs_ocr.return_value = True
        return strategy

    @pytest.fixture
    def mock_doc(self):
        """Create a mock document."""
        doc = MagicMock()
        doc.page_count = 5
        return doc

    def test_init_default_values(self):
        """Test OCRProcessor initialization with default values."""
        processor = OCRProcessor()

        assert isinstance(processor.detection_strategy, NoTextDetectionStrategy)
        assert processor.language == "eng"
        assert processor.dpi == 300
        assert processor.num_processes >= 1
        assert processor.include_pages_with_text is False

    def test_init_custom_values(self, mock_detection_strategy):
        """Test OCRProcessor initialization with custom values."""
        processor = OCRProcessor(
            detection_strategy=mock_detection_strategy,
            language="deu",
            dpi=150,
            num_processes=4,
            include_pages_with_text=True,
        )

        assert processor.detection_strategy is mock_detection_strategy
        assert processor.language == "deu"
        assert processor.dpi == 150
        assert processor.num_processes == 4
        assert processor.include_pages_with_text is True

    def test_needs_ocr_delegates_to_strategy(self, mock_detection_strategy, mock_doc):
        """Test that needs_ocr delegates to the detection strategy."""
        processor = OCRProcessor(detection_strategy=mock_detection_strategy)

        result = processor.needs_ocr(mock_doc)

        mock_detection_strategy.needs_ocr.assert_called_once_with(mock_doc)
        assert result is True

    def test_needs_ocr_returns_false_when_strategy_says_no(self, mock_doc):
        """Test needs_ocr returns False when strategy determines no OCR needed."""
        strategy = MagicMock()
        strategy.needs_ocr.return_value = False
        processor = OCRProcessor(detection_strategy=strategy)

        result = processor.needs_ocr(mock_doc)

        assert result is False


class TestOCRTextLayer:
    """The OCR text layer added to each page, and what is left as it was."""

    def test_adds_text_where_the_words_are_shown(self, scanned_pdf, expected_word_boxes):
        result = _sequential().process(str(scanned_pdf), show_progress=False)

        for page in result:
            ocr_boxes = ocr_word_boxes(page)
            assert set(ocr_boxes) == set(expected_word_boxes)
            for word, shown_box in expected_word_boxes.items():
                assert placed_on(ocr_boxes[word], shown_box), (word, ocr_boxes[word], shown_box)

    def test_keeps_pages_as_they_were(self, scanned_pdf):
        result = _sequential().process(str(scanned_pdf), show_progress=False)

        with pymupdf.open(scanned_pdf) as original:
            assert result.page_count == original.page_count
            for before, after in zip(original, result, strict=True):
                assert after.rect == before.rect
                assert after.get_pixmap(dpi=SCAN_DPI).samples == before.get_pixmap(dpi=SCAN_DPI).samples
                assert [original.extract_image(image[0])["image"] for image in before.get_images()] == [
                    result.extract_image(image[0])["image"] for image in after.get_images()
                ]

    def test_keeps_bookmarks_and_metadata(self, scanned_pdf):
        result = _sequential().process(str(scanned_pdf), show_progress=False)

        assert result.get_toc() == BOOKMARKS
        assert result.metadata["title"] == TITLE

    @pytest.mark.parametrize("rotation", [0, 90, 180, 270])
    @pytest.mark.parametrize("cropbox", [None, CROPBOX], ids=["uncropped", "cropped"])
    def test_places_text_on_rotated_and_cropped_pages(self, rotation, cropbox):
        with pymupdf.open() as doc:
            text_page(doc, rotation, cropbox)
            result = _sequential(include_pages_with_text=True).process_in_memory(doc, show_progress=False)

        page = result[0]
        shown_boxes, ocr_boxes = shown_word_boxes(page), ocr_word_boxes(page)
        assert set(ocr_boxes) == set(shown_boxes)
        for word, shown_box in shown_boxes.items():
            assert placed_on(ocr_boxes[word], shown_box), (word, ocr_boxes[word], shown_box)
        assert page.rotation == rotation

    def test_annotations_are_kept_but_not_recognized(self, tmp_path, expected_word_boxes):
        """A page is OCR'd without its annotations, which stay on it.

        Text extraction reads annotations' text as the page's, so a page with a text
        annotation counts as having text and is OCR'd only with include_pages_with_text.
        """
        path = tmp_path / "annotated.pdf"
        with pymupdf.open() as doc:
            page = scan_page(doc)
            page.add_freetext_annot(pymupdf.Rect(20, 360, 580, 440), ANNOTATION_TEXT, fontsize=36)
            doc.save(path)

        result = _sequential(include_pages_with_text=True).process(str(path), show_progress=False)

        assert set(ocr_word_boxes(result[0])) == set(expected_word_boxes)
        assert [annot.info["content"] for annot in result[0].annots()] == [ANNOTATION_TEXT]

    def test_non_pdf_input_becomes_a_pdf_with_the_text_layer(self, tmp_path, expected_word_boxes):
        image_path = tmp_path / "scan.png"
        with pymupdf.open() as doc:
            scan_page(doc).get_pixmap(dpi=SCAN_DPI).save(image_path)

        result = _sequential().process(str(image_path), show_progress=False)

        assert result.is_pdf
        assert set(ocr_word_boxes(result[0])) == set(expected_word_boxes)


class TestPagesToOCR:
    """Which pages are OCR'd."""

    def test_pages_without_text_by_default(self, mixed_pdf):
        with pymupdf.open(mixed_pdf) as doc:
            assert _sequential().pages_to_ocr(doc) == [1]

    def test_every_page_when_including_pages_with_text(self, mixed_pdf):
        with pymupdf.open(mixed_pdf) as doc:
            assert _sequential(include_pages_with_text=True).pages_to_ocr(doc) == [0, 1]

    def test_given_pages_only(self, scanned_pdf):
        with pymupdf.open(scanned_pdf) as doc:
            result = _sequential().process_in_memory(doc, show_progress=False, page_indices=[1])

        assert not ocr_word_boxes(result[0])
        assert ocr_word_boxes(result[1])

    def test_page_with_text_is_left_as_it_was(self, mixed_pdf):
        result = _sequential().process(str(mixed_pdf), show_progress=False)

        with pymupdf.open(mixed_pdf) as original:
            assert result[0].get_text() == original[0].get_text()
        assert not ocr_word_boxes(result[0])
        assert ocr_word_boxes(result[1])

    def test_document_with_text_on_every_page_is_copied_unchanged(self):
        with pymupdf.open() as doc:
            text_page(doc)
            result = _sequential().process_in_memory(doc, show_progress=False)
            assert result.get_page_text(0) == doc.get_page_text(0)
        assert not ocr_word_boxes(result[0])


class TestProcessInMemory:
    """The document returned, and the original."""

    def test_does_not_modify_the_original(self, scanned_pdf):
        with pymupdf.open(scanned_pdf) as doc:
            _sequential().process_in_memory(doc, show_progress=False)
            assert all(not page.get_text().strip() for page in doc)

    def test_in_memory_document_gives_in_memory_copy(self, scanned_pdf):
        with pymupdf.open(stream=scanned_pdf.read_bytes(), filetype="pdf") as doc:
            result = _sequential().process_in_memory(doc, show_progress=False)

        assert not result.name
        assert all(ocr_word_boxes(page) for page in result)

    def test_parallel_ocr_matches_sequential(self, scanned_pdf):
        parallel = OCRProcessor(language="eng", dpi=OCR_DPI, num_processes=2)

        with pymupdf.open(scanned_pdf) as doc:
            in_parallel = parallel.process_in_memory(doc, show_progress=False)
            in_sequence = _sequential().process_in_memory(doc, show_progress=False)

        assert [page.get_text() for page in in_parallel] == [page.get_text() for page in in_sequence]

    def test_file_backed_document_gives_file_backed_copy_removed_at_exit(self, scanned_pdf, monkeypatch):
        registered: list[Callable[[], None]] = []
        monkeypatch.setattr(atexit, "register", registered.append)

        with pymupdf.open(scanned_pdf) as doc:
            result = _sequential().process_in_memory(doc, show_progress=False)
        copy_path = Path(result.name)

        assert copy_path.exists() and copy_path != scanned_pdf
        assert len(registered) == 1
        result.close()
        registered[0]()
        assert not copy_path.exists()
        registered[0]()  # a copy already removed is no error
