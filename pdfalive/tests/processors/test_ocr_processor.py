"""Unit tests for OCR processor."""

from unittest.mock import MagicMock, patch

import pytest

from pdfalive.processors.ocr_detection import NoTextDetectionStrategy
from pdfalive.processors.ocr_processor import OCRProcessor, _ocr_page_range


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

    def test_init_custom_values(self, mock_detection_strategy):
        """Test OCRProcessor initialization with custom values."""
        processor = OCRProcessor(
            detection_strategy=mock_detection_strategy,
            language="deu",
            dpi=150,
            num_processes=4,
        )

        assert processor.detection_strategy is mock_detection_strategy
        assert processor.language == "deu"
        assert processor.dpi == 150
        assert processor.num_processes == 4

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

    @patch("pdfalive.processors.ocr_processor.pymupdf")
    def test_process_sequential_for_small_docs(self, mock_pymupdf):
        """Test that single-page documents use sequential processing."""
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_pymupdf.open.return_value = mock_doc

        processor = OCRProcessor(num_processes=4)
        mock_result_doc = MagicMock()
        processor._process_sequential = MagicMock(return_value=mock_result_doc)

        result = processor.process("/path/to/file.pdf", show_progress=False)

        processor._process_sequential.assert_called_once()
        assert result == mock_result_doc

    @patch("pdfalive.processors.ocr_processor.pymupdf")
    def test_process_in_memory_returns_new_document(self, mock_pymupdf):
        """Test that process_in_memory returns a new document."""
        # Setup mock input document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.page_count = 1

        # Setup mock pixmap and OCR
        mock_pixmap = MagicMock()
        mock_pixmap.pdfocr_tobytes.return_value = b"fake_pdf_bytes"
        mock_page.get_pixmap.return_value = mock_pixmap

        # Setup mock output document
        mock_output_doc = MagicMock()
        mock_ocr_page_doc = MagicMock()

        # First call returns output doc, second returns OCR page doc
        mock_pymupdf.open.side_effect = [mock_output_doc, mock_ocr_page_doc]
        mock_pymupdf.Matrix.return_value = MagicMock()

        processor = OCRProcessor(language="eng", dpi=300)
        result = processor.process_in_memory(mock_doc, show_progress=False)

        # Verify new document is returned (not the input doc)
        assert result == mock_output_doc
        assert result != mock_doc

    @patch("pdfalive.processors.ocr_processor.pymupdf")
    def test_process_in_memory_uses_correct_dpi(self, mock_pymupdf):
        """Test that process_in_memory uses the configured DPI."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.page_count = 1

        mock_pixmap = MagicMock()
        mock_pixmap.pdfocr_tobytes.return_value = b"fake_pdf_bytes"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_output_doc = MagicMock()
        mock_ocr_page_doc = MagicMock()
        mock_pymupdf.open.side_effect = [mock_output_doc, mock_ocr_page_doc]

        mock_matrix = MagicMock()
        mock_pymupdf.Matrix.return_value = mock_matrix

        # Use 150 DPI (zoom = 150/72 ≈ 2.08)
        processor = OCRProcessor(language="eng", dpi=150)
        processor.process_in_memory(mock_doc, show_progress=False)

        # Verify Matrix was created with correct zoom factor
        expected_zoom = 150 / 72.0
        mock_pymupdf.Matrix.assert_called_with(expected_zoom, expected_zoom)

    @patch("pdfalive.processors.ocr_processor.pymupdf")
    def test_process_in_memory_uses_correct_language(self, mock_pymupdf):
        """Test that process_in_memory uses the configured language."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.page_count = 1

        mock_pixmap = MagicMock()
        mock_pixmap.pdfocr_tobytes.return_value = b"fake_pdf_bytes"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_output_doc = MagicMock()
        mock_ocr_page_doc = MagicMock()
        mock_pymupdf.open.side_effect = [mock_output_doc, mock_ocr_page_doc]
        mock_pymupdf.Matrix.return_value = MagicMock()

        processor = OCRProcessor(language="fra", dpi=300)
        processor.process_in_memory(mock_doc, show_progress=False)

        # Verify pdfocr_tobytes was called with correct language
        mock_pixmap.pdfocr_tobytes.assert_called_with(language="fra")


class TestOCRPageRangeWorker:
    """Tests for the _ocr_page_range worker function."""

    @patch("pdfalive.processors.ocr_processor.pymupdf")
    def test_worker_calculates_page_range_correctly(self, mock_pymupdf):
        """Test that worker correctly calculates its page range."""
        mock_doc = MagicMock()
        mock_doc.page_count = 10

        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.pdfocr_tobytes.return_value = b"fake_pdf"
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_output_doc = MagicMock()
        mock_ocr_page_doc = MagicMock()

        # Return different docs for different calls
        mock_pymupdf.open.side_effect = (
            lambda *args, **kwargs: (mock_doc if args == () or args[0] != "pdf" else mock_ocr_page_doc)
            if args
            else mock_output_doc
        )
        mock_pymupdf.open.return_value = mock_doc
        mock_pymupdf.Matrix.return_value = MagicMock()

        # Need to handle multiple open() calls
        call_count = [0]

        def open_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_doc  # input doc
            elif args and args[0] == "pdf":
                return mock_ocr_page_doc  # OCR page docs
            else:
                return mock_output_doc  # output doc

        mock_pymupdf.open.side_effect = open_side_effect

        args = (0, 2, "/path/to/file.pdf", "/tmp", "eng", 300)
        start, end, path = _ocr_page_range(args)

        assert start == 0
        assert end == 5  # First half of 10 pages

    @patch("pdfalive.processors.ocr_processor.pymupdf")
    def test_worker_processes_last_chunk_correctly(self, mock_pymupdf):
        """Test that last worker gets remaining pages."""
        mock_doc = MagicMock()
        mock_doc.page_count = 10

        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.pdfocr_tobytes.return_value = b"fake_pdf"
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_output_doc = MagicMock()
        mock_ocr_page_doc = MagicMock()

        call_count = [0]

        def open_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_doc
            elif args and args[0] == "pdf":
                return mock_ocr_page_doc
            else:
                return mock_output_doc

        mock_pymupdf.open.side_effect = open_side_effect
        mock_pymupdf.Matrix.return_value = MagicMock()

        args = (1, 2, "/path/to/file.pdf", "/tmp", "eng", 300)
        start, end, path = _ocr_page_range(args)

        assert start == 5
        assert end == 10  # All remaining pages

    @patch("pdfalive.processors.ocr_processor.pymupdf")
    def test_worker_saves_output(self, mock_pymupdf):
        """Test that worker saves processed pages to output file."""
        mock_doc = MagicMock()
        mock_doc.page_count = 2

        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.pdfocr_tobytes.return_value = b"fake_pdf"
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_output_doc = MagicMock()
        mock_ocr_page_doc = MagicMock()

        call_count = [0]

        def open_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_doc
            elif args and args[0] == "pdf":
                return mock_ocr_page_doc
            else:
                return mock_output_doc

        mock_pymupdf.open.side_effect = open_side_effect
        mock_pymupdf.Matrix.return_value = MagicMock()

        args = (0, 1, "/path/to/file.pdf", "/tmp/output", "eng", 300)
        start, end, path = _ocr_page_range(args)

        # Output document should be saved
        mock_output_doc.save.assert_called_once()
        mock_output_doc.close.assert_called_once()
        mock_doc.close.assert_called_once()

        # Path should be in the output directory
        assert "/tmp/output" in path
        assert "ocr_part_0.pdf" in path
