"""Unit tests for OCR processor."""

import atexit
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
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
        processor._process_sequential_from_path = MagicMock(return_value=mock_result_doc)

        result = processor.process("/path/to/file.pdf", show_progress=False)

        processor._process_sequential_from_path.assert_called_once()
        assert result == mock_result_doc

    @patch("pdfalive.processors.ocr_processor.pymupdf")
    def test_process_in_memory_returns_new_document(self, mock_pymupdf):
        """Test that process_in_memory returns a new document."""
        # Setup mock input document (in-memory, no file path)
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.page_count = 1
        mock_doc.name = ""  # In-memory document has no name

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
        mock_doc.name = ""  # In-memory document

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
        mock_doc.name = ""  # In-memory document

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

    def test_process_in_memory_uses_parallel_for_file_backed_doc(self):
        """Test that process_in_memory uses parallel processing for file-backed documents."""
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_doc.name = "/path/to/file.pdf"  # File-backed document

        processor = OCRProcessor(num_processes=4)

        with patch.object(processor, "_process_parallel", return_value=MagicMock()) as mock_parallel:
            processor.process_in_memory(mock_doc, show_progress=False)

            mock_parallel.assert_called_once_with("/path/to/file.pdf", 4, False)

    def test_process_in_memory_uses_sequential_for_in_memory_doc(self):
        """Test that process_in_memory uses sequential processing for in-memory documents."""
        mock_doc = MagicMock()
        mock_doc.page_count = 10
        mock_doc.name = ""  # In-memory document (no file path)

        processor = OCRProcessor(num_processes=4)

        with patch.object(processor, "_process_sequential_from_doc", return_value=MagicMock()) as mock_sequential:
            processor.process_in_memory(mock_doc, show_progress=False)

            mock_sequential.assert_called_once_with(mock_doc, False)

    def test_process_in_memory_uses_sequential_for_single_page(self):
        """Test that process_in_memory uses sequential for single-page file-backed documents."""
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.name = "/path/to/file.pdf"  # File-backed but only 1 page

        processor = OCRProcessor(num_processes=4)

        with (
            patch.object(processor, "_process_sequential_from_doc", return_value=MagicMock()) as mock_sequential,
            patch.object(processor, "_process_parallel", return_value=MagicMock()) as mock_parallel,
        ):
            processor.process_in_memory(mock_doc, show_progress=False)

            # Should use sequential since num_processes would be min(4, 1) = 1
            mock_sequential.assert_called_once()
            mock_parallel.assert_not_called()

    def test_process_in_memory_limits_processes_to_page_count(self):
        """Test that process_in_memory limits parallel processes to page count."""
        mock_doc = MagicMock()
        mock_doc.page_count = 3
        mock_doc.name = "/path/to/file.pdf"

        processor = OCRProcessor(num_processes=10)

        with patch.object(processor, "_process_parallel", return_value=MagicMock()) as mock_parallel:
            processor.process_in_memory(mock_doc, show_progress=False)

            # num_processes should be min(10, 3) = 3
            mock_parallel.assert_called_once_with("/path/to/file.pdf", 3, False)


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
        mock_pymupdf.open.side_effect = lambda *args, **kwargs: (
            (mock_doc if args == () or args[0] != "pdf" else mock_ocr_page_doc) if args else mock_output_doc
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


class TestOCRProcessorParallelIntegration:
    """Integration tests for parallel OCR processing using real PDF files.

    These tests use actual temporary PDF files to test the parallel processing
    code path, since multiprocessing requires pickle-able objects (MagicMock
    cannot be pickled).
    """

    @pytest.fixture
    def multi_page_pdf_path(self, tmp_path):
        """Create a temporary multi-page PDF for testing parallel processing."""
        pdf_path = tmp_path / "test_multipage.pdf"
        doc = pymupdf.open()

        # Create 4 pages (enough to trigger parallel processing with num_processes=2)
        for i in range(4):
            page = doc.new_page(width=612, height=792)  # Standard letter size
            # Add some simple content to each page
            text_point = pymupdf.Point(72, 72)
            page.insert_text(text_point, f"Page {i + 1} content", fontsize=12)

        doc.save(str(pdf_path))
        doc.close()

        return str(pdf_path)

    def test_process_parallel_produces_correct_page_count(self, multi_page_pdf_path):
        """Test that parallel processing produces output with correct page count."""
        processor = OCRProcessor(language="eng", dpi=72, num_processes=2)

        # Process the document in parallel
        result_doc = processor._process_parallel(multi_page_pdf_path, num_processes=2, show_progress=False)

        # Verify the result has the same number of pages as input
        input_doc = pymupdf.open(multi_page_pdf_path)
        expected_page_count = input_doc.page_count
        input_doc.close()

        assert result_doc.page_count == expected_page_count
        result_doc.close()

    def test_process_in_memory_uses_parallel_for_real_file(self, multi_page_pdf_path):
        """Test that process_in_memory actually uses parallel path for file-backed docs."""
        processor = OCRProcessor(language="eng", dpi=72, num_processes=2)

        # Open document from file (has doc.name set)
        doc = pymupdf.open(multi_page_pdf_path)
        assert doc.name == multi_page_pdf_path  # Verify file path is set

        # Process the document - should use parallel path
        result_doc = processor.process_in_memory(doc, show_progress=False)

        # Verify the result has the same number of pages
        assert result_doc.page_count == doc.page_count

        doc.close()
        result_doc.close()

    def test_process_parallel_returns_file_backed_doc(self, tmp_path, monkeypatch):
        """Test that _process_parallel returns a file-backed document."""
        # Create a simple input PDF to pass as input_path
        input_pdf = tmp_path / "input.pdf"
        doc = pymupdf.open()
        for _ in range(4):
            doc.new_page()
        doc.save(str(input_pdf))
        doc.close()

        # Monkeypatch the OCR worker to create simple part PDFs without running real OCR
        def fake_ocr_page_range(args):
            process_idx, total_processes, input_path, output_dir, language, dpi = args
            part_path = Path(output_dir) / f"ocr_part_{process_idx}.pdf"
            part_doc = pymupdf.open()
            part_doc.new_page()
            part_doc.save(str(part_path))
            part_doc.close()
            # return a plausible start/end range along with path
            return process_idx, process_idx + 1, str(part_path)

        # Replace the module-level _ocr_page_range with our fake
        monkeypatch.setattr("pdfalive.processors.ocr_processor._ocr_page_range", fake_ocr_page_range)

        # Replace Pool with a simple fake that runs the function sequentially in-process
        class FakePool:
            def __init__(self, processes=None):
                pass

            def imap_unordered(self, func, args_list):
                for args in args_list:
                    yield func(args)

            def map(self, func, args_list):
                return [func(a) for a in args_list]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr("pdfalive.processors.ocr_processor.Pool", FakePool)

        ocr = OCRProcessor(num_processes=2)

        # Call the parallel processor; because we've faked the worker and Pool,
        # it will create parts and merge them into a single PDF file and return
        # a file-backed pymupdf.Document
        result_doc = ocr._process_parallel(str(input_pdf), num_processes=2, show_progress=False)

        # The returned document should be file-backed (truthy .name) and file should exist
        assert getattr(result_doc, "name", None), "Returned OCR doc is not file-backed"
        assert Path(result_doc.name).exists(), "Temporary OCR output file does not exist"

        # Cleanup
        result_doc.close()
        Path(result_doc.name).unlink()

    def test_process_parallel_registers_atexit_cleanup(self, tmp_path, monkeypatch):
        """Test that _process_parallel registers an atexit handler for the temp file."""
        input_pdf = tmp_path / "input.pdf"
        doc = pymupdf.open()
        for _ in range(4):
            doc.new_page()
        doc.save(str(input_pdf))
        doc.close()

        def fake_ocr_page_range(args):
            process_idx, total_processes, input_path, output_dir, language, dpi = args
            part_path = Path(output_dir) / f"ocr_part_{process_idx}.pdf"
            part_doc = pymupdf.open()
            part_doc.new_page()
            part_doc.save(str(part_path))
            part_doc.close()
            return process_idx, process_idx + 1, str(part_path)

        monkeypatch.setattr("pdfalive.processors.ocr_processor._ocr_page_range", fake_ocr_page_range)

        class FakePool:
            def __init__(self, processes=None):
                pass

            def imap_unordered(self, func, args_list):
                for args in args_list:
                    yield func(args)

            def map(self, func, args_list):
                return [func(a) for a in args_list]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr("pdfalive.processors.ocr_processor.Pool", FakePool)

        # Track atexit registrations
        registered_funcs: list = []
        monkeypatch.setattr(atexit, "register", lambda f: registered_funcs.append(f))

        ocr = OCRProcessor(num_processes=2)
        result_doc = ocr._process_parallel(str(input_pdf), num_processes=2, show_progress=False)
        tmp_path_str = result_doc.name

        # An atexit handler should have been registered
        assert len(registered_funcs) == 1, "Expected exactly one atexit handler to be registered"

        # The temp file should exist before cleanup
        assert Path(tmp_path_str).exists()

        # Run the registered cleanup
        result_doc.close()
        registered_funcs[0]()

        # After cleanup the temp file should be gone
        assert not Path(tmp_path_str).exists()

    def test_process_parallel_atexit_handler_tolerates_missing_file(self, tmp_path, monkeypatch):
        """Test that the atexit handler does not raise if the file is already deleted."""
        input_pdf = tmp_path / "input.pdf"
        doc = pymupdf.open()
        for _ in range(4):
            doc.new_page()
        doc.save(str(input_pdf))
        doc.close()

        def fake_ocr_page_range(args):
            process_idx, total_processes, input_path, output_dir, language, dpi = args
            part_path = Path(output_dir) / f"ocr_part_{process_idx}.pdf"
            part_doc = pymupdf.open()
            part_doc.new_page()
            part_doc.save(str(part_path))
            part_doc.close()
            return process_idx, process_idx + 1, str(part_path)

        monkeypatch.setattr("pdfalive.processors.ocr_processor._ocr_page_range", fake_ocr_page_range)

        class FakePool:
            def __init__(self, processes=None):
                pass

            def imap_unordered(self, func, args_list):
                for args in args_list:
                    yield func(args)

            def map(self, func, args_list):
                return [func(a) for a in args_list]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        monkeypatch.setattr("pdfalive.processors.ocr_processor.Pool", FakePool)

        registered_funcs: list = []
        monkeypatch.setattr(atexit, "register", lambda f: registered_funcs.append(f))

        ocr = OCRProcessor(num_processes=2)
        result_doc = ocr._process_parallel(str(input_pdf), num_processes=2, show_progress=False)
        tmp_path_str = result_doc.name

        result_doc.close()

        # Delete the file manually before running the atexit handler
        Path(tmp_path_str).unlink()
        assert not Path(tmp_path_str).exists()

        # The atexit handler should not raise
        registered_funcs[0]()
