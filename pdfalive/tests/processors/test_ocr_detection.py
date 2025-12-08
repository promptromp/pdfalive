"""Unit tests for OCR detection strategies."""

from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from pdfalive.processors.ocr_detection import (
    NoTextDetectionStrategy,
    OCRDetectionStrategy,
    _check_page_has_text,
)


class TestOCRDetectionStrategy:
    """Tests for base OCRDetectionStrategy class."""

    def test_is_abstract(self):
        """Test that OCRDetectionStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            OCRDetectionStrategy()


class TestNoTextDetectionStrategy:
    """Tests for NoTextDetectionStrategy."""

    @pytest.fixture
    def mock_doc_with_text(self):
        """Create a mock document that has extractable text."""
        doc = MagicMock()
        doc.page_count = 3
        doc.name = ""  # Empty name forces sequential processing in tests

        # Mock page with text
        page_with_text = MagicMock()
        page_with_text.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,  # text block
                    "lines": [
                        {
                            "spans": [
                                {"text": "Chapter 1: Introduction"},
                            ]
                        }
                    ],
                }
            ]
        }
        doc.__getitem__ = MagicMock(return_value=page_with_text)

        return doc

    @pytest.fixture
    def mock_doc_without_text(self):
        """Create a mock document that has no extractable text (scanned images only)."""
        doc = MagicMock()
        doc.page_count = 3
        doc.name = ""  # Empty name forces sequential processing in tests

        # Mock page with only image blocks (no text)
        page_without_text = MagicMock()
        page_without_text.get_text.return_value = {
            "blocks": [
                {
                    "type": 1,  # image block
                }
            ]
        }
        doc.__getitem__ = MagicMock(return_value=page_without_text)

        return doc

    @pytest.fixture
    def mock_doc_with_empty_text(self):
        """Create a mock document with text blocks but empty/whitespace text."""
        doc = MagicMock()
        doc.page_count = 2
        doc.name = ""  # Empty name forces sequential processing in tests

        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {"text": "   "},  # whitespace only
                                {"text": ""},  # empty
                            ]
                        }
                    ],
                }
            ]
        }
        doc.__getitem__ = MagicMock(return_value=page)

        return doc

    @pytest.fixture
    def mock_doc_empty(self):
        """Create a mock document with no pages."""
        doc = MagicMock()
        doc.page_count = 0
        doc.name = ""  # Empty name forces sequential processing in tests
        return doc

    def test_needs_ocr_when_document_has_text_on_all_pages(self, mock_doc_with_text):
        """Test that OCR is not needed when all pages have extractable text."""
        strategy = NoTextDetectionStrategy()

        result = strategy.needs_ocr(mock_doc_with_text, show_progress=False)

        assert result is False

    def test_needs_ocr_when_document_has_no_text(self, mock_doc_without_text):
        """Test that OCR is needed when document has no extractable text."""
        strategy = NoTextDetectionStrategy()

        result = strategy.needs_ocr(mock_doc_without_text, show_progress=False)

        assert result is True

    def test_needs_ocr_when_document_has_empty_text(self, mock_doc_with_empty_text):
        """Test that OCR is needed when document only has whitespace text."""
        strategy = NoTextDetectionStrategy()

        result = strategy.needs_ocr(mock_doc_with_empty_text, show_progress=False)

        assert result is True

    def test_needs_ocr_empty_document(self, mock_doc_empty):
        """Test that OCR is needed for empty document (no pages)."""
        strategy = NoTextDetectionStrategy()

        result = strategy.needs_ocr(mock_doc_empty, show_progress=False)

        assert result is True

    def test_sample_pages_limits_check(self):
        """Test that sample_pages parameter limits number of pages checked."""
        doc = MagicMock()
        doc.page_count = 100
        doc.name = ""  # Empty name forces sequential processing in tests

        # First page has no text, but later pages do
        pages = []
        for i in range(100):
            page = MagicMock()
            if i == 0:
                # First page: no text
                page.get_text.return_value = {"blocks": [{"type": 1}]}
            else:
                # Other pages: have text
                page.get_text.return_value = {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [{"spans": [{"text": "Some text"}]}],
                        }
                    ]
                }
            pages.append(page)

        doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])

        # With sample_pages=1, only first page is checked (no text) -> needs OCR
        strategy_limited = NoTextDetectionStrategy(sample_pages=1)
        assert strategy_limited.needs_ocr(doc, show_progress=False) is True

        # With sample_pages=5, checks 5 pages: 4/5 have text (80%) -> no OCR needed
        strategy_more = NoTextDetectionStrategy(sample_pages=5)
        assert strategy_more.needs_ocr(doc, show_progress=False) is False

    def test_sample_pages_exceeds_page_count(self):
        """Test that sample_pages works when it exceeds document page count."""
        doc = MagicMock()
        doc.page_count = 2
        doc.name = ""  # Empty name forces sequential processing in tests

        page = MagicMock()
        page.get_text.return_value = {"blocks": [{"type": 1}]}  # no text
        doc.__getitem__ = MagicMock(return_value=page)

        strategy = NoTextDetectionStrategy(sample_pages=100)

        result = strategy.needs_ocr(doc, show_progress=False)

        # Should check only 2 pages (the actual count) and determine OCR is needed
        assert result is True
        assert doc.__getitem__.call_count == 2

    def test_page_has_text_with_mixed_blocks(self):
        """Test _page_has_text with mix of text and image blocks."""
        strategy = NoTextDetectionStrategy()

        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {"type": 1},  # image block
                {"type": 1},  # another image
                {
                    "type": 0,  # text block with actual text
                    "lines": [{"spans": [{"text": "Real text here"}]}],
                },
            ]
        }

        result = strategy._page_has_text(page)

        assert result is True

    def test_page_has_text_missing_keys(self):
        """Test _page_has_text handles missing dictionary keys gracefully."""
        strategy = NoTextDetectionStrategy()

        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {"type": 0},  # text block but no lines key
                {"type": 0, "lines": []},  # text block with empty lines
                {"type": 0, "lines": [{}]},  # line with no spans key
                {"type": 0, "lines": [{"spans": []}]},  # spans is empty
            ]
        }

        result = strategy._page_has_text(page)

        assert result is False

    @pytest.mark.parametrize(
        "pages_with_text,total_pages,min_coverage,expected_needs_ocr",
        [
            # Below threshold - needs OCR
            (1, 100, 0.25, True),  # 1% coverage, need 25%
            (10, 100, 0.25, True),  # 10% coverage, need 25%
            (24, 100, 0.25, True),  # 24% coverage, need 25%
            # At or above threshold - no OCR needed
            (25, 100, 0.25, False),  # exactly 25%
            (50, 100, 0.25, False),  # 50% coverage
            (100, 100, 0.25, False),  # 100% coverage
            # Edge cases with different thresholds
            (1, 10, 0.0, False),  # 0% threshold, any text is enough
            (0, 10, 0.0, True),  # 0% threshold but no text at all
            (9, 10, 0.9, False),  # 90% threshold, 90% coverage
            (8, 10, 0.9, True),  # 90% threshold, 80% coverage
        ],
    )
    def test_min_text_coverage_threshold(self, pages_with_text, total_pages, min_coverage, expected_needs_ocr):
        """Test that min_text_coverage threshold works correctly."""
        doc = MagicMock()
        doc.page_count = total_pages
        doc.name = ""  # Empty name forces sequential processing in tests

        pages = []
        for i in range(total_pages):
            page = MagicMock()
            if i < pages_with_text:
                # Page with text
                page.get_text.return_value = {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [{"spans": [{"text": "Some text"}]}],
                        }
                    ]
                }
            else:
                # Page without text (image only)
                page.get_text.return_value = {"blocks": [{"type": 1}]}
            pages.append(page)

        doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])

        strategy = NoTextDetectionStrategy(min_text_coverage=min_coverage)
        result = strategy.needs_ocr(doc, show_progress=False)

        assert result is expected_needs_ocr

    def test_default_min_text_coverage(self):
        """Test that default min_text_coverage is 0.25 (25%)."""
        strategy = NoTextDetectionStrategy()
        assert strategy.min_text_coverage == 0.25

    def test_partial_text_below_threshold_needs_ocr(self):
        """Test that document with only 1 page of text out of 100 needs OCR."""
        doc = MagicMock()
        doc.page_count = 100
        doc.name = ""  # Empty name forces sequential processing in tests

        pages = []
        for i in range(100):
            page = MagicMock()
            if i == 99:  # Only last page has text (like metadata page)
                page.get_text.return_value = {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [{"spans": [{"text": "Document metadata"}]}],
                        }
                    ]
                }
            else:
                page.get_text.return_value = {"blocks": [{"type": 1}]}
            pages.append(page)

        doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])

        strategy = NoTextDetectionStrategy()
        result = strategy.needs_ocr(doc, show_progress=False)

        # 1% text coverage is below 25% threshold, so OCR is needed
        assert result is True

    def test_num_processes_parameter(self):
        """Test that num_processes parameter is stored correctly."""
        strategy_default = NoTextDetectionStrategy()
        assert strategy_default.num_processes >= 1

        strategy_custom = NoTextDetectionStrategy(num_processes=4)
        assert strategy_custom.num_processes == 4

    def test_sequential_fallback_for_in_memory_doc(self):
        """Test that in-memory documents (no file path) use sequential processing."""
        doc = MagicMock()
        doc.page_count = 10
        doc.name = ""  # Empty name indicates in-memory document

        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [{"spans": [{"text": "Some text"}]}],
                }
            ]
        }
        doc.__getitem__ = MagicMock(return_value=page)

        strategy = NoTextDetectionStrategy(num_processes=4)

        with (
            patch.object(strategy, "_needs_ocr_sequential", wraps=strategy._needs_ocr_sequential) as mock_sequential,
            patch.object(strategy, "_needs_ocr_parallel", wraps=strategy._needs_ocr_parallel) as mock_parallel,
        ):
            strategy.needs_ocr(doc, show_progress=False)

            # Sequential should be called, parallel should not
            mock_sequential.assert_called_once()
            mock_parallel.assert_not_called()

    def test_sequential_fallback_for_single_page(self):
        """Test that single-page documents use sequential processing."""
        doc = MagicMock()
        doc.page_count = 1
        doc.name = "/path/to/file.pdf"

        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [{"spans": [{"text": "Some text"}]}],
                }
            ]
        }
        doc.__getitem__ = MagicMock(return_value=page)

        # Even with multiple processes configured, should use sequential for 1 page
        strategy = NoTextDetectionStrategy(num_processes=4)

        with (
            patch.object(strategy, "_needs_ocr_sequential", wraps=strategy._needs_ocr_sequential) as mock_sequential,
            patch.object(strategy, "_needs_ocr_parallel", wraps=strategy._needs_ocr_parallel) as mock_parallel,
        ):
            strategy.needs_ocr(doc, show_progress=False)

            mock_sequential.assert_called_once()
            mock_parallel.assert_not_called()

    def test_parallel_called_for_file_backed_multipage_doc(self):
        """Test that file-backed multi-page documents use parallel processing."""
        doc = MagicMock()
        doc.page_count = 10
        doc.name = "/path/to/file.pdf"

        strategy = NoTextDetectionStrategy(num_processes=4)

        with patch.object(strategy, "_needs_ocr_parallel", return_value=False) as mock_parallel:
            result = strategy.needs_ocr(doc, show_progress=False)

            mock_parallel.assert_called_once_with("/path/to/file.pdf", 10, 4, False)
            assert result is False

    def test_num_processes_limited_by_pages_to_check(self):
        """Test that num_processes is limited by number of pages to check."""
        doc = MagicMock()
        doc.page_count = 3
        doc.name = "/path/to/file.pdf"

        strategy = NoTextDetectionStrategy(num_processes=10)

        with patch.object(strategy, "_needs_ocr_parallel", return_value=False) as mock_parallel:
            strategy.needs_ocr(doc, show_progress=False)

            # num_processes should be min(10, 3) = 3
            mock_parallel.assert_called_once_with("/path/to/file.pdf", 3, 3, False)

    def test_show_progress_parameter_passed_to_sequential(self):
        """Test that show_progress parameter is passed to sequential method."""
        doc = MagicMock()
        doc.page_count = 5
        doc.name = ""  # Empty name forces sequential processing

        page = MagicMock()
        page.get_text.return_value = {"blocks": [{"type": 0, "lines": [{"spans": [{"text": "text"}]}]}]}
        doc.__getitem__ = MagicMock(return_value=page)

        strategy = NoTextDetectionStrategy()

        with patch.object(strategy, "_needs_ocr_sequential", return_value=False) as mock_sequential:
            strategy.needs_ocr(doc, show_progress=True)
            mock_sequential.assert_called_once_with(doc, 5, True)

            mock_sequential.reset_mock()
            strategy.needs_ocr(doc, show_progress=False)
            mock_sequential.assert_called_once_with(doc, 5, False)

    def test_show_progress_parameter_passed_to_parallel(self):
        """Test that show_progress parameter is passed to parallel method."""
        doc = MagicMock()
        doc.page_count = 10
        doc.name = "/path/to/file.pdf"

        strategy = NoTextDetectionStrategy(num_processes=4)

        with patch.object(strategy, "_needs_ocr_parallel", return_value=False) as mock_parallel:
            strategy.needs_ocr(doc, show_progress=True)
            mock_parallel.assert_called_once_with("/path/to/file.pdf", 10, 4, True)

            mock_parallel.reset_mock()
            strategy.needs_ocr(doc, show_progress=False)
            mock_parallel.assert_called_once_with("/path/to/file.pdf", 10, 4, False)


class TestCheckPageHasTextWorker:
    """Tests for the _check_page_has_text worker function."""

    def test_worker_returns_page_index_and_result(self):
        """Test that worker function returns correct tuple format."""
        with patch("pdfalive.processors.ocr_detection.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = {
                "blocks": [
                    {
                        "type": 0,
                        "lines": [{"spans": [{"text": "Some text"}]}],
                    }
                ]
            }
            mock_doc.__getitem__ = MagicMock(return_value=mock_page)
            mock_pymupdf.open.return_value = mock_doc

            result = _check_page_has_text((5, "/path/to/file.pdf"))

            assert result == (5, True)
            mock_pymupdf.open.assert_called_once_with("/path/to/file.pdf")
            mock_doc.__getitem__.assert_called_once_with(5)
            mock_doc.close.assert_called_once()

    def test_worker_detects_no_text(self):
        """Test that worker correctly identifies pages without text."""
        with patch("pdfalive.processors.ocr_detection.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = {
                "blocks": [{"type": 1}]  # Image block only
            }
            mock_doc.__getitem__ = MagicMock(return_value=mock_page)
            mock_pymupdf.open.return_value = mock_doc

            result = _check_page_has_text((0, "/path/to/file.pdf"))

            assert result == (0, False)

    def test_worker_handles_empty_text(self):
        """Test that worker correctly handles whitespace-only text."""
        with patch("pdfalive.processors.ocr_detection.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = {
                "blocks": [
                    {
                        "type": 0,
                        "lines": [{"spans": [{"text": "   "}]}],  # Whitespace only
                    }
                ]
            }
            mock_doc.__getitem__ = MagicMock(return_value=mock_page)
            mock_pymupdf.open.return_value = mock_doc

            result = _check_page_has_text((0, "/path/to/file.pdf"))

            assert result == (0, False)


class TestNoTextDetectionStrategyParallelIntegration:
    """Integration tests for parallel OCR detection using real PDF files.

    These tests use actual temporary PDF files to test the parallel processing
    code path, since multiprocessing requires pickle-able objects.
    """

    @pytest.fixture
    def multi_page_pdf_with_text(self, tmp_path):
        """Create a temporary multi-page PDF with text for testing."""
        pdf_path = tmp_path / "test_with_text.pdf"
        doc = pymupdf.open()

        for i in range(4):
            page = doc.new_page(width=612, height=792)
            text_point = pymupdf.Point(72, 72)
            page.insert_text(text_point, f"Page {i + 1} has text content", fontsize=12)

        doc.save(str(pdf_path))
        doc.close()

        return str(pdf_path)

    @pytest.fixture
    def multi_page_pdf_without_text(self, tmp_path):
        """Create a temporary multi-page PDF without text (blank pages)."""
        pdf_path = tmp_path / "test_without_text.pdf"
        doc = pymupdf.open()

        for _ in range(4):
            doc.new_page(width=612, height=792)

        doc.save(str(pdf_path))
        doc.close()

        return str(pdf_path)

    def test_parallel_detection_with_text_returns_no_ocr_needed(self, multi_page_pdf_with_text):
        """Test that parallel detection correctly identifies documents with text."""
        strategy = NoTextDetectionStrategy(num_processes=2)

        doc = pymupdf.open(multi_page_pdf_with_text)
        assert doc.name == multi_page_pdf_with_text  # Verify file-backed

        result = strategy.needs_ocr(doc, show_progress=False)
        doc.close()

        assert result is False, "Document with text on all pages should not need OCR"

    def test_parallel_detection_without_text_returns_ocr_needed(self, multi_page_pdf_without_text):
        """Test that parallel detection correctly identifies documents without text."""
        strategy = NoTextDetectionStrategy(num_processes=2)

        doc = pymupdf.open(multi_page_pdf_without_text)
        assert doc.name == multi_page_pdf_without_text  # Verify file-backed

        result = strategy.needs_ocr(doc, show_progress=False)
        doc.close()

        assert result is True, "Document without text should need OCR"

    def test_parallel_detection_uses_parallel_path(self, multi_page_pdf_with_text, monkeypatch):
        """Test that detection actually uses the parallel code path for file-backed docs."""
        called = {"parallel": False}

        original_parallel = NoTextDetectionStrategy._needs_ocr_parallel

        def tracking_parallel(self, *args, **kwargs):
            called["parallel"] = True
            return original_parallel(self, *args, **kwargs)

        monkeypatch.setattr(NoTextDetectionStrategy, "_needs_ocr_parallel", tracking_parallel)

        strategy = NoTextDetectionStrategy(num_processes=2)
        doc = pymupdf.open(multi_page_pdf_with_text)

        strategy.needs_ocr(doc, show_progress=False)
        doc.close()

        assert called["parallel"], "Detection should use parallel path for file-backed multi-page document"
