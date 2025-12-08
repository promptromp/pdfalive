"""Unit tests for OCR detection strategies."""

from unittest.mock import MagicMock

import pytest

from pdfalive.processors.ocr_detection import NoTextDetectionStrategy, OCRDetectionStrategy


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
        return doc

    def test_needs_ocr_when_document_has_text_on_all_pages(self, mock_doc_with_text):
        """Test that OCR is not needed when all pages have extractable text."""
        strategy = NoTextDetectionStrategy()

        result = strategy.needs_ocr(mock_doc_with_text)

        assert result is False

    def test_needs_ocr_when_document_has_no_text(self, mock_doc_without_text):
        """Test that OCR is needed when document has no extractable text."""
        strategy = NoTextDetectionStrategy()

        result = strategy.needs_ocr(mock_doc_without_text)

        assert result is True

    def test_needs_ocr_when_document_has_empty_text(self, mock_doc_with_empty_text):
        """Test that OCR is needed when document only has whitespace text."""
        strategy = NoTextDetectionStrategy()

        result = strategy.needs_ocr(mock_doc_with_empty_text)

        assert result is True

    def test_needs_ocr_empty_document(self, mock_doc_empty):
        """Test that OCR is needed for empty document (no pages)."""
        strategy = NoTextDetectionStrategy()

        result = strategy.needs_ocr(mock_doc_empty)

        assert result is True

    def test_sample_pages_limits_check(self):
        """Test that sample_pages parameter limits number of pages checked."""
        doc = MagicMock()
        doc.page_count = 100

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
        assert strategy_limited.needs_ocr(doc) is True

        # With sample_pages=5, checks 5 pages: 4/5 have text (80%) -> no OCR needed
        strategy_more = NoTextDetectionStrategy(sample_pages=5)
        assert strategy_more.needs_ocr(doc) is False

    def test_sample_pages_exceeds_page_count(self):
        """Test that sample_pages works when it exceeds document page count."""
        doc = MagicMock()
        doc.page_count = 2

        page = MagicMock()
        page.get_text.return_value = {"blocks": [{"type": 1}]}  # no text
        doc.__getitem__ = MagicMock(return_value=page)

        strategy = NoTextDetectionStrategy(sample_pages=100)

        result = strategy.needs_ocr(doc)

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
        result = strategy.needs_ocr(doc)

        assert result is expected_needs_ocr

    def test_default_min_text_coverage(self):
        """Test that default min_text_coverage is 0.25 (25%)."""
        strategy = NoTextDetectionStrategy()
        assert strategy.min_text_coverage == 0.25

    def test_partial_text_below_threshold_needs_ocr(self):
        """Test that document with only 1 page of text out of 100 needs OCR."""
        doc = MagicMock()
        doc.page_count = 100

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
        result = strategy.needs_ocr(doc)

        # 1% text coverage is below 25% threshold, so OCR is needed
        assert result is True
