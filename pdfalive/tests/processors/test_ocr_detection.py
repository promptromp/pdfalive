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

    def test_needs_ocr_when_document_has_text(self, mock_doc_with_text):
        """Test that OCR is not needed when document has extractable text."""
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

        # With sample_pages=5, checks more pages and finds text -> no OCR needed
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
