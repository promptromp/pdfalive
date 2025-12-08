"""OCR detection strategies to determine if a PDF needs OCR processing."""

from abc import ABC, abstractmethod

import pymupdf


class OCRDetectionStrategy(ABC):
    """Base class for OCR detection strategies.

    Strategies determine whether a PDF document needs OCR processing
    based on various heuristics.
    """

    @abstractmethod
    def needs_ocr(self, doc: pymupdf.Document) -> bool:
        """Determine if the document needs OCR processing.

        Args:
            doc: PyMuPDF Document object.

        Returns:
            True if OCR is needed, False otherwise.
        """
        pass


class NoTextDetectionStrategy(OCRDetectionStrategy):
    """Simple strategy that checks if any page has extractable text.

    This strategy iterates through pages and checks if any text blocks,
    lines, or spans contain text. If no text is found anywhere in the
    document, OCR is needed.
    """

    def __init__(self, sample_pages: int | None = None) -> None:
        """Initialize the strategy.

        Args:
            sample_pages: If provided, only check this many pages (for efficiency).
                         If None, check all pages.
        """
        self.sample_pages = sample_pages

    def needs_ocr(self, doc: pymupdf.Document) -> bool:
        """Check if document has any extractable text.

        Returns True if no text is found in any page.
        """
        pages_to_check = doc.page_count
        if self.sample_pages is not None:
            pages_to_check = min(self.sample_pages, doc.page_count)

        for page_idx in range(pages_to_check):
            page = doc[page_idx]
            if self._page_has_text(page):
                return False

        return True

    def _page_has_text(self, page: pymupdf.Page) -> bool:
        """Check if a single page has any extractable text.

        Args:
            page: PyMuPDF Page object.

        Returns:
            True if the page has text, False otherwise.
        """
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            # Only check text blocks (type 0), skip image blocks (type 1)
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        return True

        return False
