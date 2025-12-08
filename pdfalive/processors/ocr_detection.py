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
    """Strategy that checks if sufficient pages have extractable text.

    This strategy iterates through pages and checks if text blocks,
    lines, or spans contain text. OCR is needed if fewer than the
    minimum required percentage of pages have text.
    """

    DEFAULT_MIN_TEXT_COVERAGE = 0.25  # 25% of pages must have text

    def __init__(
        self,
        sample_pages: int | None = None,
        min_text_coverage: float = DEFAULT_MIN_TEXT_COVERAGE,
    ) -> None:
        """Initialize the strategy.

        Args:
            sample_pages: If provided, only check this many pages (for efficiency).
                         If None, check all pages.
            min_text_coverage: Minimum fraction of pages that must have text
                              for OCR to be considered unnecessary. Default is 0.25 (25%).
                              Set to 0.0 to require only one page with text (legacy behavior).
        """
        self.sample_pages = sample_pages
        self.min_text_coverage = min_text_coverage

    def needs_ocr(self, doc: pymupdf.Document) -> bool:
        """Check if document has sufficient extractable text.

        Returns True if fewer than min_text_coverage fraction of pages have text,
        or if no pages have any text at all.
        """
        pages_to_check = doc.page_count
        if self.sample_pages is not None:
            pages_to_check = min(self.sample_pages, doc.page_count)

        if pages_to_check == 0:
            return True

        pages_with_text = 0
        for page_idx in range(pages_to_check):
            page = doc[page_idx]
            if self._page_has_text(page):
                pages_with_text += 1

        # Always need OCR if no pages have text
        if pages_with_text == 0:
            return True

        text_coverage = pages_with_text / pages_to_check
        return text_coverage < self.min_text_coverage

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
