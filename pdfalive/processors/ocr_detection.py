"""OCR detection strategies to determine if a PDF needs OCR processing."""

from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count

import pymupdf
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn


console = Console()


def _check_page_has_text(args: tuple) -> tuple[int, bool]:
    """Worker function to check if a single page has text.

    This function is designed to be called in a separate process.
    It opens the document independently and checks a single page.

    Args:
        args: Tuple of (page_index, input_path)

    Returns:
        Tuple of (page_index, has_text) for the processed page.
    """
    page_idx, input_path = args

    doc = pymupdf.open(input_path)
    page = doc[page_idx]

    page_dict = page.get_text("dict")
    has_text = False

    for block in page_dict.get("blocks", []):
        # Only check text blocks (type 0), skip image blocks (type 1)
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    has_text = True
                    break
            if has_text:
                break
        if has_text:
            break

    doc.close()
    return page_idx, has_text


class OCRDetectionStrategy(ABC):
    """Base class for OCR detection strategies.

    Strategies determine whether a PDF document needs OCR processing
    based on various heuristics.
    """

    @abstractmethod
    def needs_ocr(self, doc: pymupdf.Document, show_progress: bool = True) -> bool:
        """Determine if the document needs OCR processing.

        Args:
            doc: PyMuPDF Document object.
            show_progress: Whether to show progress indicator.

        Returns:
            True if OCR is needed, False otherwise.
        """
        pass


class NoTextDetectionStrategy(OCRDetectionStrategy):
    """Strategy that checks if sufficient pages have extractable text.

    This strategy iterates through pages and checks if text blocks,
    lines, or spans contain text. OCR is needed if fewer than the
    minimum required percentage of pages have text.

    Supports multiprocessing for faster detection on large documents.
    """

    DEFAULT_MIN_TEXT_COVERAGE = 0.25  # 25% of pages must have text

    def __init__(
        self,
        sample_pages: int | None = None,
        min_text_coverage: float = DEFAULT_MIN_TEXT_COVERAGE,
        num_processes: int | None = None,
    ) -> None:
        """Initialize the strategy.

        Args:
            sample_pages: If provided, only check this many pages (for efficiency).
                         If None, check all pages.
            min_text_coverage: Minimum fraction of pages that must have text
                              for OCR to be considered unnecessary. Default is 0.25 (25%).
                              Set to 0.0 to require only one page with text (legacy behavior).
            num_processes: Number of parallel processes for page checking.
                          Defaults to CPU count - 1.
        """
        self.sample_pages = sample_pages
        self.min_text_coverage = min_text_coverage
        self.num_processes = num_processes or max(1, cpu_count() - 1)

    def needs_ocr(self, doc: pymupdf.Document, show_progress: bool = True) -> bool:
        """Check if document has sufficient extractable text.

        Uses multiprocessing for large documents when the document is opened
        from a file path. Falls back to sequential processing for in-memory
        documents.

        Args:
            doc: PyMuPDF Document object.
            show_progress: Whether to show progress indicator.

        Returns:
            True if fewer than min_text_coverage fraction of pages have text,
            or if no pages have any text at all.
        """
        pages_to_check = doc.page_count
        if self.sample_pages is not None:
            pages_to_check = min(self.sample_pages, doc.page_count)

        if pages_to_check == 0:
            return True

        # Use parallel processing if document has a file path and enough pages
        num_processes = min(self.num_processes, pages_to_check)
        if doc.name and num_processes > 1:
            return self._needs_ocr_parallel(doc.name, pages_to_check, num_processes, show_progress)

        return self._needs_ocr_sequential(doc, pages_to_check, show_progress)

    def _needs_ocr_sequential(self, doc: pymupdf.Document, pages_to_check: int, show_progress: bool = True) -> bool:
        """Check for text sequentially (single process).

        Args:
            doc: PyMuPDF Document object.
            pages_to_check: Number of pages to check.
            show_progress: Whether to show progress indicator.

        Returns:
            True if OCR is needed, False otherwise.
        """
        pages_with_text = 0

        if show_progress:
            console.print(f"[cyan]Checking {pages_to_check} page(s) for existing text...[/cyan]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Detecting text...", total=pages_to_check)

                for page_idx in range(pages_to_check):
                    page = doc[page_idx]
                    if self._page_has_text(page):
                        pages_with_text += 1
                    progress.advance(task)
        else:
            for page_idx in range(pages_to_check):
                page = doc[page_idx]
                if self._page_has_text(page):
                    pages_with_text += 1

        # Always need OCR if no pages have text
        if pages_with_text == 0:
            return True

        text_coverage = pages_with_text / pages_to_check
        return text_coverage < self.min_text_coverage

    def _needs_ocr_parallel(
        self, input_path: str, pages_to_check: int, num_processes: int, show_progress: bool = True
    ) -> bool:
        """Check for text in parallel using multiprocessing.

        Args:
            input_path: Path to the PDF file.
            pages_to_check: Number of pages to check.
            num_processes: Number of parallel processes.
            show_progress: Whether to show progress indicator.

        Returns:
            True if OCR is needed, False otherwise.
        """
        # Prepare arguments for worker processes
        args_list = [(page_idx, input_path) for page_idx in range(pages_to_check)]

        if show_progress:
            console.print(
                f"[cyan]Checking {pages_to_check} page(s) for existing text "
                f"(using {num_processes} parallel processes)...[/cyan]"
            )

        # Process in parallel
        with Pool(processes=num_processes) as pool:
            results = []
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Detecting text...", total=pages_to_check)

                    for result in pool.imap_unordered(_check_page_has_text, args_list):
                        results.append(result)
                        progress.advance(task)
            else:
                results = pool.map(_check_page_has_text, args_list)

        # Count pages with text
        pages_with_text = sum(1 for _, has_text in results if has_text)

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
