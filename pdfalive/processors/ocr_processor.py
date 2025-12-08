"""OCR processor for extracting text from scanned PDFs."""

import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pymupdf
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from pdfalive.processors.ocr_detection import NoTextDetectionStrategy, OCRDetectionStrategy


console = Console()


def _ocr_page_range(args: tuple) -> tuple[int, int, str]:
    """Worker function to OCR a range of pages.

    This function is designed to be called in a separate process.
    It opens the document independently, renders pages to pixmaps,
    performs OCR, and creates a new PDF with the OCR text layer.

    Args:
        args: Tuple of (process_index, total_processes, input_path, output_dir, language, dpi)

    Returns:
        Tuple of (start_page, end_page, output_path) for the processed range.
    """
    process_idx, total_processes, input_path, output_dir, language, dpi = args

    doc = pymupdf.open(input_path)
    page_count = doc.page_count

    # Calculate page range for this process
    pages_per_process = page_count // total_processes
    start_page = process_idx * pages_per_process
    end_page = start_page + pages_per_process if process_idx < total_processes - 1 else page_count

    # Create output document with OCR text layer
    output_doc = pymupdf.open()

    for page_idx in range(start_page, end_page):
        page = doc[page_idx]

        # Render page to pixmap at specified DPI
        # Use a zoom factor based on DPI (72 DPI is the base)
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert pixmap to PDF with OCR text layer
        ocr_pdf_bytes = pix.pdfocr_tobytes(language=language)

        # Open the OCR'd single-page PDF and insert into output
        ocr_page_doc = pymupdf.open("pdf", ocr_pdf_bytes)
        output_doc.insert_pdf(ocr_page_doc)
        ocr_page_doc.close()

    # Save the processed portion
    output_path = Path(output_dir) / f"ocr_part_{process_idx}.pdf"
    output_doc.save(str(output_path))
    output_doc.close()
    doc.close()

    return start_page, end_page, str(output_path)


class OCRProcessor:
    """Processor for performing OCR on PDF documents.

    Uses multiprocessing to speed up OCR on multi-page documents.
    """

    def __init__(
        self,
        detection_strategy: OCRDetectionStrategy | None = None,
        language: str = "eng",
        dpi: int = 300,
        num_processes: int | None = None,
    ) -> None:
        """Initialize the OCR processor.

        Args:
            detection_strategy: Strategy to determine if OCR is needed.
                               Defaults to NoTextDetectionStrategy.
            language: Tesseract language code(s). Default is "eng".
            dpi: Resolution for OCR processing. Default is 300.
            num_processes: Number of parallel processes. Defaults to CPU count.
        """
        self.detection_strategy = detection_strategy or NoTextDetectionStrategy()
        self.language = language
        self.dpi = dpi
        self.num_processes = num_processes or max(1, cpu_count() - 1)

    def needs_ocr(self, doc: pymupdf.Document) -> bool:
        """Check if the document needs OCR processing.

        Args:
            doc: PyMuPDF Document object.

        Returns:
            True if OCR is needed according to the detection strategy.
        """
        return self.detection_strategy.needs_ocr(doc)

    def process(self, input_path: str, show_progress: bool = True) -> pymupdf.Document:
        """Perform OCR on a PDF document.

        Uses multiprocessing to parallelize OCR across pages.

        Args:
            input_path: Path to the input PDF file.
            show_progress: Whether to show progress indicator.

        Returns:
            A new PyMuPDF Document with OCR text layer added.
        """
        doc = pymupdf.open(input_path)
        page_count = doc.page_count
        doc.close()

        # Determine number of processes to use
        num_processes = min(self.num_processes, page_count)

        if num_processes <= 1:
            # Single process mode - no multiprocessing overhead
            return self._process_sequential(input_path, show_progress)

        return self._process_parallel(input_path, num_processes, show_progress)

    def _ocr_single_page(self, page: pymupdf.Page) -> bytes:
        """Perform OCR on a single page and return PDF bytes with text layer.

        Args:
            page: PyMuPDF Page object.

        Returns:
            PDF bytes containing the page with OCR text layer.
        """
        # Render page to pixmap at specified DPI
        zoom = self.dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert pixmap to PDF with OCR text layer
        return pix.pdfocr_tobytes(language=self.language)

    def _process_sequential(self, input_path: str, show_progress: bool = True) -> pymupdf.Document:
        """Process document sequentially (single process).

        Args:
            input_path: Path to the input PDF file.
            show_progress: Whether to show progress indicator.

        Returns:
            New document with OCR text layer applied.
        """
        doc = pymupdf.open(input_path)
        output_doc = pymupdf.open()

        if show_progress:
            console.print(
                f"[cyan]Performing OCR on {doc.page_count} page(s) "
                f"(using {cpu_count()} available CPU cores, DPI={self.dpi})...[/cyan]"
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Performing OCR...", total=doc.page_count)

                for page in doc:
                    ocr_pdf_bytes = self._ocr_single_page(page)
                    ocr_page_doc = pymupdf.open("pdf", ocr_pdf_bytes)
                    output_doc.insert_pdf(ocr_page_doc)
                    ocr_page_doc.close()
                    progress.advance(task)
        else:
            for page in doc:
                ocr_pdf_bytes = self._ocr_single_page(page)
                ocr_page_doc = pymupdf.open("pdf", ocr_pdf_bytes)
                output_doc.insert_pdf(ocr_page_doc)
                ocr_page_doc.close()

        doc.close()
        return output_doc

    def _process_parallel(self, input_path: str, num_processes: int, show_progress: bool = True) -> pymupdf.Document:
        """Process document in parallel using multiprocessing.

        Args:
            input_path: Path to the input PDF file.
            num_processes: Number of parallel processes.
            show_progress: Whether to show progress indicator.

        Returns:
            Merged document with OCR applied to all pages.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare arguments for worker processes
            args_list = [
                (i, num_processes, input_path, temp_dir, self.language, self.dpi) for i in range(num_processes)
            ]

            if show_progress:
                doc = pymupdf.open(input_path)
                page_count = doc.page_count
                doc.close()
                console.print(
                    f"[cyan]Performing OCR on {page_count} page(s) "
                    f"(using {num_processes} parallel processes, DPI={self.dpi})...[/cyan]"
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
                        task = progress.add_task("OCR processing...", total=num_processes)

                        for result in pool.imap_unordered(_ocr_page_range, args_list):
                            results.append(result)
                            progress.advance(task)
                else:
                    results = pool.map(_ocr_page_range, args_list)

            # Sort results by start page to maintain order
            results = sorted(results, key=lambda x: x[0])

            # Merge the processed parts back together
            if show_progress:
                console.print("[cyan]Merging OCR results...[/cyan]")

            merged_doc = pymupdf.open()
            for _, _, part_path in results:
                part_doc = pymupdf.open(part_path)
                merged_doc.insert_pdf(part_doc)
                part_doc.close()

            return merged_doc

    def process_in_memory(self, doc: pymupdf.Document, show_progress: bool = True) -> pymupdf.Document:
        """Perform OCR on a document and return a new document with OCR text layer.

        Note: This method returns a NEW document with the OCR text layer.
        The original document is not modified. The caller should use the
        returned document for subsequent operations.

        Args:
            doc: PyMuPDF Document object to process.
            show_progress: Whether to show progress indicator.

        Returns:
            A new PyMuPDF Document with OCR text layer embedded.
        """
        output_doc = pymupdf.open()

        if show_progress:
            console.print(
                f"[cyan]Performing OCR on {doc.page_count} page(s) "
                f"(using {cpu_count()} available CPU cores, DPI={self.dpi})...[/cyan]"
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Performing OCR...", total=doc.page_count)

                for page in doc:
                    ocr_pdf_bytes = self._ocr_single_page(page)
                    ocr_page_doc = pymupdf.open("pdf", ocr_pdf_bytes)
                    output_doc.insert_pdf(ocr_page_doc)
                    ocr_page_doc.close()
                    progress.advance(task)
        else:
            for page in doc:
                ocr_pdf_bytes = self._ocr_single_page(page)
                ocr_page_doc = pymupdf.open("pdf", ocr_pdf_bytes)
                output_doc.insert_pdf(ocr_page_doc)
                ocr_page_doc.close()

        return output_doc
