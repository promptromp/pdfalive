"""OCR processor: adds an invisible text layer of recognized text to the pages of scanned PDFs."""

import atexit
import os
import tempfile
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count

import pymupdf

from pdfalive.console import create_console, create_progress
from pdfalive.processors.ocr_detection import NoTextDetectionStrategy, OCRDetectionStrategy, page_has_text


console = create_console()

# The document a worker process OCRs pages of, opened once per process by _open_worker_document().
_worker_document: pymupdf.Document | None = None


def _text_layer(page: pymupdf.Page, language: str, dpi: int) -> pymupdf.Document:
    """OCR a page as displayed and return a one-page PDF holding only the recognized text.

    MuPDF's Tesseract integration writes a page showing the render with the recognized
    text drawn invisibly over it, sized to the render (not to the page). The render is
    removed, so laying that page over the original (_overlay_text_layer()) adds the text
    and nothing else.

    Args:
        page: Page to OCR.
        language: Tesseract language code(s).
        dpi: Resolution to render the page at for OCR.

    Returns:
        A one-page PDF with the page's invisible text layer, in its display orientation.
    """
    zoom = dpi / 72.0
    # Annotations stay on the page, so their text is not the page's to recognize
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, annots=False)
    layer = pymupdf.open(stream=pix.pdfocr_tobytes(language=language), filetype="pdf")
    layer_page = layer[0]
    layer_page.add_redact_annot(layer_page.rect, fill=False)
    layer_page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        text=pymupdf.PDF_REDACT_TEXT_NONE,
    )
    return layer


def _overlay_text_layer(page: pymupdf.Page, layer: pymupdf.Document) -> None:
    """Lay a text layer from _text_layer() over the page it was recognized on.

    The layer covers the page as displayed, so it is stretched over the page's visible
    area and turned with the page's rotation. show_pdf_page() places by unrotated
    coordinates, so the page's rotation is set aside meanwhile (as PyMuPDF does itself
    when adding annotations to rotated pages).

    Args:
        page: Page to add the text layer to.
        layer: The page's text layer, a one-page PDF.
    """
    rotation = page.rotation
    page.set_rotation(0)
    try:
        page.show_pdf_page(page.rect, layer, 0, keep_proportion=False, rotate=rotation)
    finally:
        page.set_rotation(rotation)


def _layer_bytes(layer: pymupdf.Document) -> bytes:
    """A text layer from _text_layer() as PDF bytes, without the removed render (garbage=1)."""
    return layer.tobytes(garbage=1)


def _open_worker_document(input_path: str) -> None:
    """Pool initializer: open the document to OCR once in each worker process."""
    global _worker_document
    _worker_document = pymupdf.open(input_path)


def _worker_text_layer(args: tuple[int, str, int]) -> bytes:
    """Worker function: OCR one page of the worker's document and return its text layer PDF.

    Args:
        args: Tuple of (page_index, language, dpi).

    Returns:
        The page's text layer from _text_layer(), as PDF bytes.
    """
    page_index, language, dpi = args
    if _worker_document is None:
        raise RuntimeError("OCR worker started without its document; use _open_worker_document as initializer")
    with _text_layer(_worker_document[page_index], language, dpi) as layer:
        return _layer_bytes(layer)


def _persist_to_temp_file(doc: pymupdf.Document) -> pymupdf.Document:
    """Save a document to a temporary file, removed at exit, and return it opened from there.

    A file-backed document (truthy `doc.name`) lets later steps process it in parallel,
    each worker opening the file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_path = tmp_file.name

    def _cleanup_temp(path: str = tmp_path) -> None:
        if os.path.exists(path):
            os.unlink(path)

    atexit.register(_cleanup_temp)

    doc.save(tmp_path)
    doc.close()
    return pymupdf.open(tmp_path)


class OCRProcessor:
    """Processor that adds an OCR text layer to the pages of a PDF.

    Each page to OCR is rendered and recognized with Tesseract, and the recognized text is
    laid over the page, invisible, where its words are shown. Nothing else changes: the
    pages keep their images, size, rotation, links and annotations, and the document its
    bookmarks and metadata. By default only pages without text are OCR'd, since OCR text
    on a page that has text would repeat it.

    Uses multiprocessing to OCR pages of file-backed documents in parallel.
    """

    def __init__(
        self,
        detection_strategy: OCRDetectionStrategy | None = None,
        language: str = "eng",
        dpi: int = 300,
        num_processes: int | None = None,
        include_pages_with_text: bool = False,
    ) -> None:
        """Initialize the OCR processor.

        Args:
            detection_strategy: Strategy to determine if OCR is needed.
                               Defaults to NoTextDetectionStrategy.
            language: Tesseract language code(s). Default is "eng".
            dpi: Resolution for OCR processing. Default is 300.
            num_processes: Number of parallel processes. Defaults to CPU count - 1.
            include_pages_with_text: OCR pages that already have text as well, adding the
                recognized text to theirs. Default is False (only pages without text).
        """
        self.detection_strategy = detection_strategy or NoTextDetectionStrategy()
        self.language = language
        self.dpi = dpi
        self.num_processes = num_processes or max(1, cpu_count() - 1)
        self.include_pages_with_text = include_pages_with_text

    def needs_ocr(self, doc: pymupdf.Document) -> bool:
        """Check if the document needs OCR processing.

        Args:
            doc: PyMuPDF Document object.

        Returns:
            True if OCR is needed according to the detection strategy.
        """
        return self.detection_strategy.needs_ocr(doc)

    def pages_to_ocr(self, doc: pymupdf.Document) -> list[int]:
        """Return the indices of the pages to OCR: those without text, or all of them.

        Args:
            doc: PyMuPDF Document object.

        Returns:
            Page indices, in page order.
        """
        if self.include_pages_with_text:
            return list(range(doc.page_count))
        return [index for index in range(doc.page_count) if not page_has_text(doc[index])]

    def process(self, input_path: str, show_progress: bool = True) -> pymupdf.Document:
        """Perform OCR on a PDF file; see process_in_memory().

        Args:
            input_path: Path to the input PDF file.
            show_progress: Whether to show progress indicator.

        Returns:
            A new PyMuPDF Document with the OCR text layer added.
        """
        with pymupdf.open(input_path) as doc:
            return self.process_in_memory(doc, show_progress)

    def process_in_memory(
        self, doc: pymupdf.Document, show_progress: bool = True, page_indices: list[int] | None = None
    ) -> pymupdf.Document:
        """Return a copy of the document as a PDF, with an OCR text layer on each page to OCR.

        The original document is not modified; a document of another format PyMuPDF opens
        (an image, for instance) is converted to PDF, since the text layer is added to PDF
        pages. The pages of a file-backed document are OCR'd in parallel (when there are
        several), and the copy is file-backed as well (a temporary file removed at exit), so
        later steps can process it in parallel too; the copy of an in-memory document stays
        in memory.

        Args:
            doc: PyMuPDF Document object to process.
            show_progress: Whether to show progress indicator.
            page_indices: The pages to OCR, as pages_to_ocr() returns them; computed when None.

        Returns:
            A new PyMuPDF Document with the OCR text layer added.
        """
        if page_indices is None:
            page_indices = self.pages_to_ocr(doc)
        # Workers open the document's file, so an in-memory document is OCR'd in this process
        num_processes = min(self.num_processes, len(page_indices)) if doc.name else 1

        if show_progress:
            skipped = doc.page_count - len(page_indices)
            console.print(
                f"Performing OCR on {len(page_indices)} page(s)"
                + (f", skipping {skipped} page(s) that already have text" if skipped else "")
                + (f" (using {num_processes} parallel processes, " if num_processes > 1 else " (")
                + f"DPI={self.dpi})...",
                style="cyan",
            )

        output = pymupdf.open(stream=doc.tobytes() if doc.is_pdf else doc.convert_to_pdf(), filetype="pdf")
        with create_progress(console, disable=not show_progress) as progress:
            task = progress.add_task("Performing OCR...", total=len(page_indices))
            for page_index, layer_pdf in self._text_layers(doc, page_indices, num_processes):
                with pymupdf.open(stream=layer_pdf, filetype="pdf") as layer:
                    _overlay_text_layer(output[page_index], layer)
                progress.advance(task)

        return _persist_to_temp_file(output) if doc.name else output

    def _text_layers(
        self, doc: pymupdf.Document, page_indices: list[int], num_processes: int
    ) -> Iterator[tuple[int, bytes]]:
        """Yield each page's index and text layer PDF from _text_layer(), in page order.

        With more than one process, workers open the document's file and OCR its pages.
        """
        if num_processes <= 1:
            for page_index in page_indices:
                with _text_layer(doc[page_index], self.language, self.dpi) as layer:
                    yield page_index, _layer_bytes(layer)
            return

        tasks = [(page_index, self.language, self.dpi) for page_index in page_indices]
        with Pool(processes=num_processes, initializer=_open_worker_document, initargs=(doc.name,)) as pool:
            yield from zip(page_indices, pool.imap(_worker_text_layer, tasks), strict=True)
