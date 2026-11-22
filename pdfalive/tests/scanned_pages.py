"""Synthetic test pages for OCR: words shown as text or as a page image, and where OCR put them."""

import pymupdf


# The font of the invisible text layer MuPDF's Tesseract integration writes.
OCR_FONT = "GlyphLessFont"
# Words drawn on the test pages, large enough for Tesseract to read reliably at OCR_DPI.
SHOWN_WORDS = (("QUARTERLY", 30, (0.1, 0.3)), ("STATISTICS", 24, (0.3, 0.75)))
OCR_DPI = 100
SCAN_DPI = 150
PAGE_WIDTH, PAGE_HEIGHT = 600, 450
CROPBOX = pymupdf.Rect(40, 30, 560, 420)
TITLE = "A Scanned Report"
BOOKMARKS = [[1, "First Part", 1], [1, "Second Part", 2]]


def insert_words(page: pymupdf.Page) -> None:
    """Write SHOWN_WORDS upright on the page as displayed, whatever its rotation and cropbox."""
    shown = page.rect
    for word, fontsize, (fx, fy) in SHOWN_WORDS:
        point = pymupdf.Point(shown.x0 + fx * shown.width, shown.y0 + fy * shown.height)
        page.insert_text(point * page.derotation_matrix, word, fontsize=fontsize, rotate=page.rotation)


def text_page(
    doc: pymupdf.Document, rotation: int = 0, cropbox: pymupdf.Rect | None = None, sideways: bool = False
) -> pymupdf.Page:
    """Add a page showing SHOWN_WORDS as text; stored with width and height swapped if ``sideways``."""
    width, height = (PAGE_HEIGHT, PAGE_WIDTH) if sideways else (PAGE_WIDTH, PAGE_HEIGHT)
    page = doc.new_page(width=width, height=height)
    if cropbox is not None:
        page.set_cropbox(cropbox)
    page.set_rotation(rotation)
    insert_words(page)
    return page


def scan_page(doc: pymupdf.Document) -> pymupdf.Page:
    """Add a page showing SHOWN_WORDS as an image only, as a scanned page does."""
    with pymupdf.open() as source:
        pix = text_page(source).get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_image(page.rect, pixmap=pix)
    return page


def shown_word_boxes(page: pymupdf.Page) -> dict[str, pymupdf.Rect]:
    """Boxes of the words a page shows as (non-OCR) text."""
    return {
        span["text"].strip(): pymupdf.Rect(span["bbox"])
        for block in page.get_text("dict")["blocks"]
        if block["type"] == 0
        for line in block["lines"]
        for span in line["spans"]
        if span["font"] != OCR_FONT
    }


def ocr_word_boxes(page: pymupdf.Page) -> dict[str, pymupdf.Rect]:
    """Boxes of the words in a page's OCR text layer."""
    return {
        span["text"].strip(): pymupdf.Rect(span["bbox"])
        for block in page.get_text("dict")["blocks"]
        if block["type"] == 0
        for line in block["lines"]
        for span in line["spans"]
        if span["font"] == OCR_FONT
    }


def placed_on(ocr_box: pymupdf.Rect, shown_box: pymupdf.Rect) -> bool:
    """Whether an OCR word lies on the shown word: centred within it, about as long.

    Tesseract's boxes hug the glyphs, while a text span's box spans the font's full
    height, so only the centre and the length along the line are compared.
    """
    centre = pymupdf.Point((ocr_box.x0 + ocr_box.x1) / 2, (ocr_box.y0 + ocr_box.y1) / 2)
    ocr_length, shown_length = max(ocr_box.width, ocr_box.height), max(shown_box.width, shown_box.height)
    return bool(shown_box.contains(centre)) and abs(ocr_length - shown_length) <= 0.1 * shown_length
