"""Reading a page's text structure as the page is displayed."""

import pymupdf


# PyMuPDF's "dict" extraction flags without TEXT_PRESERVE_IMAGES: text lines and spans
# are identical, but image bytes are not embedded (on scanned books every page carries a
# full-page image, which makes the default extraction ~35x slower). For readers of lines
# and spans only (folio detection, the per-page text check of OCR), since the block
# grouping this can change around images does not matter to them.
TEXT_ONLY_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES


def _display_matrix(page: pymupdf.Page) -> pymupdf.Matrix | None:
    """The matrix from a page's unrotated text coordinates to its display coordinates; None if not rotated.

    Built from the displayed page's size (``page.rect``), the bounds MuPDF places text in:
    ``page.rotation_matrix`` uses the raw CropBox instead, which differs when the CropBox
    extends past the MediaBox (MuPDF clips it) or the page sets ``/UserUnit``.
    """
    width, height = page.rect.width, page.rect.height
    return {
        90: pymupdf.Matrix(0, 1, -1, 0, width, 0),
        180: pymupdf.Matrix(-1, 0, 0, -1, width, height),
        270: pymupdf.Matrix(0, -1, 1, 0, 0, height),
    }.get(page.rotation)


def displayed_page_dict(page: pymupdf.Page, flags: int = pymupdf.TEXTFLAGS_DICT) -> dict:
    """Return a page's ``get_text("dict")`` structure in the page's display orientation.

    PyMuPDF reports a page's text in unrotated coordinates: on a page with a ``/Rotate``
    entry, ``width``, ``height`` and every box describe the page as stored, so a running
    header shown at the top of a page stored sideways sits at its left or right edge. For
    such a page, the boxes of blocks, lines and spans, span origins and line directions are
    mapped into display coordinates (_display_matrix()), and ``width`` and ``height`` are
    the displayed page's (``page.rect``), so a position read against the height is where
    the text is shown. A page displayed as stored (PyMuPDF normalizes ``Page.rotation`` to
    0, 90, 180 or 270) is returned as extracted, its ``width`` and ``height`` those PyMuPDF
    reports.

    Args:
        page: The page to read.
        flags: PyMuPDF text extraction flags (default: those of a plain "dict" extraction).

    Returns:
        The page's text dict, in display coordinates.
    """
    page_dict = page.get_text("dict", flags=flags)
    rotation = _display_matrix(page)
    if rotation is None:
        return page_dict

    direction_only = pymupdf.Matrix(rotation.a, rotation.b, rotation.c, rotation.d, 0, 0)
    page_dict["width"], page_dict["height"] = page.rect.width, page.rect.height
    for block in page_dict.get("blocks", []):
        block["bbox"] = tuple(pymupdf.Rect(block["bbox"]) * rotation)
        for line in block.get("lines", []):
            line["bbox"] = tuple(pymupdf.Rect(line["bbox"]) * rotation)
            line["dir"] = tuple(pymupdf.Point(line["dir"]) * direction_only)
            for span in line.get("spans", []):
                span["bbox"] = tuple(pymupdf.Rect(span["bbox"]) * rotation)
                span["origin"] = tuple(pymupdf.Point(span["origin"]) * rotation)
    return page_dict
