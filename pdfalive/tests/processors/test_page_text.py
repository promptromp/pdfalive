"""Tests for reading a page's text as displayed."""

import pymupdf
import pytest

from pdfalive.processors.page_text import displayed_page_dict


PAGE_WIDTH, PAGE_HEIGHT = 400, 600
HEADER_TEXT = "TOP HEADER"
# Where the header's baseline starts on the page as displayed.
HEADER_ORIGIN = pymupdf.Point(20, 30)
HEADER_FONT_SIZE = 12
ROTATIONS = [0, 90, 180, 270]
# Mapping a box through a rotation matrix rounds it by a few hundred-thousandths of a point.
POSITION_TOLERANCE = 0.01


def _page_with_header(doc: pymupdf.Document, rotation: int) -> pymupdf.Page:
    """A page stored portrait, shown with ``rotation``, with HEADER_TEXT upright at the top as displayed."""
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.set_rotation(rotation)
    page.insert_text(HEADER_ORIGIN * page.derotation_matrix, HEADER_TEXT, fontsize=HEADER_FONT_SIZE, rotate=rotation)
    return page


def _header_line(page_dict: dict) -> tuple[dict, dict]:
    (block,) = page_dict["blocks"]
    (line,) = block["lines"]
    (span,) = line["spans"]
    return line, span


@pytest.fixture
def displayed_header() -> tuple[dict, dict]:
    """The header line and span of an unrotated page: where the text is displayed."""
    with pymupdf.open() as doc:
        return _header_line(_page_with_header(doc, 0).get_text("dict"))


def test_page_displayed_as_stored_is_returned_as_extracted():
    with pymupdf.open() as doc:
        page = _page_with_header(doc, 0)
        assert displayed_page_dict(page) == page.get_text("dict")


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_text_is_where_it_is_displayed(rotation, displayed_header):
    expected_line, expected_span = displayed_header
    with pymupdf.open() as doc:
        page = _page_with_header(doc, rotation)
        page_dict = displayed_page_dict(page)
        displayed_size = (page.rect.width, page.rect.height)

    line, span = _header_line(page_dict)
    assert (page_dict["width"], page_dict["height"]) == displayed_size
    assert span["text"] == HEADER_TEXT
    assert span["bbox"] == pytest.approx(expected_span["bbox"], abs=POSITION_TOLERANCE)
    assert span["origin"] == pytest.approx(expected_span["origin"], abs=POSITION_TOLERANCE)
    assert line["bbox"] == pytest.approx(expected_line["bbox"], abs=POSITION_TOLERANCE)
    assert line["dir"] == pytest.approx(expected_line["dir"], abs=POSITION_TOLERANCE)


# Ways a page's displayed area can differ from its MediaBox, set up on a fresh page.
PAGE_LAYOUTS = {
    "plain": lambda doc, page: None,
    "cropbox-offset": lambda doc, page: page.set_cropbox(pymupdf.Rect(30, 40, 370, 560)),
    # MuPDF clips a CropBox to the MediaBox; Page.rotation_matrix uses the raw CropBox.
    "cropbox-past-mediabox": lambda doc, page: doc.xref_set_key(page.xref, "CropBox", "[0 0 400 700]"),
    "user-unit": lambda doc, page: doc.xref_set_key(page.xref, "UserUnit", "2"),
}
# Pixels of a grayscale render darker than this are ink.
INK_THRESHOLD = 128


def _ink_box(page: pymupdf.Page) -> pymupdf.Rect:
    """The box of everything drawn on a page, from a render at 72 DPI (display coordinates)."""
    pix = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY)
    width, samples = pix.width, pix.samples
    rows = [y for y in range(pix.height) if min(samples[y * width : (y + 1) * width]) < INK_THRESHOLD]
    columns = [x for x in range(width) if any(samples[y * width + x] < INK_THRESHOLD for y in rows)]
    return pymupdf.Rect(columns[0], rows[0], columns[-1] + 1, rows[-1] + 1)


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("layout", PAGE_LAYOUTS)
def test_text_box_covers_the_rendered_text(rotation, layout):
    """The ground truth is where the text is drawn, whatever the page's boxes and rotation."""
    with pymupdf.open() as doc:
        PAGE_LAYOUTS[layout](doc, doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT))
        page = doc[0]
        page.set_rotation(rotation)
        shown = page.rect
        origin = pymupdf.Point(shown.x0 + 0.1 * shown.width, shown.y0 + 0.2 * shown.height)
        page.insert_text(origin * page.derotation_matrix, HEADER_TEXT, fontsize=HEADER_FONT_SIZE, rotate=rotation)
        page = doc[0]
        _, span = _header_line(displayed_page_dict(page))
        ink = _ink_box(page)

    assert pymupdf.Rect(span["bbox"]).contains(pymupdf.Point((ink.x0 + ink.x1) / 2, (ink.y0 + ink.y1) / 2))
