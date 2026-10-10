"""Unit tests for printed page number (folio) detection and the printed-to-PDF page map."""

import random

import pymupdf
import pytest

from pdfalive.processors.printed_pages import (
    FOLIO_CONFIRMATION_WINDOW,
    MAX_FOLIO_EXTRAPOLATION,
    MIN_ALTERNATE_SEQUENCE_FOLIOS,
    MIN_FOLIO_PAGE_COVERAGE,
    PrintedPageMap,
    page_folio_candidates,
)


PAGE_HEIGHT = 800.0
LINE_HEIGHT = 12.0
LEFT_X, RIGHT_X = 72.0, 520.0
# Large top margin, as in PDFs whose pages are bigger than the printed trim area.
HEADER_Y = 118.0
BODY_Y = 160.0
FOOTER_Y = 670.0
LINE_SPACING = 2 * LINE_HEIGHT
# Enough body rows that a mid-page line is never among a page's edge rows.
BODY_ROWS = 4
FOOTER_FOLIO = 7
HORIZONTAL = (1.0, 0.0)
VERTICAL = (0.0, -1.0)
FOLIO = 210


def _line(y: float, *texts: str, x: float = LEFT_X, direction: tuple[float, float] = HORIZONTAL) -> dict:
    """A text line made of one span per text, laid out left to right."""
    spans = []
    for text in texts:
        width = LINE_HEIGHT * len(text)
        spans.append({"text": text, "bbox": (x, y, x + width, y + LINE_HEIGHT)})
        x += width + LINE_HEIGHT
    return {"dir": direction, "spans": spans}


def _vertical_line(x: float, top: float, bottom: float, text: str) -> dict:
    return {"dir": VERTICAL, "spans": [{"text": text, "bbox": (x, top, x + LINE_HEIGHT, bottom)}]}


def _page_dict(*lines: dict) -> dict:
    return {"height": PAGE_HEIGHT, "blocks": [{"type": 0, "lines": [line]} for line in lines]}


def _body_with(mid_page_text: str) -> list[dict]:
    """Body rows with ``mid_page_text`` in the middle, at least _EDGE_ROWS rows from both edges."""
    rows = [_line(BODY_Y + i * LINE_SPACING, f"Body line {i}") for i in range(BODY_ROWS)]
    rows.insert(BODY_ROWS // 2, _line(BODY_Y + (BODY_ROWS // 2 - 0.5) * LINE_SPACING, mid_page_text))
    return rows


class TestPageFolioCandidates:
    """Tests for page_folio_candidates()."""

    @pytest.mark.parametrize(
        "page_dict,expected",
        [
            pytest.param(
                _page_dict(_line(HEADER_Y, str(FOLIO)), _line(HEADER_Y, "8 Running Title", x=RIGHT_X)),
                {FOLIO},
                id="header-folio-beside-running-title",
            ),
            pytest.param(
                _page_dict(_line(HEADER_Y, "Chapter 9"), _line(BODY_Y, "Body"), _line(FOOTER_Y, str(FOLIO))),
                {FOLIO},
                id="footer-folio-on-chapter-opening",
            ),
            *(
                pytest.param(_page_dict(_line(BODY_Y, "Body"), _line(FOOTER_Y, text)), {FOLIO}, id=f"decorated-{text}")
                for text in (f"- {FOLIO} -", f"— {FOLIO} —", f"[{FOLIO}]", f"Page {FOLIO}")
            ),
            pytest.param(
                _page_dict(_line(HEADER_Y, f"{FOLIO} CHAPTER 1. INTRODUCTION"), _line(BODY_Y, "Body")),
                set(),
                id="number-sharing-a-line-with-running-title-is-not-read",
            ),
            pytest.param(
                _page_dict(_line(HEADER_Y, "Running title"), *_body_with("12"), _line(FOOTER_Y, "Last line")),
                set(),
                id="number-mid-page-ignored",
            ),
            pytest.param(
                _page_dict(_line(HEADER_Y, "8 Running Title"), _line(FOOTER_Y, "(3.2)")),
                set(),
                id="edge-lines-without-bare-number",
            ),
            pytest.param(
                _page_dict(_line(HEADER_Y, "1", "2"), _line(BODY_Y, "Body")),
                set(),
                id="separate-number-spans-are-not-one-number",
            ),
            pytest.param(_page_dict(_line(HEADER_Y, "0"), _line(BODY_Y, "Body")), set(), id="zero-is-not-a-folio"),
            pytest.param(_page_dict(), set(), id="empty-page"),
        ],
    )
    def test_detects_folio_lines_on_top_or_bottom_edge(self, page_dict, expected):
        assert page_folio_candidates(page_dict) == expected

    def test_punctuation_only_lines_do_not_define_page_edges(self):
        """OCR noise ("|") or a symbol glyph below the footer must not hide the folio."""
        page_dict = _page_dict(_line(HEADER_Y, "Running title"), _line(FOOTER_Y, str(FOLIO)), _line(FOOTER_Y + 40, "|"))

        assert page_folio_candidates(page_dict) == {FOLIO}

    def test_diagonal_watermark_does_not_define_page_edges(self):
        """A watermark tilted well below 45 degrees still is not a horizontal edge line."""
        tilt = (0.77, -0.64)  # about 40 degrees
        watermark = {
            "dir": tilt,
            "spans": [{"text": "CONFIDENTIAL DRAFT", "bbox": (LEFT_X, HEADER_Y, RIGHT_X, PAGE_HEIGHT)}],
        }
        page_dict = _page_dict(
            watermark, _line(HEADER_Y, "Running title"), *_body_with("12"), _line(FOOTER_Y, str(FOOTER_FOLIO))
        )

        assert page_folio_candidates(page_dict) == {FOOTER_FOLIO}

    def test_rotated_margin_text_does_not_define_page_edges(self):
        """A vertical watermark spanning the page must not turn mid-page numbers into folio candidates."""
        watermark = _vertical_line(LEFT_X / 2, HEADER_Y / 2, PAGE_HEIGHT - HEADER_Y / 2, "arXiv:0000.00000v1")
        page_dict = _page_dict(
            watermark, _line(HEADER_Y, "Running title"), *_body_with("12"), _line(FOOTER_Y, str(FOOTER_FOLIO))
        )

        assert page_folio_candidates(page_dict) == {FOOTER_FOLIO}

    def test_stamp_line_below_the_footer_does_not_hide_the_folio(self):
        """A per-page repository notice under the footer folio (one row inside the edge) is tolerated."""
        stamp = _line(FOOTER_Y + LINE_SPACING, "Downloaded from repository on 1 January")
        page_dict = _page_dict(
            _line(HEADER_Y, "Running title"), *_body_with("Body"), _line(FOOTER_Y, str(FOLIO)), stamp
        )

        assert page_folio_candidates(page_dict) == {FOLIO}

    def test_zero_height_folio_line_is_read(self):
        """Degenerate font metrics can give a line a zero-height bbox; it still sits on its own row."""
        flat_folio = {"dir": HORIZONTAL, "spans": [{"text": str(FOLIO), "bbox": (LEFT_X, FOOTER_Y, RIGHT_X, FOOTER_Y)}]}

        assert page_folio_candidates(_page_dict(_line(HEADER_Y, "Running title"), flat_folio)) == {FOLIO}

    @pytest.mark.parametrize("rotation", [90, 180, 270])
    def test_reads_folios_on_rotated_pages_as_displayed(self, rotation):
        """Text that reads horizontally on a /Rotate page is vertical in unrotated coordinates."""
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.set_rotation(rotation)
            for y, text in ((HEADER_Y, str(FOLIO)), (BODY_Y, "Body text of the page")):
                page.insert_text(pymupdf.Point(LEFT_X, y) * page.derotation_matrix, text, rotate=rotation)

            assert page_folio_candidates(page.get_text("dict"), page.rotation_matrix) == {FOLIO}

    def test_ignores_spans_without_bbox(self):
        page_dict = {"blocks": [{"lines": [{"spans": [{"text": "7"}]}]}]}
        assert page_folio_candidates(page_dict) == set()


def _consecutive_folios(first_folio: int, count: int) -> list[set[int]]:
    """Candidates for ``count`` consecutive pages numbered from ``first_folio``."""
    return [{first_folio + i} for i in range(count)]


class TestPrintedPageMapConstruction:
    """Tests for PrintedPageMap.from_folio_candidates()."""

    FRONT_MATTER_PAGES = 3
    RUN_LENGTH = 12

    def test_maps_consecutive_folios_with_constant_offset(self):
        candidates = [set()] * self.FRONT_MATTER_PAGES + _consecutive_folios(1, self.RUN_LENGTH)

        page_map = PrintedPageMap.from_folio_candidates(candidates)

        assert page_map.pdf_page_by_printed == {
            folio: folio + self.FRONT_MATTER_PAGES for folio in range(1, self.RUN_LENGTH + 1)
        }

    def test_keeps_both_offsets_across_an_omitted_page(self):
        """A printed page missing from the PDF shifts the offset; both runs are kept."""
        omitted_folio = self.RUN_LENGTH + 1
        candidates = _consecutive_folios(1, self.RUN_LENGTH) + _consecutive_folios(omitted_folio + 1, self.RUN_LENGTH)

        page_map = PrintedPageMap.from_folio_candidates(candidates)

        assert page_map.pdf_page_by_printed[self.RUN_LENGTH] == self.RUN_LENGTH
        assert page_map.pdf_page_by_printed[omitted_folio + 1] == self.RUN_LENGTH + 1
        assert omitted_folio not in page_map.pdf_page_by_printed

    def test_rejects_isolated_number_without_confirming_neighbor(self):
        stray_folio = 90
        first_run_folio = 5
        candidates = (
            [set()] * (FOLIO_CONFIRMATION_WINDOW + 1)
            + [{stray_folio}]
            + _consecutive_folios(first_run_folio, self.RUN_LENGTH)
        )

        assert stray_folio not in PrintedPageMap.from_folio_candidates(candidates).pdf_page_by_printed

    def test_stray_run_breaking_monotone_numbering_is_dropped_but_not_ambiguous(self):
        """A short off-sequence run (e.g. a numbered list's last lines) is noise."""
        first_folio, stray_run_length = 10, MIN_ALTERNATE_SEQUENCE_FOLIOS - 1
        candidates = _consecutive_folios(first_folio, self.RUN_LENGTH) + _consecutive_folios(1, stray_run_length)

        page_map = PrintedPageMap.from_folio_candidates(candidates)

        assert sorted(page_map.pdf_page_by_printed) == list(range(first_folio, first_folio + self.RUN_LENGTH))
        assert page_map.pdf_page(first_folio + 1) == page_map.pdf_page_by_printed[first_folio + 1]

    def test_alternate_numbering_sequence_makes_its_printed_pages_ambiguous(self):
        """A separately numbered part (e.g. an appendix restarting at 1) cannot be resolved by number alone."""
        appendix_length = MIN_ALTERNATE_SEQUENCE_FOLIOS
        main_run = _consecutive_folios(1, self.RUN_LENGTH * 2)
        page_map = PrintedPageMap.from_folio_candidates(main_run + _consecutive_folios(1, appendix_length))

        assert all(page_map.pdf_page(folio) is None for folio in range(1, appendix_length + 1))
        assert page_map.pdf_page(appendix_length + 1) == appendix_length + 1

    def test_scattered_same_offset_noise_does_not_span_a_large_ambiguous_range(self):
        """Off-sequence detections sharing an offset but far apart are separate runs, each too short."""
        main_run = _consecutive_folios(1, self.RUN_LENGTH * 4)
        stray_pair = [{1}, {2}]  # off-sequence, same offset in both places
        gap = [set()] * (FOLIO_CONFIRMATION_WINDOW + 1)
        candidates = main_run + stray_pair + gap + stray_pair[: MIN_ALTERNATE_SEQUENCE_FOLIOS - 2]

        page_map = PrintedPageMap.from_folio_candidates(candidates)

        assert page_map.pdf_page(1) == 1

    def test_too_few_folios_yield_an_empty_map(self):
        """A couple of edge-line numbers confirming each other is noise, not pagination."""
        stray_pages = 2
        page_count = int(stray_pages / MIN_FOLIO_PAGE_COVERAGE) + 1
        candidates = _consecutive_folios(1, stray_pages) + [set()] * (page_count - stray_pages)

        assert len(PrintedPageMap.from_folio_candidates(candidates)) == 0

    def test_keeps_one_folio_per_page(self):
        candidates = [{1}, {2, 3}, {3}, {4}]

        page_map = PrintedPageMap.from_folio_candidates(candidates)

        pdf_pages = list(page_map.pdf_page_by_printed.values())
        assert len(pdf_pages) == len(set(pdf_pages))


# Lookup scenario: folios detected on printed pages 1-3 and 6-7. Printed 4 (a blank
# verso) is omitted from the PDF, so from printed 5 on the offset is one smaller;
# printed 5 is a chapter opening without a folio.
LOOKUP_PAGE_COUNT = 40
LOOKUP_OFFSET = 13
FIRST_RUN = range(1, 4)
SECOND_RUN = range(6, 8)
LAST_FOLIO = SECOND_RUN[-1]
LAST_FOLIO_PDF_PAGE = LAST_FOLIO + LOOKUP_OFFSET - 1


class TestPrintedPageMapLookup:
    """Tests for PrintedPageMap.pdf_page()."""

    PAGE_COUNT = LOOKUP_PAGE_COUNT
    FRONT_MATTER_OFFSET = LOOKUP_OFFSET
    FIRST_RUN = FIRST_RUN
    SECOND_RUN = SECOND_RUN
    PAGE_MAP = PrintedPageMap(
        {folio: folio + LOOKUP_OFFSET for folio in FIRST_RUN}
        | {folio: folio + LOOKUP_OFFSET - 1 for folio in SECOND_RUN},
        LOOKUP_PAGE_COUNT,
    )

    @pytest.mark.parametrize(
        "printed_page,expected_pdf_page",
        [
            pytest.param(FIRST_RUN[1], FIRST_RUN[1] + LOOKUP_OFFSET, id="exact"),
            pytest.param(SECOND_RUN[0] - 1, SECOND_RUN[0] - 1 + LOOKUP_OFFSET - 1, id="missing-folio"),
            pytest.param(
                LAST_FOLIO + MAX_FOLIO_EXTRAPOLATION,
                LAST_FOLIO_PDF_PAGE + MAX_FOLIO_EXTRAPOLATION,
                id="extrapolation-limit",
            ),
            pytest.param(LAST_FOLIO + MAX_FOLIO_EXTRAPOLATION + 1, None, id="beyond-extrapolation-limit"),
            pytest.param(0, None, id="not-a-printed-page"),
        ],
    )
    def test_locates_printed_pages(self, printed_page, expected_pdf_page):
        assert self.PAGE_MAP.pdf_page(printed_page) == expected_pdf_page

    def test_printed_page_omitted_from_the_pdf_is_not_located(self):
        omitted = self.SECOND_RUN[0] - 2  # printed 4: between pages carrying folios 3 and 5 there is none
        page_map = PrintedPageMap(
            {omitted - 1: omitted - 1 + self.FRONT_MATTER_OFFSET, omitted + 1: omitted + self.FRONT_MATTER_OFFSET},
            self.PAGE_COUNT,
        )

        assert page_map.pdf_page(omitted) is None

    def test_missing_folio_stays_strictly_between_neighbouring_folios(self):
        """A part page between two folios lands on the page between them, not on a neighbour's page."""
        lower_folio, upper_folio = 37, 41  # printed 38 and 40 (blank versos) omitted from the PDF
        lower_pdf_page = 47
        page_map = PrintedPageMap({lower_folio: lower_pdf_page, upper_folio: lower_pdf_page + 2}, self.PAGE_COUNT * 2)

        assert page_map.pdf_page((lower_folio + upper_folio) // 2) == lower_pdf_page + 1

    def test_ties_take_the_following_folio_offset(self):
        """An unnumbered insert (plate) before a chapter opening: count back from the next folio."""
        lower_folio, upper_folio = 39, 41
        lower_pdf_page = 30
        upper_pdf_page = lower_pdf_page + 3  # a plate and the chapter opening (printed 40) in between
        page_map = PrintedPageMap({lower_folio: lower_pdf_page, upper_folio: upper_pdf_page}, self.PAGE_COUNT)

        assert page_map.pdf_page(lower_folio + 1) == upper_pdf_page - 1

    def test_printed_page_before_the_pdf_starts_is_not_located(self):
        """An excerpt PDF starting at printed page 45: earlier printed pages are not in the document."""
        first_folio = 45
        page_map = PrintedPageMap({first_folio: 1, first_folio + 1: 2}, self.PAGE_COUNT)

        assert page_map.pdf_page(first_folio - 2) is None

    @pytest.mark.parametrize("seed", range(5))
    def test_lookups_never_step_backwards(self, seed):
        """Printed-to-PDF lookups are monotone, whatever the gaps between detected folios."""
        rng = random.Random(seed)
        folio_count, printed_span, pdf_span = 8, 200, 250
        folios = sorted(rng.sample(range(1, printed_span), folio_count))
        pdf_pages = sorted(rng.sample(range(1, pdf_span), folio_count))
        page_map = PrintedPageMap(dict(zip(folios, pdf_pages, strict=True)), pdf_span)

        located = [page for page in map(page_map.pdf_page, range(1, printed_span)) if page is not None]

        assert located == sorted(located)

    def test_extrapolation_past_the_last_page_is_not_clamped_onto_it(self):
        """With the last folio a few pages before the end, a printed page beyond the PDF is not located."""
        last_folio, last_pdf_page = 2, self.PAGE_COUNT - 9
        page_map = PrintedPageMap({last_folio - 1: last_pdf_page - 1, last_folio: last_pdf_page}, self.PAGE_COUNT)
        beyond = last_folio + (self.PAGE_COUNT - last_pdf_page) + 1

        assert page_map.pdf_page(beyond - 1) == self.PAGE_COUNT
        assert page_map.pdf_page(beyond) is None

    def test_printed_page_past_the_last_page_is_not_located(self):
        page_map = PrintedPageMap({1: self.PAGE_COUNT - 1, 2: self.PAGE_COUNT}, self.PAGE_COUNT)

        assert page_map.pdf_page(4) is None

    def test_empty_map_locates_nothing(self):
        assert PrintedPageMap({}, self.PAGE_COUNT).pdf_page(1) is None

    def test_is_immutable(self):
        """The map copies its input and exposes a read-only view."""
        printed_page = self.FIRST_RUN[0]
        pdf_page = printed_page + self.FRONT_MATTER_OFFSET
        source = {printed_page: pdf_page}
        page_map = PrintedPageMap(source, self.PAGE_COUNT)

        source[printed_page] = pdf_page + 1
        with pytest.raises(TypeError):
            page_map.pdf_page_by_printed[printed_page] = pdf_page + 1  # type: ignore[index]

        assert page_map.pdf_page(printed_page) == pdf_page


class TestPrintedPageMapFromDocument:
    """PrintedPageMap.from_document() on a real PDF."""

    FRONT_MATTER_PAGES = 3
    NUMBERED_PAGES = 8

    @pytest.fixture
    def numbered_doc(self):
        doc = pymupdf.open()
        for _ in range(self.FRONT_MATTER_PAGES):
            doc.new_page().insert_text((LEFT_X, BODY_Y), "Front matter")
        for folio in range(1, self.NUMBERED_PAGES + 1):
            page = doc.new_page()
            page.insert_text((LEFT_X, HEADER_Y), str(folio))
            page.insert_text((RIGHT_X - LEFT_X, HEADER_Y), "Running Title")
            page.insert_text((LEFT_X, BODY_Y), "Body text of the page")
        yield doc
        doc.close()

    def test_reads_folios_printed_on_pages(self, numbered_doc):
        page_map = PrintedPageMap.from_document(numbered_doc)

        assert page_map.pdf_page_by_printed == {
            folio: folio + self.FRONT_MATTER_PAGES for folio in range(1, self.NUMBERED_PAGES + 1)
        }
