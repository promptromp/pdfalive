"""Printed page numbers (folios) read off PDF pages, and the printed-to-PDF mapping they imply.

A printed book numbers its pages. A PDF of it usually starts with unnumbered or
roman-numbered front matter, and may also omit printed pages (e.g. stripped blank
versos) or insert unnumbered ones (plates, part title pages), so the offset between
printed and PDF page numbers can drift across the document. Reading the folio printed
on each page recovers the mapping directly instead of assuming one global offset.
"""

import bisect
import re
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from types import MappingProxyType
from typing import NamedTuple

import pymupdf


# PyMuPDF's "dict" extraction flags without TEXT_PRESERVE_IMAGES: text lines and spans
# are identical, but image bytes are not embedded (on scanned books every page carries a
# full-page image, which makes the default extraction ~35x slower). Folio detection only
# reads lines, so the block grouping this can change around images does not matter here.
_TEXT_ONLY_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES

# Largest |sin| of a text line's tilt still read as horizontal: scans are deskewed to
# within a few degrees, while watermarks and margin notes are set at steep angles.
_MAX_HORIZONTAL_TILT = 0.1

# A line that can define the page's top or bottom edge has at least one letter or digit;
# punctuation-only OCR noise ("|", ";") and symbol glyphs (end-of-proof marks) do not.
_EDGE_LINE_CONTENT = re.compile(r"\w")

# Number of text rows at the page's top and at its bottom where a folio may sit: the
# outermost row (beside a running title), or one row inside a per-page stamp line such
# as a repository's "Downloaded from ... on ..." notice.
_EDGE_ROWS = 2

# An Arabic page number; printed numbering starts at 1.
_FOLIO_NUMBER = r"[1-9]\d{0,3}"

# An edge line that is a folio and nothing else, possibly decorated: "12", "- 12 -",
# "— 12 —", "[12]", "Page 12". (Numbers sharing a line with a running title are not
# read: there they are mostly chapter or section numbers.)
_FOLIO_LINE_PATTERN = re.compile(rf"[^\w]*(?:page\s+)?({_FOLIO_NUMBER})[^\w]*", re.IGNORECASE)

# A folio candidate is trusted only when a page at most this many pages away carries
# one implying the same printed-to-PDF offset. Consecutive printed pages share their
# offset, while stray numbers on an edge line (chapter, equation or list numbers, OCR
# noise) rarely form such runs.
FOLIO_CONFIRMATION_WINDOW = 2

# Confirmed folios off the book's numbering sequence make the printed pages they span
# ambiguous only when at least this many share an offset, i.e. form another numbering
# sequence (a separately numbered appendix, a second volume). Shorter off-sequence runs
# are edge-line noise, such as list or equation numbers on consecutive pages.
MIN_ALTERNATE_SEQUENCE_FOLIOS = 5

# Share of the document's pages that must carry a folio of the book's numbering
# sequence for the map to be trusted. Most pages of a paginated book carry a folio; a
# handful of edge-line numbers confirming each other is noise, not pagination.
MIN_FOLIO_PAGE_COVERAGE = 0.5

# A printed page whose folio was not detected (chapter openings, part titles) takes
# the offset of the nearest detected folio, but only within this many printed pages:
# farther away the offset may have drifted, so the page is reported as unknown.
MAX_FOLIO_EXTRAPOLATION = 10

_FIRST_PDF_PAGE = 1
_FIRST_PRINTED_PAGE = 1


class _TextLine(NamedTuple):
    top: float
    bottom: float
    text: str


def _horizontal_text_lines(page_dict: dict, rotation: pymupdf.Matrix) -> list[_TextLine]:
    """Return the page's non-empty text lines that read horizontally as displayed.

    Coordinates are mapped through ``rotation`` (the page's rotation matrix) so that a
    page with a ``/Rotate`` entry is judged as it is displayed. Tilted lines (e.g. a
    vertical or diagonal watermark) are skipped: their extent spans much of the page and
    says nothing about where a folio sits. Spans are joined with spaces, as OCR text
    layers put each word in its own span.
    """
    linear = pymupdf.Matrix(rotation.a, rotation.b, rotation.c, rotation.d, 0, 0)
    lines = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            direction = pymupdf.Point(line.get("dir", (1.0, 0.0))) * linear
            if direction.x <= 0 or abs(direction.y) > _MAX_HORIZONTAL_TILT * abs(direction):
                continue
            spans = [span for span in line.get("spans", []) if span.get("bbox")]
            text = " ".join(span.get("text", "") for span in spans).strip()
            if text:
                rects = [pymupdf.Rect(span["bbox"]) * rotation for span in spans]
                lines.append(_TextLine(top=min(r.y0 for r in rects), bottom=max(r.y1 for r in rects), text=text))
    return lines


def _overlaps_vertically(line: _TextLine, other: _TextLine) -> bool:
    """Whether two lines share a row (inclusive, so zero-height lines overlap themselves)."""
    return line.top <= other.bottom and line.bottom >= other.top


def _rows(lines: list[_TextLine]) -> list[list[_TextLine]]:
    """Group lines into rows of vertically overlapping lines, from top to bottom."""
    rows: list[list[_TextLine]] = []
    for line in sorted(lines, key=lambda line: line.top):
        if rows and any(_overlaps_vertically(line, other) for other in rows[-1]):
            rows[-1].append(line)
        else:
            rows.append([line])
    return rows


def page_folio_candidates(page_dict: dict, rotation: pymupdf.Matrix = pymupdf.Identity) -> set[int]:
    """Return the numbers that may be the folio printed on a page.

    Folios sit in the running header or footer: in one of the _EDGE_ROWS outermost rows
    of text at the top or bottom, as a line of their own (possibly decorated, "- 12 -")
    beside or instead of a running title. Rows are found relative to the page's own
    text rather than at fixed y-positions, because page margins vary widely between PDFs.

    Args:
        page_dict: The page's ``get_text("dict")`` structure.
        rotation: The page's rotation matrix (``page.rotation_matrix``).

    Returns:
        The candidate folios (usually zero or one).
    """
    lines = [line for line in _horizontal_text_lines(page_dict, rotation) if _EDGE_LINE_CONTENT.search(line.text)]
    rows = _rows(lines)
    edge_lines = [line for row in rows[:_EDGE_ROWS] + rows[-_EDGE_ROWS:] for line in row]
    return {int(match.group(1)) for line in edge_lines if (match := _FOLIO_LINE_PATTERN.fullmatch(line.text))}


def _confirmed_folios(candidates: list[set[int]]) -> list[tuple[int, int]]:
    """Return (pdf_page, folio) pairs whose offset recurs within FOLIO_CONFIRMATION_WINDOW pages."""
    offsets = [{page - folio for folio in folios} for page, folios in enumerate(candidates, start=_FIRST_PDF_PAGE)]
    confirmed: list[tuple[int, int]] = []
    for page_idx, folios in enumerate(candidates):
        nearby = range(
            max(0, page_idx - FOLIO_CONFIRMATION_WINDOW),
            min(len(candidates), page_idx + FOLIO_CONFIRMATION_WINDOW + 1),
        )
        nearby_offsets = set().union(*(offsets[idx] for idx in nearby if idx != page_idx))
        page = page_idx + _FIRST_PDF_PAGE
        confirmed.extend((page, folio) for folio in folios if page - folio in nearby_offsets)
    return confirmed


def _longest_increasing_chain(folio_pages: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the longest subsequence of (pdf_page, folio) pairs increasing in both components.

    Patience sorting over folios, with pairs ordered by PDF page and, within a page,
    by descending folio so that at most one folio per page is chosen. Strictly
    increasing folios also mean each printed number appears at most once.
    """
    ordered = sorted(folio_pages, key=lambda pair: (pair[0], -pair[1]))
    tail_folios: list[int] = []  # smallest tail folio of an increasing chain of each length
    tail_indices: list[int] = []
    predecessors: list[int | None] = []
    for idx, (_pdf_page, folio) in enumerate(ordered):
        length = bisect.bisect_left(tail_folios, folio)
        predecessors.append(tail_indices[length - 1] if length else None)
        if length == len(tail_folios):
            tail_folios.append(folio)
            tail_indices.append(idx)
        else:
            tail_folios[length] = folio
            tail_indices[length] = idx

    chain: list[tuple[int, int]] = []
    cursor = tail_indices[-1] if tail_indices else None
    while cursor is not None:
        chain.append(ordered[cursor])
        cursor = predecessors[cursor]
    return chain[::-1]


def _alternate_sequence_ranges(folio_pages: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the (first, last) printed span of each alternate numbering sequence among ``folio_pages``.

    An alternate sequence is a contiguous run of detections sharing an offset (no gap
    wider than FOLIO_CONFIRMATION_WINDOW pages) with at least MIN_ALTERNATE_SEQUENCE_FOLIOS
    detections; scattered same-offset noise does not join into one span.
    """
    pdf_pages_by_offset: defaultdict[int, list[int]] = defaultdict(list)
    for pdf_page, folio in folio_pages:
        pdf_pages_by_offset[pdf_page - folio].append(pdf_page)

    ranges: list[tuple[int, int]] = []
    for offset, pdf_pages in pdf_pages_by_offset.items():
        runs: list[list[int]] = []
        for pdf_page in sorted(pdf_pages):
            if runs and pdf_page - runs[-1][-1] <= FOLIO_CONFIRMATION_WINDOW:
                runs[-1].append(pdf_page)
            else:
                runs.append([pdf_page])
        ranges.extend((run[0] - offset, run[-1] - offset) for run in runs if len(run) >= MIN_ALTERNATE_SEQUENCE_FOLIOS)
    return ranges


class PrintedPageMap:
    """Maps printed page numbers to the 1-indexed PDF pages that carry them. Immutable."""

    def __init__(
        self,
        pdf_page_by_printed: Mapping[int, int],
        page_count: int,
        ambiguous_printed_ranges: Iterable[tuple[int, int]] = (),
    ) -> None:
        """Create a map.

        Args:
            pdf_page_by_printed: Detected folios of the book's numbering: printed page -> PDF page.
            page_count: Number of pages in the document; lookups never leave 1..page_count.
            ambiguous_printed_ranges: Inclusive printed-page spans also used by another
                numbering sequence (e.g. a separately numbered appendix); never resolved.
        """
        self._pdf_page_by_printed = dict(pdf_page_by_printed)
        self._sorted_folios = sorted(self._pdf_page_by_printed)
        self._page_count = page_count
        self._ambiguous_printed_ranges = tuple(ambiguous_printed_ranges)

    @classmethod
    def from_document(cls, doc: pymupdf.Document) -> "PrintedPageMap":
        """Build the map from the folios printed on every page of ``doc``."""
        candidates = []
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            candidates.append(
                page_folio_candidates(page.get_text("dict", flags=_TEXT_ONLY_FLAGS), page.rotation_matrix)
            )
        return cls.from_folio_candidates(candidates)

    @classmethod
    def from_folio_candidates(cls, candidates_by_page: Iterable[Collection[int]]) -> "PrintedPageMap":
        """Build the map from per-page folio candidates, in PDF page order.

        Keeps candidates confirmed by a nearby page implying the same offset, then the
        longest run of them increasing in both PDF page and folio: the book's numbering
        sequence. Confirmed detections off that run that form another numbering sequence
        make the printed pages it spans ambiguous; shorter off-run detections are noise. If the sequence covers
        less than MIN_FOLIO_PAGE_COVERAGE of the pages, the map is left empty.

        Args:
            candidates_by_page: Element i holds the folio candidates of PDF page i + 1.
        """
        candidates = [set(folios) for folios in candidates_by_page]
        page_count = len(candidates)
        confirmed = _confirmed_folios(candidates)
        chain = _longest_increasing_chain(confirmed)
        if len(chain) < MIN_FOLIO_PAGE_COVERAGE * page_count:
            return cls({}, page_count)
        off_chain = set(confirmed) - set(chain)
        return cls(
            {folio: pdf_page for pdf_page, folio in chain},
            page_count,
            _alternate_sequence_ranges(off_chain),
        )

    @property
    def pdf_page_by_printed(self) -> Mapping[int, int]:
        """Read-only view of the detected folios: printed page -> PDF page."""
        return MappingProxyType(self._pdf_page_by_printed)

    def __len__(self) -> int:
        """Number of printed pages whose folio was detected."""
        return len(self._pdf_page_by_printed)

    def _is_ambiguous(self, printed_page: int) -> bool:
        return any(first <= printed_page <= last for first, last in self._ambiguous_printed_ranges)

    def pdf_page(self, printed_page: int) -> int | None:
        """Return the PDF page carrying ``printed_page``, or None if it cannot be located.

        Exact when that page's folio was detected. Between two detected folios, the PDF
        page is interpolated linearly (exact when no page is missing in between, and
        monotone in the printed page), kept strictly between the folios' pages; when no
        page lies between them, the printed page is absent from the PDF. Beyond the
        first or last detected folio, the nearest folio's offset is applied. Lookups
        farther than MAX_FOLIO_EXTRAPOLATION printed pages from every detected folio,
        printed pages shared with another numbering sequence, and pages outside the
        document are not located.

        Known limitation: two printed pages per PDF page (2-up scans) keep only one
        folio per page, so half of their lookups are interpolated.
        """
        if printed_page < _FIRST_PRINTED_PAGE or not self._sorted_folios or self._is_ambiguous(printed_page):
            return None
        if printed_page in self._pdf_page_by_printed:
            return self._pdf_page_by_printed[printed_page]

        insert_at = bisect.bisect_left(self._sorted_folios, printed_page)
        lower = self._sorted_folios[insert_at - 1] if insert_at > 0 else None
        upper = self._sorted_folios[insert_at] if insert_at < len(self._sorted_folios) else None
        distance = min(abs(printed_page - folio) for folio in (lower, upper) if folio is not None)
        if distance > MAX_FOLIO_EXTRAPOLATION:
            return None

        if lower is not None and upper is not None:
            lower_pdf, upper_pdf = self._pdf_page_by_printed[lower], self._pdf_page_by_printed[upper]
            if upper_pdf - lower_pdf <= 1:
                return None  # no PDF page between the two folios: the printed page is absent
            fraction = (printed_page - lower) / (upper - lower)
            estimate = lower_pdf + round(fraction * (upper_pdf - lower_pdf))
            return min(max(estimate, lower_pdf + 1), upper_pdf - 1)

        nearest = lower if lower is not None else upper
        assert nearest is not None  # at least one neighbour exists: the map is not empty
        estimate = printed_page + self._pdf_page_by_printed[nearest] - nearest
        return estimate if _FIRST_PDF_PAGE <= estimate <= self._page_count else None
