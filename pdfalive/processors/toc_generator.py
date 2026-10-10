"""Table of Contents generator."""

import re
import time
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from functools import cached_property
from multiprocessing import Pool, cpu_count

import pymupdf
from langchain.chat_models.base import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from pdfalive.console import create_console, create_progress, safe_text
from pdfalive.models.toc import TOC, PostprocessedTOC, TOCEntry, TOCFeature
from pdfalive.processors.printed_pages import PrintedPageMap
from pdfalive.prompts import (
    TOC_GENERATOR_CONTINUATION_SYSTEM_PROMPT,
    TOC_GENERATOR_SYSTEM_PROMPT,
    TOC_POSTPROCESSOR_SYSTEM_PROMPT,
)
from pdfalive.structured_output import StructuredModel, invoke_structured_with_retry
from pdfalive.tokens import TokenUsage, estimate_tokens


# PDF font subset prefixes are six uppercase letters followed by '+', e.g. "ABCDEF+Times".
_FONT_SUBSET_PREFIX_LENGTH = 6
_FONT_SUBSET_PREFIX_PATTERN = re.compile(rf"^[A-Z]{{{_FONT_SUBSET_PREFIX_LENGTH}}}\+")


# Suppress PydanticSerializationUnexpectedValue warnings emitted by LangChain's
# with_structured_output() wrapper. The parsed output is correct; the warning is a
# known LangChain + Pydantic compatibility issue (the union serializer for the
# response type warns "Expected `none`" even though the value is valid).
# A context-manager approach (warnings.catch_warnings) doesn't work here because
# pydantic-core emits the warning from Rust, bypassing Python-level scoped filters.
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings", category=UserWarning, module=r"pydantic\.main"
)

# Sub-pattern for Roman numerals (I through XXXIX covers typical chapter counts)
_ROMAN_NUMERAL_RE = r"(?:X{0,3}(?:IX|IV|V?I{0,3}))"

# Regex pattern for section numbering (e.g. "1.", "1.2", "Chapter 1", "Appendix A",
# Roman numerals like "I ", "XIV.", and letter-spaced "C H A P T E R")
_SECTION_NUMBER_PATTERN = re.compile(
    r"^\s*("
    r"\d+\.|\d+\.\d+"  # Arabic: "1.", "1.2"
    r"|(?:Chapter|Section|Part|Appendix)\s"  # Named prefixes
    r"|C\s+H\s+A\s+P\s+T\s+E\s+R"  # Letter-spaced CHAPTER
    r"|" + _ROMAN_NUMERAL_RE + r"\.?\s"  # Roman numeral + optional dot + space
    r")",
    re.IGNORECASE,
)

_SECTION_PREFIX_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)+)")
_APPENDIX_PREFIX_PATTERN = re.compile(r"^\s*appendix\s+\S+\s+", re.IGNORECASE)

# Pattern for letter-spaced ALL-CAPS text (e.g. "C H A P T E R  I" or "P R E F A C E")
_LETTERSPACED_MIN_ADDITIONAL_CHARS = 3
_LETTERSPACED_PATTERN = re.compile(rf"^[A-Z](\s+[A-Z]){{{_LETTERSPACED_MIN_ADDITIONAL_CHARS},}}")

# Minimum/maximum text length for heading candidates
_HEADING_MIN_LENGTH = 3
_HEADING_MAX_LENGTH = 200

# Front matter titles to skip when detecting front matter offset
_FRONT_MATTER_TITLES = frozenset(
    {
        "contents",
        "table of contents",
        "introduction",
        "preface",
        "foreword",
        "acknowledgements",
        "acknowledgments",
        "note on the use of the book",
    }
)

# Compiled pattern to match front matter titles, optionally followed by qualifying
# phrases like "to the ...", "for the ...", "of the ...", "of volume ...".
# This prevents "Introduction to Ito-Calculus" from being classified as front matter
# while still matching "Introduction to the Second Edition".
_FRONT_MATTER_TITLE_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(t) for t in sorted(_FRONT_MATTER_TITLES)) + r")"
    r"(?:\s+(?:to the\b.*|to\s+.*\bedition\b.*|for the\b.*|of the\b.*|of volume\b.*))?$"
)

# Strong signal that a level-1 TOC entry is the first main-content chapter.
_MAIN_CONTENT_START_PATTERN = re.compile(
    r"^\s*(?:"
    r"1(?:\.|\s+)"
    r"|chapter\s+(?:1|i)\b"
    r"|part\s+(?:1|i)\b"
    r"|i\s+"
    r")",
    re.IGNORECASE,
)

# Minimum font size ratio vs body text to be considered a heading candidate (Phase 1)
_HEADING_FONT_SIZE_RATIO = 1.15

# Stricter font size ratio for Phase 2 heading candidates from remaining blocks
_HEADING_FONT_SIZE_RATIO_PHASE2 = 1.2

# Maximum number of heading candidates to add per page from Phase 2 scanning
_MAX_HEADING_CANDIDATES_PER_PAGE = 3

# Default maximum characters to preserve from each PDF text span.
DEFAULT_TEXT_SNIPPET_LENGTH = 50

# Default feature extraction limits per page/block.
DEFAULT_MAX_BLOCKS_PER_PAGE = 3
DEFAULT_MAX_LINES_PER_BLOCK = 5

# Default maximum TOC hierarchy depth requested from the LLM.
DEFAULT_MAX_TOC_DEPTH = 2

# PyMuPDF TOC level used for top-level bookmarks.
_ROOT_TOC_LEVEL = 1

# Normalized y-position threshold: spans below this are in the "bottom of page" zone
# where relaxed heading detection criteria are applied (Phase 2)
_BOTTOM_OF_PAGE_Y_THRESHOLD = 0.6

# Running header y-position threshold: features at y < this are likely running headers
_RUNNING_HEADER_Y_THRESHOLD = 0.05

# Running-header-aware search skips very top/bottom page regions.
_HEADING_SEARCH_HEADER_Y_THRESHOLD = 0.06
_HEADING_SEARCH_FOOTER_Y_THRESHOLD = 0.95

# Fallback correction checks the bottom half of the previous page.
_PREVIOUS_PAGE_BOTTOM_HEADING_Y_THRESHOLD = 0.5

# Fuzzy match: minimum length of the shorter string for substring matching
_FUZZY_MIN_SUBSTRING_LEN = 8

# Fuzzy match: minimum ratio of shorter/longer string lengths for a substring
# match to be accepted.  Blocks e.g. "introduction" (12) matching
# "2 introduction to itocalculus" (29) since 12/29 = 0.41 < 0.5.
_FUZZY_MIN_COVERAGE_RATIO = 0.5

# Front-matter offset detection ignores early level-1 entries and only treats
# offsets of this size or larger as meaningful enough to warn/correct.
_FRONT_MATTER_FIRST_MAIN_ENTRY_MIN_PAGE = 5
_SIGNIFICANT_FRONT_MATTER_OFFSET = 3

# Feature serialization formatting precision.
_FONT_SIZE_DECIMAL_PLACES = 1
_Y_POSITION_DECIMAL_PLACES = 2

# Token estimation should never report an empty feature block as zero tokens.
_MIN_ESTIMATED_BLOCK_TOKENS = 1

# PyMuPDF span flag bit indicating bold text.
_PYMUPDF_BOLD_FLAG = 16
_PYMUPDF_TEXT_BLOCK_TYPE = 0

# Reference text scan defaults for postprocessing.
DEFAULT_REFERENCE_TOC_MAX_PAGES = 20
DEFAULT_REFERENCE_TOC_UNFILTERED_PAGES = 3

# Postprocess feature summary defaults.
DEFAULT_POSTPROCESS_FEATURE_SUMMARY_MAX_ENTRIES = 80
_POSTPROCESS_HEADING_FONT_SIZE_RATIO = 1.1
_POSTPROCESS_HEADING_SUMMARY_BUDGET_RATIO = 0.8

# Safe fallback when PyMuPDF page height is unavailable.
_DEFAULT_PAGE_HEIGHT = 1.0

# Minimum process count when deriving parallelism from CPU count.
_MIN_PROCESS_COUNT = 1

# TOC reference-text filtering thresholds.
_TOC_DOT_LEADER_MIN_RUN_LENGTH = 3
_TOC_PAGE_NUMBER_MIN_SPACES = 4
_REFERENCE_TOC_OFFSET_MIN_MATCHES = 3
_REFERENCE_TOC_OFFSET_MIN_DOMINANCE_RATIO = 0.6
_REFERENCE_TOC_OFFSET_MAX_TITLE_LENGTH = 200

# Default local search window used for plain text heading checks.
_DEFAULT_HEADING_SEARCH_WINDOW = 1

# Output/prompt formatting precision.
_CONFIDENCE_DECIMAL_PLACES = 2

# Maximum number of dropped out-of-range entries named in the console warning.
_MAX_DROPPED_ENTRIES_SHOWN = 5


def _normalize_snippet(text: str) -> str:
    """Normalize text for fuzzy matching: strip numbering, punctuation, whitespace.

    Args:
        text: Raw text to normalize.

    Returns:
        Lowercased text with numbering prefixes, punctuation, and extra whitespace removed.
    """
    text = re.sub(r"^\d+(?:\.\d+)*[\.\)]*\s*", "", text)  # Strip "1. ", "3.2 ", "1) " prefixes
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_title_for_matching(title: str) -> str:
    """Normalize a title without dropping leading section numbers."""
    title = re.sub(r"[^\w\s\.]", "", title.strip().lower())
    return re.sub(r"\s+", " ", title)


def _extract_section_prefix(title: str) -> str | None:
    """Extract a dotted numeric section prefix such as '6.6' from a title."""
    match = _SECTION_PREFIX_PATTERN.match(title)
    return match.group(1) if match else None


# Leading designator tokens in normalized (punctuation-free, lowercased) titles:
# bare numbers ("34", "1 2") and label words with an optional short identifier
# ("chapter 7", "appendix b", "part iii", or a bare "chapter" when OCR loses
# the number glyph).
_LEADING_DESIGNATOR_PATTERN = re.compile(r"^(?:(?:chapter|part|appendix|section)\s+(?:\S{1,4}\s+)?|\d+\s+)")

_LEADING_NUMBER_PATTERN = re.compile(r"^\s*(\d+)")


def _strip_leading_designators(normalized_title: str) -> str:
    """Strip leading numbering/label designators from a normalized title.

    Section identity lives in the title's core text: "chapter 3 risk management"
    and "risk management" name the same section, while "translators preface" and
    "preface" do not. Never strips down to an empty string.
    """
    stripped = normalized_title
    while True:
        match = _LEADING_DESIGNATOR_PATTERN.match(stripped)
        if not match or match.end() >= len(stripped):
            return stripped
        stripped = stripped[match.end() :]


def _extract_leading_number(text: str) -> str | None:
    """Extract a leading integer designator from raw (unnormalized) text."""
    match = _LEADING_NUMBER_PATTERN.match(text)
    return match.group(1) if match else None


def _heading_text_matches_title(title: str, candidate_text: str, anchored: bool = False) -> bool:
    """Match heading text while preserving section-prefix evidence when present.

    In anchored mode (used when *locating* a heading's page), evidence must be
    structural, not incidental:
    - The title must match at the START of the candidate text. Headings begin
      blocks; a title appearing mid-block ("...see Preface to the second
      edition...") or behind extra leading words ("Translator's preface to the
      second edition") is a different or merely-referenced section.
    - A candidate that is a sub-phrase of the title (e.g. block "Differential
      forms" for title "Chapter 7 Differential forms") is accepted only if the
      candidate's own leading number designator, when present, agrees with the
      title's numbering — block "34. Differential forms" must not satisfy a
      title designated chapter 7.

    The default (unanchored) mode keeps loose containment for presence checks
    over large text spans (whole pages, page zones).
    """
    candidate_normalized = _normalize_snippet(candidate_text)
    if len(candidate_normalized) < _HEADING_MIN_LENGTH:
        return False

    search_texts = [_normalize_snippet(title)]
    appendix_without_prefix = _APPENDIX_PREFIX_PATTERN.sub("", title).strip()
    if appendix_without_prefix != title:
        search_texts.append(_normalize_snippet(appendix_without_prefix))

    candidate_leading_number = _extract_leading_number(candidate_text)
    title_numbers = re.findall(r"\d+", title)
    candidate_consistent = (
        candidate_leading_number is None or candidate_leading_number in title_numbers or not title_numbers
    )

    # Anchoring compares designator-stripped cores: a heading block may carry a
    # designator the title lacks ("Chapter 4: Numerical Differentiation" for
    # title "Numerical Differentiation") and vice versa.
    candidate_core = _strip_leading_designators(candidate_normalized)

    content_matches = False
    for search_text in search_texts:
        if len(search_text) < _HEADING_MIN_LENGTH:
            continue

        # Forward: title contained in the candidate (anchored: at its start only).
        if anchored:
            if candidate_consistent:
                # Full title at the candidate's start: heading may continue into
                # merged body text. Checked against both the normalized candidate
                # (block carries the same designator as the title) and its
                # designator-stripped core (block omits the designator).
                if candidate_normalized.startswith(search_text) or candidate_core.startswith(search_text):
                    content_matches = True
                    break
                # Relaxed variant: both sides reduced to designator-stripped
                # cores. With designators gone the cores carry all remaining
                # identity, so they must match EXACTLY — a prefix match would
                # conflate a short title with a longer different heading
                # ("Distance" vs "Distance and Angles").
                search_core = _strip_leading_designators(search_text)
                if (
                    search_core != search_text
                    and len(search_core) >= _FUZZY_MIN_SUBSTRING_LEN
                    and candidate_core == search_core
                ):
                    content_matches = True
                    break
        elif search_text in candidate_normalized:
            content_matches = True
            break

        # Reverse: candidate is a sub-phrase of the title (short heading block).
        shorter_len = min(len(search_text), len(candidate_normalized))
        longer_len = max(len(search_text), len(candidate_normalized))
        if (
            shorter_len >= _FUZZY_MIN_SUBSTRING_LEN
            and candidate_normalized in search_text
            and shorter_len / longer_len >= _FUZZY_MIN_COVERAGE_RATIO
        ):
            if anchored and not candidate_consistent:
                continue
            content_matches = True
            break

    if not content_matches:
        return False

    section_prefix = _extract_section_prefix(title)
    if section_prefix is None:
        return True

    return section_prefix in _normalize_title_for_matching(candidate_text)


def apply_toc_to_document(doc: pymupdf.Document, toc: list, output_file: str) -> None:
    """Apply a TOC (bookmarks) to a document and save it.

    This helper function is useful when you want to apply a TOC generated
    from one document (e.g., an OCR'd version) to another document
    (e.g., the original without OCR text layer).

    Args:
        doc: PyMuPDF Document to apply the TOC to.
        toc: TOC list in PyMuPDF format (list of [level, title, page] entries).
        output_file: Path to save the modified document.
    """
    doc.set_toc(toc)
    doc.save(output_file)


# Console for rich output
console = create_console()

# Default maximum tokens for features per batch to stay under context window limits.
# Must fit within the smallest common context window (128k tokens) with headroom for:
#   - System prompt (~800 tokens)
#   - User message template (~200 tokens)
#   - Structured output JSON schema (~500 tokens)
#   - Response/output tokens (~10k reserved)
#   - Safety margin (~17k)
# Token counts are computed using tiktoken (o200k_base encoding) for accuracy.
DEFAULT_MAX_TOKENS_PER_BATCH = 100000

# Default number of blocks to overlap between batches for context continuity
DEFAULT_OVERLAP_BLOCKS = 5

# Estimated token overhead for the prompt template (system + user message excluding features)
PROMPT_OVERHEAD_TOKENS = 3000

# Delay between LLM calls (in seconds) to avoid rate limiting
# Default is 10s to stay under typical rate limits (e.g., 30k input tokens/minute)
DEFAULT_REQUEST_DELAY_SECONDS = 10.0

# What the postprocessing LLM call is reported as.
_POSTPROCESS_DESCRIPTION = "TOC postprocessing"


# Pattern for lines that look like TOC entries: trailing page numbers,
# dot-leaders, or tab-separated numbers
_TOC_LINE_TRAILING_NUMBER_PATTERN = re.compile(
    rf"[.\s·…]{{{_TOC_DOT_LEADER_MIN_RUN_LENGTH},}}\s*\d+\s*$"  # dot-leaders before page number
    r"|"
    r"\t\d+\s*$"  # tab-separated page number
    r"|"
    rf"\s{{{_TOC_PAGE_NUMBER_MIN_SPACES},}}\d+\s*$"  # multiple spaces followed by a page number
)

# Maximum line length for short title-like lines in TOC filtering
_TOC_SHORT_LINE_MAX_LENGTH = 100


def _extract_toc_like_lines(page_text: str) -> str:
    """Filter page text to keep only lines that look like TOC entries.

    Keeps lines that have:
    - Trailing page numbers (with dot-leaders, tabs, or multiple spaces)
    - Section numbering patterns (e.g. "1.", "Chapter 3")
    - Short title-like lines (< 100 chars)

    Args:
        page_text: Raw text from a PDF page.

    Returns:
        Filtered text containing only TOC-like lines.
    """
    lines = page_text.split("\n")
    kept: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Keep lines with trailing page numbers (dot-leaders, etc.)
        if _TOC_LINE_TRAILING_NUMBER_PATTERN.search(stripped):
            kept.append(stripped)
            continue

        # Keep lines matching section numbering
        if _SECTION_NUMBER_PATTERN.match(stripped):
            kept.append(stripped)
            continue

        # Keep short title-like lines
        if len(stripped) < _TOC_SHORT_LINE_MAX_LENGTH:
            kept.append(stripped)

    return "\n".join(kept)


def _strip_subset_prefix(font_name: str) -> str:
    """Strip font subset prefix (e.g. 'ABCDEF+TimesNewRomanPS-BoldMT' -> 'TimesNewRomanPS-BoldMT').

    PDF fonts often include a 6-letter uppercase prefix followed by '+' for subset-embedded fonts.
    This prefix is arbitrary and varies per document, so stripping it reduces noise and enables
    font deduplication.

    Args:
        font_name: The raw font name from PyMuPDF.

    Returns:
        Font name without subset prefix.
    """
    return _FONT_SUBSET_PREFIX_PATTERN.sub("", font_name)


def serialize_features_compact(features: list) -> str:
    """Serialize features into a compact text format with a font table.

    Produces a format like:
        FONTS:
        F0=TimesNewRomanPS-BoldMT
        F1=TimesNewRomanPSMT

        P42:
        F0|14|Chapter 3: Risk Manag|.12|B
        F1|12|This is the first par|.15

    This is significantly more compact than Python str() on nested lists, reducing
    input tokens by ~50% while preserving all information the LLM needs.

    Args:
        features: Nested list of TOCFeature objects (blocks > lines > spans),
                  as produced by _extract_features().

    Returns:
        Compact string representation of the features.
    """
    # Pass 1: Collect unique font names and build font table
    font_set: dict[str, str] = {}  # stripped font name -> font ID

    for block in features:
        for line in block:
            for span in line:
                if isinstance(span, TOCFeature):
                    stripped = _strip_subset_prefix(span.font_name)
                    if stripped not in font_set:
                        font_set[stripped] = f"F{len(font_set)}"

    # Pass 2: Group spans by page and build output
    page_spans: dict[int, list[str]] = {}
    for block in features:
        for line in block:
            for span in line:
                if isinstance(span, TOCFeature):
                    stripped = _strip_subset_prefix(span.font_name)
                    font_id = font_set[stripped]

                    # Round font_size: 14.0 -> "14", 12.5 -> "12.5"
                    size = span.font_size
                    size_str = str(int(size)) if size == int(size) else str(round(size, _FONT_SIZE_DECIMAL_PLACES))

                    parts = [font_id, size_str, span.text_snippet]

                    # Only emit y_position and bold for heading-like spans
                    # (non-body spans: bold, larger size, or section-numbered)
                    if span.y_position is not None:
                        # Format: ".12" for 0.12, ".0" for 0.0
                        y_str = f"{span.y_position:.{_Y_POSITION_DECIMAL_PLACES}f}".lstrip("0") or "0"
                        parts.append(y_str)
                    if span.is_bold:
                        parts.append("B")

                    line_str = "|".join(parts)

                    if span.page_number not in page_spans:
                        page_spans[span.page_number] = []
                    page_spans[span.page_number].append(line_str)

    # Build output string
    lines: list[str] = []

    # Font table header
    if font_set:
        lines.append("FONTS:")
        for font_name, font_id in font_set.items():
            lines.append(f"{font_id}={font_name}")
        lines.append("")

    # Page groups
    for page_num in sorted(page_spans.keys()):
        lines.append(f"P{page_num}:")
        lines.extend(page_spans[page_num])

    return "\n".join(lines)


# Maximum non-heading (body) spans per page kept in the LLM payload. One body span
# per page preserves page anchoring and the font-size baseline while dropping the
# bulk of body text, which carries no TOC signal (~60% of payload tokens).
_MAX_BODY_SPANS_PER_PAGE_FOR_LLM = 1


def _is_heading_like_feature(span: TOCFeature, body_font_size: float) -> bool:
    """Classify an extracted TOCFeature as heading-like (vs body text).

    Applies the same shared criteria as _is_heading_candidate, but on
    TOCFeature objects (post-extraction) rather than raw PyMuPDF span dicts.
    """
    return _passes_heading_criteria(
        text=span.text_snippet.strip(),
        full_text_length=span.text_length,
        font_size=span.font_size,
        is_bold=bool(span.is_bold),
        body_font_size=body_font_size,
    )


def filter_features_for_llm(features: list, max_body_spans_per_page: int = _MAX_BODY_SPANS_PER_PAGE_FOR_LLM) -> list:
    """Reduce the feature payload sent to the LLM by dropping low-signal body spans.

    Keeps all heading-like spans (larger font, bold at body size, section
    numbering, letter-spaced caps) plus the first `max_body_spans_per_page`
    body spans per page. The retained body span keeps every page represented
    in the payload (page anchoring) and gives the LLM a body-font baseline —
    and is typically the running header or first paragraph line, preserving
    the y-position guidance in the prompt.

    The full (unfiltered) features must still be used for deterministic
    corrections and the postprocess summary; this filter only shrinks the
    LLM payload.

    Args:
        features: Nested list of TOCFeature objects (blocks > lines > spans).
        max_body_spans_per_page: Body spans to retain per page.

    Returns:
        A filtered nested list with the same blocks > lines > spans structure;
        empty lines and blocks are pruned.
    """
    _, body_font_size = _compute_body_font_profile(features)

    body_spans_kept: dict[int, int] = {}
    filtered: list[list] = []

    for block in features:
        filtered_block: list[list] = []
        for line in block:
            filtered_line = []
            for span in line:
                if not isinstance(span, TOCFeature):
                    continue
                if _is_heading_like_feature(span, body_font_size):
                    filtered_line.append(span)
                elif body_spans_kept.get(span.page_number, 0) < max_body_spans_per_page:
                    body_spans_kept[span.page_number] = body_spans_kept.get(span.page_number, 0) + 1
                    filtered_line.append(span)
            if filtered_line:
                filtered_block.append(filtered_line)
        if filtered_block:
            filtered.append(filtered_block)

    return filtered


def _estimate_block_tokens(block: list) -> int:
    """Estimate the token count for a single feature block in compact format.

    Builds an approximate compact-format string per block and counts tokens
    via tiktoken. This is fast enough for per-block calls (~50 chars each).

    Args:
        block: A block from the features list (list of lines, each a list of TOCFeature spans).

    Returns:
        Token count for this block.
    """
    parts: list[str] = ["P1:"]
    for line in block:
        for span in line:
            if isinstance(span, TOCFeature):
                parts.append(f"F0|12|{span.text_snippet}|.12|B")
    return max(_MIN_ESTIMATED_BLOCK_TOKENS, estimate_tokens("\n".join(parts)))


def _is_bold_font(span: dict) -> bool:
    """Check if a span uses a bold font.

    Uses both the PyMuPDF flags bitmask (bit 4 = bold) and font name heuristics.

    Args:
        span: A PyMuPDF span dict with 'flags' and 'font' keys.

    Returns:
        True if the span is bold.
    """
    flags = span.get("flags", 0)
    font_name = span.get("font", "")
    return bool(flags & _PYMUPDF_BOLD_FLAG) or "bold" in font_name.lower()


def _compute_body_font_profile(features: list) -> tuple[str, float]:
    """Determine the most common (font_name, font_size) pair across all features.

    This represents the "body text" baseline used to identify heading candidates.

    Args:
        features: Nested list of TOCFeature objects (blocks > lines > spans).

    Returns:
        Tuple of (font_name, font_size) for the most common pair.
        Falls back to ("", 0.0) if no features are found.
    """
    counter: Counter[tuple[str, float]] = Counter()
    for block in features:
        for line in block:
            for span in line:
                if isinstance(span, TOCFeature):
                    counter[(span.font_name, span.font_size)] += 1

    if not counter:
        return ("", 0.0)

    return counter.most_common(1)[0][0]


def _passes_heading_criteria(
    text: str,
    full_text_length: int,
    font_size: float,
    is_bold: bool,
    body_font_size: float,
    font_size_ratio: float = _HEADING_FONT_SIZE_RATIO,
) -> bool:
    """Shared heading classification core used for both raw spans and TOCFeatures.

    A text run is heading-like when its length is within heading bounds AND:
    - font size >= font_size_ratio * body font size, OR
    - bold at >= body font size, OR
    - it matches the section numbering pattern, OR
    - it is letter-spaced ALL-CAPS (e.g. "C H A P T E R  I").

    The bold-based check always uses the true body_font_size; only the
    size-based check uses font_size_ratio.
    """
    if len(text) < _HEADING_MIN_LENGTH or full_text_length > _HEADING_MAX_LENGTH:
        return False

    if body_font_size > 0 and font_size >= body_font_size * font_size_ratio:
        return True

    if is_bold and body_font_size > 0 and font_size >= body_font_size:
        return True

    if _SECTION_NUMBER_PATTERN.match(text):
        return True

    return bool(_LETTERSPACED_PATTERN.match(text))


def _is_heading_candidate(
    span: dict,
    body_font_name: str,
    body_font_size: float,
    font_size_ratio: float = _HEADING_FONT_SIZE_RATIO,
) -> bool:
    """Determine if a raw PyMuPDF span is likely a heading (see _passes_heading_criteria).

    Args:
        span: A PyMuPDF span dict.
        body_font_name: The most common font name (body text baseline).
        body_font_size: The most common font size (body text baseline).
        font_size_ratio: Minimum font size ratio vs body text for size-based detection.
            Defaults to _HEADING_FONT_SIZE_RATIO (1.15). Pass a higher value
            (e.g. _HEADING_FONT_SIZE_RATIO_PHASE2) for stricter Phase 2 scanning.
    """
    text = span.get("text", "").strip()
    return _passes_heading_criteria(
        text=text,
        full_text_length=len(text),
        font_size=span.get("size", 0.0),
        is_bold=_is_bold_font(span),
        body_font_size=body_font_size,
        font_size_ratio=font_size_ratio,
    )


def _merge_line_spans(spans: list[dict], text_snippet_length: int) -> dict:
    """Merge spans from one PDF line into a single synthetic span.

    PyMuPDF often splits one visual heading into multiple spans because of
    font/style changes. Treating only the first span can produce numeric-only
    headings like "3.2"; merging the line preserves the full candidate title.
    """
    if not spans:
        return {}

    text = re.sub(r"\s+", " ", " ".join(span.get("text", "").strip() for span in spans).strip())
    first = spans[0]
    bboxes = [span.get("bbox") for span in spans if span.get("bbox")]
    first_bbox = bboxes[0] if bboxes else first.get("bbox")

    return {
        "font": first.get("font", ""),
        "size": first.get("size", 0.0),
        "text": text[:text_snippet_length],
        "full_text_length": len(text),
        "bbox": first_bbox,
        "flags": first.get("flags", 0),
        "is_bold": any(_is_bold_font(span) for span in spans),
    }


def _span_to_feature(span: dict, page_number: int, page_height: float, text_snippet_length: int) -> TOCFeature:
    """Convert a raw or synthetic PyMuPDF span to a TOCFeature."""
    bbox = span.get("bbox")
    y_pos = round(bbox[1] / page_height, _Y_POSITION_DECIMAL_PLACES) if bbox and page_height > 0 else None
    text = span.get("text", "")
    return TOCFeature(
        page_number=page_number,
        font_name=span.get("font", ""),
        font_size=span.get("size", 0.0),
        text_length=span.get("full_text_length", len(text)),
        text_snippet=text[:text_snippet_length],
        y_position=y_pos,
        is_bold=span.get("is_bold", _is_bold_font(span)),
    )


def _extract_features_from_page_range(args: tuple) -> tuple[int, int, list, list]:
    """Worker function to extract features from a range of pages.

    This function is designed to be called in a separate process.
    It opens the document independently and processes its assigned pages.

    Args:
        args: Tuple of (process_index, total_processes, input_path,
                       max_blocks_per_page, max_lines_per_block, text_snippet_length)

    Returns:
        Tuple of (start_page, end_page, features_list, remaining_spans_buffer)
        for the processed range. remaining_spans_buffer contains
        (page_number, page_height, insert_index, spans) tuples for blocks beyond
        the max_blocks_per_page limit.
    """
    (
        process_idx,
        total_processes,
        input_path,
        max_blocks_per_page,
        max_lines_per_block,
        text_snippet_length,
        *optional_page_count,
    ) = args

    doc = pymupdf.open(input_path)
    page_count = min(optional_page_count[0], doc.page_count) if optional_page_count else doc.page_count

    # Calculate page range for this process
    pages_per_process = page_count // total_processes
    start_page = process_idx * pages_per_process
    end_page = start_page + pages_per_process if process_idx < total_processes - 1 else page_count

    features: list[list] = []
    remaining_spans_buffer: list[tuple[int, float, int, list[dict]]] = []

    for page_idx in range(start_page, end_page):
        page = doc[page_idx]
        page_number = page_idx + 1  # 1-indexed
        page_dict = page.get_text("dict")
        page_height = page_dict.get("height", page.rect.height) if hasattr(page, "rect") else _DEFAULT_PAGE_HEIGHT

        for block_ix, block in enumerate(page_dict["blocks"]):
            if block_ix >= max_blocks_per_page:
                # Buffer first span of remaining blocks for heading candidate scanning
                if block["type"] == _PYMUPDF_TEXT_BLOCK_TYPE:
                    lines = block.get("lines", [])
                    if lines:
                        spans = lines[0].get("spans", [])
                        if spans:
                            remaining_spans_buffer.append((page_number, page_height, len(features), spans))
                continue

            features.append([])
            if block["type"] == _PYMUPDF_TEXT_BLOCK_TYPE:
                # text block
                for line_ix, line in enumerate(block["lines"]):
                    if line_ix >= max_lines_per_block:
                        break

                    features[-1].append([])

                    for span in line["spans"]:
                        features[-1][-1].append(_span_to_feature(span, page_number, page_height, text_snippet_length))

    doc.close()
    return start_page, end_page, features, remaining_spans_buffer


def _title_match_key(title: str) -> str:
    """Normalize a title for matching postprocessed entries to first-pass ones (case/punctuation-blind)."""
    title = re.sub(r"[^\w\s]", "", title)  # strip all punctuation
    return re.sub(r"\s+", " ", title.strip().lower())


# An exact first-pass match may override a page located through the printed page numbers
# only this close to it: the same heading's two placements differ by a few pages at most
# (a misread printed page, a heading starting at the foot of a page); farther apart they
# are different occurrences of a repeated title ("Exercises", "Summary").
_MAX_FOLIO_MATCH_DISTANCE = 5


class _OriginalEntryMatcher:
    """Matches postprocessed entries to first-pass (original) entries, one-to-one.

    Each original entry restores at most one refined entry, so a title repeated per
    chapter ("Exercises") is not collapsed onto one original occurrence. Exact title
    matches are assigned first, globally and nearest position first; a fuzzy (prefix)
    match can then claim an original that no entry matched exactly.
    """

    def __init__(self, original_toc: TOC) -> None:
        self._pages_by_key: dict[str, list[tuple[int, int]]] = {}  # title key -> [(page_number, level)]
        for entry in original_toc.entries:
            self._pages_by_key.setdefault(_title_match_key(entry.title), []).append((entry.page_number, entry.level))
        self._claimed: set[tuple[str, int, int]] = set()  # (title key, page, level) already matched

    def claim_exact(self, requests: Sequence[tuple[str, int, int, bool]]) -> dict[int, int]:
        """Assign same-title, same-level originals to requests, nearest first.

        Each request is (title, level, position, position_is_located): a position
        located through the printed page numbers accepts only originals within
        _MAX_FOLIO_MATCH_DISTANCE; an estimated position accepts any. Same-level only:
        avoids restoring new same-title child entries to unrelated parent or
        front-matter pages.

        Returns:
            Request index -> original page, for the requests that were matched.
        """
        candidates = sorted(
            (abs(page - position), index, key, page, level)
            for index, (title, level, position, position_is_located) in enumerate(requests)
            for key in (_title_match_key(title),)
            for page, original_level in self._pages_by_key.get(key, [])
            if original_level == level
            and (not position_is_located or abs(page - position) <= _MAX_FOLIO_MATCH_DISTANCE)
        )
        assigned: dict[int, int] = {}
        for _distance, index, key, page, level in candidates:
            if index not in assigned and (key, page, level) not in self._claimed:
                self._claimed.add((key, page, level))
                assigned[index] = page
        return assigned

    def has_title(self, title: str, level: int) -> bool:
        """Whether any original entry has exactly this title (normalized) at this level."""
        return any(
            original_level == level for _page, original_level in self._pages_by_key.get(_title_match_key(title), [])
        )

    def claim_fuzzy(self, title: str, level: int, position: int) -> int | None:
        """Claim the best unclaimed prefix-aligned original at the same level; return its page or None.

        Among equally good candidates the one closest to ``position`` wins.
        """
        key = _title_match_key(title)
        # Collect ALL prefix-aligned matches at the same level, then pick the
        # best one (highest coverage ratio = shorter/longer).
        # Titles are compared on their designator-stripped cores, and the
        # shorter core must be a PREFIX of the longer one: this accepts
        # end-truncated snippets ("Chapter 3 Risk Manag") and printed-TOC
        # designator drops ("Distance" vs "1, §2. Distance"), but rejects
        # leading-word variants ("Preface..." vs "Translator's preface...")
        # which name different sections.
        key_core = _strip_leading_designators(key)
        key_digits = re.findall(r"\d+", key)
        best: tuple[float, int, str, int] | None = None  # (ratio, -distance to position, orig_key, orig_page)

        for orig_key, matches in self._pages_by_key.items():
            # Numbering is identity: when BOTH titles carry numbers and they
            # disagree, they are different sections even if their cores
            # match ("1.8 Exercises" vs "2.9 Exercises"). A missing number
            # on one side is fine (printed TOCs often drop designators).
            orig_digits = re.findall(r"\d+", orig_key)
            if key_digits and orig_digits and key_digits != orig_digits:
                continue
            orig_core = _strip_leading_designators(orig_key)
            shorter, longer = sorted((key_core, orig_core), key=len)
            if len(shorter) < _FUZZY_MIN_SUBSTRING_LEN:
                continue
            if not longer.startswith(shorter):
                continue
            ratio = len(shorter) / len(longer)
            if ratio < _FUZZY_MIN_COVERAGE_RATIO:
                continue
            for orig_page, orig_level in matches:
                if orig_level != level or (orig_key, orig_page, orig_level) in self._claimed:
                    continue
                candidate = (ratio, -abs(orig_page - position), orig_key, orig_page)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

        if best is None:
            return None
        _ratio, _distance, best_key, best_page = best
        self._claimed.add((best_key, best_page, level))
        return best_page


class TOCGenerator:
    """Class to generate table of contents for a PDF document."""

    def __init__(
        self,
        doc: pymupdf.Document,
        llm: BaseChatModel,
        num_processes: int | None = None,
    ) -> None:
        """Initialize the TOC generator.

        Args:
            doc: PyMuPDF Document object.
            llm: LangChain chat model for TOC inference.
            num_processes: Number of parallel processes for feature extraction.
                          Defaults to CPU count - 1.
        """
        self.doc = doc
        self.llm = llm
        self.num_processes = num_processes or max(_MIN_PROCESS_COUNT, cpu_count() - 1)
        self._structured_models: dict[type[BaseModel], StructuredModel] = {}

    def run(
        self,
        output_file: str | None,
        force: bool = False,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
        postprocess: bool = False,
    ) -> TokenUsage:
        """Generate the table of contents.

        Args:
            output_file: Path to save the modified PDF with TOC, or None to only
                set the TOC on the in-memory document without saving.
            force: If True, overwrite existing TOC. Otherwise raise if TOC exists.
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.
            postprocess: If True, run a postprocessing step to clean up and improve the TOC.

        Returns:
            TokenUsage statistics from the LLM calls.

        Raises:
            ValueError: If document has existing TOC and force=False.
        """
        if self._check_for_existing_toc() and not force:
            # TODO: can also use any existing toc to guide LLM generation.
            raise ValueError(
                "The input document already has a Table of Contents. Use `--force` to force TOC generation."
            )

        features = self._extract_features(self.doc)

        # Send a reduced payload to the LLM: body text beyond one span per page
        # carries no TOC signal. The full features are kept for the deterministic
        # corrections and postprocess summary below.
        llm_features = filter_features_for_llm(features)
        toc, usage = self._extract_toc(llm_features, request_delay=request_delay)

        # Deterministic correction: fix entries that point to running headers
        # instead of actual section starts (e.g., when a section starts near the
        # bottom of page N but the feature extraction only picked up the running
        # header at the top of page N+1).
        toc = self._correct_running_header_pages(toc, features)

        # Optionally postprocess the TOC to clean up duplicates, fix errors, etc.
        if postprocess:
            toc, postprocess_usage = self._postprocess_toc(toc, features)
            usage = usage + postprocess_usage

        # Deduplicate and enforce page number monotonicity after all corrections.
        # Postprocessing may produce duplicate entries (e.g., "Exercises" added once
        # per chapter from a printed TOC but all mapping to the same page).
        toc = toc.deduplicate()
        toc = self._drop_out_of_range_entries(toc)
        toc = toc.sort_by_page()
        toc = toc.sanitize_hierarchy()
        self.doc.set_toc(toc.to_list())
        if output_file is not None:
            self.doc.save(output_file)

        return usage

    def _structured_model(self, schema: type[BaseModel]) -> StructuredModel:
        """The LLM wrapped for ``schema``'s structured output, built once per schema.

        include_raw=True exposes the provider's actual token usage (including
        reasoning tokens) on the raw message instead of relying on estimates.
        """
        if schema not in self._structured_models:
            self._structured_models[schema] = self.llm.with_structured_output(schema, include_raw=True)
        return self._structured_models[schema]

    def _check_for_existing_toc(self) -> list:
        """Check if the document already has a TOC."""
        return self.doc.get_toc()

    def _drop_out_of_range_entries(self, toc: TOC) -> TOC:
        """Drop entries whose page number does not exist in the document, with a console warning.

        LLM page numbers are not guaranteed to be in range: an entry the pipeline
        could not place through the document itself (no first-pass match and no
        locatable printed page) keeps the LLM's estimate, which can overshoot the
        last page when a front-matter offset is applied to a PDF that omits printed
        pages. Such an entry has no known destination; pointing it at the nearest
        valid page would produce a bookmark to unrelated content, so it is dropped.
        """
        in_range, out_of_range = toc.partition_by_page_range(self.doc.page_count)
        if out_of_range:
            shown = ", ".join(
                f"'{entry.title}' (page {entry.page_number})" for entry in out_of_range[:_MAX_DROPPED_ENTRIES_SHOWN]
            )
            hidden_count = len(out_of_range) - _MAX_DROPPED_ENTRIES_SHOWN
            more = f" and {hidden_count} more" if hidden_count > 0 else ""
            noun = "entry" if len(out_of_range) == 1 else "entries"
            warning = (
                f"Dropped {len(out_of_range)} {noun} with page numbers outside the document "
                f"(1-{self.doc.page_count}): {shown}{more}"
            )
            console.print(safe_text("  ", (warning, "yellow")))
        return in_range

    def _extract_features(
        self,
        doc: pymupdf.Document,
        max_pages: int | None = None,
        max_blocks_per_page: int = DEFAULT_MAX_BLOCKS_PER_PAGE,
        max_lines_per_block: int = DEFAULT_MAX_LINES_PER_BLOCK,
        text_snippet_length: int = DEFAULT_TEXT_SNIPPET_LENGTH,
        show_progress: bool = True,
    ) -> list:
        """Extract features from the document to generate TOC entries.

        Features are indexed by page, block, line, and span.
        They include attributes such as: font name, size, text length, and a text snippet.

        Uses multiprocessing for large documents to speed up extraction.

        Args:
            doc: PyMuPDF Document object.
            max_pages: Maximum number of pages to process (None for all).
            max_blocks_per_page: Maximum blocks to extract per page.
            max_lines_per_block: Maximum lines to extract per block.
            text_snippet_length: Maximum characters for text snippets.
            show_progress: Whether to show progress indicator.

        Returns:
            Nested list of TOCFeature objects.
        """
        # For documents opened in memory (not from file), use sequential processing
        if not doc.name:
            return self._extract_features_sequential(
                doc, max_pages, max_blocks_per_page, max_lines_per_block, text_snippet_length, show_progress
            )

        page_count = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
        num_processes = min(self.num_processes, page_count)

        if num_processes <= _MIN_PROCESS_COUNT:
            return self._extract_features_sequential(
                doc, max_pages, max_blocks_per_page, max_lines_per_block, text_snippet_length, show_progress
            )

        return self._extract_features_parallel(
            doc.name,
            page_count,
            num_processes,
            max_blocks_per_page,
            max_lines_per_block,
            text_snippet_length,
            show_progress,
        )

    def _extract_features_sequential(
        self,
        doc: pymupdf.Document,
        max_pages: int | None = None,
        max_blocks_per_page: int = DEFAULT_MAX_BLOCKS_PER_PAGE,
        max_lines_per_block: int = DEFAULT_MAX_LINES_PER_BLOCK,
        text_snippet_length: int = DEFAULT_TEXT_SNIPPET_LENGTH,
        show_progress: bool = True,
    ) -> list:
        """Extract features sequentially (single process) using a two-phase approach.

        Phase 1: Extract first N blocks per page (enriched with y_position and is_bold),
        and buffer raw span data from remaining blocks.
        Phase 2: Compute body font profile, filter buffered spans for heading candidates,
        and insert them in page order.

        Args:
            doc: PyMuPDF Document object.
            max_pages: Maximum number of pages to process.
            max_blocks_per_page: Maximum blocks per page.
            max_lines_per_block: Maximum lines per block.
            text_snippet_length: Maximum characters for snippets.
            show_progress: Whether to show progress indicator.

        Returns:
            Nested list of TOCFeature objects.
        """
        features: list[list] = []
        page_count = doc.page_count if max_pages is None else min(max_pages, doc.page_count)

        # Buffer for remaining-block spans: list of (page_number, page_height, insert_index, spans)
        remaining_spans_buffer: list[tuple[int, float, int, list[dict]]] = []

        def _process_page(page, page_number: int) -> None:
            page_dict = page.get_text("dict")
            page_height = page_dict.get("height", page.rect.height) if hasattr(page, "rect") else _DEFAULT_PAGE_HEIGHT

            for block_ix, block in enumerate(page_dict["blocks"]):
                if block_ix >= max_blocks_per_page:
                    # Buffer first span of remaining blocks for heading candidate scanning
                    if block["type"] == _PYMUPDF_TEXT_BLOCK_TYPE:
                        lines = block.get("lines", [])
                        if lines:
                            spans = lines[0].get("spans", [])
                            if spans:
                                remaining_spans_buffer.append((page_number, page_height, len(features), spans))
                    continue

                features.append([])
                if block["type"] == _PYMUPDF_TEXT_BLOCK_TYPE:
                    # text block
                    for line_ix, line in enumerate(block["lines"]):
                        if line_ix >= max_lines_per_block:
                            break

                        features[-1].append([])

                        for span in line["spans"]:
                            features[-1][-1].append(
                                _span_to_feature(span, page_number, page_height, text_snippet_length)
                            )

        if show_progress:
            console.print(f"Extracting features from {page_count} page(s)...", style="cyan")
            with create_progress(console) as progress:
                task = progress.add_task("Extracting features...", total=page_count)

                for ix, page in enumerate(doc):
                    if max_pages is not None and ix >= max_pages:
                        break
                    _process_page(page, ix + 1)
                    progress.advance(task)
        else:
            for ix, page in enumerate(doc):
                if max_pages is not None and ix >= max_pages:
                    break
                _process_page(page, ix + 1)

        # Phase 2: Compute body font profile and scan remaining blocks for heading candidates
        # Uses stricter criteria than Phase 1 to limit false positives from body text,
        # but relaxes criteria for spans near the bottom of pages to catch section headings
        # that start late on a page (which otherwise get misattributed to running headers
        # on the next page).
        if remaining_spans_buffer:
            body_font_name, body_font_size = _compute_body_font_profile(features)

            # Group heading candidates by their insert_index for in-order insertion
            # Process in reverse order so insert indices remain valid
            candidates_by_index: dict[int, list[list]] = {}
            # Track per-page candidate count to enforce cap
            page_candidate_count: dict[int, int] = {}

            for page_number, page_height, insert_idx, spans in remaining_spans_buffer:
                if page_candidate_count.get(page_number, 0) >= _MAX_HEADING_CANDIDATES_PER_PAGE:
                    continue
                span = _merge_line_spans(spans, text_snippet_length)
                if not span:
                    continue
                bbox = span.get("bbox", None)
                y_pos = round(bbox[1] / page_height, _Y_POSITION_DECIMAL_PLACES) if bbox and page_height > 0 else None

                # Use relaxed criteria (Phase 1 ratio) for bottom-of-page spans
                # to catch section headings that start late on a page
                if y_pos is not None and y_pos > _BOTTOM_OF_PAGE_Y_THRESHOLD:
                    ratio = _HEADING_FONT_SIZE_RATIO
                else:
                    ratio = _HEADING_FONT_SIZE_RATIO_PHASE2

                if _is_heading_candidate(span, body_font_name, body_font_size, font_size_ratio=ratio):
                    feature = _span_to_feature(span, page_number, page_height, text_snippet_length)
                    if insert_idx not in candidates_by_index:
                        candidates_by_index[insert_idx] = []
                    candidates_by_index[insert_idx].append([[feature]])
                    page_candidate_count[page_number] = page_candidate_count.get(page_number, 0) + 1

            # Insert heading candidate blocks at the right positions (reverse order to preserve indices)
            for idx in sorted(candidates_by_index.keys(), reverse=True):
                for block in reversed(candidates_by_index[idx]):
                    features.insert(idx, block)

        return features

    def _extract_features_parallel(
        self,
        input_path: str,
        page_count: int,
        num_processes: int,
        max_blocks_per_page: int = DEFAULT_MAX_BLOCKS_PER_PAGE,
        max_lines_per_block: int = DEFAULT_MAX_LINES_PER_BLOCK,
        text_snippet_length: int = DEFAULT_TEXT_SNIPPET_LENGTH,
        show_progress: bool = True,
    ) -> list:
        """Extract features in parallel using multiprocessing.

        Args:
            input_path: Path to the PDF file.
            page_count: Total number of pages to process.
            num_processes: Number of parallel processes.
            max_blocks_per_page: Maximum blocks per page.
            max_lines_per_block: Maximum lines per block.
            text_snippet_length: Maximum characters for snippets.
            show_progress: Whether to show progress indicator.

        Returns:
            Merged list of features from all processes.
        """
        if show_progress:
            console.print(
                f"Extracting features from {page_count} page(s) (using {num_processes} parallel processes)...",
                style="cyan",
            )

        # Prepare arguments for worker processes
        args_list = [
            (i, num_processes, input_path, max_blocks_per_page, max_lines_per_block, text_snippet_length, page_count)
            for i in range(num_processes)
        ]

        # Process in parallel
        with Pool(processes=num_processes) as pool:
            results = []
            if show_progress:
                with create_progress(console) as progress:
                    task = progress.add_task("Extracting features...", total=num_processes)

                    for result in pool.imap_unordered(_extract_features_from_page_range, args_list):
                        results.append(result)
                        progress.advance(task)
            else:
                results = pool.map(_extract_features_from_page_range, args_list)

        # Sort results by start page to maintain order
        results = sorted(results, key=lambda x: x[0])

        # Merge features and remaining spans buffers from all processes
        if show_progress:
            console.print("Merging extracted features...", style="cyan")

        all_features: list = []
        all_remaining_spans: list[tuple[int, float, int, list[dict]]] = []
        feature_offset = 0

        for _, _, features, remaining_spans in results:
            all_features.extend(features)
            # Adjust insert indices by the cumulative feature offset
            for page_number, page_height, insert_idx, spans in remaining_spans:
                all_remaining_spans.append((page_number, page_height, insert_idx + feature_offset, spans))
            feature_offset += len(features)

        # Phase 2: Compute body font profile and scan for heading candidates
        # Uses stricter criteria than Phase 1 to limit false positives,
        # but relaxes criteria for bottom-of-page spans to catch late-starting sections.
        if all_remaining_spans:
            body_font_name, body_font_size = _compute_body_font_profile(all_features)

            candidates_by_index: dict[int, list[list]] = {}
            page_candidate_count: dict[int, int] = {}

            for page_number, page_height, insert_idx, spans in all_remaining_spans:
                if page_candidate_count.get(page_number, 0) >= _MAX_HEADING_CANDIDATES_PER_PAGE:
                    continue
                span = _merge_line_spans(spans, text_snippet_length)
                if not span:
                    continue
                bbox = span.get("bbox", None)
                y_pos = round(bbox[1] / page_height, _Y_POSITION_DECIMAL_PLACES) if bbox and page_height > 0 else None

                # Use relaxed criteria (Phase 1 ratio) for bottom-of-page spans
                if y_pos is not None and y_pos > _BOTTOM_OF_PAGE_Y_THRESHOLD:
                    ratio = _HEADING_FONT_SIZE_RATIO
                else:
                    ratio = _HEADING_FONT_SIZE_RATIO_PHASE2

                if _is_heading_candidate(span, body_font_name, body_font_size, font_size_ratio=ratio):
                    feature = _span_to_feature(span, page_number, page_height, text_snippet_length)
                    if insert_idx not in candidates_by_index:
                        candidates_by_index[insert_idx] = []
                    candidates_by_index[insert_idx].append([[feature]])
                    page_candidate_count[page_number] = page_candidate_count.get(page_number, 0) + 1

            for idx in sorted(candidates_by_index.keys(), reverse=True):
                for block in reversed(candidates_by_index[idx]):
                    all_features.insert(idx, block)

        return all_features

    def _extract_toc(
        self,
        features: list,
        max_depth: int = DEFAULT_MAX_TOC_DEPTH,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> tuple[TOC, TokenUsage]:
        """Infer TOC entries from extracted features using the LLM.

        This method handles pagination automatically when features exceed the
        token limit, splitting them into batches and merging results.

        Args:
            features: Nested list of TOCFeature objects extracted from the document.
            max_depth: Maximum depth level for TOC entries.
            max_tokens_per_batch: Maximum tokens per LLM call (for pagination).
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.

        Returns:
            A tuple of (TOC, TokenUsage) with the generated TOC and usage statistics.
        """
        return self._extract_toc_paginated(
            features,
            max_depth=max_depth,
            max_tokens_per_batch=max_tokens_per_batch,
            request_delay=request_delay,
        )

    def _batch_features(
        self,
        features: list,
        max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        overlap_blocks: int = DEFAULT_OVERLAP_BLOCKS,
    ) -> Iterator[list]:
        """Split features into batches that fit within token limits.

        Args:
            features: Nested list of features (blocks containing lines containing spans).
            max_tokens: Maximum estimated tokens per batch (for features only).
            overlap_blocks: Number of blocks to overlap between batches for context.

        Yields:
            Batches of features, each estimated to be under max_tokens.
        """
        if not features:
            yield []
            return

        # Account for prompt overhead in the effective limit
        effective_max_tokens = max_tokens - PROMPT_OVERHEAD_TOKENS

        current_batch: list = []
        current_tokens = 0

        for block in features:
            block_tokens = _estimate_block_tokens(block)

            # If adding this block would exceed limit and we have content, yield current batch
            if current_tokens + block_tokens > effective_max_tokens and current_batch:
                yield current_batch

                # Start new batch with overlap from end of previous batch
                overlap_start = max(0, len(current_batch) - overlap_blocks)
                current_batch = current_batch[overlap_start:]
                current_tokens = estimate_tokens(serialize_features_compact(current_batch))

            current_batch.append(block)
            current_tokens += block_tokens

        # Yield final batch if non-empty
        if current_batch:
            yield current_batch

    def _extract_toc_paginated(
        self,
        features: list,
        max_depth: int = DEFAULT_MAX_TOC_DEPTH,
        max_tokens_per_batch: int = DEFAULT_MAX_TOKENS_PER_BATCH,
        overlap_blocks: int = DEFAULT_OVERLAP_BLOCKS,
        request_delay: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> tuple[TOC, TokenUsage]:
        """Extract TOC using pagination for large documents.

        Splits features into batches, makes separate LLM calls for each,
        and merges the results.

        Args:
            features: Nested list of TOCFeature objects.
            max_depth: Maximum TOC depth level.
            max_tokens_per_batch: Maximum tokens per LLM call.
            overlap_blocks: Number of blocks to overlap between batches.
            request_delay: Delay in seconds between LLM calls to avoid rate limiting.

        Returns:
            A tuple of (merged TOC, TokenUsage statistics).
        """
        usage = TokenUsage()
        merged_toc = TOC(entries=[])

        batches = list(self._batch_features(features, max_tokens_per_batch, overlap_blocks))
        total_batches = len(batches)

        console.print(f"Processing {total_batches} batch(es) of features...", style="bold")

        for batch_idx, batch in enumerate(batches):
            is_first_batch = batch_idx == 0
            batch_description = f"batch {batch_idx + 1}/{total_batches}"

            # Add delay between requests (except for the first one)
            if not is_first_batch and request_delay > 0:
                console.print(f"  Waiting {request_delay:.1f}s before next request...", style="dim")
                time.sleep(request_delay)

            # Select appropriate prompt
            system_prompt = TOC_GENERATOR_SYSTEM_PROMPT if is_first_batch else TOC_GENERATOR_CONTINUATION_SYSTEM_PROMPT

            # Build user message with batch context
            batch_context = ""
            if not is_first_batch:
                batch_context = f"\n\nThis is batch {batch_idx + 1} of {total_batches}."

            features_text = serialize_features_compact(batch)

            user_content = f"""
                Generate a table of contents based on the document features given below.
                Limit the TOC to a maximum depth of {max_depth} levels.{batch_context}
                \n\n
                ------------------------
                {features_text}
            """

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]

            # Estimate input tokens
            input_text = system_prompt + user_content
            input_tokens = estimate_tokens(input_text)

            # Make LLM call with retry logic
            batch_toc = invoke_structured_with_retry(
                console,
                self._structured_model(TOC),
                messages,
                TOC,
                usage=usage,
                description=batch_description,
                input_tokens=input_tokens,
            )

            # Log entries found
            entries_found = len(batch_toc.entries)
            if entries_found > 0:
                console.print(f"  Found {entries_found} TOC entries in this batch", style="cyan")

            # Merge results
            merged_toc = merged_toc.merge(batch_toc)

        console.print(f"All batches processed. Total TOC entries: {len(merged_toc.entries)}", style="bold green")

        return merged_toc, usage

    def _extract_reference_toc_text(
        self,
        max_pages: int = DEFAULT_REFERENCE_TOC_MAX_PAGES,
        unfiltered_pages: int = DEFAULT_REFERENCE_TOC_UNFILTERED_PAGES,
    ) -> str:
        """Extract text from the first few pages that may contain a printed TOC.

        Scans the first N pages of the document. The first `unfiltered_pages` are
        included in full (title/copyright/TOC pages are typically here). Remaining
        pages are filtered to keep only TOC-like lines, reducing token usage.

        Args:
            max_pages: Maximum number of pages to scan from the beginning.
            unfiltered_pages: Number of initial pages to include without filtering.

        Returns:
            Concatenated text from the first pages, with page markers.
        """
        pages_to_scan = min(max_pages, self.doc.page_count)
        reference_texts = []

        for page_idx in range(pages_to_scan):
            page = self.doc[page_idx]
            page_text = page.get_text("text")
            if not page_text.strip():
                continue

            if page_idx < unfiltered_pages:
                reference_texts.append(f"--- Page {page_idx + 1} ---\n{page_text}")
            else:
                filtered = _extract_toc_like_lines(page_text)
                if filtered.strip():
                    reference_texts.append(f"--- Page {page_idx + 1} ---\n{filtered}")

        return "\n\n".join(reference_texts)

    def _postprocess_toc(
        self,
        toc: TOC,
        features: list,
        max_pages_for_reference_toc: int = DEFAULT_REFERENCE_TOC_MAX_PAGES,
    ) -> tuple[TOC, TokenUsage]:
        """Postprocess a generated TOC using LLM to clean up and improve entries.

        This method takes a previously generated TOC and refines it by:
        - Removing duplicate entries
        - Fixing typos in titles
        - Adjusting page numbers based on any printed TOC found in the document
        - Adding missing entries discovered from a printed TOC
        - Removing false positives

        Args:
            toc: The previously generated TOC to postprocess.
            features: The document features used for the original extraction.
            max_pages_for_reference_toc: Maximum pages to scan for a printed TOC.

        Returns:
            A tuple of (refined TOC, TokenUsage) with the improved TOC.
        """
        usage = TokenUsage()

        # Extract reference text from first pages (may contain printed TOC)
        reference_text = self._extract_reference_toc_text(max_pages=max_pages_for_reference_toc)

        # Format the generated TOC for the prompt
        toc_entries_str = "\n".join(
            f"- {entry.title} (page {entry.page_number}, level {entry.level}, "
            f"confidence {entry.confidence:.{_CONFIDENCE_DECIMAL_PLACES}f})"
            for entry in toc.entries
        )

        # Build a compact representation of features for context
        # Only include a summary to keep token count reasonable
        features_summary = self._summarize_features_for_postprocessing(features)

        # Detect front matter offset to warn the LLM
        offset_note = self._detect_page_offset_note(toc, reference_text)

        user_content = f"""
Please review and refine the following automatically generated Table of Contents.

## Generated TOC (to be refined)

The page numbers below are **PDF page numbers** (physical position in the PDF file).
For existing entries, prefer keeping these page numbers unchanged.

{toc_entries_str if toc_entries_str else "(No entries were generated)"}

## Reference Text from First Pages (may contain printed TOC)

{offset_note}

{reference_text if reference_text else "(No text extracted from first pages)"}

## Document Features Summary

{features_summary}

Please return a cleaned and improved TOC. It is very important that you add any missing \
sections or chapters visible in the printed TOC above. Every `page_number` in your output \
must be a PDF page number (not a printed page number); `printed_page` holds the printed TOC's \
number exactly as printed.
"""

        messages = [
            SystemMessage(content=TOC_POSTPROCESSOR_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        # Estimate input tokens
        input_text = TOC_POSTPROCESSOR_SYSTEM_PROMPT + user_content
        input_tokens = estimate_tokens(input_text)

        # Make LLM call with retry logic
        console.print("Postprocessing TOC...", style="bold")
        postprocessed_toc = invoke_structured_with_retry(
            console,
            self._structured_model(PostprocessedTOC),
            messages,
            PostprocessedTOC,
            usage=usage,
            description=_POSTPROCESS_DESCRIPTION,
            input_tokens=input_tokens,
        )

        # Place the postprocessed entries on PDF pages and verify them.
        refined_toc = self._correct_postprocessed_page_numbers(toc, postprocessed_toc, reference_text=reference_text)

        console.print(
            f"Postprocessing complete. Refined TOC has {len(refined_toc.entries)} entries (was {len(toc.entries)})",
            style="bold green",
        )

        return refined_toc, usage

    def _extract_reference_toc_page_entries(self, reference_text: str) -> list[tuple[str, int]]:
        """Extract printed TOC title/page pairs from reference text.

        This intentionally only accepts lines with an explicit trailing Arabic
        page number. More ambiguous OCR layouts are left to the LLM/postprocess
        prompt rather than guessed deterministically.
        """
        entries: list[tuple[str, int]] = []
        for line in reference_text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > _REFERENCE_TOC_OFFSET_MAX_TITLE_LENGTH:
                continue

            match = _TOC_LINE_TRAILING_NUMBER_PATTERN.search(stripped)
            if not match:
                continue

            page_match = re.search(r"(\d+)\s*$", stripped)
            if not page_match:
                continue

            title = stripped[: page_match.start()].strip(" .·…\t")
            if len(_normalize_snippet(title)) < _FUZZY_MIN_SUBSTRING_LEN:
                continue
            entries.append((title, int(page_match.group(1))))

        return entries

    def _titles_match_for_offset_detection(self, left: str, right: str) -> bool:
        """Return True when two TOC titles are a strong enough offset signal."""
        left_norm = _normalize_snippet(left)
        right_norm = _normalize_snippet(right)
        if len(left_norm) < _FUZZY_MIN_SUBSTRING_LEN or len(right_norm) < _FUZZY_MIN_SUBSTRING_LEN:
            return False
        if left_norm == right_norm:
            return True

        shorter_len = min(len(left_norm), len(right_norm))
        longer_len = max(len(left_norm), len(right_norm))
        if shorter_len < _FUZZY_MIN_SUBSTRING_LEN:
            return False
        if not (left_norm in right_norm or right_norm in left_norm):
            return False
        return shorter_len / longer_len >= _FUZZY_MIN_COVERAGE_RATIO

    def _detect_reference_toc_offset(self, toc: TOC, reference_text: str) -> int:
        """Detect printed-to-PDF offset from repeated printed TOC title/page matches."""
        reference_entries = self._extract_reference_toc_page_entries(reference_text)
        if not reference_entries or not toc.entries:
            return 0

        offsets: list[int] = []
        for reference_title, printed_page in reference_entries:
            for entry in toc.entries:
                if self._titles_match_for_offset_detection(reference_title, entry.title):
                    offset = entry.page_number - printed_page
                    if offset >= _SIGNIFICANT_FRONT_MATTER_OFFSET:
                        offsets.append(offset)
                    break

        if not offsets:
            return 0

        offset_counts = Counter(offsets)
        best_offset, best_count = offset_counts.most_common(1)[0]
        if best_count < _REFERENCE_TOC_OFFSET_MIN_MATCHES:
            return 0
        if best_count / len(offsets) < _REFERENCE_TOC_OFFSET_MIN_DOMINANCE_RATIO:
            return 0
        return best_offset

    def _detect_front_matter_offset(self, toc: TOC, reference_text: str = "") -> int:
        """Detect the number of front matter pages from Phase 1 TOC data.

        First try to infer a high-confidence offset from repeated matches
        between printed TOC lines and generated entries. If that signal is not
        available, use generated level-1 entries and front-matter anchors.

        Prefer the first numbered level-1 main-content entry (e.g. "1. ...",
        "Chapter I ...") because book title pages can look like chapter
        headings. Fall back to the first non-front-matter level-1 entry past
        the front-matter threshold.

        Args:
            toc: The generated TOC with correct PDF page numbers.

        Returns:
            Estimated front matter offset (0 if no significant front matter detected).
        """
        if reference_text:
            reference_offset = self._detect_reference_toc_offset(toc, reference_text)
            if reference_offset:
                return reference_offset

        candidates = [
            entry
            for entry in toc.entries
            if entry.level == _ROOT_TOC_LEVEL and entry.page_number > _FRONT_MATTER_FIRST_MAIN_ENTRY_MIN_PAGE
        ]

        front_matter_anchor_page = 0
        for entry in candidates:
            title_clean = _title_match_key(entry.title)
            if _FRONT_MATTER_TITLE_PATTERN.match(title_clean):
                front_matter_anchor_page = max(front_matter_anchor_page, entry.page_number)

        candidates_after_front_matter = [
            entry
            for entry in candidates
            if not front_matter_anchor_page or entry.page_number > front_matter_anchor_page
        ]

        for entry in candidates_after_front_matter:
            if _MAIN_CONTENT_START_PATTERN.match(_normalize_title_for_matching(entry.title)):
                return entry.page_number - 1

        for entry in candidates_after_front_matter:
            title_clean = _title_match_key(entry.title)
            if not _FRONT_MATTER_TITLE_PATTERN.match(title_clean):
                return entry.page_number - 1
        return 0

    def _detect_page_offset_note(self, toc: TOC, reference_text: str) -> str:
        """Detect the offset between PDF page numbers and printed page numbers.

        Looks at the first chapter-level entry in the generated TOC to estimate
        the front matter offset, then generates an explicit warning for the LLM.

        Args:
            toc: The generated TOC with correct PDF page numbers.
            reference_text: The extracted reference text from first pages.

        Returns:
            A warning string to include in the postprocessor prompt, or empty string
            if no offset is detected.
        """
        estimated_offset = self._detect_front_matter_offset(toc, reference_text=reference_text)

        if estimated_offset < _SIGNIFICANT_FRONT_MATTER_OFFSET:
            return ""

        first_chapter_page = estimated_offset + 1

        return (
            f"**Note**: This document has approximately {estimated_offset} pages of front matter. "
            f"The printed page numbers below (if any) start counting AFTER the front matter, "
            f'so printed page "1" corresponds to approximately PDF page {first_chapter_page}. '
            f"For existing entries, keep their PDF page numbers unchanged. "
            f"For new entries from the printed TOC, convert to PDF page numbers by adding {estimated_offset}: "
            f"printed page N \u2192 PDF page N + {estimated_offset}."
        )

    def _page_contains_heading(
        self, page_number: int, title: str, window: int = _DEFAULT_HEADING_SEARCH_WINDOW
    ) -> bool:
        """Check if a heading's text appears on or near the given page.

        Args:
            page_number: 1-indexed PDF page number to check.
            title: The heading title to search for.
            window: Number of pages before/after to also check.

        Returns:
            True if the title text is found on or near the page.
        """
        search_text = _normalize_snippet(title)
        if len(search_text) < _HEADING_MIN_LENGTH:
            return False  # Too short to match reliably

        start = max(0, page_number - 1 - window)
        end = min(self.doc.page_count, page_number + window)

        for page_idx in range(start, end):
            page_text = self.doc[page_idx].get_text("text")
            if _heading_text_matches_title(title, page_text):
                return True

        return False

    def _find_heading_page(
        self, title: str, start_page: int, end_page: int, hint_page: int | None = None
    ) -> int | None:
        """Find the page where a heading appears as an actual heading, not a running header.

        Searches pages in [start_page, end_page) for the title text in a
        non-running-header position. Uses block-level text
        concatenation to handle OCR text where each word is a separate span.

        When hint_page is provided, searches outward from that page so the
        closest match wins (avoids false matches in distant front matter).

        Args:
            title: The heading title to search for.
            start_page: First page to search (1-indexed, inclusive).
            end_page: Last page to search (1-indexed, exclusive).
            hint_page: Page to start searching from (1-indexed). If None,
                searches sequentially from start_page.

        Returns:
            The 1-indexed page number where the heading is found, or None.
        """
        search_text = _normalize_snippet(title)
        if len(search_text) < _HEADING_MIN_LENGTH:
            return None

        start_idx = max(0, start_page - 1)
        end_idx = min(self.doc.page_count, end_page - 1)

        # Build page order: nearest to hint_page first (later page wins ties), else
        # sequential. Ordering the range itself, rather than stepping outward from
        # the hint, still covers the whole range when the hint lies outside it.
        page_order = list(range(start_idx, end_idx))
        if hint_page is not None:
            center = hint_page - 1  # 0-indexed
            page_order.sort(key=lambda idx: (abs(idx - center), idx < center))

        for page_idx in page_order:
            page = self.doc[page_idx]
            page_dict = page.get_text("dict")
            page_height = page_dict.get("height", _DEFAULT_PAGE_HEIGHT)

            for block in page_dict.get("blocks", []):
                if block.get("type") != _PYMUPDF_TEXT_BLOCK_TYPE:
                    continue
                block_spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
                if not block_spans:
                    continue
                first_bbox = block_spans[0].get("bbox")
                if not first_bbox or page_height <= 0:
                    continue
                y_pos = first_bbox[1] / page_height
                # Skip running headers and footers.
                if y_pos < _HEADING_SEARCH_HEADER_Y_THRESHOLD or y_pos > _HEADING_SEARCH_FOOTER_Y_THRESHOLD:
                    continue
                block_text = " ".join(s.get("text", "") for s in block_spans)
                # Anchored: only structural heading evidence may relocate a page.
                if _heading_text_matches_title(title, block_text, anchored=True):
                    return page_idx + 1  # 1-indexed

        return None

    def _page_has_heading_in_bottom(
        self,
        page_number: int,
        title: str,
        y_threshold: float = _PREVIOUS_PAGE_BOTTOM_HEADING_Y_THRESHOLD,
    ) -> bool:
        """Check if heading text appears in the bottom portion of a given page.

        Searches the actual PDF page structure (blocks/spans) for text matching
        the title at a vertical position below y_threshold.

        Args:
            page_number: 1-indexed PDF page number.
            title: Heading title to search for.
            y_threshold: Normalized y-position threshold (0.0=top, 1.0=bottom).
                Only spans with y > y_threshold are searched.

        Returns:
            True if the title is found in the bottom portion of the page.
        """
        page_idx = page_number - 1
        if page_idx < 0 or page_idx >= self.doc.page_count:
            return False

        page = self.doc[page_idx]
        page_dict = page.get_text("dict")
        page_height = page_dict.get("height", _DEFAULT_PAGE_HEIGHT)

        search_text = _normalize_snippet(title)
        if len(search_text) < _HEADING_MIN_LENGTH:
            return False

        for block in page_dict.get("blocks", []):
            if block.get("type") != _PYMUPDF_TEXT_BLOCK_TYPE:
                continue
            # Concatenate all text in the block and check y-position from the
            # first span. This handles OCR text where headings are split across
            # many single-word spans within one block.
            block_spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
            if not block_spans:
                continue
            first_bbox = block_spans[0].get("bbox")
            if not first_bbox or page_height <= 0:
                continue
            y_pos = first_bbox[1] / page_height
            if y_pos <= y_threshold:
                continue
            block_text = " ".join(s.get("text", "") for s in block_spans)
            if _heading_text_matches_title(title, block_text):
                return True
        return False

    def _correct_running_header_pages(self, toc: TOC, features: list) -> TOC:
        """Correct TOC entries that point to running headers instead of actual headings.

        For each entry, checks if the title text appears only at running header
        position (y < 0.05) in the extracted features. If so, searches page N-1
        for the actual heading in the bottom portion and corrects the page number.

        This is a deterministic correction that requires no additional LLM calls.

        Args:
            toc: The generated TOC to correct.
            features: The extracted features used for TOC generation.

        Returns:
            A corrected TOC with running header misattributions fixed.
        """
        if not toc.entries:
            return toc

        # Build page-zone text index: group feature text by page and y-zone.
        # In OCR text each word is a separate span, so matching individual spans
        # against multi-word titles fails. Instead, concatenate all feature text
        # per page in two zones: "header" (y < RUNNING_HEADER_Y_THRESHOLD) and
        # "body" (y >= RUNNING_HEADER_Y_THRESHOLD), then match against the
        # concatenated text.
        page_header_text: dict[int, str] = {}  # page -> concatenated header zone text
        page_body_text: dict[int, str] = {}  # page -> concatenated body zone text
        for block in features:
            for line in block:
                for span in line:
                    if isinstance(span, TOCFeature) and span.y_position is not None:
                        norm = _normalize_snippet(span.text_snippet)
                        if not norm:
                            continue
                        if span.y_position < _RUNNING_HEADER_Y_THRESHOLD:
                            page_header_text.setdefault(span.page_number, "")
                            page_header_text[span.page_number] += " " + norm
                        else:
                            page_body_text.setdefault(span.page_number, "")
                            page_body_text[span.page_number] += " " + norm

        corrected = []
        corrections = 0

        for entry in toc.entries:
            page = entry.page_number
            entry_text = _normalize_snippet(entry.title)

            if len(entry_text) < _HEADING_MIN_LENGTH or page <= 1:
                corrected.append(entry)
                continue

            # Check if the title appears in the header zone of this page
            header_text = page_header_text.get(page, "")
            in_header = bool(header_text.strip()) and (entry_text in header_text or header_text.strip() in entry_text)

            # Check if the title also appears in the body zone of this page
            body_text = page_body_text.get(page, "")
            in_body = bool(body_text) and (entry_text in body_text or body_text.strip() in entry_text)

            # Only flag as suspicious if title IS in header zone but NOT in body zone
            is_running_header_only = in_header and not in_body

            if not is_running_header_only:
                corrected.append(entry)
                continue

            # Suspicious entry: check if heading exists on previous page
            # First try: check body features on page N-1
            prev_page = page - 1
            prev_body = page_body_text.get(prev_page, "")
            found_on_prev = bool(prev_body) and (entry_text in prev_body or prev_body.strip() in entry_text)

            # Fallback: search the actual PDF for heading in bottom of previous page
            if not found_on_prev:
                found_on_prev = self._page_has_heading_in_bottom(prev_page, entry.title)

            if found_on_prev:
                corrected.append(entry.model_copy(update={"page_number": prev_page}))
                corrections += 1
            else:
                corrected.append(entry)

        if corrections > 0:
            console.print(
                f"  Corrected {corrections} entries from running headers to actual section starts", style="yellow"
            )

        return TOC(entries=corrected)

    def _correct_postprocessed_page_numbers(
        self, original_toc: TOC, refined_toc: PostprocessedTOC, reference_text: str = ""
    ) -> TOC:
        """Correct page numbers for entries added or modified by the postprocessor.

        Uses a deterministic strategy:
        1. Detect the front matter offset from Phase 1 data.
        2. Index the original entries by normalized title.
        3. Place each refined entry: matched entries (in both original and refined)
           get their original PDF page back; other entries citing a printed-TOC page
           (``printed_page``) are placed through the printed page numbers found on
           the pages; the rest keep the postprocessor's page.
        4. With significant front matter, verify every entry by searching the PDF
           text for its heading within ±offset pages.

        Args:
            original_toc: The pre-postprocessing TOC with correct PDF page numbers.
            refined_toc: The postprocessed TOC that may have shifted page numbers.
            reference_text: Text of the document's first pages (printed TOC), used to
                detect the front matter offset.

        Returns:
            The corrected TOC with PDF page numbers restored.
        """
        if not refined_toc.entries:
            return TOC(entries=[])

        # Step 1: Detect front matter offset from Phase 1 data
        offset = self._detect_front_matter_offset(original_toc, reference_text=reference_text)

        # Step 2: Index the original entries for one-to-one matching.
        originals = _OriginalEntryMatcher(original_toc)

        # Step 3: Place each refined entry, by the strongest evidence available:
        #         1. an exact first-pass match (the first pass saw the heading there);
        #         2. the printed-TOC page it cites, located through the printed page
        #            numbers on the pages themselves (correct where a single
        #            front-matter offset drifts);
        #         3. a fuzzy (prefix) first-pass match;
        #         4. the postprocessor's own estimate (verified below).
        #         An exact-title repeat left over once every same-title original was
        #         claimed, with nothing else locating it, is dropped as a duplicate.
        #         Exact matches are assigned globally, nearest first, with the printed
        #         page (when locatable) as each entry's position.
        entries = refined_toc.entries
        printed_pdf_pages = [
            None if entry.printed_page is None else self._printed_page_map.pdf_page(entry.printed_page)
            for entry in entries
        ]
        positions = [
            entry.page_number if printed is None else printed
            for entry, printed in zip(entries, printed_pdf_pages, strict=True)
        ]
        exact_pages = originals.claim_exact(
            [
                (entry.title, entry.level, position, printed is not None)
                for entry, position, printed in zip(entries, positions, printed_pdf_pages, strict=True)
            ]
        )

        placed: list[TOCEntry] = []
        restored_count = 0
        mapped_count = 0
        for index, (entry, printed_pdf_page, position) in enumerate(
            zip(entries, printed_pdf_pages, positions, strict=True)
        ):
            if index in exact_pages:
                page = exact_pages[index]
                restored_count += entry.page_number != page
            elif printed_pdf_page is not None:
                page = printed_pdf_page
                mapped_count += entry.page_number != page
            elif originals.has_title(entry.title, entry.level):
                # Every same-title original was claimed by another refined entry and
                # nothing else locates this one: the postprocessor repeated an entry.
                continue
            elif (fuzzy_page := originals.claim_fuzzy(entry.title, entry.level, position)) is not None:
                page = fuzzy_page
                restored_count += entry.page_number != page
            else:
                page = entry.page_number
            placed.append(entry.to_toc_entry(page_number=page))

        if any(entry.printed_page is not None for entry in entries) and not self._printed_page_map:
            console.print("  No printed page numbers detected on the pages; keeping page estimates", style="dim")

        # Step 4: Verify ALL entries using running-header-aware page search: a heading
        # found on a page outranks every page source. This catches matched entries whose
        # initial extraction was wrong, estimates, and printed pages the LLM misread (or
        # a printed TOC that is off by a page). Search ±offset around each entry's page.
        verified_count = 0
        corrected: list[TOCEntry] = []
        for placed_entry in placed:
            if offset >= _SIGNIFICANT_FRONT_MATTER_OFFSET:
                verified_page = self._verify_heading_page(placed_entry, window=offset)
                if verified_page != placed_entry.page_number:
                    placed_entry = placed_entry.model_copy(update={"page_number": verified_page})
                    verified_count += 1
            corrected.append(placed_entry)

        # Log corrections
        if restored_count > 0:
            console.print(f"  Restored page numbers for {restored_count} existing entries", style="yellow")
        if mapped_count > 0:
            console.print(f"  Placed {mapped_count} entries via the document's printed page numbers", style="yellow")
        if verified_count > 0:
            console.print(
                f"  Verified and corrected {verified_count} entries via heading search (±{offset} page window)",
                style="yellow",
            )

        return TOC(entries=corrected)

    def _verify_heading_page(self, entry: TOCEntry, window: int) -> int:
        """Return the page within ±window of the entry's page where its heading is found, or its page.

        Searches outward from the nearest in-document page: an entry claimed past the
        last page (printed page + offset overshooting a PDF that omits printed pages)
        is most likely among the final pages.
        """
        if len(_normalize_snippet(entry.title)) < _FUZZY_MIN_SUBSTRING_LEN:
            return entry.page_number
        center = min(max(entry.page_number, 1), self.doc.page_count)
        search_start = max(1, center - window)
        search_end = min(self.doc.page_count, center + window) + 1
        found_page = self._find_heading_page(entry.title, search_start, search_end, hint_page=center)
        return entry.page_number if found_page is None else found_page

    @cached_property
    def _printed_page_map(self) -> PrintedPageMap:
        """The document's printed-to-PDF page map, built on first use (one pass over all pages)."""
        return PrintedPageMap.from_document(self.doc)

    def _summarize_features_for_postprocessing(
        self, features: list, max_entries: int = DEFAULT_POSTPROCESS_FEATURE_SUMMARY_MAX_ENTRIES
    ) -> str:
        """Create a compact summary of features for postprocessing context.

        Prioritizes heading-like spans (bold, larger fonts, section-numbered) and
        samples evenly across the document. Uses compact font references to reduce
        token usage.

        Args:
            features: Nested list of TOCFeature objects.
            max_entries: Maximum number of feature entries to include.

        Returns:
            A string summary of the most relevant features.
        """
        # Flatten and separate heading-like vs body spans
        heading_spans: list[TOCFeature] = []
        body_spans: list[TOCFeature] = []

        body_font_name, body_font_size = _compute_body_font_profile(features)

        for block in features:
            for line in block:
                for span in line:
                    if isinstance(span, TOCFeature):
                        is_heading = (
                            span.is_bold
                            or (
                                body_font_size > 0
                                and span.font_size > body_font_size * _POSTPROCESS_HEADING_FONT_SIZE_RATIO
                            )
                            or bool(_SECTION_NUMBER_PATTERN.match(span.text_snippet.strip()))
                        )
                        if is_heading:
                            heading_spans.append(span)
                        else:
                            body_spans.append(span)

        if not heading_spans and not body_spans:
            return "(No features available)"

        # Allocate most entries to heading spans, rest to body for context
        heading_budget = min(len(heading_spans), int(max_entries * _POSTPROCESS_HEADING_SUMMARY_BUDGET_RATIO))
        body_budget = min(len(body_spans), max_entries - heading_budget)

        def _sample(spans: list[TOCFeature], n: int) -> list[TOCFeature]:
            if len(spans) <= n:
                return spans
            step = len(spans) / n
            return [spans[int(i * step)] for i in range(n)]

        sampled = _sample(heading_spans, heading_budget) + _sample(body_spans, body_budget)
        # Sort by page number for readability
        sampled.sort(key=lambda s: (s.page_number, s.y_position or 0))

        # Build compact summary using stripped font names
        summary_lines = []
        for span in sampled:
            font = _strip_subset_prefix(span.font_name)
            size = (
                int(span.font_size)
                if span.font_size == int(span.font_size)
                else round(span.font_size, _FONT_SIZE_DECIMAL_PLACES)
            )
            parts = [f"P{span.page_number}"]
            if span.y_position is not None:
                y_str = f"{span.y_position:.{_Y_POSITION_DECIMAL_PLACES}f}".lstrip("0") or "0"
                parts[0] += f"@{y_str}"
            parts.append(f"{font} {size}")
            if span.is_bold:
                parts.append("B")
            parts.append(f'"{span.text_snippet}"')
            summary_lines.append(" ".join(parts))

        return "\n".join(summary_lines)
