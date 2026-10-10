"""Table of Contents entry data model."""

import re

from pydantic import BaseModel, Field, field_validator


# A page number as printed in a table of contents, in ASCII digits.
_ARABIC_NUMBER = re.compile(r"[0-9]+")


class TOCEntry(BaseModel):
    """A single entry in a Table of Contents."""

    title: str = Field(description="Title of the TOC entry")
    page_number: int = Field(description="Page number of the TOC entry (1-indexed)")
    level: int = Field(description="Hierarchical level of the TOC entry")
    confidence: float = Field(description="Confidence score for the TOC entry (0.0 to 1.0)")

    def __str__(self) -> str:
        return f"TOCEntry(level={self.level}, title='{self.title}', page_number={self.page_number}, confidence={self.confidence})"  # noqa: E501

    def to_list(self) -> list:
        """Convert TOCEntry to list format compatible with PyMuPDF `set_toc()`."""
        return [self.level, self.title, self.page_number]

    @classmethod
    def from_list(cls, toc_list: list) -> "TOCEntry":
        """Create TOCEntry from list format returned by PyMuPDF `get_toc()`."""
        level, title, page_number = toc_list

        return cls(title=title, page_number=page_number, level=level, confidence=1.0)


class TOC(BaseModel):
    """Table of Contents model."""

    entries: list[TOCEntry]

    def to_list(self) -> list:
        """Convert TOC to list format compatible with PyMuPDF `set_toc()`."""
        return [entry.to_list() for entry in self.entries]

    def sanitize_hierarchy(self) -> "TOC":
        """Sanitize TOC hierarchy to satisfy PyMuPDF's set_toc() constraints.

        PyMuPDF requires:
        - The first entry must have level 1
        - Each entry's level can be at most 1 higher than the previous entry's level

        This method adjusts levels that violate these constraints while preserving
        the relative hierarchy as much as possible.

        Returns:
            A new TOC with valid hierarchy levels.
        """
        if not self.entries:
            return self

        sanitized: list[TOCEntry] = []
        for i, entry in enumerate(self.entries):
            if i == 0:
                # First entry must be level 1
                new_level = 1
            else:
                prev_level = sanitized[i - 1].level
                # Level can increase by at most 1 from previous
                new_level = min(entry.level, prev_level + 1)

            # Level must be at least 1
            new_level = max(1, new_level)

            if new_level != entry.level:
                sanitized.append(entry.model_copy(update={"level": new_level}))
            else:
                sanitized.append(entry)

        return TOC(entries=sanitized)

    def partition_by_page_range(self, page_count: int) -> tuple["TOC", list[TOCEntry]]:
        """Split entries by whether their page number exists in a document of ``page_count`` pages.

        Entries are 1-indexed, so a page outside 1..page_count has no real
        destination. PyMuPDF's set_toc() only validates such pages loosely: it
        raises "page number out of range" for rows outside -1..page_count, but
        never checks the final row (silently clamping it to the last page) and
        maps page 0 to page 1, so an out-of-range entry either crashes the
        write or yields a bookmark to unrelated content.

        Args:
            page_count: Number of pages in the document the TOC will be applied to.

        Returns:
            A tuple of (TOC with the in-range entries, list of out-of-range entries),
            both preserving the original entry order.
        """
        in_range: list[TOCEntry] = []
        out_of_range: list[TOCEntry] = []
        for entry in self.entries:
            (in_range if 1 <= entry.page_number <= page_count else out_of_range).append(entry)
        return TOC(entries=in_range), out_of_range

    def sort_by_page(self) -> "TOC":
        """Sort TOC entries by page number to enforce monotonicity.

        Entries are sorted by (page_number, level) so that page numbers are
        non-decreasing.  Ties on the same page are broken by level so that
        parent entries come before their children.

        Returns:
            A new TOC with entries sorted by page number.
        """
        if not self.entries:
            return self
        sorted_entries = sorted(self.entries, key=lambda e: (e.page_number, e.level))
        return TOC(entries=sorted_entries)

    def deduplicate(self) -> "TOC":
        """Remove duplicate entries with the same (page_number, title).

        Keeps the first occurrence of each duplicate. This is useful after
        postprocessing, where the LLM may add multiple identical entries
        (e.g., "Exercises" appearing once per chapter in a printed TOC but
        all mapping to the same page after page number correction).

        Returns:
            A new TOC with duplicates removed, preserving order.
        """
        seen: set[tuple[int, str]] = set()
        unique: list[TOCEntry] = []
        for entry in self.entries:
            key = (entry.page_number, entry.title)
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        return TOC(entries=unique)

    def merge(self, other: "TOC") -> "TOC":
        """Merge another TOC into this one, handling duplicates.

        When entries have the same page_number and title, the entry from `self`
        (the earlier batch) is preferred. Entries are sorted by page_number
        after merging.

        Args:
            other: Another TOC to merge with this one.

        Returns:
            A new TOC containing entries from both, with duplicates removed.
        """
        # Use (page_number, title) as key for deduplication
        # Prefer entries from self (earlier batch)
        seen: dict[tuple[int, str], TOCEntry] = {}

        for entry in self.entries:
            key = (entry.page_number, entry.title)
            if key not in seen:
                seen[key] = entry

        for entry in other.entries:
            key = (entry.page_number, entry.title)
            if key not in seen:
                seen[key] = entry

        # Sort by page_number, then by level (for stable ordering)
        merged_entries = sorted(seen.values(), key=lambda e: (e.page_number, e.level))

        return TOC(entries=merged_entries)


class PostprocessedTOCEntry(TOCEntry):
    """A TOC entry returned by the postprocessor, citing its printed-TOC page when listed there."""

    printed_page: int | None = Field(
        default=None,
        description=(
            "Page number exactly as printed for this entry in the document's own printed table of contents, "
            "if it is listed there with an Arabic page number; otherwise null"
        ),
    )

    @field_validator("printed_page", mode="before")
    @classmethod
    def _non_arabic_printed_page_is_absent(cls, value: object) -> object:
        """Read anything but a plain Arabic page number (e.g. "xii" from front matter) as absent.

        Providers without strict schema enforcement may return the printed TOC's page
        verbatim (roman numerals, other scripts' digits, fractions); such values carry no
        folio to map, and must not fail the whole response.
        """
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else None
        if isinstance(value, str) and _ARABIC_NUMBER.fullmatch(value.strip()):
            return int(value.strip())
        return None

    def to_toc_entry(self, page_number: int) -> TOCEntry:
        """Return the plain TOC entry for this entry, placed on ``page_number``."""
        return TOCEntry.model_validate(self.model_dump(exclude={"printed_page"}) | {"page_number": page_number})


class PostprocessedTOC(BaseModel):
    """Table of Contents returned by the postprocessor (its structured-output schema)."""

    entries: list[PostprocessedTOCEntry]


class TOCFeature(BaseModel):
    """Feature used for TOC generation.

    These features are extracted from the PDF document and used to identify potential TOC entries.

    """

    page_number: int = Field(description="Page number of the feature (1-indexed)")
    font_name: str = Field(description="Font name of the text span")
    font_size: float = Field(description="Font size of the text span")
    text_length: int = Field(description="Length of the text span")
    text_snippet: str = Field(description="Snippet of the text span (truncated)")
    y_position: float | None = Field(default=None, description="Normalized vertical position (0.0=top, 1.0=bottom)")
    is_bold: bool | None = Field(default=None, description="Whether the span uses a bold font")

    def __str__(self) -> str:
        # Nb. This format is used in the LLM prompt and so is kept compact. Prompt instructions include details.
        base = f"({self.page_number}, '{self.font_name}', {self.font_size}, {self.text_length}, '{self.text_snippet}'"
        extras = []
        if self.y_position is not None:
            extras.append(f"y={self.y_position}")
        if self.is_bold is not None:
            extras.append(f"bold={self.is_bold}")
        if extras:
            return base + ", " + ", ".join(extras) + ")"
        return base + ")"

    def __repr__(self) -> str:
        return self.__str__()
