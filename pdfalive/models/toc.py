"""Table of Contents entry data model."""

from pydantic import BaseModel, Field


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
