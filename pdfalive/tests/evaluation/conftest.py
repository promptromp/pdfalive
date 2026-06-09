"""Shared fixtures for evaluation tests."""

from collections.abc import Callable

import pytest

from pdfalive.evaluation.metrics import GoldenEntry
from pdfalive.models.toc import TOC, TOCEntry


DEFAULT_TITLE = "Chapter 1: Distance and Angles"
DEFAULT_PAGE_NUMBER = 14
DEFAULT_LEVEL = 1
DEFAULT_CONFIDENCE = 0.9


@pytest.fixture
def make_golden_entry() -> Callable[..., GoldenEntry]:
    """Factory fixture for GoldenEntry objects with sensible defaults."""

    def _make(
        title: str = DEFAULT_TITLE,
        page_number: int = DEFAULT_PAGE_NUMBER,
        level: int = DEFAULT_LEVEL,
    ) -> GoldenEntry:
        return GoldenEntry(title=title, page_number=page_number, level=level)

    return _make


@pytest.fixture
def make_toc_entry() -> Callable[..., TOCEntry]:
    """Factory fixture for TOCEntry objects with sensible defaults."""

    def _make(
        title: str = DEFAULT_TITLE,
        page_number: int = DEFAULT_PAGE_NUMBER,
        level: int = DEFAULT_LEVEL,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> TOCEntry:
        return TOCEntry(title=title, page_number=page_number, level=level, confidence=confidence)

    return _make


@pytest.fixture
def make_toc(make_toc_entry: Callable[..., TOCEntry]) -> Callable[..., TOC]:
    """Factory fixture for TOC objects built from (title, page_number, level) tuples."""

    def _make(*specs: tuple[str, int, int]) -> TOC:
        return TOC(entries=[make_toc_entry(title=t, page_number=p, level=lvl) for t, p, lvl in specs])

    return _make
