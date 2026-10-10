"""Shared fixtures for the test suite."""

import os
from collections.abc import Callable
from typing import Any, cast

import pytest
from langsmith import utils as langsmith_utils

from pdfalive.evaluation.metrics import GoldenEntry
from pdfalive.models.toc import TOC, TOCEntry


# Environment variables langsmith reads to decide whether to trace, in lookup order.
_TRACING_ENV_VARS = ("LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")
_TRACING_DISABLED = "false"
# Legacy v1 tracing switches: with tracing otherwise off, LangChain refuses to run under them.
_LEGACY_TRACING_ENV_VARS = ("LANGCHAIN_TRACING", "LANGCHAIN_HANDLER")


def pytest_configure(config: pytest.Config) -> None:
    """Keep the test suite from sending LangSmith traces, whatever the developer's environment.

    The CLI commands are ``@traceable``: with ``LANGSMITH_TRACING`` set, every CLI test
    would otherwise post a run to the developer's LangSmith project and consume its
    trace quota. Disabling it through the environment (once, before any test) also
    covers worker threads, spawned processes and fixtures of every scope.
    """
    for name in _LEGACY_TRACING_ENV_VARS:
        os.environ.pop(name, None)
    for name in _TRACING_ENV_VARS:
        os.environ[name] = _TRACING_DISABLED
    # langsmith caches env lookups; its typing overloads hide the lru_cache wrapper's cache_clear.
    cast(Any, langsmith_utils.get_env_var).cache_clear()


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
