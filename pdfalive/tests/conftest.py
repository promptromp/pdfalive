"""Shared fixtures for the test suite."""

import os
import time
from collections.abc import Callable
from typing import Any, cast

import pytest
from langsmith import utils as langsmith_utils
from rich.console import Console

from pdfalive.console import TerminalConsole
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

# File names Rich would parse as markup, each a valid single path component on
# every OS (no "/", "\" or ":"): a bracketed author name starting with a lowercase
# particle (Rich tags start with a lowercase letter, "#", "/" or "@", so a
# capitalized "[Author]" is safe), an explicit style tag, a link tag, and a
# color tag.
MARKUP_LIKE_FILENAMES = (
    "[van Author] Some Title.pdf",
    "[bold]x.pdf",
    "[link=x]y.pdf",
    "[#ff0000]x.pdf",
)

# Markup-like text for values that are not file names (titles, queries, model
# identifiers, reasoning, exception messages): the file names, plus a bare
# closing tag (raises MarkupError), a URL link tag, an emoji shortcode (":A:"),
# and trailing backslashes, which rich.markup.escape() mishandles.
MARKUP_LIKE_TEXTS = (
    *MARKUP_LIKE_FILENAMES,
    "[/]weird",
    "[link=http://x]y",
    "Maps f:A:B",
    "one trailing backslash \\",
    "two trailing backslashes \\\\",
    "three trailing backslashes \\\\\\",
)

# Single-line outside text carrying characters that would act on the terminal or
# the displayed line, each paired with how pdfalive shows it: conceal (ESC [8m),
# erase line (ESC [2K, and through the single-character C1 control sequence
# introducer), a newline and a tab, a right-to-left override that disguises an
# executable as a PDF, and an undecodable byte (surrogateescape).
CONTROL_SEQUENCE_TEXTS = (
    ("conceal\x1b[8m.pdf", "conceal\\x1b[8m.pdf"),
    ("erase line\x1b[2K.pdf", "erase line\\x1b[2K.pdf"),
    ("c1 erase line\x9b2K.pdf", "c1 erase line\\x9b2K.pdf"),
    ("line\nbreak.pdf", "line\\nbreak.pdf"),
    ("tab\there.pdf", "tab\\there.pdf"),
    ("invoice‮fdp.exe", "invoice\\u202efdp.exe"),
    ("undecodable \udce9.pdf", "undecodable \\xe9.pdf"),
)
CONTROL_SEQUENCE_IDS = ("conceal", "erase-line", "c1-erase-line", "newline", "tab", "rlo-spoof", "undecodable-byte")

# Wide enough that Rich never wraps a console line mid-assertion.
WIDE_CONSOLE_WIDTH = 1000


@pytest.fixture(params=MARKUP_LIKE_FILENAMES)
def markup_like_filename(request: pytest.FixtureRequest) -> str:
    """Each of MARKUP_LIKE_FILENAMES in turn: a test using this fixture runs once per name."""
    return request.param


@pytest.fixture(params=MARKUP_LIKE_TEXTS)
def markup_like_text(request: pytest.FixtureRequest) -> str:
    """Each of MARKUP_LIKE_TEXTS in turn: a test using this fixture runs once per string."""
    return request.param


@pytest.fixture(params=CONTROL_SEQUENCE_TEXTS, ids=CONTROL_SEQUENCE_IDS)
def control_sequence_text(request: pytest.FixtureRequest) -> tuple[str, str]:
    """Each (raw, as shown) pair of CONTROL_SEQUENCE_TEXTS in turn."""
    return request.param


@pytest.fixture
def recording_console() -> Console:
    """A console like production's that records output for export_text(), at a fixed width.

    Tests monkeypatch it over a module's ``console`` attribute to assert on what
    that module printed.
    """
    return TerminalConsole(record=True, width=WIDE_CONSOLE_WIDTH)


@pytest.fixture
def stock_recording_console() -> Console:
    """A recording console with Rich's defaults (emoji substitution on), for pinning down Rich behavior."""
    return Console(record=True, width=WIDE_CONSOLE_WIDTH)


@pytest.fixture
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip retry backoff waits; tenacity's default sleep strategy calls time.sleep."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


class _MessageAttributeError(Exception):
    """An SDK-style error: ``str()`` is empty and the text lives in a ``message`` attribute of any type."""

    def __init__(self, message: object) -> None:
        super().__init__()
        self.message = message


@pytest.fixture
def make_message_attribute_error() -> Callable[[object], Exception]:
    """Factory for SDK-style errors whose ``str()`` is empty, with the given ``message`` attribute."""
    return _MessageAttributeError


@pytest.fixture
def non_string_error_message() -> dict[str, dict[str, str]]:
    """A non-str ``message`` payload, as some SDK errors carry."""
    return {"error": {"code": "rate_limit_exceeded"}}


@pytest.fixture
def non_string_message_error(
    make_message_attribute_error: Callable[[object], Exception], non_string_error_message: dict[str, dict[str, str]]
) -> Exception:
    """An SDK-style error whose ``message`` attribute is ``non_string_error_message``."""
    return make_message_attribute_error(non_string_error_message)


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
