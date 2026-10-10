"""Terminal output shared by the CLI and the processors: consoles, progress bars, safe text.

pdfalive prints text it does not control: file paths, model identifiers, TOC
titles, LLM rename suggestions, exception messages. Such text must reach the
terminal exactly as it is:

- Never parsed as Rich markup (``[van Author] Some Title.pdf`` would lose its
  bracketed part, a stray ``[/]`` raises ``MarkupError``) or for ``:name:``
  emoji shortcodes (``:A:`` in ``Maps f:A:B``). Escaping markup is no
  substitute: separately escaped fragments can pair up into a tag once
  joined, and ``rich.markup.escape`` mishandles trailing backslashes.
- With characters that would act on the terminal or the displayed line made
  visible: control characters (an ESC sequence could hide or erase output),
  bidi embedding/override/isolate controls and line/paragraph separators (an
  RLO can disguise ``invoice\\u202efdp.exe`` as ``invoiceexe.pdf``), and lone
  surrogates (undecodable bytes from argv or file names, which crash a UTF-8
  terminal). Each is replaced by a Python-style escape (ESC becomes ``\\x1b``,
  an undecodable byte 0xE9 becomes ``\\xe9``), shown in ESCAPE_STYLE so it
  cannot be mistaken for a literal backslash sequence. A single-line value
  also has its newlines and tabs escaped.

Two choke points apply these rules. Every string printed on a console from
:func:`create_console` (including plain-string table cells) goes through
:meth:`TerminalConsole.render_str`, which never parses markup or emoji and
makes unsafe characters visible. Styled output is composed with
:func:`safe_text`, the single point where outside text becomes ``Text``; it
treats each value as a single line unless asked otherwise. :func:`visible` is
the plain-string form, for messages that are not printed on a console (Click
errors).

Ruff's banned-api rule (TID251) keeps the Rich APIs that create or fall back to
an unconfigured console (``Console``, the global-console helpers, and the
progress, live, status, prompt, logging, traceback and pretty modules), or that
parse markup or ANSI codes (``rich.markup``, ``Text.from_markup``,
``Text.from_ansi``), out of every other module.
"""

import re
from bisect import bisect_right
from collections.abc import Iterator
from itertools import accumulate

from rich.cells import cell_len
from rich.console import Console, HighlighterType, JustifyMethod, OverflowMethod
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.style import Style, StyleType
from rich.text import Text


# The console type, for annotations elsewhere: an alias cannot be called, so
# consoles are still created only by create_console().
type ConsoleType = Console

# A part of a safe_text() line: plain text, or (text, style).
type TextPart = str | tuple[str, StyleType]

# Style of the visible escape that replaces an unsafe character.
ESCAPE_STYLE = "reverse"

# Marks where truncate_visible() cut a value.
ELLIPSIS = "…"

# C0 control characters except tab and newline, DEL, and C1 control characters.
_CONTROL_CHARACTERS = "\x00-\x08\x0b-\x1f\x7f-\x9f"
# Layout characters that only a multi-line value may keep (newline) or that a
# single-line value must not contain (newline, tab).
_TAB = "\t"
_NEWLINE = "\n"
# Bidi controls and separators that reorder or break the displayed line:
# LRE, RLE, PDF, LRO, RLO (U+202A-U+202E), LRI, RLI, FSI, PDI (U+2066-U+2069),
# LINE SEPARATOR (U+2028) and PARAGRAPH SEPARATOR (U+2029). Other format
# characters, such as soft hyphens and zero-width joiners, are legitimate in titles.
_BIDI_CONTROLS_AND_SEPARATORS = "‪-‮⁦-⁩  "
# Lone surrogates. Python's surrogateescape error handler maps each undecodable
# byte 0x80-0xFF of argv or a file name to U+DC80-U+DCFF.
_SURROGATES = "\ud800-\udfff"
_SURROGATE_ESCAPE_BYTES = range(0xDC80, 0xDD00)
_SURROGATE_ESCAPE_OFFSET = 0xDC00

_UNSAFE_IN_SINGLE_LINE = re.compile(
    f"[{_CONTROL_CHARACTERS}{_TAB}{_NEWLINE}{_BIDI_CONTROLS_AND_SEPARATORS}{_SURROGATES}]"
)
_UNSAFE_IN_MULTILINE = re.compile(f"[{_CONTROL_CHARACTERS}{_TAB}{_BIDI_CONTROLS_AND_SEPARATORS}{_SURROGATES}]")
_WINDOWS_LINE_BREAK = "\r\n"


def _escape(character: str) -> str:
    """The visible escape for an unsafe character."""
    code_point = ord(character)
    if code_point in _SURROGATE_ESCAPE_BYTES:
        # Show the undecodable byte itself.
        return f"\\x{code_point - _SURROGATE_ESCAPE_OFFSET:02x}"
    return character.encode("unicode_escape").decode("ascii")


def _visible_runs(value: str, *, multiline: bool) -> Iterator[tuple[str, bool]]:
    """Split value into (text, is_escape) runs, with each unsafe character replaced by its escape."""
    if multiline:
        value = value.replace(_WINDOWS_LINE_BREAK, _NEWLINE)
    unsafe = _UNSAFE_IN_MULTILINE if multiline else _UNSAFE_IN_SINGLE_LINE
    position = 0
    for match in unsafe.finditer(value):
        if match.start() > position:
            yield value[position : match.start()], False
        yield _escape(match.group()), True
        position = match.end()
    if position < len(value):
        yield value[position:], False


def visible(value: str, *, multiline: bool = False) -> str:
    """value with unsafe characters replaced by visible escapes, for plain-string messages.

    Args:
        value: Text the program does not control.
        multiline: Keep newlines (normalizing CRLF to LF) instead of escaping them.
    """
    return "".join(run for run, _ in _visible_runs(value, multiline=multiline))


def truncate_visible(value: str, max_width: int) -> str:
    """Cut value so that it fits max_width terminal cells once shown, ending a cut with an ellipsis.

    Widths are measured on the shown form, so an unsafe character counts as its
    visible escape, and an escape is never cut in half.
    """
    shown_widths = list(accumulate(cell_len(visible(character)) for character in value))
    if not shown_widths or shown_widths[-1] <= max_width:
        return value
    kept = bisect_right(shown_widths, max_width - cell_len(ELLIPSIS))
    return value[:kept] + ELLIPSIS


def safe_text(*parts: TextPart, multiline: bool = False) -> Text:
    """Compose terminal output from plain parts and ``(text, style)`` parts.

    Each part is shown verbatim, with unsafe characters replaced by visible
    escapes in ESCAPE_STYLE. Every message that includes text the program does
    not control is built here.

    Args:
        parts: Plain text, or (text, style) pairs.
        multiline: Keep newlines in the parts; by default each value is one line.
    """
    text = Text()
    for part in parts:
        value, style = (part, "") if isinstance(part, str) else part
        for run, is_escape in _visible_runs(value, multiline=multiline):
            start = len(text)
            text.append(run, style=style)
            if is_escape:
                text.stylize(ESCAPE_STYLE, start, len(text))
    return text


class TerminalConsole(Console):
    """A console whose printed strings are shown verbatim and terminal-safe.

    Strings, including plain-string table cells, are never parsed as markup or
    emoji shortcodes, whatever a caller passes, and unsafe characters are made
    visible as in safe_text() (newlines are kept). Construct it through
    create_console(); tests build their own with record/width options.
    """

    def render_str(
        self,
        text: str,
        *,
        style: str | Style = "",
        justify: JustifyMethod | None = None,
        overflow: OverflowMethod | None = None,
        emoji: bool | None = None,
        markup: bool | None = None,
        highlight: bool | None = None,
        highlighter: HighlighterType | None = None,
    ) -> Text:
        runs = list(_visible_runs(text, multiline=True))
        rendered = super().render_str(
            "".join(run for run, _ in runs),
            style=style,
            justify=justify,
            overflow=overflow,
            emoji=False,
            markup=False,
            highlight=highlight,
            highlighter=highlighter,
        )
        position = 0
        for run, is_escape in runs:
            if is_escape:
                rendered.stylize(ESCAPE_STYLE, position, position + len(run))
            position += len(run)
        return rendered


def create_console() -> TerminalConsole:
    """Create the console a pdfalive module prints on."""
    return TerminalConsole()


def create_progress(console: Console) -> Progress:
    """Create pdfalive's progress bar (spinner, description, bar, percentage), drawn on console.

    The description is shown as plain text, never parsed as markup.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}", style="progress.description", markup=False),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
