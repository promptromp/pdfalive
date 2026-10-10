"""Tests for pdfalive's consoles, progress bars and safe text."""

import io

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pdfalive.console import (
    ELLIPSIS,
    ESCAPE_STYLE,
    TerminalConsole,
    contains_unsafe_characters,
    create_progress,
    safe_text,
    truncate_visible,
    visible,
)


# A styled segment that follows the value under test, where markup would put a closing tag.
FOLLOWING_SEGMENT = "(next)"
FOLLOWING_STYLE = "yellow"
VALUE_STYLE = "bold cyan"

# Progress descriptions are program text, but must still print as they are.
MARKUP_LIKE_DESCRIPTION = "[bold]Detecting text[/bold] :A:"
PROGRESS_TEMPLATE_MARKUP = "[progress.description]"

# (raw, shown) for characters made visible in every value, single-line or not.
ALWAYS_ESCAPED = [
    ("\x00", "\\x00"),
    ("\x07", "\\x07"),
    ("\r", "\\r"),
    ("\x1b[8m", "\\x1b[8m"),
    ("\x1b[2K", "\\x1b[2K"),
    ("\x7f", "\\x7f"),
    ("\x9b2K", "\\x9b2K"),
    ("\t", "\\t"),
    ("\u202e", "\\u202e"),
    ("\u2066", "\\u2066"),
    ("\u2028", "\\u2028"),
    ("\u2029", "\\u2029"),
    ("\udce9", "\\xe9"),
    ("\ud800", "\\ud800"),
]
ALWAYS_ESCAPED_IDS = [
    "nul",
    "bell",
    "carriage-return",
    "conceal",
    "erase-line",
    "delete",
    "c1-csi",
    "tab",
    "rlo",
    "lri",
    "line-separator",
    "paragraph-separator",
    "undecodable-byte",
    "lone-surrogate",
]

# Format characters that legitimately occur in titles and are kept.
KEPT_FORMAT_CHARACTERS = "soft\u00adhyphen zero\u200dwidth\u200cjoiners"

# A value long enough to need truncation, and the width it is cut to.
TRUNCATION_WIDTH = 10


def test_console_prints_strings_without_parsing_markup_or_shortcodes(
    recording_console: Console, markup_like_text: str
) -> None:
    """A future console.print(f"... {outside_text}") prints the text as it is instead of restyling it."""
    recording_console.print(markup_like_text)

    assert recording_console.export_text() == f"{markup_like_text}\n"


def test_text_segments_render_verbatim_on_a_stock_console(
    stock_recording_console: Console, markup_like_text: str
) -> None:
    """The premise of building output as Text: its segments are neither parsed as markup nor
    emoji-substituted, whatever the console's settings, including before a styled segment."""
    stock_recording_console.print(
        Text.assemble((markup_like_text, VALUE_STYLE), " ", (FOLLOWING_SEGMENT, FOLLOWING_STYLE))
    )

    assert stock_recording_console.export_text() == f"{markup_like_text} {FOLLOWING_SEGMENT}\n"


class TestSafeText:
    @pytest.mark.parametrize("multiline", [False, True])
    @pytest.mark.parametrize(("raw", "shown"), ALWAYS_ESCAPED, ids=ALWAYS_ESCAPED_IDS)
    def test_makes_unsafe_characters_visible(self, raw: str, shown: str, multiline: bool) -> None:
        assert safe_text(raw, (raw, VALUE_STYLE), multiline=multiline).plain == shown * 2

    def test_styles_escapes_distinctly(self) -> None:
        """An escape for a real control character cannot be mistaken for a literal backslash sequence."""
        text = safe_text(("ok\x1b", VALUE_STYLE), "\\x1b")

        escape_spans = [text.plain[span.start : span.end] for span in text.spans if span.style == ESCAPE_STYLE]
        assert escape_spans == ["\\x1b"]

    def test_single_line_values_escape_newlines(self) -> None:
        assert safe_text("a\nb").plain == "a\\nb"

    def test_multiline_values_keep_newlines_and_normalize_crlf(self) -> None:
        assert safe_text("a\r\nb\nc", multiline=True).plain == "a\nb\nc"

    def test_keeps_format_characters_legitimate_in_titles(self) -> None:
        assert safe_text(KEPT_FORMAT_CHARACTERS).plain == KEPT_FORMAT_CHARACTERS

    def test_applies_part_styles(self, markup_like_text: str) -> None:
        text = safe_text((markup_like_text, VALUE_STYLE), " ", (FOLLOWING_SEGMENT, FOLLOWING_STYLE))

        styled = [(text.plain[span.start : span.end], span.style) for span in text.spans]
        assert styled == [(markup_like_text, VALUE_STYLE), (FOLLOWING_SEGMENT, FOLLOWING_STYLE)]

    def test_printed_control_sequences_cannot_reach_the_terminal(
        self, recording_console: Console, control_sequence_text: tuple[str, str]
    ) -> None:
        raw, shown = control_sequence_text

        recording_console.print(safe_text((raw, VALUE_STYLE)))

        assert recording_console.export_text() == f"{shown}\n"


class TestVisible:
    """The plain-string form, for messages not printed on a console."""

    @pytest.mark.parametrize(("raw", "shown"), ALWAYS_ESCAPED, ids=ALWAYS_ESCAPED_IDS)
    def test_matches_safe_text(self, raw: str, shown: str) -> None:
        assert visible(raw) == shown

    def test_multiline_keeps_newlines(self) -> None:
        assert visible("a\r\nb", multiline=True) == "a\nb"


class TestContainsUnsafeCharacters:
    """The one definition of an unsafe character, shared with checks outside the console (new file names)."""

    @pytest.mark.parametrize("raw", [raw for raw, _shown in ALWAYS_ESCAPED], ids=ALWAYS_ESCAPED_IDS)
    def test_every_escaped_character_is_unsafe(self, raw: str) -> None:
        assert contains_unsafe_characters(f"before {raw} after")

    def test_a_newline_is_unsafe(self) -> None:
        assert contains_unsafe_characters("line\nbreak")

    def test_markup_like_text_is_safe(self, markup_like_text: str) -> None:
        assert not contains_unsafe_characters(markup_like_text)


class TestTerminalConsole:
    """Every string printed on the console, including plain-string table cells, is made safe."""

    @pytest.mark.parametrize(("raw", "shown"), ALWAYS_ESCAPED, ids=ALWAYS_ESCAPED_IDS)
    def test_plain_strings_are_made_safe(self, recording_console: Console, raw: str, shown: str) -> None:
        recording_console.print(f"value {raw}")

        assert recording_console.export_text() == f"value {shown}\n"

    @pytest.mark.parametrize(("raw", "shown"), ALWAYS_ESCAPED, ids=ALWAYS_ESCAPED_IDS)
    def test_plain_string_table_cells_are_made_safe(self, recording_console: Console, raw: str, shown: str) -> None:
        table = Table()
        table.add_column("Value")
        table.add_row(raw)

        recording_console.print(table)

        assert shown in recording_console.export_text()

    def test_plain_strings_keep_their_newlines(self, recording_console: Console) -> None:
        recording_console.print("first\r\nsecond")

        assert recording_console.export_text() == "first\nsecond\n"

    def test_escapes_in_plain_strings_are_styled_distinctly(self, recording_console: Console) -> None:
        rendered = recording_console.render_str("ok\x1b")

        assert [rendered.plain[span.start : span.end] for span in rendered.spans if span.style == ESCAPE_STYLE] == [
            "\\x1b"
        ]

    def test_markup_and_emoji_stay_off_even_when_requested(
        self, recording_console: Console, markup_like_text: str
    ) -> None:
        recording_console.print(markup_like_text, markup=True, emoji=True)

        assert recording_console.export_text() == f"{markup_like_text}\n"

    def test_undecodable_bytes_do_not_crash_a_utf8_terminal(self) -> None:
        """Regression: a lone surrogate from argv or a file name raised UnicodeEncodeError on a UTF-8 stream."""
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        console = TerminalConsole(file=stream)

        console.print("name \udce9.pdf")

        stream.flush()
        assert stream.buffer.getvalue() == b"name \\xe9.pdf\n"  # type: ignore[attr-defined]


class TestTruncateVisible:
    def test_leaves_a_value_that_fits(self) -> None:
        value = "x" * TRUNCATION_WIDTH

        assert truncate_visible(value, TRUNCATION_WIDTH) == value

    @pytest.mark.parametrize("character", ["x", "長"], ids=["narrow", "wide"])
    def test_cuts_to_the_width_with_an_ellipsis(self, character: str) -> None:
        value = character * TRUNCATION_WIDTH * 2

        shown = truncate_visible(value, TRUNCATION_WIDTH)

        assert shown.endswith(ELLIPSIS)
        assert value.startswith(shown.removesuffix(ELLIPSIS))
        assert cell_len(shown) <= TRUNCATION_WIDTH

    def test_never_cuts_an_escape_in_half(self) -> None:
        """Width counts an unsafe character as its visible escape, and the cut falls before it."""
        prefix = "x" * (TRUNCATION_WIDTH - len(ELLIPSIS) - 1)
        value = f"{prefix}\x1b[2K and more"

        shown = truncate_visible(value, TRUNCATION_WIDTH)

        assert shown == f"{prefix}{ELLIPSIS}"
        assert cell_len(safe_text(shown).plain) <= TRUNCATION_WIDTH


def test_progress_bar_shows_its_description_as_plain_text(recording_console: Console) -> None:
    """Descriptions are never parsed as markup or emoji shortcodes, and the bar works on a TerminalConsole."""
    with create_progress(recording_console) as progress:
        task = progress.add_task(MARKUP_LIKE_DESCRIPTION, total=1)
        progress.advance(task)

    output = recording_console.export_text()
    assert MARKUP_LIKE_DESCRIPTION in output
    assert PROGRESS_TEMPLATE_MARKUP not in output
