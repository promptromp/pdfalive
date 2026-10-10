"""Comparison forms that neutralize PDF text-extraction artifacts in titles.

TeX-produced PDFs extract some characters in forms no title is written in:
ligature glyphs ("Deﬁnitions") and accents set as separate spacing characters
next to their letter ("K¨ahler", "Poincar´e", "ˇCech", "C¸etin"). The forms
here make such text compare equal to the plain, composed or accent-less
spellings an LLM writes ("Definitions", "Kähler", "Kahler", "Čech", "Çetin").
They remove the accents from both sides rather than guess which letter a
spacing accent belongs to, so the result does not depend on where extraction
placed the accent.

The neutralization is confined to Latin script: only an explicit set of TeX
spacing-accent characters is removed, and combining marks are dropped only
from Latin letters. Other scripts' marks (Cyrillic "й", Greek accents, kana
voicing marks), marks on symbols ("≠"), and compatibility characters ("S²",
"H₁", "™") keep their identity.

- ``artifact_neutral_form`` neutralizes the artifacts, keeping case and
  punctuation.
- ``matching_form`` is that form casefolded. All title matching starts from it.
- ``identity_form`` is the artifact-neutral form without apostrophes, since an
  apostrophe is often extracted as an acute ("O´Neill"). Bookmark identity
  uses it.

These forms are only ever compared: nothing here is sent to the LLM or written
to a bookmark. The TOC pipeline (``processors``), the TOC model (``models``) and
the evaluation metrics (``evaluation``) all build their title keys from these
primitives, each with its own punctuation policy.
"""

import re
import unicodedata
from collections.abc import Iterable


# The Latin ligatures of Unicode's Alphabetic Presentation Forms block.
_LIGATURE_EXPANSIONS = {
    "\N{LATIN SMALL LIGATURE FF}": "ff",
    "\N{LATIN SMALL LIGATURE FI}": "fi",
    "\N{LATIN SMALL LIGATURE FL}": "fl",
    "\N{LATIN SMALL LIGATURE FFI}": "ffi",
    "\N{LATIN SMALL LIGATURE FFL}": "ffl",
    "\N{LATIN SMALL LIGATURE LONG S T}": "st",
    "\N{LATIN SMALL LIGATURE ST}": "st",
}

# TeX's accent glyphs as PDF extraction yields them (via the Adobe Glyph List),
# plus the modifier-letter forms of the same accents. The ASCII circumflex,
# tilde and grave are absent: TeX's circumflex and tilde extract as U+02C6 and
# U+02DC, the ASCII characters are notation ("L^p", "x ~ y", a code
# backtick), and the keys' punctuation stripping deletes them anyway.
_SPACING_ACCENTS = frozenset(
    {
        "\N{DIAERESIS}",
        "\N{ACUTE ACCENT}",
        "\N{MODIFIER LETTER CIRCUMFLEX ACCENT}",
        "\N{SMALL TILDE}",
        "\N{MACRON}",
        "\N{BREVE}",
        "\N{DOT ABOVE}",
        "\N{RING ABOVE}",
        "\N{DOUBLE ACUTE ACCENT}",
        "\N{CEDILLA}",
        "\N{OGONEK}",
        "\N{CARON}",
        "\N{MODIFIER LETTER MACRON}",
        "\N{MODIFIER LETTER ACUTE ACCENT}",
        "\N{MODIFIER LETTER GRAVE ACCENT}",
    }
)
# TeX builds the cedilla and the ogonek below their letter, so extraction can
# place them after it ("C¸etin"); every other accent comes before its letter.
_AFTER_LETTER_ACCENTS = frozenset({"\N{CEDILLA}", "\N{OGONEK}"})

# TeX sets accents on i and j over dotless glyphs (\'\i, \"\i), so the letter
# after a removed accent may be dotless ("na¨ıve", "Mart´ınez").
_DOTLESS_LETTERS = {
    "\N{LATIN SMALL LETTER DOTLESS I}": "i",
    "\N{LATIN SMALL LETTER DOTLESS J}": "j",
}

# Latin letters: ASCII, the Latin-1 Supplement letters, Latin Extended-A and -B,
# and Latin Extended Additional.
_LATIN_LETTER = (
    "[A-Za-z"
    "\N{LATIN CAPITAL LETTER A WITH GRAVE}-\N{LATIN CAPITAL LETTER O WITH DIAERESIS}"
    "\N{LATIN CAPITAL LETTER O WITH STROKE}-\N{LATIN SMALL LETTER O WITH DIAERESIS}"
    "\N{LATIN SMALL LETTER O WITH STROKE}-\N{LATIN SMALL LETTER Y WITH STROKE}"
    "\N{LATIN CAPITAL LETTER A WITH RING BELOW}-\N{LATIN SMALL LETTER Y WITH LOOP}]"
)
# Unicode's Combining Diacritical Marks block.
_COMBINING_DIACRITICS = "[\N{COMBINING GRAVE ACCENT}-\N{COMBINING LATIN SMALL LETTER X}]"


def _character_class(characters: Iterable[str]) -> str:
    """A regex character class matching any of ``characters``."""
    return "[" + "".join(re.escape(character) for character in sorted(characters)) + "]"


_LIGATURE_PATTERN = re.compile(_character_class(_LIGATURE_EXPANSIONS))
_ACCENT = _character_class(_SPACING_ACCENTS)
_SPACING_ACCENT_PATTERN = re.compile(_ACCENT)
_BEFORE_LETTER_ACCENT = _character_class(_SPACING_ACCENTS - _AFTER_LETTER_ACCENTS)
_DOTLESS_LETTER = _character_class(_DOTLESS_LETTERS)
# A run of spacing accents, with any whitespace a span break left between it
# and a neighboring letter ("K ¨ahler"), and the dotless letter that a run
# ending in a before-letter accent was set over ("na¨ıve", not "Kas¸ık").
_SPACING_ACCENT_RUN_PATTERN = re.compile(
    rf"(?:(?<=\w)\s+)?(?:{_ACCENT}*{_BEFORE_LETTER_ACCENT}\s*(?P<base>{_DOTLESS_LETTER})|{_ACCENT}+(?:\s+(?=\w))?)"
)
# A Latin letter with its combining diacritics (in decomposed text).
_ACCENTED_LATIN_LETTER_PATTERN = re.compile(f"(?P<base>{_LATIN_LETTER}){_COMBINING_DIACRITICS}+")
_EXTRACTION_ARTIFACT_PATTERN = re.compile(_character_class([*_LIGATURE_EXPANSIONS, *_SPACING_ACCENTS]))

# Punctuation policies for comparison keys. Each removes every character that
# is neither a word character nor whitespace, except as named.
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
# Keeps every dot, for checks that read section numbering ("6.6", "1.").
PUNCTUATION_EXCEPT_DOTS_PATTERN = re.compile(r"[^\w\s.]")
# Apostrophes, which join the parts of one word ("O'Neill", "Hilbert's").
APOSTROPHE_PATTERN = re.compile("['\N{RIGHT SINGLE QUOTATION MARK}\N{MODIFIER LETTER APOSTROPHE}]")

_WHITESPACE_RUN_PATTERN = re.compile(r"\s+")


def _expand_ligature(match: re.Match[str]) -> str:
    return _LIGATURE_EXPANSIONS[match.group()]


def _unaccented_base(match: re.Match[str]) -> str:
    base = match.group("base") or ""
    return _DOTLESS_LETTERS.get(base, base)


def artifact_neutral_form(text: str) -> str:
    """Text with extraction artifacts neutralized, keeping case and punctuation.

    Expands Latin ligatures, removes TeX spacing accents (with whitespace a
    span break left beside them) and the combining diacritics of Latin
    letters, dotting a dotless i/j that TeX set an accent over, and returns
    NFC. "K¨ahler", "K ¨ahler", "Kähler" (composed or decomposed) and "Kahler"
    all become "Kahler"; "Deﬁnitions" becomes "Definitions"; "Case x ≠ 0"
    keeps its "≠".
    """
    decomposed = unicodedata.normalize("NFD", _LIGATURE_PATTERN.sub(_expand_ligature, text))
    # Every accent run holds a spacing accent. Most text has none, and finding
    # that with a plain character search is far cheaper than scanning for runs.
    without_spacing_accents = (
        _SPACING_ACCENT_RUN_PATTERN.sub(_unaccented_base, decomposed)
        if _SPACING_ACCENT_PATTERN.search(decomposed)
        else decomposed
    )
    without_latin_diacritics = _ACCENTED_LATIN_LETTER_PATTERN.sub(_unaccented_base, without_spacing_accents)
    return unicodedata.normalize("NFC", without_latin_diacritics)


def matching_form(text: str) -> str:
    """The form every title comparison starts from: artifact-neutral, then casefolded."""
    return artifact_neutral_form(text).casefold()


def identity_form(text: str) -> str:
    """Artifact-neutral form without apostrophes: "O´Neill" and "O'Neill" share it, case and symbols stay."""
    return APOSTROPHE_PATTERN.sub("", artifact_neutral_form(text))


def has_extraction_artifacts(text: str) -> bool:
    """Whether text contains a ligature glyph or a TeX spacing accent."""
    return _EXTRACTION_ARTIFACT_PATTERN.search(text) is not None


def strip_punctuation(text: str, pattern: re.Pattern[str] = PUNCTUATION_PATTERN, replacement: str = "") -> str:
    """Remove (or replace) the punctuation ``pattern`` matches, then collapse whitespace."""
    return _WHITESPACE_RUN_PATTERN.sub(" ", pattern.sub(replacement, text)).strip()


def title_key(title: str) -> str:
    """Key for title-to-title matching: matching form without punctuation."""
    return strip_punctuation(matching_form(title))
