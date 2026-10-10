"""Comparison forms that neutralize PDF text-extraction artifacts.

TeX-produced PDFs extract some characters in forms no title is written in:
ligature glyphs ("Deﬁnitions") and accents set as separate spacing characters
next to their letter ("K¨ahler", "Poincar´e", "ˇCech", "C¸etin"). The forms
here make such text compare equal to the plain, composed or accent-less
spellings an LLM writes ("Definitions", "Kähler", "Kahler", "Čech", "Çetin").
They do so by removing accents from both sides rather than by guessing which
letter a spacing accent belongs to, so the result does not depend on where
extraction placed the accent:

- ``artifact_neutral_form`` expands Latin ligatures, drops Latin diacritics
  and spacing accents and dots TeX's dotless i/j, keeping case and
  punctuation. Bookmark identity uses it.
- ``matching_form`` is that form casefolded. All title matching starts from it.

Only Latin combining diacritics (U+0300–U+036F) are dropped and only canonical
normalization is applied, so super- and subscripts ("S²", "H₁"), kana voicing
marks, Hangul and symbols such as "™" keep their identity.

These forms are only ever compared: nothing here is sent to the LLM or written
to a bookmark. The TOC pipeline (``processors``), the TOC model (``models``) and
the evaluation metrics (``evaluation``) all build their title keys from these
primitives, each with its own punctuation policy.
"""

import re
import unicodedata
from collections.abc import Iterable
from typing import Final


_CANONICAL_DECOMPOSITION: Final = "NFD"
_CANONICAL_COMPOSITION: Final = "NFC"
_COMPATIBILITY_COMPOSITION: Final = "NFKC"

# Latin ligatures from Unicode's Alphabetic Presentation Forms block
# (ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ), expanded to the letters they join.
_FIRST_LATIN_LIGATURE = "\N{LATIN SMALL LIGATURE FF}"
_LAST_LATIN_LIGATURE = "\N{LATIN SMALL LIGATURE ST}"
_LIGATURE_EXPANSIONS = {
    chr(codepoint): unicodedata.normalize(_COMPATIBILITY_COMPOSITION, chr(codepoint))
    for codepoint in range(ord(_FIRST_LATIN_LIGATURE), ord(_LAST_LATIN_LIGATURE) + 1)
}

# TeX sets accents on i and j over dotless glyphs (\'\i, \"\i), so once the
# accent is removed the letter is left dotless ("na¨ıve", "Mart´ınez").
_DOTLESS_LETTERS = {
    "\N{LATIN SMALL LETTER DOTLESS I}": "i",
    "\N{LATIN SMALL LETTER DOTLESS J}": "j",
}

_LETTER_NEUTRALIZATIONS = _LIGATURE_EXPANSIONS | _DOTLESS_LETTERS

# Spacing accents: Unicode's spacing modifier symbols (category Sk: "¨", "´",
# "`", "˜", "¸", ...) plus the accent modifier letters (category Lm) that PDF
# extraction yields for TeX's circumflex, caron, macron, acute and grave
# glyphs. ASCII "^" is excluded: TeX's circumflex extracts as U+02C6, while "^"
# is notation ("L^p"). Every Sk character beyond the Basic Multilingual Plane
# is an emoji skin-tone modifier, so only the BMP is scanned.
_SPACING_MODIFIER_SYMBOL_CATEGORY = "Sk"
_BASIC_MULTILINGUAL_PLANE_SIZE = 0x10000
_NOTATION_CIRCUMFLEX = "^"
_MODIFIER_LETTER_ACCENTS = (
    "\N{MODIFIER LETTER CIRCUMFLEX ACCENT}",
    "\N{CARON}",
    "\N{MODIFIER LETTER MACRON}",
    "\N{MODIFIER LETTER ACUTE ACCENT}",
    "\N{MODIFIER LETTER GRAVE ACCENT}",
)
_SPACING_ACCENTS = (
    frozenset(
        chr(codepoint)
        for codepoint in range(_BASIC_MULTILINGUAL_PLANE_SIZE)
        if unicodedata.category(chr(codepoint)) == _SPACING_MODIFIER_SYMBOL_CATEGORY
    )
    - {_NOTATION_CIRCUMFLEX}
) | frozenset(_MODIFIER_LETTER_ACCENTS)

# Latin combining diacritics: Unicode's Combining Diacritical Marks block.
_LATIN_COMBINING_DIACRITICS = "\N{COMBINING GRAVE ACCENT}-\N{COMBINING LATIN SMALL LETTER X}"


def _character_class(characters: Iterable[str], ranges: str = "") -> str:
    """A regex character class matching any of ``characters`` or the given ranges."""
    return "[" + ranges + "".join(re.escape(character) for character in sorted(characters)) + "]"


_LETTER_NEUTRALIZATION_PATTERN = re.compile(_character_class(_LETTER_NEUTRALIZATIONS))
_ACCENT_PATTERN = re.compile(_character_class(_SPACING_ACCENTS, ranges=_LATIN_COMBINING_DIACRITICS))
_EXTRACTION_ARTIFACT_PATTERN = re.compile(_character_class([*_LIGATURE_EXPANSIONS, *_SPACING_ACCENTS]))

# Punctuation policies for comparison keys. Each removes every character that
# is neither a word character nor whitespace, except as named.
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
# Keeps every dot, for checks that read section numbering ("6.6", "1.").
PUNCTUATION_EXCEPT_DOTS_PATTERN = re.compile(r"[^\w\s.]")
# Keeps dots inside numbers, so "2.11" and "21.1" stay distinct: matches a
# non-dot punctuation character, or a dot not between two digits.
PUNCTUATION_OUTSIDE_NUMBERS_PATTERN = re.compile(r"[^\w\s.]|\.(?:(?<!\d\.)|(?!\d))")
# Apostrophes, which join the parts of one word ("O'Neill", "Hilbert's").
APOSTROPHE_PATTERN = re.compile("['\N{RIGHT SINGLE QUOTATION MARK}\N{MODIFIER LETTER APOSTROPHE}]")

_WHITESPACE_RUN_PATTERN = re.compile(r"\s+")
_SINGLE_SPACE = " "


def _neutralize_letter(match: re.Match[str]) -> str:
    return _LETTER_NEUTRALIZATIONS[match.group()]


def artifact_neutral_form(text: str) -> str:
    """Text with extraction artifacts neutralized: ligatures expanded, accents removed, dotless i/j dotted.

    "K¨ahler", "Kähler" (composed or decomposed) and "Kahler" all become
    "Kahler"; "Deﬁnitions" becomes "Definitions". Case and punctuation are
    kept, so "Case x > 0" and "Case x < 0" stay distinct.
    """
    neutral_letters = _LETTER_NEUTRALIZATION_PATTERN.sub(_neutralize_letter, text)
    decomposed = unicodedata.normalize(_CANONICAL_DECOMPOSITION, neutral_letters)
    return unicodedata.normalize(_CANONICAL_COMPOSITION, _ACCENT_PATTERN.sub("", decomposed))


def matching_form(text: str) -> str:
    """The form every title comparison starts from: artifact-neutral, then casefolded."""
    return artifact_neutral_form(text).casefold()


def has_extraction_artifacts(text: str) -> bool:
    """Whether text contains a ligature glyph or a spacing accent."""
    return _EXTRACTION_ARTIFACT_PATTERN.search(text) is not None


def strip_punctuation(text: str, pattern: re.Pattern[str] = PUNCTUATION_PATTERN, replacement: str = "") -> str:
    """Remove (or replace) the punctuation ``pattern`` matches, then collapse whitespace."""
    return _WHITESPACE_RUN_PATTERN.sub(_SINGLE_SPACE, pattern.sub(replacement, text)).strip()


def title_key(title: str) -> str:
    """Key for title-to-title matching: matching form without punctuation, keeping numbering distinct."""
    return strip_punctuation(matching_form(title), PUNCTUATION_OUTSIDE_NUMBERS_PATTERN)
