"""Handling of PDF text-extraction artifacts in titles: matching forms and display repair.

TeX-produced PDFs extract some characters in forms no title is written in:
ligature glyphs ("Deﬁnitions") and accents set as separate spacing characters
next to their letter ("K¨ahler", "Poincar´e", "ˇCech", "C¸etin"). This module
handles them in two separate ways.

Matching. The comparison forms make such text compare equal to the plain,
composed or accent-less spellings an LLM writes ("Definitions", "Kähler",
"Kahler", "Čech", "Çetin"). They remove accents from both sides rather than
guess which letter a spacing accent belongs to, so the result does not depend
on where extraction placed the accent:

- ``artifact_neutral_form`` expands Latin ligatures, drops Latin diacritics
  and spacing accents and dots TeX's dotless i/j, keeping case and
  punctuation. Bookmark identity uses it.
- ``matching_form`` is that form casefolded. All title matching starts from it.

Only Latin combining diacritics (U+0300–U+036F) are dropped and only canonical
normalization is applied, so super- and subscripts ("S²", "H₁"), kana voicing
marks, Hangul and symbols such as "™" keep their identity. These forms are
only ever compared, never sent to the LLM or written to a bookmark.

Display. ``repair_extracted_text`` repairs bookmark titles the LLM copied
from extracted text: it expands ligatures and puts a spacing accent back on
its letter where that letter is unambiguous, leaving any other artifact as
extracted. It is never used for matching.

The TOC pipeline (``processors``), the TOC model (``models``) and the
evaluation metrics (``evaluation``) all build their title keys from the
matching primitives here, each with its own punctuation policy.
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


# --- Display repair of bookmark titles --------------------------------------
#
# The forms above remove accents; bookmark titles should instead read as the
# document prints them. repair_extracted_text() puts a spacing accent back on
# its letter only where that letter is unambiguous and otherwise leaves the
# artifact as extracted, so it never writes a wrong letter.

# TeX's accent glyphs as PDF extraction yields them (via the Adobe Glyph
# List), with their modifier-letter variants, mapped to the combining mark
# each one stands for. ASCII "^" and "~" are absent: TeX's circumflex and
# tilde extract as U+02C6 and U+02DC, and the ASCII characters are notation.
_ACCENT_COMBINING_MARKS = {
    "\N{GRAVE ACCENT}": "\N{COMBINING GRAVE ACCENT}",
    "\N{MODIFIER LETTER GRAVE ACCENT}": "\N{COMBINING GRAVE ACCENT}",
    "\N{ACUTE ACCENT}": "\N{COMBINING ACUTE ACCENT}",
    "\N{MODIFIER LETTER ACUTE ACCENT}": "\N{COMBINING ACUTE ACCENT}",
    "\N{MODIFIER LETTER CIRCUMFLEX ACCENT}": "\N{COMBINING CIRCUMFLEX ACCENT}",
    "\N{SMALL TILDE}": "\N{COMBINING TILDE}",
    "\N{MACRON}": "\N{COMBINING MACRON}",
    "\N{MODIFIER LETTER MACRON}": "\N{COMBINING MACRON}",
    "\N{BREVE}": "\N{COMBINING BREVE}",
    "\N{DOT ABOVE}": "\N{COMBINING DOT ABOVE}",
    "\N{DIAERESIS}": "\N{COMBINING DIAERESIS}",
    "\N{RING ABOVE}": "\N{COMBINING RING ABOVE}",
    "\N{DOUBLE ACUTE ACCENT}": "\N{COMBINING DOUBLE ACUTE ACCENT}",
    "\N{CARON}": "\N{COMBINING CARON}",
    "\N{CEDILLA}": "\N{COMBINING CEDILLA}",
    "\N{OGONEK}": "\N{COMBINING OGONEK}",
}
_ACCENT_RUN_PATTERN = re.compile(_character_class(_ACCENT_COMBINING_MARKS) + "+")
_LIGATURE_PATTERN = re.compile(_character_class(_LIGATURE_EXPANSIONS))

# Acute and grave accents double as typed apostrophes ("Hilbert´s", "d´Alembert").
_ACUTE_ACCENTS = frozenset({"\N{ACUTE ACCENT}", "\N{MODIFIER LETTER ACUTE ACCENT}"})
_GRAVE_ACCENTS = frozenset({"\N{GRAVE ACCENT}", "\N{MODIFIER LETTER GRAVE ACCENT}"})
_APOSTROPHE_LIKE_ACCENTS = _ACUTE_ACCENTS | _GRAVE_ACCENTS
_APOSTROPHE = "'"

# TeX emits every accent ahead of its letter except the cedilla and the
# ogonek, which extraction can place on either side ("Fran¸cois", "C¸etin").
# Those attach to whichever neighbor is one of the letters that commonly carry
# them (ç ş ţ; ą ę į ǫ ų), and only when exactly one neighbor is.
_EITHER_SIDE_ACCENT_BASES = {
    "\N{CEDILLA}": frozenset("cst"),
    "\N{OGONEK}": frozenset("aeiou"),
}

# Acute and grave compose only onto vowels: before a consonant the accent is
# an apostrophe or ambiguous ("O´Neill").
_VOWELS = frozenset("aeiouy")


def _expand_ligature(match: re.Match[str]) -> str:
    return _LIGATURE_EXPANSIONS[match.group()]


def _base_letter(letter: str) -> str:
    """A letter without its accents, dotted and casefolded."""
    return _DOTLESS_LETTERS.get(letter, unicodedata.normalize(_CANONICAL_DECOMPOSITION, letter)[0]).casefold()


def _is_vowel(letter: str) -> bool:
    return _base_letter(letter) in _VOWELS


def _compose(base: str, accents: str) -> str | None:
    """``base`` with ``accents`` (outermost first, as TeX emits them) as one precomposed letter, if Unicode has it."""
    marks = "".join(_ACCENT_COMBINING_MARKS[accent] for accent in reversed(accents))
    composed = unicodedata.normalize(_CANONICAL_COMPOSITION, _DOTLESS_LETTERS.get(base, base) + marks)
    is_single_letter = len(composed) == 1
    return composed if is_single_letter else None


def _is_apostrophe_position(before: str, after: str) -> bool:
    """Whether an acute or grave between these letters is a typed apostrophe.

    Between two consonants ("Hilbert´s", "Euler´sche") or between a lowercase
    and an uppercase letter ("d´Alembert") no accented letter is plausible.
    """
    if not before or not after:
        return False
    between_consonants = not _is_vowel(before) and not _is_vowel(after)
    return between_consonants or (before.islower() and after.isupper())


def _repaired_accent_run(before: str, accents: str, after: str, following: str) -> str | None:
    """Replacement for ``before + accents + after``, or None to leave the artifact as extracted.

    ``before`` and ``after`` are the neighboring letters ("" if none) and
    ``following`` is the character after ``after``.
    """
    if len(accents) == 1 and accents in _APOSTROPHE_LIKE_ACCENTS and _is_apostrophe_position(before, after):
        return before + _APOSTROPHE + after
    if len(accents) == 1 and accents in _EITHER_SIDE_ACCENT_BASES:
        bases = _EITHER_SIDE_ACCENT_BASES[accents]
        before_fits = bool(before) and _base_letter(before) in bases
        after_fits = bool(after) and _base_letter(after) in bases
        if before_fits == after_fits:
            return None
        composed = _compose(before if before_fits else after, accents)
        if composed is None:
            return None
        return composed + after if before_fits else before + composed
    if not after or any(accent in _EITHER_SIDE_ACCENT_BASES for accent in accents):
        return None
    if any(accent in _APOSTROPHE_LIKE_ACCENTS for accent in accents) and not _is_vowel(after):
        return None
    # A grave starting a word stands only for a one-letter word ("`a" -> "à");
    # before a longer word it is a code backtick ("`apply`").
    if any(accent in _GRAVE_ACCENTS for accent in accents) and not before and following.isalpha():
        return None
    composed = _compose(after, accents)
    return None if composed is None else before + composed


def repair_extracted_text(text: str) -> str:
    """Best-effort repair of PDF text-extraction artifacts for display (bookmark titles).

    Expands Latin ligatures, puts each TeX spacing accent back on its letter
    ("K¨ahler" -> "Kähler", "C¸etin" -> "Çetin", "na¨ıve" -> "naïve",
    "Vi´ˆet" -> "Việt") and reads an acute or grave in an apostrophe position
    as an apostrophe ("Hilbert´s" -> "Hilbert's"). An accent whose letter is
    ambiguous is left as extracted. Case and punctuation are kept, and the
    result is NFC.

    This is for display only; matching uses matching_form(), which removes
    accents instead of placing them.
    """
    text = unicodedata.normalize(_CANONICAL_COMPOSITION, _LIGATURE_PATTERN.sub(_expand_ligature, text))
    pieces: list[str] = []
    position = 0
    for run in _ACCENT_RUN_PATTERN.finditer(text):
        start, end = run.span()
        before = text[start - 1] if start > position and text[start - 1].isalpha() else ""
        after = text[end] if end < len(text) and text[end].isalpha() else ""
        replacement = _repaired_accent_run(before, run.group(), after, text[end + 1 : end + 2])
        if replacement is None:
            continue
        pieces.append(text[position : start - len(before)])
        pieces.append(replacement)
        position = end + len(after)
    pieces.append(text[position:])
    return unicodedata.normalize(_CANONICAL_COMPOSITION, "".join(pieces))
