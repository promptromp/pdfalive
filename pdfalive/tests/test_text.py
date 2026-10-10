"""Tests for the comparison forms that neutralize PDF text-extraction artifacts."""

import pytest

from pdfalive.text import (
    APOSTROPHE_PATTERN,
    PUNCTUATION_EXCEPT_DOTS_PATTERN,
    PUNCTUATION_PATTERN,
    artifact_neutral_form,
    has_extraction_artifacts,
    identity_form,
    matching_form,
    strip_punctuation,
    title_key,
)


# One heading spelled the ways PDF extraction and the LLM produce it.
KAHLER_COMPOSED, KAHLER_DECOMPOSED, KAHLER_EXTRACTED, KAHLER_SPLIT_SPAN, KAHLER_PLAIN, KAHLER_UPPER = (
    KAHLER_SPELLINGS
) = (
    "Kähler Potentials",
    "Ka\N{COMBINING DIAERESIS}hler Potentials",
    "K\N{DIAERESIS}ahler Potentials",
    # The accent in a span of its own, joined to its neighbors with a space.
    "K \N{DIAERESIS}ahler Potentials",
    "Kahler Potentials",
    "KÄHLER POTENTIALS",
)
DEFINITIONS_SPELLINGS = ("Definitions and Notation", "De\N{LATIN SMALL LIGATURE FI}nitions and Notation")

# (extracted or variant spelling, clean spelling) pairs that must match.
EQUIVALENT_SPELLINGS = [
    ("A\N{LATIN SMALL LIGATURE FFI}ne Maps", "Affine Maps"),
    ("Re\N{LATIN SMALL LIGATURE FL}exive Spaces", "Reflexive Spaces"),
    ("E\N{LATIN SMALL LIGATURE FF}ective Bounds", "Effective Bounds"),
    ("Shu\N{LATIN SMALL LIGATURE FFL}e Products", "Shuffle Products"),
    ("Fa\N{LATIN SMALL LIGATURE ST} Decay", "Fast Decay"),
    ("Poincar\N{ACUTE ACCENT}e Maps", "Poincaré Maps"),
    ("Poincar\N{ACUTE ACCENT}e Maps", "Poincare Maps"),
    ("Erd\N{DOUBLE ACUTE ACCENT}os", "Erdős"),
    ("\N{RING ABOVE}Angstr\N{DIAERESIS}om", "Ångström"),
    ("H\N{MODIFIER LETTER CIRCUMFLEX ACCENT}opital", "Hôpital"),
    ("\N{CARON}Cech Cohomology", "Čech Cohomology"),
    ("\N{MODIFIER LETTER MACRON}a", "ā"),
    ("\N{MODIFIER LETTER ACUTE ACCENT}e", "é"),
    ("\N{MODIFIER LETTER GRAVE ACCENT}a", "à"),
    # The accent's position does not matter: a cedilla extracted after its letter.
    ("C\N{CEDILLA}etin Algebras", "Çetin Algebras"),
    # Stacked accents.
    ("Vi\N{ACUTE ACCENT}\N{MODIFIER LETTER CIRCUMFLEX ACCENT}et Lattices", "Viết Lattices"),
    # TeX's dotless i under an accent that precedes it.
    ("na\N{DIAERESIS}\N{LATIN SMALL LETTER DOTLESS I}ve", "naïve"),
    # A cedilla extracted after its letter leaves a following dotless i alone.
    ("C\N{CEDILLA}\N{LATIN SMALL LETTER DOTLESS I}nar", "Ç\N{LATIN SMALL LETTER DOTLESS I}nar"),
    ("Kas\N{CEDILLA}\N{LATIN SMALL LETTER DOTLESS I}k", "Kaş\N{LATIN SMALL LETTER DOTLESS I}k"),
    # Accents in a span of their own, joined to their neighbors with a space.
    ("Poincar \N{ACUTE ACCENT}e Maps", "Poincaré Maps"),
    ("C\N{CEDILLA} etin Algebras", "Çetin Algebras"),
    ("na\N{DIAERESIS} \N{LATIN SMALL LETTER DOTLESS I}ve", "naïve"),
    # A Greek spacing accent that canonically decomposes to a TeX one.
    ("a\N{GREEK OXIA}b", "ab"),
]

# Pairs that must stay distinct: compatibility forms, marks outside Latin script
# or on symbols, and symbols themselves keep their identity.
DISTINCT_PAIRS = [
    ("S\N{SUPERSCRIPT TWO}", "S2"),
    ("H\N{SUPERSCRIPT ONE}", "H\N{SUBSCRIPT ONE}"),
    ("\N{HIRAGANA LETTER GA}", "\N{HIRAGANA LETTER KA}"),
    ("\N{HIRAGANA LETTER KA}\N{KATAKANA-HIRAGANA VOICED SOUND MARK}", "\N{HIRAGANA LETTER KA}"),
    ("Мой", "Мои"),
    ("ёж", "еж"),
    ("\N{GREEK SMALL LETTER ALPHA WITH TONOS}", "\N{GREEK SMALL LETTER ALPHA}"),
    ("\N{GREEK TONOS}A", "A"),
    ("Case x \N{NOT EQUAL TO} 0", "Case x = 0"),
    ("A \N{NOT AN ELEMENT OF} B", "A \N{ELEMENT OF} B"),
    ("A \N{NOT A SUBSET OF} B", "A \N{SUBSET OF} B"),
    ("\N{TRADE MARK SIGN}", "TM"),
    ("Case x > 0", "Case x < 0"),
    ("C++", "C#"),
    ("A \N{UNION} B", "A \N{INTERSECTION} B"),
]

# Text matching_form leaves unchanged.
UNCHANGED_TEXT = [
    "",
    "differential forms",
    "s\N{SUPERSCRIPT TWO} and h\N{SUBSCRIPT ONE}",
    "l^p spaces",
    "\N{HIRAGANA LETTER GA}\N{HIRAGANA LETTER KA}",
    "\N{HANGUL SYLLABLE HAN}\N{HANGUL SYLLABLE GUG}",
    "\N{TRADE MARK SIGN} and \N{NUMERO SIGN}",
    "мой и ёж",
    "case x \N{NOT EQUAL TO} 0",
    # ASCII circumflex and grave are notation, not accents.
    "the `with` statement",
]

IDEMPOTENCE_INPUTS = [variant for variant, _ in EQUIVALENT_SPELLINGS] + [
    "\N{GREEK OXIA}",
    "\N{GREEK VARIA}",
    "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}stanbul",
    *KAHLER_SPELLINGS,
]


class TestMatchingForm:
    @pytest.mark.parametrize("spellings", [KAHLER_SPELLINGS, DEFINITIONS_SPELLINGS])
    def test_spellings_of_one_heading_compare_equal(self, spellings: tuple[str, ...]) -> None:
        assert len({matching_form(spelling) for spelling in spellings}) == 1

    @pytest.mark.parametrize(("variant", "clean"), EQUIVALENT_SPELLINGS)
    def test_variant_matches_clean_spelling(self, variant: str, clean: str) -> None:
        assert matching_form(variant) == matching_form(clean)

    @pytest.mark.parametrize(("left", "right"), DISTINCT_PAIRS)
    def test_distinct_text_stays_distinct(self, left: str, right: str) -> None:
        assert matching_form(left) != matching_form(right)

    @pytest.mark.parametrize("text", UNCHANGED_TEXT)
    def test_leaves_other_text_unchanged(self, text: str) -> None:
        assert matching_form(text) == text

    @pytest.mark.parametrize("text", IDEMPOTENCE_INPUTS)
    def test_is_idempotent(self, text: str) -> None:
        assert matching_form(matching_form(text)) == matching_form(text)


class TestArtifactNeutralForm:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("6.4 K\N{DIAERESIS}ahler Potentials", "6.4 Kahler Potentials"),
            ("CHAPTER 3: De\N{LATIN SMALL LIGATURE FI}nitions", "CHAPTER 3: Definitions"),
            ("Case x > 0", "Case x > 0"),
            ("L^p Spaces", "L^p Spaces"),
        ],
    )
    def test_neutralizes_artifacts_keeping_case_and_punctuation(self, text: str, expected: str) -> None:
        assert artifact_neutral_form(text) == expected


class TestHasExtractionArtifacts:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("K\N{DIAERESIS}ahler", True),
            ("De\N{LATIN SMALL LIGATURE FI}nitions", True),
            ("Probl`emes", False),
            ("The `with` Statement", False),
            ("\N{CARON}Cech", True),
            ("Kähler", False),
            ("Ka\N{COMBINING DIAERESIS}hler", False),
            ("L^p Spaces", False),
            ("Hilbert's Theorem", False),
        ],
    )
    def test_detects_ligatures_and_spacing_accents(self, text: str, expected: bool) -> None:
        assert has_extraction_artifacts(text) is expected


class TestStripPunctuation:
    @pytest.mark.parametrize(
        ("text", "pattern", "replacement", "expected"),
        [
            ("  floating-point  numbers ", PUNCTUATION_PATTERN, "", "floatingpoint numbers"),
            ("  floating-point  numbers ", PUNCTUATION_PATTERN, " ", "floating point numbers"),
            ("6.6. term structure", PUNCTUATION_EXCEPT_DOTS_PATTERN, "", "6.6. term structure"),
            ("o'neill and hilbert’s", APOSTROPHE_PATTERN, "", "oneill and hilberts"),
        ],
    )
    def test_applies_policy_and_collapses_whitespace(self, text: str, pattern, replacement: str, expected: str) -> None:
        assert strip_punctuation(text, pattern, replacement) == expected


class TestTitleKey:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (KAHLER_COMPOSED, KAHLER_EXTRACTED),
            (KAHLER_PLAIN, KAHLER_SPLIT_SPAN),
            ("Chapter 3: Definitions", "CHAPTER 3 De\N{LATIN SMALL LIGATURE FI}nitions"),
            # A grave extracted as an ASCII backtick, which the key strips as punctuation.
            ("Probl`emes", "Problèmes"),
            ("a\N{GREEK VARIA}b", "ab"),
            ("O'Neill Spaces", "O\N{ACUTE ACCENT}Neill Spaces"),
            ("d'Alembert Operators", "d\N{ACUTE ACCENT}Alembert Operators"),
            ("Euler'sche Formeln", "Euler\N{ACUTE ACCENT}sche Formeln"),
        ],
    )
    def test_spellings_of_one_title_share_a_key(self, left: str, right: str) -> None:
        assert title_key(left) == title_key(right)


class TestIdentityForm:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (KAHLER_COMPOSED, KAHLER_EXTRACTED),
            (KAHLER_COMPOSED, KAHLER_SPLIT_SPAN),
            ("De\N{LATIN SMALL LIGATURE FI}nitions", "Definitions"),
            ("O'Neill Spaces", "O\N{ACUTE ACCENT}Neill Spaces"),
            ("Hilbert’s Theorem", "Hilbert\N{ACUTE ACCENT}s Theorem"),
        ],
    )
    def test_artifact_spellings_share_a_form(self, left: str, right: str) -> None:
        assert identity_form(left) == identity_form(right)

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("Exercises", "EXERCISES"),
            ("Case x > 0", "Case x < 0"),
            ("Case x \N{NOT EQUAL TO} 0", "Case x = 0"),
            ("2.11 Exercises", "21.1 Exercises"),
            ("C++", "C#"),
        ],
    )
    def test_case_punctuation_and_symbols_stay_distinct(self, left: str, right: str) -> None:
        assert identity_form(left) != identity_form(right)
