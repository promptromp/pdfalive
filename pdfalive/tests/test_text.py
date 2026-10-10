"""Tests for the comparison forms that neutralize PDF text-extraction artifacts."""

import pytest

from pdfalive.text import (
    APOSTROPHE_PATTERN,
    PUNCTUATION_EXCEPT_DOTS_PATTERN,
    PUNCTUATION_OUTSIDE_NUMBERS_PATTERN,
    PUNCTUATION_PATTERN,
    artifact_neutral_form,
    has_extraction_artifacts,
    matching_form,
    repair_extracted_text,
    strip_punctuation,
    title_key,
)


# One heading spelled the ways PDF extraction and the LLM produce it.
KAHLER_SPELLINGS = (
    "Kähler Potentials",
    "Ka\N{COMBINING DIAERESIS}hler Potentials",
    "K\N{DIAERESIS}ahler Potentials",
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
    ("Probl\N{GRAVE ACCENT}emes", "Problèmes"),
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
    # TeX's dotless i under an accent.
    ("na\N{DIAERESIS}\N{LATIN SMALL LETTER DOTLESS I}ve", "naïve"),
    # Greek spacing accents that canonically decompose to Latin ones.
    ("a\N{GREEK OXIA}b", "ab"),
    ("a\N{GREEK VARIA}b", "ab"),
]

# Pairs that must stay distinct: compatibility forms, non-Latin marks and symbols keep their identity.
DISTINCT_PAIRS = [
    ("S\N{SUPERSCRIPT TWO}", "S2"),
    ("H\N{SUPERSCRIPT ONE}", "H\N{SUBSCRIPT ONE}"),
    ("\N{HIRAGANA LETTER GA}", "\N{HIRAGANA LETTER KA}"),
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
            ("Probl`emes", True),
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
            ("2.11: exercises.", PUNCTUATION_OUTSIDE_NUMBERS_PATTERN, "", "2.11 exercises"),
            ("o'neill and hilbert’s", APOSTROPHE_PATTERN, "", "oneill and hilberts"),
        ],
    )
    def test_applies_policy_and_collapses_whitespace(self, text: str, pattern, replacement: str, expected: str) -> None:
        assert strip_punctuation(text, pattern, replacement) == expected


class TestTitleKey:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("6.4 Kähler Potentials", "6.4 K\N{DIAERESIS}ahler Potentials"),
            ("6.4 Kahler Potentials", "6.4 K\N{DIAERESIS}ahler Potentials"),
            ("Chapter 3: Definitions", "CHAPTER 3 De\N{LATIN SMALL LIGATURE FI}nitions"),
            ("O'Neill Spaces", "O\N{ACUTE ACCENT}Neill Spaces"),
            ("d'Alembert Operators", "d\N{ACUTE ACCENT}Alembert Operators"),
            ("Euler'sche Formeln", "Euler\N{ACUTE ACCENT}sche Formeln"),
        ],
    )
    def test_spellings_of_one_title_share_a_key(self, left: str, right: str) -> None:
        assert title_key(left) == title_key(right)

    @pytest.mark.parametrize(("left", "right"), [("2.11 Exercises", "21.1 Exercises"), ("1.10 Notes", "11.0 Notes")])
    def test_numbering_stays_distinct(self, left: str, right: str) -> None:
        assert title_key(left) != title_key(right)


# (extracted title, title as the document prints it)
DISPLAY_REPAIRS = [
    ("De\N{LATIN SMALL LIGATURE FI}nitions and Notation", "Definitions and Notation"),
    ("A\N{LATIN SMALL LIGATURE FFI}ne Maps", "Affine Maps"),
    ("Fa\N{LATIN SMALL LIGATURE ST} Decay", "Fast Decay"),
    ("6.4 K\N{DIAERESIS}ahler Potentials", "6.4 Kähler Potentials"),
    ("Poincar\N{ACUTE ACCENT}e Maps", "Poincaré Maps"),
    ("POINCAR\N{ACUTE ACCENT}E MAPS", "POINCARÉ MAPS"),
    ("\N{ACUTE ACCENT}Etale Covers", "Étale Covers"),
    ("\N{CARON}Cech Cohomology", "Čech Cohomology"),
    ("Erd\N{DOUBLE ACUTE ACCENT}os Numbers", "Erdős Numbers"),
    ("\N{RING ABOVE}Angstr\N{DIAERESIS}om", "Ångström"),
    ("H\N{MODIFIER LETTER CIRCUMFLEX ACCENT}opital", "Hôpital"),
    ("Probl`emes Ouverts", "Problèmes Ouverts"),
    ("`a tous", "à tous"),
    # A cedilla attaches to whichever neighbor commonly carries one.
    ("Fran\N{CEDILLA}cois", "François"),
    ("C\N{CEDILLA}etin Algebras", "Çetin Algebras"),
    ("Aperc\N{CEDILLA}u", "Aperçu"),
    # Stacked accents, outermost first.
    ("Vi\N{ACUTE ACCENT}\N{MODIFIER LETTER CIRCUMFLEX ACCENT}et Lattices", "Viết Lattices"),
    # TeX's dotless i under an accent.
    ("na\N{DIAERESIS}\N{LATIN SMALL LETTER DOTLESS I}ve", "naïve"),
    ("Mart\N{ACUTE ACCENT}\N{LATIN SMALL LETTER DOTLESS I}nez", "Martínez"),
    # An acute or grave in an apostrophe position is an apostrophe.
    ("Hilbert\N{ACUTE ACCENT}s Theorem", "Hilbert's Theorem"),
    ("Euler\N{ACUTE ACCENT}sche Formeln", "Euler'sche Formeln"),
    ("d\N{ACUTE ACCENT}Alembert Operators", "d'Alembert Operators"),
    ("Why It Doesn`t Converge", "Why It Doesn't Converge"),
    # Decomposed accents are composed.
    ("Ka\N{COMBINING DIAERESIS}hler", "Kähler"),
]

# Titles the repair leaves as they are: ambiguous artifacts, notation and clean text.
DISPLAY_UNCHANGED = [
    "O\N{ACUTE ACCENT}Neill Spaces",
    "`apply` Basics",
    "Mie\N{CEDILLA}kisz",
    "\N{MACRON}B Operators",
    "L^p Spaces",
    "x ~ y",
    "S\N{SUPERSCRIPT TWO} and H\N{SUBSCRIPT ONE}",
    "I\N{LATIN SMALL LETTER S WITH CEDILLA}\N{LATIN SMALL LETTER DOTLESS I}k",
    "CHAPTER 3: Overview",
]


class TestRepairExtractedText:
    @pytest.mark.parametrize(("extracted", "printed"), DISPLAY_REPAIRS)
    def test_repairs_artifacts(self, extracted: str, printed: str) -> None:
        assert repair_extracted_text(extracted) == printed

    @pytest.mark.parametrize("title", DISPLAY_UNCHANGED)
    def test_leaves_ambiguous_and_clean_text(self, title: str) -> None:
        assert repair_extracted_text(title) == title

    @pytest.mark.parametrize(
        "title", [extracted for extracted, _ in DISPLAY_REPAIRS] + DISPLAY_UNCHANGED + ["a\N{GREEK OXIA}b"]
    )
    def test_is_idempotent(self, title: str) -> None:
        assert repair_extracted_text(repair_extracted_text(title)) == repair_extracted_text(title)

    @pytest.mark.parametrize(("extracted", "printed"), DISPLAY_REPAIRS)
    def test_repaired_title_still_matches(self, extracted: str, printed: str) -> None:
        assert title_key(repair_extracted_text(extracted)) == title_key(extracted)
