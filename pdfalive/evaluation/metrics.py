"""Pure functions for scoring a generated TOC against a golden (ground truth) TOC.

All functions here are deterministic and side-effect free so they can be unit
tested directly and reused by any evaluation runner.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from pdfalive.models.toc import TOC, TOCEntry


# Minimum normalized-title similarity for a golden/generated pair to count as a match.
DEFAULT_SIMILARITY_THRESHOLD = 0.8

# Page tolerance used by the within-tolerance accuracy in summaries and CLI gates.
SUMMARY_PAGE_TOLERANCE = 1

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Roman numerals i through xxxix, restricted to i/v/x so ordinary words made of
# roman letters ("mix", "did", "civil") are never mistaken for a numeral.
_ROMAN_NUMERAL_TOKEN = r"(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})"

# One leading section-numbering token of an already-normalized title: a naming
# keyword (optionally carrying a single-letter label, as in "appendix a"), an
# arabic number, or a roman numeral.
_SECTION_PREFIX_TOKEN_PATTERN = re.compile(
    rf"^(?:(?:chapter|section|part|appendix|lecture)(?:\s+[a-z]\b)?|\d+|{_ROMAN_NUMERAL_TOKEN})(?:\s+|$)"
)

# A pair that only matches once section numbering is stripped scores at most
# this, so an exact full-title match always outranks it during greedy pairing.
_SECTION_STRIPPED_MATCH_WEIGHT = 0.95


class GoldenEntry(BaseModel):
    """A single ground-truth TOC entry from a golden data file."""

    title: str = Field(description="Expected title of the TOC entry")
    page_number: int = Field(description="Expected 1-indexed PDF page number")
    level: int = Field(description="Expected hierarchy level (1 = top)")

    @classmethod
    def from_toc_entry(cls, entry: TOCEntry) -> "GoldenEntry":
        """Create a golden entry from a generated TOCEntry (drops confidence)."""
        return cls(title=entry.title, page_number=entry.page_number, level=entry.level)


def normalize_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, no punctuation, collapsed whitespace."""
    title = _PUNCTUATION_PATTERN.sub(" ", title.lower())
    return _WHITESPACE_PATTERN.sub(" ", title).strip()


def strip_section_prefix(normalized_title: str) -> str:
    """Strip leading section numbering from an already-normalized title.

    "chapter 1 distance and angles" and "1 1 exercises" become "distance and
    angles" and "exercises". A title that is nothing but numbering is returned
    unchanged rather than emptied.
    """
    stripped = normalized_title
    while True:
        candidate = _SECTION_PREFIX_TOKEN_PATTERN.sub("", stripped, count=1)
        if candidate == stripped or not candidate:
            break
        stripped = candidate
    return stripped


def _normalized_similarity(left_norm: str, right_norm: str) -> float:
    """Similarity ratio in [0, 1] between two already-normalized titles."""
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _matching_similarity(left: "_TitleForms", right: "_TitleForms") -> float:
    """Similarity used for pairing, tolerant of dropped section numbering.

    Models disagree on whether a heading's printed number belongs in the title
    ("Chapter 1: Distance and Angles" vs "Distance and Angles"). Scoring only
    full titles charges that one stylistic choice twice — once as a missing
    golden entry and once as a spurious generated one — so a pair that agrees
    after stripping the numbering still matches, at a slightly discounted
    score that keeps numbered titles pairing with their own counterparts.
    """
    full = _normalized_similarity(left.normalized, right.normalized)
    if full == 1.0:
        return full
    stripped = _normalized_similarity(left.section_stripped, right.section_stripped)
    return max(full, stripped * _SECTION_STRIPPED_MATCH_WEIGHT)


@dataclass(frozen=True)
class _TitleForms:
    """The comparison forms of one title, computed once per entry."""

    normalized: str
    section_stripped: str

    @classmethod
    def of(cls, title: str) -> "_TitleForms":
        normalized = normalize_title(title)
        return cls(normalized=normalized, section_stripped=strip_section_prefix(normalized))


def title_similarity(left: str, right: str) -> float:
    """Similarity in [0, 1] between two titles, as used for golden matching."""
    return _matching_similarity(_TitleForms.of(left), _TitleForms.of(right))


@dataclass(frozen=True)
class MatchedPair:
    """A golden entry paired with the generated entry that matched it."""

    golden: GoldenEntry
    generated: TOCEntry
    similarity: float

    @property
    def page_delta(self) -> int:
        return abs(self.golden.page_number - self.generated.page_number)

    @property
    def level_matches(self) -> bool:
        return self.golden.level == self.generated.level


@dataclass
class EvalReport:
    """Scoring result of a generated TOC against golden entries."""

    matched: list[MatchedPair]
    missing: list[GoldenEntry]
    spurious: list[TOCEntry]

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        # An empty denominator means there was nothing to get wrong.
        if denominator == 0:
            return 1.0
        return numerator / denominator

    @property
    def golden_count(self) -> int:
        return len(self.matched) + len(self.missing)

    @property
    def generated_count(self) -> int:
        return len(self.matched) + len(self.spurious)

    @property
    def precision(self) -> float:
        if self.generated_count == 0:
            # No output: perfect only when nothing was expected either.
            return 1.0 if self.golden_count == 0 else 0.0
        return self._ratio(len(self.matched), self.generated_count)

    @property
    def recall(self) -> float:
        return self._ratio(len(self.matched), self.golden_count)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def _matched_fraction(self, agreeing: int) -> float:
        if not self.matched:
            # No matches: perfect only for the truly empty case. Otherwise these
            # accuracies must not read 1.0 (a threshold gate would pass a case
            # where the pipeline matched nothing at all).
            return 1.0 if self.golden_count == 0 and self.generated_count == 0 else 0.0
        return agreeing / len(self.matched)

    def page_accuracy(self, tolerance: int = 0) -> float:
        """Fraction of matched pairs whose page numbers agree within tolerance."""
        return self._matched_fraction(sum(1 for pair in self.matched if pair.page_delta <= tolerance))

    @property
    def level_accuracy(self) -> float:
        """Fraction of matched pairs whose hierarchy levels agree."""
        return self._matched_fraction(sum(1 for pair in self.matched if pair.level_matches))

    def summary(self) -> dict[str, float | int]:
        """Flat metric dictionary, suitable for tabular display or JSON output."""
        return {
            "golden_entries": self.golden_count,
            "generated_entries": self.generated_count,
            "matched": len(self.matched),
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "page_accuracy_exact": self.page_accuracy(),
            "page_accuracy_within_1": self.page_accuracy(tolerance=SUMMARY_PAGE_TOLERANCE),
            "level_accuracy": self.level_accuracy,
        }


def evaluate_toc(
    golden: list[GoldenEntry],
    generated: TOC,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> EvalReport:
    """Match generated TOC entries to golden entries and score the result.

    Matching is greedy: all candidate pairs above the similarity threshold are
    ranked by (similarity desc, page distance asc) and consumed so that each
    golden and each generated entry is used at most once. Page distance is the
    tiebreaker so repeated titles (e.g. "Exercises" once per chapter) pair with
    the occurrence closest to their expected page.

    Args:
        golden: Ground-truth entries.
        generated: The TOC produced by the generation pipeline.
        similarity_threshold: Minimum normalized-title similarity for a match.

    Returns:
        An EvalReport with matched pairs, missing golden entries, and spurious
        generated entries.
    """
    # Normalize once per entry, not once per pair.
    golden_forms = [_TitleForms.of(entry.title) for entry in golden]
    generated_forms = [_TitleForms.of(entry.title) for entry in generated.entries]

    # Candidates sort naturally: similarity stored negated so plain tuple order
    # ranks by (similarity desc, page_delta asc, stable indices).
    candidates: list[tuple[float, int, int, int]] = []  # (-similarity, page_delta, golden_idx, generated_idx)
    for golden_idx, golden_entry in enumerate(golden):
        for generated_idx, generated_entry in enumerate(generated.entries):
            similarity = _matching_similarity(golden_forms[golden_idx], generated_forms[generated_idx])
            if similarity >= similarity_threshold:
                page_delta = abs(golden_entry.page_number - generated_entry.page_number)
                candidates.append((-similarity, page_delta, golden_idx, generated_idx))

    candidates.sort()

    matched: list[MatchedPair] = []
    used_golden: set[int] = set()
    used_generated: set[int] = set()
    for negated_similarity, _, golden_idx, generated_idx in candidates:
        if golden_idx in used_golden or generated_idx in used_generated:
            continue
        used_golden.add(golden_idx)
        used_generated.add(generated_idx)
        matched.append(
            MatchedPair(
                golden=golden[golden_idx],
                generated=generated.entries[generated_idx],
                similarity=-negated_similarity,
            )
        )

    missing = [entry for idx, entry in enumerate(golden) if idx not in used_golden]
    spurious = [entry for idx, entry in enumerate(generated.entries) if idx not in used_generated]

    return EvalReport(matched=matched, missing=missing, spurious=spurious)
