"""Tests for TOC evaluation metrics."""

from collections.abc import Callable

import pytest

from pdfalive.evaluation.metrics import (
    GoldenEntry,
    evaluate_toc,
    normalize_title,
    title_similarity,
)
from pdfalive.models.toc import TOC, TOCEntry


SIMILARITY_THRESHOLD = 0.8

CHAPTER_1 = ("Chapter 1: Distance and Angles", 14, 1)
SECTION_1_1 = ("1, §1. Lines", 14, 2)
SECTION_1_2 = ("1, §2. Distance", 21, 2)
CHAPTER_2 = ("Chapter 2: Coordinates", 78, 1)
INDEX = ("Index", 404, 1)


class TestNormalizeTitle:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Chapter 1: Distance and Angles", "chapter 1 distance and angles"),
            ("  1, §1.  Lines ", "1 1 lines"),
            ("INDEX", "index"),
            ("Rates of Change", "rates of change"),
            ("", ""),
        ],
    )
    def test_normalizes_case_punctuation_and_whitespace(self, raw: str, expected: str) -> None:
        assert normalize_title(raw) == expected


class TestTitleSimilarity:
    @pytest.mark.parametrize(
        ("left", "right", "minimum"),
        [
            ("Chapter 1: Distance and Angles", "chapter 1 distance and angles", 1.0),
            ("Chapter 1: Distance and Angles", "Chapter 1: Distence and Angles", 0.9),  # typo
            ("Isometries", "Isometries as Compositions of Reflections", 0.0),
        ],
    )
    def test_similarity_lower_bounds(self, left: str, right: str, minimum: float) -> None:
        assert title_similarity(left, right) >= minimum

    def test_unrelated_titles_score_below_threshold(self) -> None:
        assert title_similarity("Chapter 1: Distance and Angles", "Index") < SIMILARITY_THRESHOLD

    def test_is_symmetric(self) -> None:
        left, right = "Chapter 2: Coordinates", "Chapter 2 Coordinates."
        assert title_similarity(left, right) == title_similarity(right, left)


class TestEvaluateToc:
    @pytest.fixture
    def golden(self, make_golden_entry: Callable[..., GoldenEntry]) -> list[GoldenEntry]:
        return [
            make_golden_entry(title=t, page_number=p, level=lvl)
            for t, p, lvl in [CHAPTER_1, SECTION_1_1, SECTION_1_2, CHAPTER_2, INDEX]
        ]

    def test_identical_tocs_score_perfectly(self, golden: list[GoldenEntry], make_toc: Callable[..., TOC]) -> None:
        generated = make_toc(CHAPTER_1, SECTION_1_1, SECTION_1_2, CHAPTER_2, INDEX)

        report = evaluate_toc(golden, generated)

        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0
        assert report.page_accuracy() == 1.0
        assert report.level_accuracy == 1.0
        assert not report.missing
        assert not report.spurious

    def test_missing_entry_reduces_recall_and_is_reported(
        self, golden: list[GoldenEntry], make_toc: Callable[..., TOC]
    ) -> None:
        generated = make_toc(CHAPTER_1, SECTION_1_1, SECTION_1_2, CHAPTER_2)  # Index missing

        report = evaluate_toc(golden, generated)

        assert report.precision == 1.0
        assert report.recall == pytest.approx(4 / 5)
        assert [entry.title for entry in report.missing] == ["Index"]

    def test_spurious_entry_reduces_precision_and_is_reported(
        self, golden: list[GoldenEntry], make_toc: Callable[..., TOC]
    ) -> None:
        running_header = ("DISTANCE AND ANGLES [1, S1]", 16, 1)
        generated = make_toc(CHAPTER_1, SECTION_1_1, SECTION_1_2, CHAPTER_2, INDEX, running_header)

        report = evaluate_toc(golden, generated)

        assert report.recall == 1.0
        assert report.precision == pytest.approx(5 / 6)
        assert [entry.title for entry in report.spurious] == ["DISTANCE AND ANGLES [1, S1]"]

    def test_typo_in_title_still_matches(self, golden: list[GoldenEntry], make_toc: Callable[..., TOC]) -> None:
        typoed = ("Chapter 1: Distence and Angles", 14, 1)
        generated = make_toc(typoed, SECTION_1_1, SECTION_1_2, CHAPTER_2, INDEX)

        report = evaluate_toc(golden, generated)

        assert report.recall == 1.0
        assert report.precision == 1.0

    def test_page_off_by_one_matches_but_lowers_exact_page_accuracy(
        self, golden: list[GoldenEntry], make_toc: Callable[..., TOC]
    ) -> None:
        shifted_index = ("Index", 405, 1)
        generated = make_toc(CHAPTER_1, SECTION_1_1, SECTION_1_2, CHAPTER_2, shifted_index)

        report = evaluate_toc(golden, generated)

        assert report.recall == 1.0
        assert report.page_accuracy() == pytest.approx(4 / 5)
        assert report.page_accuracy(tolerance=1) == 1.0

    def test_duplicate_titles_match_by_page_proximity(
        self, make_golden_entry: Callable[..., GoldenEntry], make_toc: Callable[..., TOC]
    ) -> None:
        golden = [
            make_golden_entry(title="Exercises", page_number=20, level=2),
            make_golden_entry(title="Exercises", page_number=90, level=2),
        ]
        generated = make_toc(("Exercises", 91, 2), ("Exercises", 20, 2))

        report = evaluate_toc(golden, generated)

        assert report.recall == 1.0
        # The page-91 generated entry must pair with the page-90 golden entry,
        # leaving the page-20 pair exact.
        assert report.page_accuracy() == pytest.approx(1 / 2)
        assert report.page_accuracy(tolerance=1) == 1.0

    def test_level_mismatch_lowers_level_accuracy(
        self, golden: list[GoldenEntry], make_toc: Callable[..., TOC]
    ) -> None:
        demoted_chapter_2 = ("Chapter 2: Coordinates", 78, 2)
        generated = make_toc(CHAPTER_1, SECTION_1_1, SECTION_1_2, demoted_chapter_2, INDEX)

        report = evaluate_toc(golden, generated)

        assert report.level_accuracy == pytest.approx(4 / 5)

    def test_empty_generated_toc_scores_zero(self, golden: list[GoldenEntry]) -> None:
        report = evaluate_toc(golden, TOC(entries=[]))

        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.f1 == 0.0
        assert len(report.missing) == len(golden)

    def test_both_empty_scores_perfectly(self) -> None:
        report = evaluate_toc([], TOC(entries=[]))

        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0

    def test_summary_dict_contains_all_metrics(self, golden: list[GoldenEntry], make_toc: Callable[..., TOC]) -> None:
        generated = make_toc(CHAPTER_1, SECTION_1_1, SECTION_1_2, CHAPTER_2, INDEX)

        summary = evaluate_toc(golden, generated).summary()

        expected_keys = {
            "golden_entries",
            "generated_entries",
            "matched",
            "precision",
            "recall",
            "f1",
            "page_accuracy_exact",
            "page_accuracy_within_1",
            "level_accuracy",
        }
        assert expected_keys <= summary.keys()


class TestGoldenEntry:
    def test_loads_from_golden_json_shape(self) -> None:
        entry = GoldenEntry.model_validate({"level": 2, "title": "1, §1. Lines", "page_number": 14})

        assert entry.title == "1, §1. Lines"
        assert entry.page_number == 14
        assert entry.level == 2

    def test_from_toc_entry(self, make_toc_entry: Callable[..., TOCEntry]) -> None:
        toc_entry = make_toc_entry(title="Index", page_number=404, level=1)

        golden = GoldenEntry.from_toc_entry(toc_entry)

        assert golden == GoldenEntry(title="Index", page_number=404, level=1)


class TestZeroMatchAccuracies:
    def test_page_and_level_accuracy_are_zero_when_nothing_matched(
        self, make_golden_entry: Callable[..., GoldenEntry], make_toc: Callable[..., TOC]
    ) -> None:
        """A case with golden entries but no matches must not report perfect accuracies."""
        golden = [make_golden_entry(title="Chapter 1: Distance and Angles")]
        generated = make_toc(("Completely Unrelated Heading", 99, 1))

        report = evaluate_toc(golden, generated)

        assert report.page_accuracy() == 0.0
        assert report.page_accuracy(tolerance=1) == 0.0
        assert report.level_accuracy == 0.0
