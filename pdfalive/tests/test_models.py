"""Unit tests for data models."""

import pytest

from pdfalive.models.rename import FIRST_FILE_ID, RenameOp, RenameResult
from pdfalive.models.toc import TOC, PostprocessedTOCEntry, TOCEntry


class TestTOCEntry:
    """Tests for TOCEntry model."""

    @pytest.fixture
    def sample_entry(self):
        return TOCEntry(title="Chapter 1: Introduction", page_number=1, level=1, confidence=0.95)

    def test_to_list(self, sample_entry):
        """Test conversion to PyMuPDF-compatible list format."""
        result = sample_entry.to_list()

        assert result == [1, "Chapter 1: Introduction", 1]

    @pytest.mark.parametrize(
        "toc_list,expected_level,expected_title,expected_page",
        [
            ([1, "Chapter 1", 5], 1, "Chapter 1", 5),
            ([2, "Section 1.1", 10], 2, "Section 1.1", 10),
        ],
    )
    def test_from_list(self, toc_list, expected_level, expected_title, expected_page):
        """Test creation from PyMuPDF list format."""
        entry = TOCEntry.from_list(toc_list)

        assert entry.level == expected_level
        assert entry.title == expected_title
        assert entry.page_number == expected_page
        assert entry.confidence == 1.0

    def test_str_representation(self, sample_entry):
        """Test string representation."""
        result = str(sample_entry)

        assert "level=1" in result
        assert "Chapter 1: Introduction" in result
        assert "page_number=1" in result
        assert "confidence=0.95" in result


class TestTOC:
    """Tests for TOC model."""

    @pytest.fixture
    def sample_entries(self):
        return [
            TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.9),
            TOCEntry(title="Section 1.1", page_number=3, level=2, confidence=0.85),
            TOCEntry(title="Chapter 2", page_number=10, level=1, confidence=0.95),
        ]

    @pytest.fixture
    def sample_toc(self, sample_entries):
        return TOC(entries=sample_entries)

    def test_to_list(self, sample_toc):
        """Test conversion to PyMuPDF-compatible nested list format."""
        result = sample_toc.to_list()

        expected = [
            [1, "Chapter 1", 1],
            [2, "Section 1.1", 3],
            [1, "Chapter 2", 10],
        ]
        assert result == expected

    def test_empty_toc(self):
        """Test empty TOC."""
        toc = TOC(entries=[])

        assert toc.to_list() == []

    def test_merge_non_overlapping(self):
        """Test merging two TOCs with no overlapping entries."""
        toc1 = TOC(
            entries=[
                TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.9),
                TOCEntry(title="Chapter 2", page_number=10, level=1, confidence=0.85),
            ]
        )
        toc2 = TOC(
            entries=[
                TOCEntry(title="Chapter 3", page_number=20, level=1, confidence=0.9),
                TOCEntry(title="Chapter 4", page_number=30, level=1, confidence=0.95),
            ]
        )

        merged = TOC.merged([toc1, toc2])

        assert len(merged.entries) == 4
        assert merged.entries[0].title == "Chapter 1"
        assert merged.entries[1].title == "Chapter 2"
        assert merged.entries[2].title == "Chapter 3"
        assert merged.entries[3].title == "Chapter 4"

    def test_merge_with_duplicates_prefers_earlier(self):
        """Test that merging prefers entries from the first TOC when duplicates exist."""
        toc1 = TOC(
            entries=[
                TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.9),
                TOCEntry(title="Overlapping Chapter", page_number=10, level=1, confidence=0.85),
            ]
        )
        toc2 = TOC(
            entries=[
                TOCEntry(title="Overlapping Chapter", page_number=10, level=1, confidence=0.95),
                TOCEntry(title="Chapter 3", page_number=20, level=1, confidence=0.9),
            ]
        )

        merged = TOC.merged([toc1, toc2])

        assert len(merged.entries) == 3
        # The overlapping entry should have confidence from toc1 (0.85)
        overlapping = next(e for e in merged.entries if e.title == "Overlapping Chapter")
        assert overlapping.confidence == 0.85

    def test_merge_sorts_by_page_number(self):
        """Test that merged entries are sorted by page number."""
        toc1 = TOC(
            entries=[
                TOCEntry(title="Chapter 2", page_number=10, level=1, confidence=0.9),
            ]
        )
        toc2 = TOC(
            entries=[
                TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.9),
                TOCEntry(title="Chapter 3", page_number=20, level=1, confidence=0.9),
            ]
        )

        merged = TOC.merged([toc1, toc2])

        assert merged.entries[0].page_number == 1
        assert merged.entries[1].page_number == 10
        assert merged.entries[2].page_number == 20

    def test_merge_empty_tocs(self):
        """Test merging empty TOCs."""
        toc1 = TOC(entries=[])
        toc2 = TOC(entries=[])

        merged = TOC.merged([toc1, toc2])

        assert len(merged.entries) == 0

    def test_merge_into_empty_toc(self):
        """Test merging into an empty TOC."""
        toc1 = TOC(entries=[])
        toc2 = TOC(
            entries=[
                TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.9),
            ]
        )

        merged = TOC.merged([toc1, toc2])

        assert len(merged.entries) == 1
        assert merged.entries[0].title == "Chapter 1"

    def test_merge_with_different_levels_same_page(self):
        """Test merging entries with different levels on the same page."""
        toc1 = TOC(
            entries=[
                TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.9),
            ]
        )
        toc2 = TOC(
            entries=[
                TOCEntry(title="Section 1.1", page_number=1, level=2, confidence=0.85),
            ]
        )

        merged = TOC.merged([toc1, toc2])

        # Both should be kept since they have different titles
        assert len(merged.entries) == 2
        # Sorted by page_number, then level
        assert merged.entries[0].level == 1
        assert merged.entries[1].level == 2


class TestTOCIdentityDuplicates:
    """deduplicate() and merged() treat extraction-artifact spellings of one title on one page as one bookmark."""

    PAGE = 42
    OTHER_PAGE = 43
    LEVEL = 2
    FIRST_LEVEL, LATER_LEVEL = 1, 2
    FIRST_CONFIDENCE, LATER_CONFIDENCE = 0.9, 0.6

    # (clean spelling, extracted spelling) of one title.
    SAME_TITLE_SPELLINGS = [
        ("6.4 Kähler Potentials", "6.4 K\N{DIAERESIS}ahler Potentials"),
        ("3.1 Definitions and Notation", "3.1 De\N{LATIN SMALL LIGATURE FI}nitions and Notation"),
        ("5.1 Çetin Algebras", "5.1 C\N{CEDILLA}etin Algebras"),
        # An apostrophe extracted as an acute accent.
        ("8.2 O'Neill Spaces", "8.2 O\N{ACUTE ACCENT}Neill Spaces"),
    ]

    # Different bookmarks: numbering, case, punctuation, symbols, and diacritics
    # between two artifact-free titles.
    DISTINCT_TITLES = [
        ("2.11 Exercises", "21.1 Exercises"),
        ("Case x > 0", "Case x < 0"),
        ("Case x \N{NOT EQUAL TO} 0", "Case x = 0"),
        ("C++ Templates", "C# Templates"),
        ("A \N{UNION} B", "A \N{INTERSECTION} B"),
        ("Exercises", "EXERCISES"),
        ("Solving x\N{COMBINING DOT ABOVE} = ax", "Solving x\N{COMBINING DIAERESIS} = ax"),
        ("X\N{COMBINING MACRON} Statistics", "X Statistics"),
        ("Kähler Potentials", "Kahler Potentials"),
    ]

    @pytest.mark.parametrize(("clean", "extracted"), SAME_TITLE_SPELLINGS)
    @pytest.mark.parametrize("clean_first", [True, False])
    def test_deduplicate_keeps_clean_spelling(self, make_toc, clean: str, extracted: str, clean_first: bool) -> None:
        titles = (clean, extracted) if clean_first else (extracted, clean)
        toc = make_toc(*[(title, self.PAGE, self.LEVEL) for title in titles])

        assert [entry.title for entry in toc.deduplicate().entries] == [clean]

    @pytest.mark.parametrize(("clean", "extracted"), SAME_TITLE_SPELLINGS)
    def test_duplicate_keeps_first_entry_level_and_confidence(self, make_toc_entry, clean: str, extracted: str) -> None:
        first = make_toc_entry(
            title=extracted, page_number=self.PAGE, level=self.FIRST_LEVEL, confidence=self.FIRST_CONFIDENCE
        )
        later = make_toc_entry(
            title=clean, page_number=self.PAGE, level=self.LATER_LEVEL, confidence=self.LATER_CONFIDENCE
        )

        (kept,) = TOC(entries=[first, later]).deduplicate().entries

        assert (kept.title, kept.level, kept.confidence) == (clean, self.FIRST_LEVEL, self.FIRST_CONFIDENCE)

    @pytest.mark.parametrize(("clean", "extracted"), SAME_TITLE_SPELLINGS)
    def test_merged_prefers_earlier_batch_with_clean_title(self, make_toc_entry, clean: str, extracted: str) -> None:
        earlier = TOC(entries=[make_toc_entry(title=extracted, page_number=self.PAGE, level=self.FIRST_LEVEL)])
        later = TOC(entries=[make_toc_entry(title=clean, page_number=self.PAGE, level=self.LATER_LEVEL)])

        (kept,) = TOC.merged([earlier, later]).entries

        assert (kept.title, kept.level) == (clean, self.FIRST_LEVEL)

    def test_first_entry_kept_when_every_spelling_has_artifacts(self, make_toc) -> None:
        first, second = "K\N{DIAERESIS}ahler", "Ka\N{DIAERESIS}hler"
        toc = make_toc((first, self.PAGE, self.LEVEL), (second, self.PAGE, self.LEVEL))

        assert [entry.title for entry in toc.deduplicate().entries] == [first]

    @pytest.mark.parametrize(("first", "second"), DISTINCT_TITLES)
    def test_distinct_titles_are_kept(self, make_toc, first: str, second: str) -> None:
        toc = make_toc((first, self.PAGE, self.LEVEL), (second, self.PAGE, self.LEVEL))

        assert [entry.title for entry in toc.deduplicate().entries] == [first, second]

    @pytest.mark.parametrize(("clean", "extracted"), SAME_TITLE_SPELLINGS)
    def test_same_title_on_different_pages_is_kept(self, make_toc, clean: str, extracted: str) -> None:
        toc = make_toc((clean, self.PAGE, self.LEVEL), (extracted, self.OTHER_PAGE, self.LEVEL))

        assert len(toc.deduplicate().entries) == len(toc.entries)


class TestTOCSanitizeHierarchy:
    """Tests for TOC.sanitize_hierarchy() — ensures PyMuPDF set_toc() compatibility."""

    def _entry(self, title: str, page: int, level: int) -> TOCEntry:
        return TOCEntry(title=title, page_number=page, level=level, confidence=0.9)

    def test_empty_toc(self):
        """Empty TOC is returned unchanged."""
        toc = TOC(entries=[])
        assert toc.sanitize_hierarchy().entries == []

    def test_valid_hierarchy_unchanged(self):
        """A valid hierarchy passes through without modification."""
        toc = TOC(
            entries=[
                self._entry("Ch 1", 1, 1),
                self._entry("Sec 1.1", 2, 2),
                self._entry("Ch 2", 10, 1),
                self._entry("Sec 2.1", 11, 2),
                self._entry("Sub 2.1.1", 12, 3),
            ]
        )
        result = toc.sanitize_hierarchy()
        levels = [e.level for e in result.entries]
        assert levels == [1, 2, 1, 2, 3]

    def test_first_entry_promoted_to_level_1(self):
        """First entry with level > 1 is promoted to level 1."""
        toc = TOC(
            entries=[
                self._entry("Sec 1.1", 1, 2),
                self._entry("Ch 2", 10, 1),
            ]
        )
        result = toc.sanitize_hierarchy()
        assert result.entries[0].level == 1
        assert result.entries[1].level == 1

    def test_level_jump_clamped(self):
        """A jump from level 1 to level 3 is clamped to level 2."""
        toc = TOC(
            entries=[
                self._entry("Ch 1", 1, 1),
                self._entry("Deep sub", 2, 3),
            ]
        )
        result = toc.sanitize_hierarchy()
        assert result.entries[0].level == 1
        assert result.entries[1].level == 2

    def test_multiple_jumps_cascaded(self):
        """Multiple consecutive jumps are each clamped relative to previous."""
        toc = TOC(
            entries=[
                self._entry("Ch 1", 1, 1),
                self._entry("Sub 1", 2, 4),
                self._entry("Sub 2", 3, 5),
            ]
        )
        result = toc.sanitize_hierarchy()
        levels = [e.level for e in result.entries]
        assert levels == [1, 2, 3]

    def test_level_decrease_preserved(self):
        """Decreasing levels (e.g., level 3 -> level 1) are not clamped."""
        toc = TOC(
            entries=[
                self._entry("Ch 1", 1, 1),
                self._entry("Sec 1.1", 2, 2),
                self._entry("Sub 1.1.1", 3, 3),
                self._entry("Ch 2", 10, 1),
            ]
        )
        result = toc.sanitize_hierarchy()
        levels = [e.level for e in result.entries]
        assert levels == [1, 2, 3, 1]

    def test_all_level_2_entries(self):
        """All entries at level 2 — first gets promoted, rest clamped to max prev+1."""
        toc = TOC(
            entries=[
                self._entry("A", 1, 2),
                self._entry("B", 5, 2),
                self._entry("C", 10, 2),
            ]
        )
        result = toc.sanitize_hierarchy()
        levels = [e.level for e in result.entries]
        assert levels == [1, 2, 2]

    def test_preserves_other_fields(self):
        """Sanitization only changes level, not title/page/confidence."""
        entry = TOCEntry(title="My Section", page_number=42, level=3, confidence=0.75)
        toc = TOC(entries=[entry])
        result = toc.sanitize_hierarchy()
        sanitized = result.entries[0]
        assert sanitized.level == 1  # changed
        assert sanitized.title == "My Section"  # preserved
        assert sanitized.page_number == 42  # preserved
        assert sanitized.confidence == 0.75  # preserved

    @pytest.mark.parametrize(
        "input_levels,expected_levels",
        [
            ([1], [1]),
            ([2], [1]),
            ([3], [1]),
            ([1, 2], [1, 2]),
            ([1, 3], [1, 2]),
            ([2, 3], [1, 2]),
            ([1, 1, 1], [1, 1, 1]),
            ([1, 2, 3, 4], [1, 2, 3, 4]),
            ([1, 4, 2, 5], [1, 2, 2, 3]),
            ([3, 5, 1, 4], [1, 2, 1, 2]),
        ],
    )
    def test_parametrized_level_sequences(self, input_levels, expected_levels):
        """Parametrized tests for various level sequences."""
        entries = [self._entry(f"Entry {i}", i + 1, lvl) for i, lvl in enumerate(input_levels)]
        toc = TOC(entries=entries)
        result = toc.sanitize_hierarchy()
        actual = [e.level for e in result.entries]
        assert actual == expected_levels


class TestTOCSortByPage:
    """Tests for TOC.sort_by_page() — ensures page number monotonicity."""

    def _entry(self, title: str, page: int, level: int) -> TOCEntry:
        return TOCEntry(title=title, page_number=page, level=level, confidence=0.9)

    def test_empty_toc(self):
        """Empty TOC is returned unchanged."""
        toc = TOC(entries=[])
        assert toc.sort_by_page().entries == []

    def test_single_entry(self):
        """Single-entry TOC is returned unchanged."""
        toc = TOC(entries=[self._entry("Ch 1", 5, 1)])
        result = toc.sort_by_page()
        assert len(result.entries) == 1
        assert result.entries[0].page_number == 5

    def test_already_sorted(self):
        """Already-sorted entries are preserved in order."""
        toc = TOC(
            entries=[
                self._entry("Ch 1", 1, 1),
                self._entry("Sec 1.1", 3, 2),
                self._entry("Ch 2", 10, 1),
            ]
        )
        result = toc.sort_by_page()
        pages = [e.page_number for e in result.entries]
        assert pages == [1, 3, 10]

    def test_unsorted_entries(self):
        """Unsorted entries are reordered by page number."""
        toc = TOC(
            entries=[
                self._entry("Ch 2", 10, 1),
                self._entry("Ch 1", 1, 1),
                self._entry("Ch 3", 20, 1),
            ]
        )
        result = toc.sort_by_page()
        pages = [e.page_number for e in result.entries]
        assert pages == [1, 10, 20]

    def test_backward_jump_corrected(self):
        """Backward page jump (e.g., page 12 followed by page 10) is fixed by sorting."""
        toc = TOC(
            entries=[
                self._entry("1 Preliminaries", 12, 1),
                self._entry("2 Introduction to Ito-Calculus", 10, 1),
                self._entry("3 Advanced Topics", 30, 1),
            ]
        )
        result = toc.sort_by_page()
        pages = [e.page_number for e in result.entries]
        assert pages == [10, 12, 30]

    def test_ties_broken_by_level(self):
        """Entries on the same page are ordered by level (parent before child)."""
        toc = TOC(
            entries=[
                self._entry("Sec 1.1", 5, 2),
                self._entry("Ch 1", 5, 1),
                self._entry("Sub 1.1.1", 5, 3),
            ]
        )
        result = toc.sort_by_page()
        levels = [e.level for e in result.entries]
        assert levels == [1, 2, 3]

    def test_preserves_other_fields(self):
        """Sorting only changes order, not entry contents."""
        entry = TOCEntry(title="My Section", page_number=42, level=2, confidence=0.75)
        toc = TOC(entries=[self._entry("Ch 1", 50, 1), entry])
        result = toc.sort_by_page()
        assert result.entries[0].title == "My Section"
        assert result.entries[0].confidence == 0.75

    @pytest.mark.parametrize(
        "input_pages,expected_pages",
        [
            ([1, 2, 3], [1, 2, 3]),
            ([3, 1, 2], [1, 2, 3]),
            ([5, 5, 5], [5, 5, 5]),
            ([10, 5, 20, 1], [1, 5, 10, 20]),
        ],
    )
    def test_parametrized_page_sequences(self, input_pages, expected_pages):
        """Parametrized tests for various page sequences."""
        entries = [self._entry(f"Entry {i}", p, 1) for i, p in enumerate(input_pages)]
        toc = TOC(entries=entries)
        result = toc.sort_by_page()
        actual = [e.page_number for e in result.entries]
        assert actual == expected_pages


class TestTOCPartitionByPageRange:
    """Tests for TOC.partition_by_page_range() — PyMuPDF set_toc() rejects pages outside the document."""

    PAGE_COUNT = 240

    @pytest.mark.parametrize(
        "input_pages,expected_kept_pages,expected_dropped_pages",
        [
            ([], [], []),
            ([1, 120, PAGE_COUNT], [1, 120, PAGE_COUNT], []),
            ([1, PAGE_COUNT + 1, PAGE_COUNT + 13], [1], [PAGE_COUNT + 1, PAGE_COUNT + 13]),
            ([0, -1, 5], [5], [0, -1]),
            ([PAGE_COUNT + 1], [], [PAGE_COUNT + 1]),
        ],
    )
    def test_splits_entries_by_page_range(self, make_toc, input_pages, expected_kept_pages, expected_dropped_pages):
        """Entries on pages 1..page_count are kept; all others are returned as dropped."""
        toc = make_toc(*[(f"Entry {i}", page, 1) for i, page in enumerate(input_pages)])

        kept, dropped = toc.partition_by_page_range(self.PAGE_COUNT)

        assert [e.page_number for e in kept.entries] == expected_kept_pages
        assert [e.page_number for e in dropped] == expected_dropped_pages

    def test_preserves_entry_order_and_fields(self, make_toc_entry):
        """Partitioning neither reorders nor modifies entries."""
        entries = [
            make_toc_entry(title="Chapter 30", page_number=214, level=1, confidence=0.8),
            make_toc_entry(title="Index", page_number=self.PAGE_COUNT + 13, level=1, confidence=0.7),
            make_toc_entry(title="30.1 Polynomial", page_number=210, level=2, confidence=0.6),
        ]

        kept, dropped = TOC(entries=entries).partition_by_page_range(self.PAGE_COUNT)

        assert kept.entries == [entries[0], entries[2]]
        assert dropped == [entries[1]]


class TestPostprocessedTOCEntry:
    """Tests for PostprocessedTOCEntry, the postprocessor's entry schema."""

    @pytest.fixture
    def entry_fields(self):
        return {"title": "1.2 Notation", "page_number": 15, "level": 2, "confidence": 0.8}

    @pytest.mark.parametrize(
        "printed_page,expected",
        [
            pytest.param(12, 12, id="arabic-int"),
            pytest.param(" 12 ", 12, id="arabic-string"),
            pytest.param("xii", None, id="roman-numeral-is-absent"),
            pytest.param("١٢", None, id="non-ascii-digits-are-absent"),
            pytest.param("²", None, id="superscript-is-absent"),
            pytest.param(12.0, 12, id="integral-float"),
            pytest.param(12.5, None, id="fraction-is-absent"),
            pytest.param(True, None, id="bool-is-absent"),
            pytest.param("", None, id="empty-is-absent"),
            pytest.param(None, None, id="absent"),
        ],
    )
    def test_printed_page_accepts_only_arabic_numbers(self, entry_fields, printed_page, expected):
        assert PostprocessedTOCEntry(**entry_fields, printed_page=printed_page).printed_page == expected

    def test_to_toc_entry_keeps_every_field_but_the_printed_page(self, entry_fields):
        new_page = entry_fields["page_number"] + 3

        toc_entry = PostprocessedTOCEntry(**entry_fields, printed_page=4).to_toc_entry(page_number=new_page)

        assert type(toc_entry) is TOCEntry
        assert toc_entry.model_dump() == entry_fields | {"page_number": new_page}


class TestRenameOp:
    """Tests for RenameOp model."""

    FILE_ID = 1
    INPUT_NAME = "old_file.pdf"
    OUTPUT_NAME = "new_file.pdf"
    CONFIDENCE = 0.85

    @pytest.fixture
    def sample_op(self, make_rename_op):
        return make_rename_op(self.FILE_ID, self.INPUT_NAME, self.OUTPUT_NAME, confidence=self.CONFIDENCE)

    def test_basic_creation(self, sample_op):
        assert (sample_op.file_id, sample_op.input_filename, sample_op.output_filename, sample_op.confidence) == (
            self.FILE_ID,
            self.INPUT_NAME,
            self.OUTPUT_NAME,
            self.CONFIDENCE,
        )

    def test_str_representation(self, sample_op):
        result = str(sample_op)

        for part in (f"#{self.FILE_ID}", self.INPUT_NAME, self.OUTPUT_NAME, str(self.CONFIDENCE)):
            assert part in result

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_valid_confidence_values(self, make_rename_op, confidence):
        op = make_rename_op(self.FILE_ID, self.INPUT_NAME, self.OUTPUT_NAME, confidence=confidence)

        assert op.confidence == confidence

    def test_default_reasoning(self):
        """Test that reasoning defaults to empty string."""
        op = RenameOp(
            file_id=self.FILE_ID,
            input_filename=self.INPUT_NAME,
            output_filename=self.OUTPUT_NAME,
            confidence=self.CONFIDENCE,
        )

        assert op.reasoning == ""


class TestRenameResult:
    """Tests for RenameResult model."""

    @pytest.fixture
    def sample_operations(self, make_rename_op):
        return [
            make_rename_op(file_id, f"file{file_id}.pdf", f"renamed{file_id}.pdf")
            for file_id in range(FIRST_FILE_ID, FIRST_FILE_ID + 2)
        ]

    def test_len(self, sample_operations):
        assert len(RenameResult(operations=sample_operations)) == len(sample_operations)

    def test_empty_result(self):
        result = RenameResult()

        assert len(result) == 0
        assert result.operations == []
