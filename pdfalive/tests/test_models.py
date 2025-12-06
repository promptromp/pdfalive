"""Unit tests for data models."""

import pytest

from pdfalive.models.toc import TOC, TOCEntry


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

        merged = toc1.merge(toc2)

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

        merged = toc1.merge(toc2)

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

        merged = toc1.merge(toc2)

        assert merged.entries[0].page_number == 1
        assert merged.entries[1].page_number == 10
        assert merged.entries[2].page_number == 20

    def test_merge_empty_tocs(self):
        """Test merging empty TOCs."""
        toc1 = TOC(entries=[])
        toc2 = TOC(entries=[])

        merged = toc1.merge(toc2)

        assert len(merged.entries) == 0

    def test_merge_into_empty_toc(self):
        """Test merging into an empty TOC."""
        toc1 = TOC(entries=[])
        toc2 = TOC(
            entries=[
                TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.9),
            ]
        )

        merged = toc1.merge(toc2)

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

        merged = toc1.merge(toc2)

        # Both should be kept since they have different titles
        assert len(merged.entries) == 2
        # Sorted by page_number, then level
        assert merged.entries[0].level == 1
        assert merged.entries[1].level == 2
