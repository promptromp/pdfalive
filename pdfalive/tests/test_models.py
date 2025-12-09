"""Unit tests for data models."""

import pytest

from pdfalive.models.rename import RenameOp, RenameResult
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


class TestRenameOp:
    """Tests for RenameOp model."""

    @pytest.fixture
    def sample_op(self):
        return RenameOp(
            input_filename="old_file.pdf",
            output_filename="new_file.pdf",
            confidence=0.85,
            reasoning="Applied naming convention",
        )

    def test_basic_creation(self, sample_op):
        """Test basic model creation."""
        assert sample_op.input_filename == "old_file.pdf"
        assert sample_op.output_filename == "new_file.pdf"
        assert sample_op.confidence == 0.85
        assert sample_op.reasoning == "Applied naming convention"

    def test_str_representation(self, sample_op):
        """Test string representation."""
        result = str(sample_op)

        assert "old_file.pdf" in result
        assert "new_file.pdf" in result
        assert "0.85" in result

    @pytest.mark.parametrize(
        "confidence",
        [0.0, 0.5, 1.0],
    )
    def test_valid_confidence_values(self, confidence):
        """Test valid confidence values."""
        op = RenameOp(
            input_filename="a.pdf",
            output_filename="b.pdf",
            confidence=confidence,
            reasoning="test",
        )
        assert op.confidence == confidence

    def test_default_reasoning(self):
        """Test that reasoning defaults to empty string."""
        op = RenameOp(
            input_filename="a.pdf",
            output_filename="b.pdf",
            confidence=0.9,
        )
        assert op.reasoning == ""


class TestRenameResult:
    """Tests for RenameResult model."""

    @pytest.fixture
    def sample_operations(self):
        return [
            RenameOp(
                input_filename="file1.pdf",
                output_filename="renamed1.pdf",
                confidence=0.9,
                reasoning="Test 1",
            ),
            RenameOp(
                input_filename="file2.pdf",
                output_filename="renamed2.pdf",
                confidence=0.8,
                reasoning="Test 2",
            ),
        ]

    @pytest.fixture
    def sample_result(self, sample_operations):
        return RenameResult(operations=sample_operations)

    def test_len(self, sample_result):
        """Test __len__ method."""
        assert len(sample_result) == 2

    def test_empty_result(self):
        """Test empty result."""
        result = RenameResult()
        assert len(result) == 0
        assert result.operations == []

    def test_operations_list(self, sample_result):
        """Test accessing operations list."""
        assert len(sample_result.operations) == 2
        assert sample_result.operations[0].input_filename == "file1.pdf"
        assert sample_result.operations[1].input_filename == "file2.pdf"

    def test_merge_non_overlapping(self):
        """Test merging two RenameResults with no overlapping filenames."""
        result1 = RenameResult(
            operations=[
                RenameOp(input_filename="file1.pdf", output_filename="new1.pdf", confidence=0.9, reasoning="Test 1"),
                RenameOp(input_filename="file2.pdf", output_filename="new2.pdf", confidence=0.85, reasoning="Test 2"),
            ]
        )
        result2 = RenameResult(
            operations=[
                RenameOp(input_filename="file3.pdf", output_filename="new3.pdf", confidence=0.9, reasoning="Test 3"),
                RenameOp(input_filename="file4.pdf", output_filename="new4.pdf", confidence=0.95, reasoning="Test 4"),
            ]
        )

        merged = result1.merge(result2)

        assert len(merged) == 4
        filenames = {op.input_filename for op in merged.operations}
        assert filenames == {"file1.pdf", "file2.pdf", "file3.pdf", "file4.pdf"}

    def test_merge_with_duplicates_prefers_earlier(self):
        """Test that merging prefers operations from the first result when duplicates exist."""
        result1 = RenameResult(
            operations=[
                RenameOp(
                    input_filename="file1.pdf",
                    output_filename="first_rename.pdf",
                    confidence=0.85,
                    reasoning="First",
                ),
            ]
        )
        result2 = RenameResult(
            operations=[
                RenameOp(
                    input_filename="file1.pdf",
                    output_filename="second_rename.pdf",
                    confidence=0.95,
                    reasoning="Second",
                ),
                RenameOp(
                    input_filename="file2.pdf",
                    output_filename="new2.pdf",
                    confidence=0.9,
                    reasoning="Test",
                ),
            ]
        )

        merged = result1.merge(result2)

        assert len(merged) == 2
        # The overlapping entry should have output_filename from result1
        file1_op = next(op for op in merged.operations if op.input_filename == "file1.pdf")
        assert file1_op.output_filename == "first_rename.pdf"
        assert file1_op.confidence == 0.85

    def test_merge_empty_results(self):
        """Test merging empty RenameResults."""
        result1 = RenameResult(operations=[])
        result2 = RenameResult(operations=[])

        merged = result1.merge(result2)

        assert len(merged) == 0

    def test_merge_into_empty_result(self):
        """Test merging into an empty RenameResult."""
        result1 = RenameResult(operations=[])
        result2 = RenameResult(
            operations=[
                RenameOp(input_filename="file1.pdf", output_filename="new1.pdf", confidence=0.9, reasoning="Test"),
            ]
        )

        merged = result1.merge(result2)

        assert len(merged) == 1
        assert merged.operations[0].input_filename == "file1.pdf"

    def test_merge_from_empty_result(self):
        """Test merging an empty RenameResult into a non-empty one."""
        result1 = RenameResult(
            operations=[
                RenameOp(input_filename="file1.pdf", output_filename="new1.pdf", confidence=0.9, reasoning="Test"),
            ]
        )
        result2 = RenameResult(operations=[])

        merged = result1.merge(result2)

        assert len(merged) == 1
        assert merged.operations[0].input_filename == "file1.pdf"
