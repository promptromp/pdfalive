"""Unit tests for RenameProcessor."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pdfalive.models.rename import RenameOp, RenameResult
from pdfalive.processors.rename_processor import RenameProcessor


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MagicMock()


@pytest.fixture
def sample_rename_result():
    """Sample rename result from LLM."""
    return RenameResult(
        operations=[
            RenameOp(
                input_filename="old_file.pdf",
                output_filename="New File Name.pdf",
                confidence=0.9,
                reasoning="Applied user naming convention",
            ),
            RenameOp(
                input_filename="another_file.pdf",
                output_filename="Another New Name.pdf",
                confidence=0.85,
                reasoning="Extracted title from filename",
            ),
        ]
    )


class TestRenameProcessor:
    """Tests for RenameProcessor class."""

    def test_init(self, mock_llm):
        """Test processor initialization."""
        processor = RenameProcessor(llm=mock_llm)

        assert processor.llm == mock_llm

    def test_extract_filenames_from_paths(self, mock_llm):
        """Test extracting filenames from full paths."""
        processor = RenameProcessor(llm=mock_llm)
        paths = [
            Path("/home/user/docs/file1.pdf"),
            Path("/var/data/file2.pdf"),
            Path("relative/path/file3.pdf"),
        ]

        filenames = processor._extract_filenames(paths)

        assert filenames == ["file1.pdf", "file2.pdf", "file3.pdf"]

    def test_extract_filenames_preserves_order(self, mock_llm):
        """Test that filename extraction preserves order."""
        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/z/zebra.pdf"), Path("/a/apple.pdf"), Path("/m/mango.pdf")]

        filenames = processor._extract_filenames(paths)

        assert filenames == ["zebra.pdf", "apple.pdf", "mango.pdf"]

    def test_build_path_mapping(self, mock_llm):
        """Test building mapping from filename to original path."""
        processor = RenameProcessor(llm=mock_llm)
        paths = [
            Path("/home/user/docs/file1.pdf"),
            Path("/var/data/file2.pdf"),
        ]

        mapping = processor._build_path_mapping(paths)

        assert mapping["file1.pdf"] == Path("/home/user/docs/file1.pdf")
        assert mapping["file2.pdf"] == Path("/var/data/file2.pdf")

    def test_generate_renames_calls_llm(self, mock_llm, sample_rename_result):
        """Test that generate_renames calls the LLM with correct messages."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = sample_rename_result
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/docs/old_file.pdf"), Path("/docs/another_file.pdf")]
        query = "Rename to title case"

        result = processor.generate_renames(paths, query)

        mock_llm.with_structured_output.assert_called_once_with(RenameResult)
        mock_structured_llm.invoke.assert_called_once()
        assert len(result.operations) == 2

    def test_generate_renames_includes_query_in_prompt(self, mock_llm, sample_rename_result):
        """Test that the user query is included in the LLM prompt."""
        messages_received = []

        def capture_invoke(messages):
            messages_received.append(messages)
            return sample_rename_result

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = capture_invoke
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/docs/file.pdf")]
        query = "Rename to author-title format"

        processor.generate_renames(paths, query)

        assert len(messages_received) == 1
        user_message = messages_received[0][1].content
        assert "author-title format" in user_message

    def test_generate_renames_includes_filenames_in_prompt(self, mock_llm, sample_rename_result):
        """Test that filenames are included in the LLM prompt."""
        messages_received = []

        def capture_invoke(messages):
            messages_received.append(messages)
            return sample_rename_result

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = capture_invoke
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/docs/my_special_file.pdf")]
        query = "Add prefix"

        processor.generate_renames(paths, query)

        user_message = messages_received[0][1].content
        assert "my_special_file.pdf" in user_message

    def test_resolve_full_paths(self, mock_llm):
        """Test resolving rename operations to full paths."""
        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/docs/old_file.pdf"), Path("/data/another.pdf")]
        operations = [
            RenameOp(
                input_filename="old_file.pdf",
                output_filename="new_file.pdf",
                confidence=0.9,
                reasoning="test",
            ),
            RenameOp(
                input_filename="another.pdf",
                output_filename="renamed.pdf",
                confidence=0.8,
                reasoning="test",
            ),
        ]

        resolved = processor._resolve_full_paths(operations, paths)

        assert resolved[0] == (Path("/docs/old_file.pdf"), Path("/docs/new_file.pdf"))
        assert resolved[1] == (Path("/data/another.pdf"), Path("/data/renamed.pdf"))

    def test_resolve_full_paths_preserves_directory(self, mock_llm):
        """Test that resolved paths keep the original directory."""
        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/very/long/path/to/file.pdf")]
        operations = [
            RenameOp(
                input_filename="file.pdf",
                output_filename="renamed.pdf",
                confidence=0.9,
                reasoning="test",
            ),
        ]

        resolved = processor._resolve_full_paths(operations, paths)

        assert resolved[0][1] == Path("/very/long/path/to/renamed.pdf")

    def test_resolve_full_paths_handles_missing_files(self, mock_llm):
        """Test that missing files in operations are skipped."""
        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/docs/real_file.pdf")]
        operations = [
            RenameOp(
                input_filename="nonexistent.pdf",
                output_filename="new.pdf",
                confidence=0.9,
                reasoning="test",
            ),
        ]

        resolved = processor._resolve_full_paths(operations, paths)

        assert len(resolved) == 0

    def test_apply_renames_creates_files(self, mock_llm, tmp_path):
        """Test that apply_renames actually renames files."""
        # Create test files
        file1 = tmp_path / "old_name.pdf"
        file1.touch()

        processor = RenameProcessor(llm=mock_llm)
        renames = [(file1, tmp_path / "new_name.pdf")]

        processor.apply_renames(renames)

        assert not file1.exists()
        assert (tmp_path / "new_name.pdf").exists()

    def test_apply_renames_multiple_files(self, mock_llm, tmp_path):
        """Test applying renames to multiple files."""
        file1 = tmp_path / "file1.pdf"
        file2 = tmp_path / "file2.pdf"
        file1.touch()
        file2.touch()

        processor = RenameProcessor(llm=mock_llm)
        renames = [
            (file1, tmp_path / "renamed1.pdf"),
            (file2, tmp_path / "renamed2.pdf"),
        ]

        processor.apply_renames(renames)

        assert not file1.exists()
        assert not file2.exists()
        assert (tmp_path / "renamed1.pdf").exists()
        assert (tmp_path / "renamed2.pdf").exists()

    def test_apply_renames_raises_on_missing_source(self, mock_llm, tmp_path):
        """Test that apply_renames raises error for missing source file."""
        processor = RenameProcessor(llm=mock_llm)
        renames = [(tmp_path / "nonexistent.pdf", tmp_path / "new.pdf")]

        with pytest.raises(FileNotFoundError):
            processor.apply_renames(renames)

    def test_apply_renames_raises_on_existing_target(self, mock_llm, tmp_path):
        """Test that apply_renames raises error if target already exists."""
        source = tmp_path / "source.pdf"
        target = tmp_path / "target.pdf"
        source.touch()
        target.touch()

        processor = RenameProcessor(llm=mock_llm)
        renames = [(source, target)]

        with pytest.raises(FileExistsError):
            processor.apply_renames(renames)


class TestRenameProcessorEdgeCases:
    """Tests for edge cases in RenameProcessor."""

    def test_empty_file_list(self, mock_llm):
        """Test handling empty file list."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = RenameResult(operations=[])
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        result = processor.generate_renames([], "some query")

        assert len(result.operations) == 0

    def test_duplicate_filenames_different_paths(self, mock_llm):
        """Test handling files with same name in different directories."""
        processor = RenameProcessor(llm=mock_llm)
        paths = [
            Path("/dir1/file.pdf"),
            Path("/dir2/file.pdf"),
        ]

        # This should raise an error since filenames must be unique
        with pytest.raises(ValueError, match="duplicate"):
            processor._build_path_mapping(paths)

    def test_special_characters_in_filename(self, mock_llm, sample_rename_result):
        """Test handling filenames with special characters."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = RenameResult(
            operations=[
                RenameOp(
                    input_filename="file with spaces & special.pdf",
                    output_filename="clean_filename.pdf",
                    confidence=0.9,
                    reasoning="Cleaned special chars",
                ),
            ]
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/docs/file with spaces & special.pdf")]

        result = processor.generate_renames(paths, "clean names")

        assert result.operations[0].output_filename == "clean_filename.pdf"


class TestRenameProcessorIntegration:
    """Integration tests for RenameProcessor."""

    def test_full_workflow(self, mock_llm, tmp_path):
        """Test complete workflow from generation to application."""
        # Setup files
        file1 = tmp_path / "old_file.pdf"
        file2 = tmp_path / "another_file.pdf"
        file1.touch()
        file2.touch()

        # Setup LLM response
        rename_result = RenameResult(
            operations=[
                RenameOp(
                    input_filename="old_file.pdf",
                    output_filename="New File.pdf",
                    confidence=0.9,
                    reasoning="Applied naming",
                ),
                RenameOp(
                    input_filename="another_file.pdf",
                    output_filename="Another New.pdf",
                    confidence=0.85,
                    reasoning="Applied naming",
                ),
            ]
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = rename_result
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Execute workflow
        processor = RenameProcessor(llm=mock_llm)
        paths = [file1, file2]
        result = processor.generate_renames(paths, "rename files")

        # Resolve and apply
        resolved = processor._resolve_full_paths(result.operations, paths)
        processor.apply_renames(resolved)

        # Verify
        assert not file1.exists()
        assert not file2.exists()
        assert (tmp_path / "New File.pdf").exists()
        assert (tmp_path / "Another New.pdf").exists()
