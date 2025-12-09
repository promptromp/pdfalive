"""Unit tests for RenameProcessor."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pdfalive.models.rename import RenameOp, RenameResult
from pdfalive.processors.rename_processor import (
    PROMPT_OVERHEAD_TOKENS,
    RenameProcessor,
)
from pdfalive.tokens import TokenUsage


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

        result, usage = processor.generate_renames(paths, query)

        mock_llm.with_structured_output.assert_called_once_with(RenameResult)
        mock_structured_llm.invoke.assert_called_once()
        assert len(result.operations) == 2
        assert isinstance(usage, TokenUsage)

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

    def test_generate_renames_returns_token_usage(self, mock_llm, sample_rename_result):
        """Test that generate_renames returns token usage statistics."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = sample_rename_result
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        paths = [Path("/docs/file.pdf")]
        query = "Add prefix"

        result, usage = processor.generate_renames(paths, query)

        assert isinstance(usage, TokenUsage)
        assert usage.llm_calls == 1
        assert usage.input_tokens > 0
        assert usage.output_tokens >= 0

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


class TestRenameProcessorBatching:
    """Tests for RenameProcessor batching functionality."""

    def test_batch_filenames_single_batch(self, mock_llm):
        """Test that small file lists stay in a single batch."""
        processor = RenameProcessor(llm=mock_llm)
        filenames = ["file1.pdf", "file2.pdf", "file3.pdf"]

        batches = list(processor._batch_filenames(filenames, max_tokens=10000))

        assert len(batches) == 1
        assert batches[0] == filenames

    def test_batch_filenames_multiple_batches(self, mock_llm):
        """Test that large file lists are split into multiple batches."""
        processor = RenameProcessor(llm=mock_llm)
        # Create many long filenames to force batching
        filenames = [f"very_long_filename_number_{i:04d}_with_lots_of_text.pdf" for i in range(100)]

        # Use a very small max_tokens to force multiple batches
        batches = list(processor._batch_filenames(filenames, max_tokens=500 + PROMPT_OVERHEAD_TOKENS))

        assert len(batches) > 1
        # All filenames should be in some batch
        all_batched = []
        for batch in batches:
            all_batched.extend(batch)
        assert set(all_batched) == set(filenames)

    def test_batch_filenames_empty_list(self, mock_llm):
        """Test batching empty file list."""
        processor = RenameProcessor(llm=mock_llm)

        batches = list(processor._batch_filenames([]))

        assert len(batches) == 1
        assert batches[0] == []

    def test_batch_filenames_preserves_order(self, mock_llm):
        """Test that batching preserves filename order."""
        processor = RenameProcessor(llm=mock_llm)
        filenames = [f"file_{i:03d}.pdf" for i in range(50)]

        batches = list(processor._batch_filenames(filenames, max_tokens=500 + PROMPT_OVERHEAD_TOKENS))

        # Flatten batches and check order
        all_batched = []
        for batch in batches:
            all_batched.extend(batch)
        assert all_batched == filenames

    def test_generate_renames_batched_merges_results(self, mock_llm):
        """Test that batched generation merges results correctly."""
        # Create mock that returns different results for each batch
        batch_results = [
            RenameResult(
                operations=[
                    RenameOp(
                        input_filename="very_long_filename_to_force_batching_file1.pdf",
                        output_filename="new1.pdf",
                        confidence=0.9,
                        reasoning="Batch 1",
                    ),
                ]
            ),
            RenameResult(
                operations=[
                    RenameOp(
                        input_filename="very_long_filename_to_force_batching_file2.pdf",
                        output_filename="new2.pdf",
                        confidence=0.85,
                        reasoning="Batch 2",
                    ),
                ]
            ),
        ]
        call_count = [0]

        def mock_invoke(messages):
            result = batch_results[call_count[0]]
            call_count[0] += 1
            return result

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = mock_invoke
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        # Use long filenames to force batching with small token limit
        paths = [
            Path("/docs/very_long_filename_to_force_batching_file1.pdf"),
            Path("/docs/very_long_filename_to_force_batching_file2.pdf"),
        ]
        query = "Rename files"

        # Force multiple batches with very small token limit (just enough for prompt overhead + 1 file)
        # Each filename is ~50 chars = ~17 tokens, so set limit to fit only 1 file
        result, usage = processor.generate_renames(
            paths, query, max_tokens_per_batch=PROMPT_OVERHEAD_TOKENS + 20, request_delay=0
        )

        # Should have called LLM twice (once per batch)
        assert mock_structured_llm.invoke.call_count == 2
        # Results should be merged
        assert len(result.operations) == 2
        filenames = {op.input_filename for op in result.operations}
        expected = {
            "very_long_filename_to_force_batching_file1.pdf",
            "very_long_filename_to_force_batching_file2.pdf",
        }
        assert filenames == expected

    def test_generate_renames_batched_tracks_usage(self, mock_llm, sample_rename_result):
        """Test that batched generation tracks token usage across batches."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = sample_rename_result
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        # Create enough files to force multiple batches
        paths = [Path(f"/docs/file{i}.pdf") for i in range(10)]
        query = "Rename files"

        # Force multiple batches with small token limit
        result, usage = processor.generate_renames(paths, query, max_tokens_per_batch=200 + PROMPT_OVERHEAD_TOKENS)

        # Should track usage across all batches
        assert usage.llm_calls >= 1
        assert usage.input_tokens > 0

    def test_generate_renames_uses_continuation_prompt_for_later_batches(self, mock_llm):
        """Test that continuation prompt is used for batches after the first."""
        messages_received = []

        def capture_invoke(messages):
            messages_received.append(messages)
            return RenameResult(operations=[])

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = capture_invoke
        mock_llm.with_structured_output.return_value = mock_structured_llm

        processor = RenameProcessor(llm=mock_llm)
        # Create files with long names to force multiple batches with small token limit
        paths = [
            Path("/docs/very_long_filename_to_force_multiple_batches_file1.pdf"),
            Path("/docs/very_long_filename_to_force_multiple_batches_file2.pdf"),
        ]
        query = "Rename files"

        # Use very small token limit to force batching
        processor.generate_renames(paths, query, max_tokens_per_batch=PROMPT_OVERHEAD_TOKENS + 25, request_delay=0)

        # Should have multiple calls
        assert len(messages_received) >= 2

        # First batch should use main prompt
        first_system = messages_received[0][0].content
        assert "CONTINUATION" not in first_system

        # Later batches should use continuation prompt
        second_system = messages_received[1][0].content
        assert "CONTINUATION" in second_system


class TestRenameProcessorEdgeCases:
    """Tests for edge cases in RenameProcessor."""

    def test_empty_file_list(self, mock_llm):
        """Test handling empty file list."""
        processor = RenameProcessor(llm=mock_llm)
        result, usage = processor.generate_renames([], "some query")

        assert len(result.operations) == 0
        assert usage.llm_calls == 0

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

        result, usage = processor.generate_renames(paths, "clean names")

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
        result, usage = processor.generate_renames(paths, "rename files")

        # Resolve and apply
        resolved = processor._resolve_full_paths(result.operations, paths)
        processor.apply_renames(resolved)

        # Verify
        assert not file1.exists()
        assert not file2.exists()
        assert (tmp_path / "New File.pdf").exists()
        assert (tmp_path / "Another New.pdf").exists()
        assert usage.llm_calls == 1
