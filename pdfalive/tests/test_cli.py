"""Tests for CLI commands."""

import os
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from pdfalive.cli import _save_inplace, cli


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


class TestGenerateTocInplace:
    """Tests for generate-toc --inplace flag."""

    def test_missing_output_and_inplace_raises_error(self, runner: CliRunner) -> None:
        """Test that missing both OUTPUT_FILE and --inplace raises an error."""
        with runner.isolated_filesystem():
            # Create a dummy input file so Click's exists=True check passes
            Path("input.pdf").write_bytes(b"%PDF-1.4 dummy")
            result = runner.invoke(cli, ["generate-toc", "input.pdf"])
            assert result.exit_code != 0
            assert "Either OUTPUT_FILE must be provided or --inplace must be set" in result.output

    def test_both_output_and_inplace_raises_error(self, runner: CliRunner) -> None:
        """Test that providing both OUTPUT_FILE and --inplace raises an error."""
        with runner.isolated_filesystem():
            Path("input.pdf").write_bytes(b"%PDF-1.4 dummy")
            result = runner.invoke(cli, ["generate-toc", "input.pdf", "output.pdf", "--inplace"])
            assert result.exit_code != 0
            assert "Cannot specify both OUTPUT_FILE and --inplace" in result.output

    def test_help_shows_inplace_option(self, runner: CliRunner) -> None:
        """Test that --help shows the --inplace option."""
        result = runner.invoke(cli, ["generate-toc", "--help"])
        assert result.exit_code == 0
        assert "--inplace" in result.output
        assert "Modify the input file in place" in result.output


class TestExtractTextInplace:
    """Tests for extract-text --inplace flag."""

    def test_missing_output_and_inplace_raises_error(self, runner: CliRunner) -> None:
        """Test that missing both OUTPUT_FILE and --inplace raises an error."""
        with runner.isolated_filesystem():
            Path("input.pdf").write_bytes(b"%PDF-1.4 dummy")
            result = runner.invoke(cli, ["extract-text", "input.pdf"])
            assert result.exit_code != 0
            assert "Either OUTPUT_FILE must be provided or --inplace must be set" in result.output

    def test_both_output_and_inplace_raises_error(self, runner: CliRunner) -> None:
        """Test that providing both OUTPUT_FILE and --inplace raises an error."""
        with runner.isolated_filesystem():
            Path("input.pdf").write_bytes(b"%PDF-1.4 dummy")
            result = runner.invoke(cli, ["extract-text", "input.pdf", "output.pdf", "--inplace"])
            assert result.exit_code != 0
            assert "Cannot specify both OUTPUT_FILE and --inplace" in result.output

    def test_help_shows_inplace_option(self, runner: CliRunner) -> None:
        """Test that --help shows the --inplace option."""
        result = runner.invoke(cli, ["extract-text", "--help"])
        assert result.exit_code == 0
        assert "--inplace" in result.output
        assert "Modify the input file in place" in result.output


class TestSaveInplace:
    """Tests for the _save_inplace helper function."""

    def test_replaces_target_with_temp_file(self, tmp_path: Path) -> None:
        """Test that _save_inplace replaces target file with temp file contents."""
        # Create original file with some content
        target_file = tmp_path / "original.pdf"
        target_file.write_text("original content")

        # Create temp file with new content
        temp_file = tmp_path / "temp.pdf"
        temp_file.write_text("new content")

        _save_inplace(str(temp_file), str(target_file))

        # Target should have new content
        assert target_file.read_text() == "new content"
        # Temp file should be gone (moved)
        assert not temp_file.exists()

    def test_preserves_file_permissions(self, tmp_path: Path) -> None:
        """Test that _save_inplace preserves the original file's permissions."""
        # Create original file with specific permissions
        target_file = tmp_path / "original.pdf"
        target_file.write_text("original content")
        original_mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP  # 0o640
        os.chmod(target_file, original_mode)

        # Create temp file (will have different default permissions)
        temp_file = tmp_path / "temp.pdf"
        temp_file.write_text("new content")

        _save_inplace(str(temp_file), str(target_file))

        # Check permissions are preserved
        result_mode = os.stat(target_file).st_mode & 0o777
        assert result_mode == original_mode
