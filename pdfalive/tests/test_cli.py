"""Tests for CLI commands."""

import os
import stat
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from click.testing import CliRunner
from rich.cells import cell_len
from rich.console import Console

import pdfalive.cli as cli_module
from pdfalive.cli import (
    _HIGH_CONFIDENCE_THRESHOLD,
    _MEDIUM_CONFIDENCE_THRESHOLD,
    _RENAME_REASONING_PREVIEW_WIDTH,
    _confidence_style,
    _save_inplace,
    cli,
)
from pdfalive.evaluation.metrics import EvalReport, GoldenEntry
from pdfalive.evaluation.runner import EvalCase, GoldenFile
from pdfalive.models.rename import RenameOp, RenameResult
from pdfalive.models.toc import TOCEntry
from pdfalive.processors.rename_processor import RenameProcessor
from pdfalive.tokens import TokenUsage


# A --input-file line naming a missing file, with an erase-line sequence in it, and how errors show it.
HIDDEN_PATH_LINE = "missing\x1b[2K.pdf"
HIDDEN_PATH_LINE_SHOWN = "missing\\x1b[2K.pdf"

# Rich draws the cell borders of table body rows with this character.
TABLE_ROW_BORDER = "│"

# Offset from a confidence threshold that lands just below it.
CONFIDENCE_STEP = 0.01

# Rich's single-cell ellipsis, appended to truncated reasoning.
ELLIPSIS = "…"


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every CLI test from an empty temporary working directory.

    Replaces Click's deprecated ``CliRunner.isolated_filesystem`` (removed in
    Click 9) and keeps cwd-based config discovery from picking up a
    ``pdfalive.toml`` that happens to live in the developer's working directory.

    The directory is a sibling of ``tmp_path`` rather than ``tmp_path`` itself,
    so anything the CLI writes relative to the cwd stays out of the tests that
    assert over ``tmp_path`` contents to prove no temp file leaked.
    """
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


class TestGenerateTocInplace:
    """Tests for generate-toc --inplace flag."""

    def test_missing_output_and_inplace_raises_error(self, runner: CliRunner) -> None:
        """Test that missing both OUTPUT_FILE and --inplace raises an error."""
        # Create a dummy input file so Click's exists=True check passes
        Path("input.pdf").write_bytes(b"%PDF-1.4 dummy")
        result = runner.invoke(cli, ["generate-toc", "input.pdf"])
        assert result.exit_code != 0
        assert "Either OUTPUT_FILE must be provided or --inplace must be set" in result.output

    def test_both_output_and_inplace_raises_error(self, runner: CliRunner) -> None:
        """Test that providing both OUTPUT_FILE and --inplace raises an error."""
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
        Path("input.pdf").write_bytes(b"%PDF-1.4 dummy")
        result = runner.invoke(cli, ["extract-text", "input.pdf"])
        assert result.exit_code != 0
        assert "Either OUTPUT_FILE must be provided or --inplace must be set" in result.output

    def test_both_output_and_inplace_raises_error(self, runner: CliRunner) -> None:
        """Test that providing both OUTPUT_FILE and --inplace raises an error."""
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


class TestRenameInputFile:
    """Tests for rename --input-file option."""

    def test_help_shows_input_file_option(self, runner: CliRunner) -> None:
        """Test that --help shows the --input-file option."""
        result = runner.invoke(cli, ["rename", "--help"])
        assert result.exit_code == 0
        assert "--input-file" in result.output
        assert "-f" in result.output
        assert "text file" in result.output

    def test_missing_both_input_files_and_input_file_raises_error(self, runner: CliRunner) -> None:
        """Test that missing both INPUT_FILES and --input-file raises an error."""
        result = runner.invoke(cli, ["rename", "-q", "Add prefix"])
        assert result.exit_code != 0
        assert "Either INPUT_FILES arguments or --input-file option must be provided" in result.output

    def test_both_input_files_and_input_file_raises_error(self, runner: CliRunner) -> None:
        """Test that providing both INPUT_FILES and --input-file raises an error."""
        Path("test.pdf").write_bytes(b"%PDF-1.4 dummy")
        Path("paths.txt").write_text("test.pdf\n")
        result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt", "test.pdf"])
        assert result.exit_code != 0
        assert "Cannot specify both INPUT_FILES arguments and --input-file option" in result.output

    def test_input_file_not_found_raises_error(self, runner: CliRunner) -> None:
        """Test that a non-existent --input-file raises an error."""
        result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "nonexistent.txt"])
        assert result.exit_code != 0
        # Click's exists=True validation catches this
        assert "does not exist" in result.output.lower() or "no such file" in result.output.lower()

    def test_input_file_with_nonexistent_path_raises_error(self, runner: CliRunner) -> None:
        """Test that a path in --input-file that doesn't exist raises an error."""
        Path("paths.txt").write_text("nonexistent.pdf\n")
        result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])
        assert result.exit_code != 0
        assert "File not found" in result.output
        assert "nonexistent.pdf" in result.output
        assert "line 1" in result.output

    def test_missing_path_error_shows_control_characters_visibly(self, runner: CliRunner) -> None:
        """Click prints its error messages itself, so outside text in them is made visible first."""
        Path("paths.txt").write_text(f"{HIDDEN_PATH_LINE}\n")

        result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])

        assert result.exit_code != 0
        assert f"File not found: '{HIDDEN_PATH_LINE_SHOWN}'" in result.output
        assert HIDDEN_PATH_LINE not in result.output

    def test_input_file_empty_raises_error(self, runner: CliRunner) -> None:
        """Test that an empty --input-file raises an error."""
        Path("paths.txt").write_text("")
        result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])
        assert result.exit_code != 0
        assert "No valid file paths found" in result.output

    def test_input_file_only_comments_and_blanks_raises_error(self, runner: CliRunner) -> None:
        """Test that --input-file with only comments and blank lines raises an error."""
        Path("paths.txt").write_text("# This is a comment\n\n# Another comment\n   \n")
        result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])
        assert result.exit_code != 0
        assert "No valid file paths found" in result.output

    @patch("pdfalive.cli.RenameProcessor")
    @patch("pdfalive.cli.build_chat_model")
    def test_input_file_skips_comments_and_blank_lines(
        self, mock_build_chat_model: MagicMock, mock_processor_cls: MagicMock, runner: CliRunner
    ) -> None:
        """Test that --input-file correctly skips comments and blank lines."""
        # Make processor.generate_renames return empty result to short-circuit
        mock_processor = mock_processor_cls.return_value
        mock_result = MagicMock()
        mock_result.operations = []
        mock_processor.generate_renames.return_value = (mock_result, MagicMock())

        Path("file1.pdf").write_bytes(b"%PDF-1.4 dummy")
        Path("file2.pdf").write_bytes(b"%PDF-1.4 dummy")

        input_content = "# This is a comment\nfile1.pdf\n\n# Another comment\nfile2.pdf\n\n"
        Path("paths.txt").write_text(input_content)

        result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])
        # Should show "2 file(s)" (parsed correctly, skipping comments)
        assert "2" in result.output and "file" in result.output


class TestReasoningEffort:
    """Tests for the --reasoning-effort option reaching the chat model."""

    @pytest.mark.parametrize("command", ["generate-toc", "rename"])
    def test_option_appears_in_help(self, runner: CliRunner, command: str) -> None:
        result = runner.invoke(cli, [command, "--help"])

        assert result.exit_code == 0
        assert "--reasoning-effort" in result.output

    @patch("pdfalive.cli.RenameProcessor")
    @patch("pdfalive.cli.build_chat_model")
    def test_effort_is_forwarded_to_the_model(
        self, mock_build_chat_model: MagicMock, mock_processor_cls: MagicMock, runner: CliRunner
    ) -> None:
        mock_result = MagicMock()
        mock_result.operations = []
        mock_processor_cls.return_value.generate_renames.return_value = (mock_result, MagicMock())
        Path("file1.pdf").write_bytes(b"%PDF-1.4 dummy")

        runner.invoke(cli, ["rename", "-q", "Add prefix", "--reasoning-effort", "xhigh", "file1.pdf"])

        mock_build_chat_model.assert_called_once_with("gpt-5.6", "xhigh")

    @patch("pdfalive.cli.RenameProcessor")
    @patch("pdfalive.cli.build_chat_model")
    def test_effort_defaults_to_none(
        self, mock_build_chat_model: MagicMock, mock_processor_cls: MagicMock, runner: CliRunner
    ) -> None:
        """Without the option the model is built exactly as it was before."""
        mock_result = MagicMock()
        mock_result.operations = []
        mock_processor_cls.return_value.generate_renames.return_value = (mock_result, MagicMock())
        Path("file1.pdf").write_bytes(b"%PDF-1.4 dummy")

        runner.invoke(cli, ["rename", "-q", "Add prefix", "file1.pdf"])

        mock_build_chat_model.assert_called_once_with("gpt-5.6", None)

    def test_config_file_supplies_the_default(self, runner: CliRunner) -> None:
        Path("pdfalive.toml").write_text('[global]\nreasoning-effort = "medium"\n')

        result = runner.invoke(cli, ["generate-toc", "--help"])

        assert result.exit_code == 0
        assert "medium" in result.output


class TestConfigIntegration:
    """Tests for config file integration with CLI."""

    def test_config_option_in_help(self, runner: CliRunner) -> None:
        """Test that --config option appears in main help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "-c" in result.output
        assert "TOML config file" in result.output

    def test_explicit_config_file_not_found(self, runner: CliRunner) -> None:
        """Test that explicit non-existent config file raises error."""
        result = runner.invoke(cli, ["--config", "nonexistent.toml", "--help"])
        assert result.exit_code != 0
        assert "Config file not found" in result.output

    def test_config_file_sets_defaults_for_rename(self, runner: CliRunner) -> None:
        """Test that config file sets default values for rename command."""
        # Create config file with rename query
        config_content = """
[rename]
query = "Rename to [Author] Title.pdf"
"""
        Path("pdfalive.toml").write_text(config_content)

        # Check that --help shows the default value from config
        result = runner.invoke(cli, ["rename", "--help"])
        assert result.exit_code == 0
        # The default should now be shown in help
        assert "Rename to [Author] Title.pdf" in result.output

    def test_config_file_sets_defaults_for_generate_toc(self, runner: CliRunner) -> None:
        """Test that config file sets default values for generate-toc command."""
        # Create config file
        config_content = """
[generate-toc]
model-identifier = "custom-model"
ocr-dpi = 150
"""
        Path("pdfalive.toml").write_text(config_content)

        # Check that --help shows the default values from config
        result = runner.invoke(cli, ["generate-toc", "--help"])
        assert result.exit_code == 0
        assert "custom-model" in result.output
        assert "150" in result.output

    def test_cli_args_override_config(self, runner: CliRunner) -> None:
        """Test that CLI arguments override config file values."""
        # Create config file
        config_content = """
[generate-toc]
model-identifier = "config-model"
"""
        Path("pdfalive.toml").write_text(config_content)

        # Create a dummy input file
        Path("input.pdf").write_bytes(b"%PDF-1.4 dummy")

        # Run with explicit --model-identifier to override config
        # This won't actually run the command (no valid PDF), but we can check
        # that the option parsing works correctly
        result = runner.invoke(
            cli,
            ["generate-toc", "input.pdf", "output.pdf", "--model-identifier", "cli-model"],
        )
        # The command will fail because the PDF is invalid, but we just want to
        # verify the option parsing worked (config + CLI override)
        # Check that no error about invalid option or missing config occurred
        assert "Invalid value" not in result.output
        assert "config" not in result.output.lower() or "Config file not found" not in result.output

    def test_explicit_config_path_option(self, runner: CliRunner) -> None:
        """Test that explicit --config path is used."""
        # Create config file in a subdirectory
        config_dir = Path("configs")
        config_dir.mkdir()
        config_file = config_dir / "custom.toml"
        config_content = """
[rename]
query = "Custom config query"
"""
        config_file.write_text(config_content)

        # Use explicit config path
        result = runner.invoke(cli, ["--config", str(config_file), "rename", "--help"])
        assert result.exit_code == 0
        assert "Custom config query" in result.output

    def test_hidden_config_file_detected(self, runner: CliRunner) -> None:
        """Test that .pdfalive.toml is auto-detected."""
        # Create hidden config file
        config_content = """
[rename]
query = "Hidden config query"
"""
        Path(".pdfalive.toml").write_text(config_content)

        result = runner.invoke(cli, ["rename", "--help"])
        assert result.exit_code == 0
        assert "Hidden config query" in result.output

    def test_global_settings_apply_to_commands(self, runner: CliRunner) -> None:
        """Test that global settings are applied to relevant commands."""
        config_content = """
[global]
model-identifier = "global-model"
"""
        Path("pdfalive.toml").write_text(config_content)

        # Check generate-toc picks up global setting
        result = runner.invoke(cli, ["generate-toc", "--help"])
        assert result.exit_code == 0
        assert "global-model" in result.output

        # Check rename picks up global setting
        result = runner.invoke(cli, ["rename", "--help"])
        assert result.exit_code == 0
        assert "global-model" in result.output

    def test_invalid_toml_shows_error(self, runner: CliRunner) -> None:
        """Test that invalid TOML shows an appropriate error."""
        Path("pdfalive.toml").write_text("invalid toml [[[")

        # Use a subcommand to trigger the config loading
        result = runner.invoke(cli, ["rename", "--help"])
        assert result.exit_code != 0
        assert "Error loading config file" in result.output


def _create_valid_pdf(path: Path) -> None:
    """Create a minimal valid PDF file at the given path."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test content")
    doc.save(str(path))
    doc.close()


class TestGenerateTocTempFileCleanup:
    """Tests for temp file cleanup in generate-toc --inplace mode.

    These tests verify that:
    - On failure, inplace temp files are cleaned up (no leaked files)
    - On failure, the original input file is NEVER deleted
    - In non-inplace mode, user-specified output paths are NEVER deleted on failure
    """

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_inplace_failure_cleans_up_temp_file(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """On inplace failure, the temp file must be removed."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)

        # Make OCR say no OCR is needed, then TOCGenerator.run raises
        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.side_effect = RuntimeError("LLM exploded")

        result = runner.invoke(cli, ["generate-toc", "--inplace", str(input_pdf)])
        assert result.exit_code != 0

        # The only file in the directory should be the original input
        remaining_files = list(tmp_path.iterdir())
        assert remaining_files == [input_pdf], (
            f"Expected only input file, but found: {[f.name for f in remaining_files]}"
        )

    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_inplace_failure_preserves_original_input(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """On inplace failure, the original input file must be untouched."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)
        original_content = input_pdf.read_bytes()

        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.side_effect = RuntimeError("LLM exploded")

        result = runner.invoke(cli, ["generate-toc", "--inplace", str(input_pdf)])
        assert result.exit_code != 0

        # Original file must still exist with identical content
        assert input_pdf.exists()
        assert input_pdf.read_bytes() == original_content

    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_non_inplace_failure_does_not_delete_output_path(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """In non-inplace mode, a failure must NOT delete the user-specified output path."""
        input_pdf = tmp_path / "input.pdf"
        output_pdf = tmp_path / "output.pdf"
        _create_valid_pdf(input_pdf)
        # Pre-create output file to simulate overwriting an existing file
        output_pdf.write_bytes(b"pre-existing output content")

        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.side_effect = RuntimeError("LLM exploded")

        result = runner.invoke(cli, ["generate-toc", str(input_pdf), str(output_pdf)])
        assert result.exit_code != 0

        # The pre-existing output file must NOT have been deleted
        assert output_pdf.exists()
        assert output_pdf.read_bytes() == b"pre-existing output content"

    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_inplace_ocr_failure_cleans_up_temp_file(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """If OCR processing fails in inplace mode, temp file must be cleaned up."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)

        # OCR says it's needed, then process_in_memory raises
        mock_ocr_cls.return_value.needs_ocr.return_value = True
        mock_ocr_cls.return_value.process_in_memory.side_effect = RuntimeError("OCR failed")

        result = runner.invoke(cli, ["generate-toc", "--inplace", str(input_pdf)])
        assert result.exit_code != 0

        # Only the original input should remain
        remaining_files = list(tmp_path.iterdir())
        assert remaining_files == [input_pdf]
        assert input_pdf.exists()


class TestExtractTextTempFileCleanup:
    """Tests for temp file cleanup in extract-text --inplace mode.

    These tests verify that:
    - On failure, inplace temp files are cleaned up
    - On failure, the original input file is NEVER deleted
    - In non-inplace mode, user-specified output paths are NEVER deleted on failure
    - When no OCR is needed in inplace mode, temp file is still cleaned up
    """

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    @patch("pdfalive.cli.OCRProcessor")
    def test_inplace_failure_cleans_up_temp_file(
        self,
        mock_ocr_cls: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """On inplace failure, the temp file must be removed."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)

        # OCR needed, but process_in_memory raises
        mock_ocr_cls.return_value.needs_ocr.return_value = True
        mock_ocr_cls.return_value.process_in_memory.side_effect = RuntimeError("OCR failed")

        result = runner.invoke(cli, ["extract-text", "--inplace", str(input_pdf)])
        assert result.exit_code != 0

        remaining_files = list(tmp_path.iterdir())
        assert remaining_files == [input_pdf]

    @patch("pdfalive.cli.OCRProcessor")
    def test_inplace_failure_preserves_original_input(
        self,
        mock_ocr_cls: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """On inplace failure, the original input file must be untouched."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)
        original_content = input_pdf.read_bytes()

        mock_ocr_cls.return_value.needs_ocr.return_value = True
        mock_ocr_cls.return_value.process_in_memory.side_effect = RuntimeError("OCR failed")

        result = runner.invoke(cli, ["extract-text", "--inplace", str(input_pdf)])
        assert result.exit_code != 0

        assert input_pdf.exists()
        assert input_pdf.read_bytes() == original_content

    @patch("pdfalive.cli.OCRProcessor")
    def test_non_inplace_failure_does_not_delete_output_path(
        self,
        mock_ocr_cls: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """In non-inplace mode, a failure must NOT delete the user-specified output path."""
        input_pdf = tmp_path / "input.pdf"
        output_pdf = tmp_path / "output.pdf"
        _create_valid_pdf(input_pdf)
        output_pdf.write_bytes(b"pre-existing output content")

        mock_ocr_cls.return_value.needs_ocr.return_value = True
        mock_ocr_cls.return_value.process_in_memory.side_effect = RuntimeError("OCR failed")

        result = runner.invoke(cli, ["extract-text", str(input_pdf), str(output_pdf)])
        assert result.exit_code != 0

        assert output_pdf.exists()
        assert output_pdf.read_bytes() == b"pre-existing output content"

    @patch("pdfalive.cli.OCRProcessor")
    def test_inplace_no_ocr_needed_cleans_up_temp_file(
        self,
        mock_ocr_cls: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """When no OCR is needed in inplace mode, the temp file must still be cleaned up."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)

        mock_ocr_cls.return_value.needs_ocr.return_value = False

        result = runner.invoke(cli, ["extract-text", "--inplace", str(input_pdf)])
        assert result.exit_code == 0

        # Only the original input should remain (temp file cleaned up)
        remaining_files = list(tmp_path.iterdir())
        assert remaining_files == [input_pdf]


class TestConfidenceStyle:
    """Tests for the rename preview's confidence color-coding."""

    @pytest.mark.parametrize(
        ("confidence", "expected_style"),
        [
            (_HIGH_CONFIDENCE_THRESHOLD, "green"),
            (_HIGH_CONFIDENCE_THRESHOLD - CONFIDENCE_STEP, "yellow"),
            (_MEDIUM_CONFIDENCE_THRESHOLD, "yellow"),
            (_MEDIUM_CONFIDENCE_THRESHOLD - CONFIDENCE_STEP, "red"),
            (None, "red"),
        ],
    )
    def test_thresholds(self, confidence: float | None, expected_style: str) -> None:
        assert _confidence_style(confidence) == expected_style


def _table_rows(output: str) -> list[list[str]]:
    """Cell texts of the non-blank body rows of the Rich tables in exported console output."""
    rows = []
    for line in output.splitlines():
        if line.startswith(TABLE_ROW_BORDER):
            cells = [cell.strip() for cell in line.strip(TABLE_ROW_BORDER).split(TABLE_ROW_BORDER)]
            if any(cells):
                rows.append(cells)
    return rows


class TestOutsideTextIsPrintedSafely:
    """Text the CLI does not control is never parsed as Rich markup, and its control characters are made visible.

    Regression: input paths were interpolated into console markup, so
    ``[van Author] Some Title.pdf`` printed as `` Some Title.pdf`` (Rich parsed
    ``[van Author]`` as a style tag), and a ``[/]`` raised MarkupError.

    File paths come from ``markup_like_filename`` (names that are valid path
    components); every other value comes from ``markup_like_text``.
    """

    MODEL_IDENTIFIER = "test-model"
    PLAIN_PDF_NAME = "input.pdf"
    OUTPUT_DIRNAME = "out"
    OCR_DPI = 150
    RENAMED_PREFIX = "renamed "
    RENAME_CONFIDENCE = 0.95
    RENAME_QUERY = "Add a prefix"
    RENAME_REASONING = "Applied the naming instruction"
    DECLINE_CONFIRMATION = "n\n"
    EVAL_MODE = "replay"
    EVAL_MIN_F1 = 1.0
    SPURIOUS_TITLE_PREFIX = "Extra "

    @pytest.fixture
    def cli_console(self, recording_console: Console, monkeypatch: pytest.MonkeyPatch) -> Console:
        """Record everything the CLI module prints."""
        monkeypatch.setattr(cli_module, "console", recording_console)
        return recording_console

    @pytest.fixture
    def markup_like_pdf(self, tmp_path: Path, markup_like_filename: str) -> Path:
        """A valid PDF with a markup-like file name."""
        path = tmp_path / markup_like_filename
        _create_valid_pdf(path)
        return path

    @pytest.fixture
    def markup_like_output(self, tmp_path: Path, markup_like_filename: str) -> Path:
        """An output path with a markup-like file name; never written, as the processors are mocked."""
        return tmp_path / self.OUTPUT_DIRNAME / markup_like_filename

    @pytest.fixture
    def plain_pdf(self, tmp_path: Path) -> Path:
        """A valid PDF with an ordinary name, for tests whose markup-like values are not paths."""
        path = tmp_path / self.PLAIN_PDF_NAME
        _create_valid_pdf(path)
        return path

    @pytest.fixture
    def suggest_rename(self) -> Iterator[Callable[[RenameOp], None]]:
        """Patch the chat model; the yielded function sets the rename the real RenameProcessor gets back."""
        with patch("pdfalive.cli.build_chat_model") as mock_build_chat_model:
            structured_llm = mock_build_chat_model.return_value.with_structured_output.return_value

            def suggest(op: RenameOp) -> None:
                structured_llm.invoke.return_value = RenameResult(operations=[op])

            yield suggest

    def _rename_op(self, source: Path, reasoning: str, output_filename: str | None = None) -> RenameOp:
        return RenameOp(
            input_filename=source.name,
            output_filename=output_filename or f"{self.RENAMED_PREFIX}{source.name}",
            confidence=self.RENAME_CONFIDENCE,
            reasoning=reasoning,
        )

    def _preview_row(self, op: RenameOp) -> list[str]:
        """The rename preview table row the CLI should print for op."""
        return [op.input_filename, op.output_filename, f"{op.confidence:.0%}", op.reasoning]

    @pytest.mark.parametrize("inplace", [False, True])
    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_generate_toc_paths(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        markup_like_pdf: Path,
        markup_like_output: Path,
        inplace: bool,
    ) -> None:
        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.return_value = TokenUsage()
        destination = ["--inplace"] if inplace else [str(markup_like_output)]

        result = runner.invoke(
            cli,
            ["generate-toc", str(markup_like_pdf), *destination, "--model-identifier", self.MODEL_IDENTIFIER],
        )

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        in_place = " (in place)" if inplace else ""
        assert f"Generating TOC for {markup_like_pdf}{in_place} using model {self.MODEL_IDENTIFIER}..." in output
        done = (
            f"All done. Modified {markup_like_pdf} in place."
            if inplace
            else f"All done. Saved to {markup_like_output}."
        )
        assert done in output

    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_generate_toc_option_values(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        plain_pdf: Path,
        markup_like_text: str,
    ) -> None:
        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.return_value = TokenUsage()
        options = ["--model-identifier", markup_like_text, "--ocr-language", markup_like_text]

        result = runner.invoke(
            cli, ["generate-toc", str(plain_pdf), "--inplace", *options, "--ocr-dpi", str(self.OCR_DPI)]
        )

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        assert f"using model {markup_like_text}..." in output
        assert f"Options: ocr ({markup_like_text}, {self.OCR_DPI} DPI)" in output

    @pytest.mark.parametrize("inplace", [False, True])
    @patch("pdfalive.cli.OCRProcessor")
    def test_extract_text_paths(
        self,
        mock_ocr_cls: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        markup_like_pdf: Path,
        markup_like_output: Path,
        inplace: bool,
    ) -> None:
        mock_ocr_cls.return_value.needs_ocr.return_value = True
        destination = ["--inplace"] if inplace else [str(markup_like_output)]

        result = runner.invoke(cli, ["extract-text", str(markup_like_pdf), *destination])

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        in_place = " (in place)" if inplace else ""
        assert f"Processing {markup_like_pdf}{in_place}..." in output
        done = (
            f"All done. Modified {markup_like_pdf} in place."
            if inplace
            else f"All done. Saved to {markup_like_output}."
        )
        assert done in output

    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_generate_toc_output_path_control_sequences(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        tmp_path: Path,
        plain_pdf: Path,
        control_sequence_text: tuple[str, str],
    ) -> None:
        raw, visible = control_sequence_text
        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.return_value = TokenUsage()
        output_path = tmp_path / self.OUTPUT_DIRNAME / raw  # never written, as the processors are mocked

        result = runner.invoke(cli, ["generate-toc", str(plain_pdf), str(output_path)])

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        assert f"All done. Saved to {tmp_path / self.OUTPUT_DIRNAME / visible}." in output
        assert raw not in output

    def test_rename_preview_control_sequences(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        plain_pdf: Path,
        control_sequence_text: tuple[str, str],
    ) -> None:
        """A suggested file name and reasoning carrying terminal control sequences are shown escaped."""
        raw, visible = control_sequence_text
        op = self._rename_op(plain_pdf, reasoning=raw, output_filename=raw)
        suggest_rename(op)

        result = runner.invoke(
            cli, ["rename", "-q", self.RENAME_QUERY, str(plain_pdf)], input=self.DECLINE_CONFIRMATION
        )

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        assert [op.input_filename, visible, f"{op.confidence:.0%}", visible] in _table_rows(output)
        assert raw not in output

    @pytest.mark.parametrize("character", ["x", "長"], ids=["narrow", "wide"])
    def test_rename_preview_truncates_long_reasoning_to_the_cell_width(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        plain_pdf: Path,
        character: str,
    ) -> None:
        reasoning = character * _RENAME_REASONING_PREVIEW_WIDTH * 2
        op = self._rename_op(plain_pdf, reasoning=reasoning)
        suggest_rename(op)

        result = runner.invoke(
            cli, ["rename", "-q", self.RENAME_QUERY, str(plain_pdf)], input=self.DECLINE_CONFIRMATION
        )

        assert result.exit_code == 0, result.output
        [row] = _table_rows(cli_console.export_text())
        shown = row[-1]
        assert shown.endswith(ELLIPSIS)
        # A wide character that does not fit before the ellipsis leaves a padding space instead.
        assert reasoning.startswith(shown.removesuffix(ELLIPSIS).rstrip())
        assert cell_len(shown) <= _RENAME_REASONING_PREVIEW_WIDTH

    @patch("pdfalive.cli.OCRProcessor")
    def test_extract_text_language(
        self,
        mock_ocr_cls: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        plain_pdf: Path,
        markup_like_text: str,
    ) -> None:
        mock_ocr_cls.return_value.needs_ocr.return_value = False

        result = runner.invoke(
            cli,
            [
                "extract-text",
                str(plain_pdf),
                "--inplace",
                "--ocr-language",
                markup_like_text,
                "--ocr-dpi",
                str(self.OCR_DPI),
            ],
        )

        assert result.exit_code == 0, result.output
        assert f"Language: {markup_like_text}, DPI: {self.OCR_DPI}" in cli_console.export_text()

    def test_rename_preview_filenames(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        markup_like_pdf: Path,
    ) -> None:
        op = self._rename_op(markup_like_pdf, reasoning=self.RENAME_REASONING)
        suggest_rename(op)

        result = runner.invoke(
            cli, ["rename", "-q", self.RENAME_QUERY, str(markup_like_pdf)], input=self.DECLINE_CONFIRMATION
        )

        assert result.exit_code == 0, result.output
        assert self._preview_row(op) in _table_rows(cli_console.export_text())

    def test_rename_preview_text_values(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        plain_pdf: Path,
        markup_like_text: str,
    ) -> None:
        op = self._rename_op(plain_pdf, reasoning=markup_like_text)
        suggest_rename(op)

        result = runner.invoke(
            cli,
            ["rename", "-q", markup_like_text, "--model-identifier", markup_like_text, str(plain_pdf)],
            input=self.DECLINE_CONFIRMATION,
        )

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        assert f"using model {markup_like_text}..." in output
        assert f"Query: {markup_like_text}" in output
        assert self._preview_row(op) in _table_rows(output)

    def test_rename_apply_error(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        markup_like_pdf: Path,
    ) -> None:
        op = self._rename_op(markup_like_pdf, reasoning=self.RENAME_REASONING)
        suggest_rename(op)
        target = markup_like_pdf.parent / op.output_filename
        target.write_bytes(markup_like_pdf.read_bytes())  # occupy the target so the rename fails

        result = runner.invoke(cli, ["rename", "-q", self.RENAME_QUERY, "--yes", str(markup_like_pdf)])

        assert result.exit_code == 0, result.output
        error_lines = [line for line in cli_console.export_text().splitlines() if line.startswith("Error:")]
        assert len(error_lines) == 1
        assert str(markup_like_pdf) in error_lines[0]
        assert str(target) in error_lines[0]

    @patch.object(RenameProcessor, "generate_renames")
    @patch("pdfalive.cli.build_chat_model")
    def test_rename_generation_error(
        self,
        mock_build_chat_model: MagicMock,
        mock_generate_renames: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        plain_pdf: Path,
        markup_like_text: str,
    ) -> None:
        mock_generate_renames.side_effect = ValueError(markup_like_text)

        result = runner.invoke(cli, ["rename", "-q", self.RENAME_QUERY, str(plain_pdf)])

        assert result.exit_code == 1
        assert f"Error: {markup_like_text}" in cli_console.export_text()

    @patch("pdfalive.cli.run_eval_case")
    @patch("pdfalive.cli.discover_cases")
    def test_eval_report(
        self,
        mock_discover_cases: MagicMock,
        mock_run_eval_case: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        tmp_path: Path,
        plain_pdf: Path,
        markup_like_text: str,
        make_golden_entry: Callable[..., GoldenEntry],
        make_toc_entry: Callable[..., TOCEntry],
    ) -> None:
        """Case names and TOC titles reach the summary table and the report lines verbatim."""
        case = EvalCase(
            name=markup_like_text,
            pdf_path=plain_pdf,
            golden_path=tmp_path / "golden.json",
            cassette_path=tmp_path / "cassette.json",
            golden=GoldenFile(name=markup_like_text, source_pdf=str(plain_pdf), entries=[]),
        )
        missing = make_golden_entry(title=markup_like_text)
        spurious = make_toc_entry(title=f"{self.SPURIOUS_TITLE_PREFIX}{markup_like_text}")
        mock_discover_cases.return_value = [case]
        mock_run_eval_case.return_value = EvalReport(matched=[], missing=[missing], spurious=[spurious])

        result = runner.invoke(
            cli,
            ["eval", "--evals-dir", str(tmp_path), "--mode", self.EVAL_MODE, "--min-f1", str(self.EVAL_MIN_F1)],
        )

        assert result.exit_code == 1  # nothing matched, so F1 is below the threshold
        output = cli_console.export_text()
        assert f"Evaluating case {case.name} ({self.EVAL_MODE} mode)..." in output
        assert [row[0] for row in _table_rows(output)] == [case.name]
        assert f"{case.name}: 1 missing golden entries:" in output
        assert f"  - {missing.title} (page {missing.page_number}, level {missing.level})" in output
        assert f"{case.name}: 1 spurious generated entries:" in output
        assert f"  - {spurious.title} (page {spurious.page_number}, level {spurious.level})" in output
        assert f"Threshold failure: {case.name}: F1" in output
