"""Tests for CLI commands."""

import os
import stat
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from click.testing import CliRunner
from langchain_core.exceptions import OutputParserException
from rich.cells import cell_len
from rich.console import Console

import pdfalive.cli as cli_module
from pdfalive.cli import (
    _EXIT_FAILURE,
    _HIGH_CONFIDENCE_THRESHOLD,
    _MEDIUM_CONFIDENCE_THRESHOLD,
    _RENAME_REASONING_PREVIEW_WIDTH,
    _confidence_style,
    _save_inplace,
    cli,
)
from pdfalive.evaluation.metrics import EvalReport, GoldenEntry
from pdfalive.evaluation.runner import EvalCase, GoldenFile
from pdfalive.models.rename import FIRST_FILE_ID, RenameOp, RenameResult
from pdfalive.models.toc import TOCEntry
from pdfalive.processors.rename_processor import REJECTED_PATH, REJECTED_UNKNOWN_INPUT, RenameProcessor
from pdfalive.retry import LLMCallError
from pdfalive.tokens import TokenUsage


# Click's exit status for a usage error, and the CLI's for an LLM call that failed for good.
USAGE_ERROR_EXIT_CODE = 2
LLM_FAILURE_EXIT_CODE = 1


# A --input-file line naming a missing file, with an erase-line sequence in it, and how errors show it.
HIDDEN_PATH_LINE = "missing\x1b[2K.pdf"
HIDDEN_PATH_LINE_SHOWN = "missing\\x1b[2K.pdf"

# Rich draws the cell borders of table body rows with this character.
TABLE_ROW_BORDER = "│"

# Offset from a confidence threshold that lands just below it.
CONFIDENCE_STEP = 0.01

# Rich's single-cell ellipsis, appended to truncated reasoning.
ELLIPSIS = "…"

# The rename used by the CLI tests: a query, the suggested new name, and the answer
# that declines the confirmation prompt.
RENAME_QUERY = "Add a prefix"
NEW_NAME = "renamed.pdf"
DECLINE_CONFIRMATION = "n\n"

# A second input, beside PLAIN_PDF_NAME.
OTHER_PDF_NAME = "other.pdf"
# A suggested name the rename command always rejects (it would leave the directory).
UNSAFE_TARGET_NAME = "../outside.pdf"
# A directory symlinked to the inputs' directory, giving a file another spelling.
SYMLINKED_DIRECTORY_NAME = "linked"
# Batches that complete before one fails, and the tokens each reports.
COMPLETED_BATCHES = 2
BATCH_TOKENS = 100

# A rename query only a config file in the (isolated) home directory sets.
HOME_CONFIG_QUERY = "Query from the home config"

# Heading of the token usage summary, and the options that show or hide it.
TOKEN_USAGE_HEADING = "Token Usage:"
SHOW_TOKEN_USAGE_FLAG = "--show-token-usage"
NO_SHOW_TOKEN_USAGE_FLAG = "--no-show-token-usage"


def _unwrapped(output: str) -> str:
    """Help output with Click's line wrapping undone, so multi-word defaults can be matched."""
    return " ".join(output.split())


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


class TestOutputChoice:
    """OUTPUT_FILE versus --inplace, decided the same way (``is not None``) by validation and the output path."""

    INPUT_NAME = "input.pdf"
    EMPTY_OUTPUT = ""

    @pytest.fixture
    def input_pdf(self) -> Path:
        path = Path(self.INPUT_NAME)
        path.write_bytes(b"%PDF-1.4 dummy")
        return path

    @pytest.mark.parametrize("command", ["generate-toc", "extract-text"])
    @pytest.mark.parametrize(
        ("inplace", "message"),
        [(True, "Cannot specify both OUTPUT_FILE and --inplace"), (False, "OUTPUT_FILE must not be empty")],
        ids=["with-inplace", "without-inplace"],
    )
    def test_an_empty_output_file_is_a_usage_error(
        self, runner: CliRunner, input_pdf: Path, command: str, inplace: bool, message: str
    ) -> None:
        """Regression: an empty OUTPUT_FILE with --inplace passed the falsy checks and was then used as the output."""
        result = runner.invoke(cli, [command, str(input_pdf), self.EMPTY_OUTPUT, *(["--inplace"] if inplace else [])])

        assert result.exit_code == USAGE_ERROR_EXIT_CODE
        assert message in result.output
        assert list(Path().iterdir()) == [input_pdf]


class TestLlmFailuresAreReported:
    """An LLM call the retry policy gives up on is an Error: line and exit status 1, not a traceback."""

    FAILURE = LLMCallError("LLM call for batch 1/1 failed: StructuredOutputError: The answer held no TOC")
    CASE_NAME = "case"

    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_generate_toc(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        tmp_path: Path,
    ) -> None:
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)
        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.side_effect = self.FAILURE

        result = runner.invoke(cli, ["generate-toc", "--inplace", str(input_pdf)])

        assert result.exit_code == LLM_FAILURE_EXIT_CODE
        assert f"Error: {self.FAILURE}" in cli_console.export_text()
        assert list(tmp_path.iterdir()) == [input_pdf]  # the in-place temp file is gone too

    @patch.object(RenameProcessor, "generate_renames")
    @patch("pdfalive.cli.build_chat_model")
    def test_rename(
        self,
        mock_build_chat_model: MagicMock,
        mock_generate_renames: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        tmp_path: Path,
    ) -> None:
        """Regression: rename caught only ValueError, so a StructuredOutputError or refusal was a traceback."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)
        mock_generate_renames.side_effect = self.FAILURE

        result = runner.invoke(cli, ["rename", "-q", "Rename", str(input_pdf)])

        assert result.exit_code == LLM_FAILURE_EXIT_CODE
        assert f"Error: {self.FAILURE}" in cli_console.export_text()

    @patch("pdfalive.cli.run_eval_case")
    @patch("pdfalive.cli.discover_cases")
    def test_eval(
        self,
        mock_discover_cases: MagicMock,
        mock_run_eval_case: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        tmp_path: Path,
    ) -> None:
        input_pdf = tmp_path / "input.pdf"
        mock_discover_cases.return_value = [
            EvalCase(
                name=self.CASE_NAME,
                pdf_path=input_pdf,
                golden_path=tmp_path / "golden.json",
                cassette_path=tmp_path / "cassette.json",
                golden=GoldenFile(name=self.CASE_NAME, source_pdf=str(input_pdf), entries=[]),
            )
        ]
        mock_run_eval_case.side_effect = self.FAILURE

        result = runner.invoke(cli, ["eval", "--evals-dir", str(tmp_path)])

        assert result.exit_code == LLM_FAILURE_EXIT_CODE
        assert f"Error: {self.FAILURE}" in cli_console.export_text()


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
        assert "medium" in _unwrapped(result.output)


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
        assert "Rename to [Author] Title.pdf" in _unwrapped(result.output)

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
        assert "custom-model" in _unwrapped(result.output)
        assert "150" in _unwrapped(result.output)

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
        assert "Custom config query" in _unwrapped(result.output)

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
        assert "Hidden config query" in _unwrapped(result.output)

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
        assert "global-model" in _unwrapped(result.output)

        # Check rename picks up global setting
        result = runner.invoke(cli, ["rename", "--help"])
        assert result.exit_code == 0
        assert "global-model" in _unwrapped(result.output)

    def test_config_discovery_reads_the_isolated_home(self, runner: CliRunner, isolated_home: Path) -> None:
        """Tests run with an empty temporary home (see conftest), so a developer's own config cannot affect them."""
        assert Path.home() == isolated_home
        (isolated_home / "pdfalive.toml").write_text(f'[rename]\nquery = "{HOME_CONFIG_QUERY}"\n')

        result = runner.invoke(cli, ["rename", "--help"])

        assert result.exit_code == 0
        assert HOME_CONFIG_QUERY in _unwrapped(result.output)

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


@pytest.fixture(params=["error", "ctrl-c"])
def run_failure(request: pytest.FixtureRequest) -> BaseException:
    """A failure partway through a command: an ordinary exception, or Ctrl-C."""
    failures: dict[str, BaseException] = {"error": RuntimeError("processing failed"), "ctrl-c": KeyboardInterrupt()}
    return failures[request.param]


class TestInplaceTempFileLifetime:
    """The --inplace temporary file is removed on every way out, however early the command stops."""

    INPUT_NAME = "input.pdf"

    @pytest.mark.parametrize("command", ["generate-toc", "extract-text"])
    def test_a_failure_before_any_processing_leaves_no_temp_file(
        self, tmp_path: Path, run_failure: BaseException, command: str
    ) -> None:
        """Regression: the temp file was created before the cleanup's try, so a failure in between leaked it."""
        input_pdf = tmp_path / self.INPUT_NAME
        _create_valid_pdf(input_pdf)

        with patch("pdfalive.cli._in_place_note", side_effect=run_failure):
            result = CliRunner().invoke(cli, [command, "--inplace", str(input_pdf)])

        assert result.exit_code != 0
        assert list(tmp_path.iterdir()) == [input_pdf]


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
        run_failure: BaseException,
    ) -> None:
        """On inplace failure, including Ctrl-C, the temp file must be removed."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)

        # Make OCR say no OCR is needed, then TOCGenerator.run fails
        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.side_effect = run_failure

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
        run_failure: BaseException,
    ) -> None:
        """If OCR processing fails or is interrupted in inplace mode, temp file must be cleaned up."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)

        # OCR says it's needed, then process_in_memory fails
        mock_ocr_cls.return_value.needs_ocr.return_value = True
        mock_ocr_cls.return_value.process_in_memory.side_effect = run_failure

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
        run_failure: BaseException,
    ) -> None:
        """On inplace failure, including Ctrl-C, the temp file must be removed."""
        input_pdf = tmp_path / "input.pdf"
        _create_valid_pdf(input_pdf)

        # OCR needed, but process_in_memory fails
        mock_ocr_cls.return_value.needs_ocr.return_value = True
        mock_ocr_cls.return_value.process_in_memory.side_effect = run_failure

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
        ],
    )
    def test_thresholds(self, confidence: float, expected_style: str) -> None:
        assert _confidence_style(confidence) == expected_style


# File name of the ordinary input PDF used by tests whose hostile values are not paths.
PLAIN_PDF_NAME = "input.pdf"


@pytest.fixture
def cli_console(recording_console: Console, monkeypatch: pytest.MonkeyPatch) -> Console:
    """Record everything the CLI module prints."""
    monkeypatch.setattr(cli_module, "console", recording_console)
    return recording_console


@pytest.fixture
def plain_pdf(tmp_path: Path) -> Path:
    """A valid PDF with an ordinary name."""
    path = tmp_path / PLAIN_PDF_NAME
    _create_valid_pdf(path)
    return path


@pytest.fixture
def rename_llm() -> Iterator[MagicMock]:
    """Patch the chat model the rename command builds; returns its structured-output runnable."""
    with patch("pdfalive.cli.build_chat_model") as mock_build_chat_model:
        yield mock_build_chat_model.return_value.with_structured_output.return_value


@pytest.fixture
def suggest_rename(
    rename_llm: MagicMock, make_structured_answer: Callable[..., dict[str, Any]]
) -> Callable[[RenameOp], None]:
    """Set the rename suggestion the real RenameProcessor gets back from the patched chat model."""

    def suggest(op: RenameOp) -> None:
        rename_llm.invoke.return_value = make_structured_answer(RenameResult(operations=[op]))

    return suggest


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
    OUTPUT_DIRNAME = "out"
    OCR_DPI = 150
    RENAMED_PREFIX = "renamed "
    RENAME_REASONING = "Applied the naming instruction"
    EVAL_MODE = "replay"
    EVAL_MIN_F1 = 1.0
    SPURIOUS_TITLE_PREFIX = "Extra "

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

    def _preview_row(self, source: Path, op: RenameOp) -> list[str]:
        """The rename preview table row the CLI should print for op, the suggestion for source."""
        return [source.name, op.output_filename, f"{op.confidence:.0%}", op.reasoning]

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

    def test_rename_preview_shows_control_sequences_in_reasoning_escaped(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
        control_sequence_text: tuple[str, str],
    ) -> None:
        raw, visible = control_sequence_text
        op = make_rename_op(FIRST_FILE_ID, plain_pdf.name, f"{self.RENAMED_PREFIX}{plain_pdf.name}", reasoning=raw)
        suggest_rename(op)

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)], input=DECLINE_CONFIRMATION)

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        assert [plain_pdf.name, op.output_filename, f"{op.confidence:.0%}", visible] in _table_rows(output)
        assert raw not in output

    def test_suggested_names_with_control_sequences_are_shown_escaped(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
        control_sequence_text: tuple[str, str],
    ) -> None:
        """Rejected (control characters) or planned (an RLO, an undecodable byte), the name is shown escaped."""
        raw, visible = control_sequence_text
        suggest_rename(make_rename_op(FIRST_FILE_ID, plain_pdf.name, raw))

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)], input=DECLINE_CONFIRMATION)

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        assert visible in output
        assert raw not in output

    @pytest.mark.parametrize("character", ["x", "長"], ids=["narrow", "wide"])
    def test_rename_preview_truncates_long_reasoning_to_the_cell_width(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
        character: str,
    ) -> None:
        reasoning = character * _RENAME_REASONING_PREVIEW_WIDTH * 2
        op = make_rename_op(
            FIRST_FILE_ID, plain_pdf.name, f"{self.RENAMED_PREFIX}{plain_pdf.name}", reasoning=reasoning
        )
        suggest_rename(op)

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)], input=DECLINE_CONFIRMATION)

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
        make_rename_op: Callable[..., RenameOp],
        markup_like_pdf: Path,
    ) -> None:
        op = make_rename_op(
            FIRST_FILE_ID,
            markup_like_pdf.name,
            f"{self.RENAMED_PREFIX}{markup_like_pdf.name}",
            reasoning=self.RENAME_REASONING,
        )
        suggest_rename(op)

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(markup_like_pdf)], input=DECLINE_CONFIRMATION)

        assert result.exit_code == 0, result.output
        assert self._preview_row(markup_like_pdf, op) in _table_rows(cli_console.export_text())

    def test_rename_preview_text_values(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
        markup_like_text: str,
    ) -> None:
        op = make_rename_op(
            FIRST_FILE_ID, plain_pdf.name, f"{self.RENAMED_PREFIX}{plain_pdf.name}", reasoning=markup_like_text
        )
        suggest_rename(op)

        result = runner.invoke(
            cli,
            ["rename", "-q", markup_like_text, "--model-identifier", markup_like_text, str(plain_pdf)],
            input=DECLINE_CONFIRMATION,
        )

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        assert f"using model {markup_like_text}..." in output
        assert f"Query: {markup_like_text}" in output
        assert self._preview_row(plain_pdf, op) in _table_rows(output)

    def test_rename_apply_error(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        markup_like_pdf: Path,
    ) -> None:
        op = make_rename_op(
            FIRST_FILE_ID,
            markup_like_pdf.name,
            f"{self.RENAMED_PREFIX}{markup_like_pdf.name}",
            reasoning=self.RENAME_REASONING,
        )
        suggest_rename(op)
        target = markup_like_pdf.parent / op.output_filename
        target.write_bytes(markup_like_pdf.read_bytes())  # occupy the target so the rename fails

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, "--yes", str(markup_like_pdf)])

        assert result.exit_code == _EXIT_FAILURE, result.output
        error_lines = [line for line in cli_console.export_text().splitlines() if line.startswith("Error:")]
        assert len(error_lines) == 1
        assert str(markup_like_pdf) in error_lines[0]
        assert str(target) in error_lines[0]

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


class TestRenameCommand:
    """Tests for the rename command's handling of suggestions, errors and token usage."""

    SUBDIRECTORIES = ("first", "second")

    def test_suggestions_that_would_leave_the_directory_are_skipped_and_reported(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
    ) -> None:
        """Regression: a suggested "../name" was joined to the source's directory, moving the file out of it."""
        suggest_rename(make_rename_op(FIRST_FILE_ID, plain_pdf.name, UNSAFE_TARGET_NAME))

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, "--yes", str(plain_pdf)])

        assert result.exit_code == _EXIT_FAILURE  # unattended, and nothing could be applied
        assert plain_pdf.exists()
        assert not (plain_pdf.parent / UNSAFE_TARGET_NAME).exists()
        output = cli_console.export_text()
        assert "Skipped 1 suggestion(s):" in output
        assert f"  - {plain_pdf.name} -> {UNSAFE_TARGET_NAME}: {REJECTED_PATH}" in output
        assert "No valid rename operations to apply." in output

    def test_files_sharing_a_name_in_different_directories_are_renamed_together(
        self,
        make_structured_answer,
        runner: CliRunner,
        cli_console: Console,
        rename_llm: MagicMock,
        make_rename_op: Callable[..., RenameOp],
        tmp_path: Path,
    ) -> None:
        """Regression: inputs had to have unique file names, because suggestions were matched by name."""
        sources = [tmp_path / directory / PLAIN_PDF_NAME for directory in self.SUBDIRECTORIES]
        for source in sources:
            source.parent.mkdir()
            _create_valid_pdf(source)
        rename_llm.invoke.return_value = make_structured_answer(
            RenameResult(
                operations=[
                    make_rename_op(file_id, PLAIN_PDF_NAME, f"{directory} {NEW_NAME}")
                    for file_id, directory in enumerate(self.SUBDIRECTORIES, start=FIRST_FILE_ID)
                ]
            )
        )

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, "--yes", *map(str, sources)])

        assert result.exit_code == 0, result.output
        for source, directory in zip(sources, self.SUBDIRECTORIES, strict=True):
            assert not source.exists()
            assert (source.parent / f"{directory} {NEW_NAME}").exists()

    def test_a_suggestion_whose_id_names_no_file_is_shown_with_the_name_it_gave(
        self,
        make_structured_answer,
        runner: CliRunner,
        cli_console: Console,
        rename_llm: MagicMock,
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
        markup_like_text: str,
    ) -> None:
        """Without a trustworthy id the input cannot be named, so the echoed name helps tell which file was meant."""
        op = make_rename_op(FIRST_FILE_ID + 1, markup_like_text, NEW_NAME)  # the request lists one file
        rename_llm.invoke.return_value = make_structured_answer(RenameResult(operations=[op]))

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)], input=DECLINE_CONFIRMATION)

        assert result.exit_code == 0, result.output
        expected = f'  - file id {op.file_id} ("{markup_like_text}") -> {NEW_NAME}: {REJECTED_UNKNOWN_INPUT}'
        assert expected in cli_console.export_text()

    def test_inputs_sharing_a_name_are_shown_with_their_directories(
        self,
        make_structured_answer,
        runner: CliRunner,
        cli_console: Console,
        rename_llm: MagicMock,
        make_rename_op: Callable[..., RenameOp],
        tmp_path: Path,
    ) -> None:
        """Both the preview and the skipped list name each input so the two can be told apart."""
        sources = [tmp_path / directory / PLAIN_PDF_NAME for directory in self.SUBDIRECTORIES]
        for source in sources:
            source.parent.mkdir()
            _create_valid_pdf(source)
        planned, rejected = (
            make_rename_op(file_id, PLAIN_PDF_NAME, new_name)
            for file_id, new_name in enumerate((NEW_NAME, UNSAFE_TARGET_NAME), start=FIRST_FILE_ID)
        )
        rename_llm.invoke.return_value = make_structured_answer(RenameResult(operations=[planned, rejected]))
        shown = [f"{directory}/{PLAIN_PDF_NAME}" for directory in self.SUBDIRECTORIES]

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, *map(str, sources)], input=DECLINE_CONFIRMATION)

        assert result.exit_code == 0, result.output
        output = cli_console.export_text()
        assert [row[:2] for row in _table_rows(output)] == [[shown[0], NEW_NAME]]
        assert f"  - {shown[1]} -> {UNSAFE_TARGET_NAME}: {REJECTED_PATH}" in output

    @pytest.mark.parametrize("show_token_usage", [True, False])
    @pytest.mark.parametrize("outcome", ["no-suggestions", "all-rejected", "declined", "applied"])
    def test_token_usage_follows_the_flag_on_every_path_past_the_llm(
        self,
        make_structured_answer,
        runner: CliRunner,
        cli_console: Console,
        rename_llm: MagicMock,
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
        outcome: str,
        show_token_usage: bool,
    ) -> None:
        """Regression: no suggestions, or none valid, returned before reporting the LLM calls' token usage."""
        operations = {
            "no-suggestions": [],
            "all-rejected": [make_rename_op(FIRST_FILE_ID, plain_pdf.name, UNSAFE_TARGET_NAME)],
            "declined": [make_rename_op(FIRST_FILE_ID, plain_pdf.name, NEW_NAME)],
            "applied": [make_rename_op(FIRST_FILE_ID, plain_pdf.name, NEW_NAME)],
        }[outcome]
        rename_llm.invoke.return_value = make_structured_answer(RenameResult(operations=operations))
        confirmation = ["--yes"] if outcome == "applied" else []
        token_usage_flag = SHOW_TOKEN_USAGE_FLAG if show_token_usage else NO_SHOW_TOKEN_USAGE_FLAG

        result = runner.invoke(
            cli,
            ["rename", "-q", RENAME_QUERY, token_usage_flag, *confirmation, str(plain_pdf)],
            input=DECLINE_CONFIRMATION,
        )

        assert result.exit_code == 0, result.output
        assert (TOKEN_USAGE_HEADING in cli_console.export_text()) is show_token_usage

    def test_config_file_can_turn_token_usage_off(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
    ) -> None:
        Path("pdfalive.toml").write_text("[global]\nshow-token-usage = false\n")
        suggest_rename(make_rename_op(FIRST_FILE_ID, plain_pdf.name, NEW_NAME))

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)], input=DECLINE_CONFIRMATION)

        assert result.exit_code == 0, result.output
        assert TOKEN_USAGE_HEADING not in cli_console.export_text()


class TestRenameExitPaths:
    """Every way out of the rename command after the LLM calls reports their token usage and an honest status."""

    END_OF_INPUT = ""

    def test_token_usage_is_reported_when_the_prompt_is_aborted(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
    ) -> None:
        """Regression: Ctrl-C or end of input at the confirmation prompt skipped the token usage summary."""
        suggest_rename(make_rename_op(FIRST_FILE_ID, plain_pdf.name, NEW_NAME))

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)], input=self.END_OF_INPUT)

        assert result.exit_code != 0  # Click reports the abort
        assert TOKEN_USAGE_HEADING in cli_console.export_text()
        assert plain_pdf.exists()

    def test_token_usage_is_reported_when_applying_raises(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
    ) -> None:
        suggest_rename(make_rename_op(FIRST_FILE_ID, plain_pdf.name, NEW_NAME))

        with patch.object(RenameProcessor, "apply_renames", side_effect=RuntimeError("disk vanished")):
            result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, "--yes", str(plain_pdf)])

        assert isinstance(result.exception, RuntimeError)
        assert TOKEN_USAGE_HEADING in cli_console.export_text()

    def test_a_failed_rename_exits_with_the_failure_status(
        self,
        runner: CliRunner,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
    ) -> None:
        """Regression: renames that failed after confirmation still exited 0."""
        suggest_rename(make_rename_op(FIRST_FILE_ID, plain_pdf.name, NEW_NAME))
        (plain_pdf.parent / NEW_NAME).touch()  # occupy the target so the rename fails

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, "--yes", str(plain_pdf)])

        assert result.exit_code == _EXIT_FAILURE

    def test_the_same_file_in_different_spellings_is_counted_and_renamed_once(
        self,
        runner: CliRunner,
        cli_console: Console,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
        tmp_path: Path,
    ) -> None:
        """Regression: the same file passed twice was rejected as a duplicate name in different directories."""
        linked_directory = tmp_path / SYMLINKED_DIRECTORY_NAME
        linked_directory.symlink_to(plain_pdf.parent, target_is_directory=True)
        spellings = [
            plain_pdf,
            Path(os.path.relpath(plain_pdf)),
            plain_pdf.parent / ".." / plain_pdf.parent.name / plain_pdf.name,
            linked_directory / plain_pdf.name,
        ]
        suggest_rename(make_rename_op(FIRST_FILE_ID, plain_pdf.name, NEW_NAME))

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, "--yes", *map(str, spellings)])

        assert result.exit_code == 0, result.output
        assert (plain_pdf.parent / NEW_NAME).exists()
        output = cli_console.export_text()
        assert "Renaming 1 file(s)" in output
        assert "Successfully renamed 1 file(s)." in output

    def test_an_unattended_run_that_rejects_every_suggestion_fails(
        self,
        runner: CliRunner,
        suggest_rename: Callable[[RenameOp], None],
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
    ) -> None:
        """Under --yes nobody reads the skipped list, so a run where every suggestion was rejected fails."""
        suggest_rename(make_rename_op(FIRST_FILE_ID, plain_pdf.name, UNSAFE_TARGET_NAME))

        unattended = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, "--yes", str(plain_pdf)])
        interactive = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)], input=DECLINE_CONFIRMATION)

        assert unattended.exit_code == _EXIT_FAILURE
        assert interactive.exit_code == 0

    @pytest.mark.parametrize(
        ("outcome", "exit_code"),
        [
            ("no-suggestions", 0),
            ("unchanged-names-only", 0),
            ("all-rejected", _EXIT_FAILURE),
            ("some-rejected-the-rest-applied", _EXIT_FAILURE),
        ],
    )
    def test_unattended_exit_status(
        self,
        make_structured_answer,
        runner: CliRunner,
        rename_llm: MagicMock,
        make_rename_op: Callable[..., RenameOp],
        plain_pdf: Path,
        outcome: str,
        exit_code: int,
    ) -> None:
        """Under --yes any rejected suggestion fails the run, applied or not; unchanged names are not rejections.

        Regression: a run whose suggestions all kept the current names exited 1, as if they had been rejected.
        """
        other_pdf = plain_pdf.with_name(OTHER_PDF_NAME)
        _create_valid_pdf(other_pdf)
        operations = {
            "no-suggestions": [],
            "unchanged-names-only": [make_rename_op(FIRST_FILE_ID, plain_pdf.name, plain_pdf.name)],
            "all-rejected": [make_rename_op(FIRST_FILE_ID, plain_pdf.name, UNSAFE_TARGET_NAME)],
            "some-rejected-the-rest-applied": [
                make_rename_op(FIRST_FILE_ID, plain_pdf.name, UNSAFE_TARGET_NAME),
                make_rename_op(FIRST_FILE_ID + 1, other_pdf.name, NEW_NAME),
            ],
        }[outcome]
        rename_llm.invoke.return_value = make_structured_answer(RenameResult(operations=operations))

        result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, "--yes", str(plain_pdf), str(other_pdf)])

        assert result.exit_code == exit_code, result.output
        assert (plain_pdf.parent / NEW_NAME).exists() is (outcome == "some-rejected-the-rest-applied")

    def test_token_usage_of_completed_batches_survives_a_failing_one(
        self, runner: CliRunner, cli_console: Console, plain_pdf: Path
    ) -> None:
        """Regression: a batch failing after others had completed lost their (billed) token usage."""

        def two_batches_then_failure(paths, query, *, usage: TokenUsage, **options):
            for batch in range(COMPLETED_BATCHES):
                usage.add_call(input_tokens=BATCH_TOKENS, output_tokens=BATCH_TOKENS, description=f"batch {batch}")
            raise RuntimeError("the third batch failed")

        with (
            patch("pdfalive.cli.build_chat_model"),
            patch.object(RenameProcessor, "generate_renames", side_effect=two_batches_then_failure),
        ):
            result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)])

        assert isinstance(result.exception, RuntimeError)
        output = cli_console.export_text()
        assert TOKEN_USAGE_HEADING in output
        assert f"LLM calls: {COMPLETED_BATCHES}" in output

    def test_errors_in_llm_output_are_not_reported_as_input_errors(
        self, runner: CliRunner, cli_console: Console, rename_llm: MagicMock, plain_pdf: Path
    ) -> None:
        """Regression: `except ValueError` also caught structured-output errors (which subclass ValueError)."""
        with patch.object(RenameProcessor, "generate_renames", side_effect=OutputParserException("not json")):
            result = runner.invoke(cli, ["rename", "-q", RENAME_QUERY, str(plain_pdf)])

        assert isinstance(result.exception, OutputParserException)
        assert "Error:" not in cli_console.export_text()


class TestShowTokenUsageFlag:
    """--show-token-usage / --no-show-token-usage on the LLM-using commands that report usage."""

    @pytest.mark.parametrize("command", ["generate-toc", "rename"])
    def test_off_switch_appears_in_help(self, runner: CliRunner, command: str) -> None:
        result = runner.invoke(cli, [command, "--help"])

        assert result.exit_code == 0
        assert NO_SHOW_TOKEN_USAGE_FLAG in result.output

    @pytest.mark.parametrize(("flags", "shown"), [([], True), ([NO_SHOW_TOKEN_USAGE_FLAG], False)])
    @patch("pdfalive.cli.TOCGenerator")
    @patch("pdfalive.cli.build_chat_model")
    @patch("pdfalive.cli.OCRProcessor")
    def test_generate_toc_follows_the_flag(
        self,
        mock_ocr_cls: MagicMock,
        mock_build_chat_model: MagicMock,
        mock_toc_cls: MagicMock,
        runner: CliRunner,
        cli_console: Console,
        plain_pdf: Path,
        flags: list[str],
        shown: bool,
    ) -> None:
        mock_ocr_cls.return_value.needs_ocr.return_value = False
        mock_toc_cls.return_value.run.return_value = TokenUsage()

        result = runner.invoke(cli, ["generate-toc", str(plain_pdf), "--inplace", *flags])

        assert result.exit_code == 0, result.output
        assert (TOKEN_USAGE_HEADING in cli_console.export_text()) is shown
