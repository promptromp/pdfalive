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
        with runner.isolated_filesystem():
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
        with runner.isolated_filesystem():
            Path("paths.txt").write_text("nonexistent.pdf\n")
            result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])
            assert result.exit_code != 0
            assert "File not found" in result.output
            assert "nonexistent.pdf" in result.output
            assert "line 1" in result.output

    def test_input_file_empty_raises_error(self, runner: CliRunner) -> None:
        """Test that an empty --input-file raises an error."""
        with runner.isolated_filesystem():
            Path("paths.txt").write_text("")
            result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])
            assert result.exit_code != 0
            assert "No valid file paths found" in result.output

    def test_input_file_only_comments_and_blanks_raises_error(self, runner: CliRunner) -> None:
        """Test that --input-file with only comments and blank lines raises an error."""
        with runner.isolated_filesystem():
            Path("paths.txt").write_text("# This is a comment\n\n# Another comment\n   \n")
            result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])
            assert result.exit_code != 0
            assert "No valid file paths found" in result.output

    @pytest.mark.slow
    def test_input_file_skips_comments_and_blank_lines(self, runner: CliRunner) -> None:
        """Test that --input-file correctly skips comments and blank lines."""
        with runner.isolated_filesystem():
            # Create test files
            Path("file1.pdf").write_bytes(b"%PDF-1.4 dummy")
            Path("file2.pdf").write_bytes(b"%PDF-1.4 dummy")

            # Create input file with comments and blank lines
            input_content = """# This is a comment
file1.pdf

# Another comment
file2.pdf

"""
            Path("paths.txt").write_text(input_content)

            # The command will fail when trying to actually process (no valid PDF/LLM)
            # but we can verify it parsed the file correctly by checking the file count
            result = runner.invoke(cli, ["rename", "-q", "Add prefix", "-f", "paths.txt"])
            # Should show "2 file(s)" (parsed correctly, skipping comments)
            assert "2" in result.output and "file" in result.output


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
        with runner.isolated_filesystem():
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
        with runner.isolated_filesystem():
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
        with runner.isolated_filesystem():
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
        with runner.isolated_filesystem():
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
        with runner.isolated_filesystem():
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
        with runner.isolated_filesystem():
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
        with runner.isolated_filesystem():
            Path("pdfalive.toml").write_text("invalid toml [[[")

            # Use a subcommand to trigger the config loading
            result = runner.invoke(cli, ["rename", "--help"])
            assert result.exit_code != 0
            assert "Error loading config file" in result.output
