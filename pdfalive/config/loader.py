"""Configuration file loading and path resolution."""

import tomllib
from pathlib import Path
from typing import Any

from pdfalive.config.models import PdfAliveConfig


# Config file names to search for, in order of preference
CONFIG_FILE_NAMES = ["pdfalive.toml", ".pdfalive.toml"]


def find_config_file(start_dir: Path | None = None) -> Path | None:
    """Find configuration file by searching standard locations.

    Search order:
    1. Current working directory (or start_dir if provided)
    2. User's home directory
    3. ~/.config/pdfalive/

    For each location, checks for both 'pdfalive.toml' and '.pdfalive.toml'.

    Args:
        start_dir: Directory to start searching from. Defaults to cwd.

    Returns:
        Path to the config file if found, None otherwise.
    """
    search_dirs = [
        start_dir or Path.cwd(),
        Path.home(),
        Path.home() / ".config" / "pdfalive",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for filename in CONFIG_FILE_NAMES:
            config_path = search_dir / filename
            if config_path.is_file():
                return config_path

    return None


def load_config(config_path: Path) -> PdfAliveConfig:
    """Load and validate configuration from a TOML file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        Validated PdfAliveConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        tomllib.TOMLDecodeError: If the TOML is malformed.
        pydantic.ValidationError: If the config doesn't match the schema.
    """
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return PdfAliveConfig.model_validate(data)


def _config_to_default_map(config: PdfAliveConfig) -> dict[str, dict[str, Any]]:
    """Convert a PdfAliveConfig to Click's default_map format.

    The default_map is a nested dict where:
    - Top-level keys are command names
    - Values are dicts mapping option names (with underscores) to their values

    Global settings are merged into each command's defaults, with command-specific
    settings taking precedence.

    Args:
        config: Validated configuration object.

    Returns:
        Dictionary suitable for use as Click's ctx.default_map.
    """
    default_map: dict[str, dict[str, Any]] = {}

    # Get global settings as base (convert TOML kebab-case to Click's underscore style)
    global_defaults: dict[str, Any] = {}
    if config.global_.model_identifier is not None:
        global_defaults["model_identifier"] = config.global_.model_identifier
    if config.global_.show_token_usage is not None:
        global_defaults["show_token_usage"] = config.global_.show_token_usage

    # Process generate-toc command
    generate_toc_defaults = dict(global_defaults)  # Start with global defaults
    generate_toc = config.generate_toc
    if generate_toc.force is not None:
        generate_toc_defaults["force"] = generate_toc.force
    if generate_toc.request_delay is not None:
        generate_toc_defaults["request_delay"] = generate_toc.request_delay
    if generate_toc.ocr is not None:
        generate_toc_defaults["ocr"] = generate_toc.ocr
    if generate_toc.ocr_language is not None:
        generate_toc_defaults["ocr_language"] = generate_toc.ocr_language
    if generate_toc.ocr_dpi is not None:
        generate_toc_defaults["ocr_dpi"] = generate_toc.ocr_dpi
    if generate_toc.ocr_output is not None:
        generate_toc_defaults["ocr_output"] = generate_toc.ocr_output
    if generate_toc.postprocess is not None:
        generate_toc_defaults["postprocess"] = generate_toc.postprocess
    if generate_toc.inplace is not None:
        generate_toc_defaults["inplace"] = generate_toc.inplace
    # Command-specific model/token settings override global
    if generate_toc.model_identifier is not None:
        generate_toc_defaults["model_identifier"] = generate_toc.model_identifier
    if generate_toc.show_token_usage is not None:
        generate_toc_defaults["show_token_usage"] = generate_toc.show_token_usage
    if generate_toc_defaults:
        default_map["generate-toc"] = generate_toc_defaults

    # Process extract-text command (no global settings apply - no LLM usage)
    extract_text_defaults: dict[str, Any] = {}
    extract_text = config.extract_text
    if extract_text.ocr_language is not None:
        extract_text_defaults["ocr_language"] = extract_text.ocr_language
    if extract_text.ocr_dpi is not None:
        extract_text_defaults["ocr_dpi"] = extract_text.ocr_dpi
    if extract_text.force is not None:
        extract_text_defaults["force"] = extract_text.force
    if extract_text.inplace is not None:
        extract_text_defaults["inplace"] = extract_text.inplace
    if extract_text_defaults:
        default_map["extract-text"] = extract_text_defaults

    # Process rename command
    rename_defaults = dict(global_defaults)  # Start with global defaults
    rename = config.rename
    if rename.query is not None:
        rename_defaults["query"] = rename.query
    if rename.yes is not None:
        rename_defaults["yes"] = rename.yes
    # Command-specific model/token settings override global
    if rename.model_identifier is not None:
        rename_defaults["model_identifier"] = rename.model_identifier
    if rename.show_token_usage is not None:
        rename_defaults["show_token_usage"] = rename.show_token_usage
    if rename_defaults:
        default_map["rename"] = rename_defaults

    return default_map


def load_config_as_default_map(config_path: Path | None = None) -> dict[str, dict[str, Any]] | None:
    """Load config file and convert to Click's default_map format.

    If config_path is None, attempts to auto-detect a config file.
    If no config file is found or provided, returns None.

    Args:
        config_path: Explicit path to config file, or None to auto-detect.

    Returns:
        Dictionary suitable for use as Click's ctx.default_map, or None if
        no config file was found.

    Raises:
        FileNotFoundError: If an explicit config_path is provided but doesn't exist.
        tomllib.TOMLDecodeError: If the TOML is malformed.
        pydantic.ValidationError: If the config doesn't match the schema.
    """
    if config_path is None:
        config_path = find_config_file()
        if config_path is None:
            return None

    config = load_config(config_path)
    return _config_to_default_map(config)
