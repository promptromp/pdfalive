# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pdfalive is a Python library and CLI tool that uses LLMs to enhance PDF files. It provides:
- **Automatic Table of Contents (bookmarks) generation** for PDFs using LLM inference
- **OCR processing** for scanned PDFs using Tesseract integration

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the CLI - TOC generation
uv run pdfalive generate-toc examples/example.pdf output.pdf --force

# Run the CLI - OCR text extraction
uv run pdfalive extract-text input.pdf output.pdf

# Linting
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy pdfalive

# Run tests
uv run pytest
```

## Architecture

The codebase follows a processor pattern where document operations are encapsulated in processor classes.

```
pdfalive/
├── cli.py                 # Click-based CLI entry point
├── tokens.py              # Token counting utilities
├── prompts.py             # LLM system prompts
├── config/
│   ├── __init__.py        # Config module exports
│   ├── models.py          # Pydantic models for config validation
│   └── loader.py          # TOML loading, path resolution, default_map conversion
├── models/
│   ├── toc.py             # TOC, TOCEntry, TOCFeature models
│   ├── page_content.py    # PageContent model
│   └── rename.py          # RenameOp, RenameResult models
├── processors/
│   ├── toc_generator.py   # TOCGenerator processor
│   ├── ocr_processor.py   # OCRProcessor for text extraction
│   ├── ocr_detection.py   # OCR detection strategies
│   └── rename_processor.py # RenameProcessor for file renaming
└── tests/                 # Unit tests
```

**CLI Commands:**
- `generate-toc` - Main command for TOC generation (with optional automatic OCR)
- `extract-text` - OCR-only command for text extraction from scanned PDFs
- `rename` - Intelligent file renaming using LLM inference. Supports reading paths from a file via `-f`/`--input-file` option for handling many files or long filenames.

**Processor Classes:**
- `TOCGenerator` - Extracts font/text features from PDF pages, sends to LLM for TOC inference, writes bookmarks back to PDF. Supports multiprocessing, intelligent batching for large documents, and retry logic with exponential backoff.
- `OCRProcessor` - Performs OCR on scanned PDFs using PyMuPDF's Tesseract integration. Supports multiprocessing for parallel page processing.
- `OCRDetectionStrategy` / `NoTextDetectionStrategy` - Strategy pattern for determining if a document needs OCR.
- `RenameProcessor` - Uses LLM to generate intelligent file rename suggestions based on user instructions. Supports batch renaming with confirmation preview.

**Model Classes:**
- `TOC` / `TOCEntry` - Pydantic models for structured LLM output with confidence scores
- `TOCFeature` - Compact representation of page features sent to the LLM
- `PageContent` - Data model for page representation
- `RenameOp` / `RenameResult` - Pydantic models for file rename operations with confidence scores and reasoning

**Key Integration Points:**
- PyMuPDF (`pymupdf`) for PDF reading/writing and OCR via Tesseract
- LangChain for LLM abstraction with `init_chat_model()` and `with_structured_output()` for typed responses
- Tenacity for retry logic with exponential backoff
- Rich for terminal progress indicators
- Default model: `gpt-5.3` (configurable via `--model-identifier`)

**TOC Generation Strategy:**
The `TOCGenerator._extract_features()` method extracts font metadata (name, size) and text snippets from the first few blocks/lines of each page. This condensed representation is sent to the LLM which identifies chapter/section headings based on font patterns and returns structured `TOCEntry` objects with confidence scores. For large documents, features are batched with overlap for context continuity.

**OCR Integration:**
When `--ocr` is enabled (default), `generate-toc` automatically detects if a PDF needs OCR by checking for extractable text, performs OCR if needed, then proceeds with TOC generation on the text layer.

**Configuration System:**
The CLI supports TOML configuration files (`pdfalive.toml` or `.pdfalive.toml`) for setting default option values. The config module (`pdfalive/config/`) handles:
- `models.py` - Pydantic models for validating config structure with kebab-case alias support
- `loader.py` - File discovery (cwd > home > ~/.config/pdfalive/), TOML parsing, and conversion to Click's `default_map` format

The config is loaded via an eager callback on the `--config` option in `cli.py`. Global settings apply to all LLM-using commands, and command-specific settings override globals. CLI arguments always take precedence over config file values.

## Development Guidelines

- always prefer placing imports at top of files rather than inline. Especially when writing unit-test. only do otherwise to avoid circular dependencies in rare cases. In those cases, mention explicitly why you are doing this in a comment on the relevant code line.
- When writing unit-tests, use variables and/or pytest fixtures (e.g. via conftest.py and `@pytest.fixture` decorator) for fixture values and objects, rather than repeating literal values in test setup and assertions. Prefer using pytest's `@pytest.mark.parametrize` decorator when you wish to test different values or combinations of values rather than creating repetitive standalone test cases.
- When making changes, always make sure formatting, linting, type checks, and tests work afterwards. We use ruff, mypy and pytest for these, and can run them via uv, e.g. `uv run ruff ...`, `uv run mypy`, etc.
- When finished making substantial changes to functionality and/or API (e.g. CLI usage) make sure to update documentation - README.md, CLAUDE.md and docs/ markdown files should all be kept up to date. Changing any CLI configuration options should also result in update the config/ submodule which lets us use TOML configuration files for defaults.
