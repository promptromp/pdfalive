# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pdfalive is a Python library and CLI tool that uses LLMs to enhance PDF files. Currently, it provides automatic Table of Contents (bookmarks) generation for PDFs.

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the CLI
uv run pdfalive generate-toc examples/example.pdf output.pdf --force

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

**Core Flow:**
1. `cli.py` - Click-based CLI entry point, initializes PDF document and LLM
2. `processors/toc_generator.py` - `TOCGenerator` extracts font/text features from PDF pages, sends to LLM for TOC inference, then writes bookmarks back to PDF
3. `models/` - Pydantic models for structured LLM output (`TOCEntry`, `TOC`) and prompts

**Key Integration Points:**
- PyMuPDF (`pymupdf`) for PDF reading/writing
- LangChain for LLM abstraction with `init_chat_model()` and `with_structured_output()` for typed responses
- Default model: `claude-sonnet-4-5-20250929`

**TOC Generation Strategy:**
The `TOCGenerator._extract_features()` method extracts font metadata (name, size) and text snippets from the first few blocks/lines of each page. This condensed representation is sent to the LLM which identifies chapter/section headings based on font patterns and returns structured `TOCEntry` objects with confidence scores.
- when making changes, always make sure formatting, linting, type checks, and tests work afterwards. We use ruff, mypy and pytest for these, and can run them via uv, e.g. `uv run ruff ...`, `uv run mypy`, etc.
- When writing unit-tests, use variables and/or pytest fixtures (e.g. via conftext.py and \@pytest.fixture decorator) for fixture values and objects, rather than repeating literal values in test setup and assertions. Prefer using pytest's `\@pytest.mark.parametrize` decorator when you wish to test different values or combinations of values rather than creating repeatitive standalone test cases.
