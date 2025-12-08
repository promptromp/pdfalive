![pdfalive logo](https://github.com/promptromp/pdfalive/raw/main/docs/assets/pdfalive.png)

--------------------------------------------------------------------------------


[![CI](https://github.com/promptromp/pdfalive/actions/workflows/ci.yml/badge.svg)](https://github.com/promptromp/pdfalive/actions/workflows/ci.yml)
[![GitHub License](https://img.shields.io/github/license/promptromp/pdfalive)](https://github.com/promptromp/pdfalive/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/pdfalive)](https://pypi.org/project/pdfalive/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pdfalive)](https://pypi.org/project/pdfalive/)

*pdfalive*: A Python library and set of CLI tools to bring PDF files alive with the magic of LLMs.

Features:

* Automatically generate a Table of Contents via PDF Bookmarks for PDF file using LLMs. Supports arbitrarily large files with intelligent batching.
* Automatically detect if OCR is needed to parse text from raster data. If needed, performs OCR via Tesseract OCR library.
* Choose which LLM to use from any vendor. Supports using local models via `ollama` as well. Retry logic included for handling rate limits.

## Installation

the [tesseract](https://github.com/tesseract-ocr/tesseract) library is required for OCR. This is used for PDFs where text is not parsed. On MacOS, can install via Homebrew:

	brew install tesseract

You can then install the pdfalive package via pip for example:

	pip install pdfalive


## Usage

To use the CLIs described below, you can install the python package (`pip install pdfalive`), or run the cli directly using [uvx](https://docs.astral.sh/uv/guides/tools/):

	uvx pdfalive generate-toc input.pdf output.pdf

More detailed examples of the CLI sub-commands are provided below.
You can also use `--help` on the main command-line and any of the sub-commands to get an idea of the different options supported.

### generate-toc

Automatically generate clickable Table of Contents (using PDF bookmarks) for a PDF file. The tool extracts font and text features from the PDF and uses an LLM to intelligently identify chapter and section headings.

Basic usage:

	pdfalive generate-toc input.pdf output.pdf

**Choosing an LLM:** By default we use the latest OpenAI model, but you can use any LLM supported by LangChain:

	pdfalive generate-toc --model-identifier 'claude-sonnet-4-5' input.pdf output.pdf

Set the appropriate API key for your provider (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

**Scanned PDFs:** OCR is enabled by default. If your PDF is a scanned document without extractable text, OCR will be performed automatically to extract text before TOC generation.

By default, the OCR text layer is discarded after TOC generation (preserving original file size). To include the OCR text layer in the output (making it searchable):

	pdfalive generate-toc --ocr-output scanned.pdf output.pdf

To disable automatic OCR detection entirely:

	pdfalive generate-toc --no-ocr input.pdf output.pdf

**Other useful options:**

- `--force` - Overwrite existing TOC if the PDF already has bookmarks
- `--ocr-language` - Set OCR language (default: `eng`). Use Tesseract language codes like `deu`, `fra`, etc.

### extract-text

Extract text from scanned PDFs using OCR and save to a new PDF with an embedded text layer:

	pdfalive extract-text input.pdf output.pdf

This is useful when you want a searchable/selectable text layer without generating a TOC.


## Development

We use `uv` to manage the library. To install locally can run e.g. with:

	uv sync
	uv pip install -e .

We use `ruff` for formatting and linting, `mypy` for static type checking, and `pytest` for running unit-tests. We also use [pre-commit](https://pre-commit.com/) for ensuring high-quality commits.
