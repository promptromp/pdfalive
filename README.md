![pdfalive logo](https://github.com/promptromp/pdfalive/raw/main/docs/assets/pdfalive.png)

--------------------------------------------------------------------------------

[![CI](https://github.com/promptromp/pdfalive/actions/workflows/ci.yml/badge.svg)](https://github.com/promptromp/pdfalive/actions/workflows/ci.yml)
[![GitHub License](https://img.shields.io/github/license/promptromp/pdfalive)](https://github.com/promptromp/pdfalive/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/pdfalive)](https://pypi.org/project/pdfalive/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pdfalive)](https://pypi.org/project/pdfalive/)

A Python library and CLI toolkit that brings PDF files alive with the power of LLMs.

## Highlights

| Feature | Details |
|---------|---------|
| **Automatic TOC generation** | Generate clickable Table of Contents (PDF bookmarks) using LLM inference with intelligent batching for arbitrarily large documents |
| **Smart OCR detection** | Automatically detects scanned PDFs and performs OCR via [Tesseract](https://github.com/tesseract-ocr/tesseract) when needed |
| **Intelligent file renaming** | Batch rename files using natural language instructions with LLM-powered inference and confidence scoring |
| **Multi-provider LLM support** | Use any LLM provider via [LangChain](https://github.com/langchain-ai/langchain): OpenAI, Anthropic, local models via [Ollama](https://ollama.ai/), and more |
| **TOC postprocessing** | Optional second LLM pass cross-references against printed TOC pages to fix typos, remove duplicates, and correct hierarchy |
| **TOML configuration** | Set persistent defaults for any CLI option via `pdfalive.toml` config files with per-command sections |
| **Built-in resilience** | Automatic retry logic with exponential backoff for handling API rate limits |

## Installation

[Tesseract](https://github.com/tesseract-ocr/tesseract) is required for OCR functionality. On macOS:

```bash
brew install tesseract
```

Install pdfalive via [pip](https://pip.pypa.io/):

```bash
pip install pdfalive
```

Or run directly without installation using [uvx](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx pdfalive generate-toc input.pdf output.pdf
```

## Usage

Use `--help` on any command for detailed options:

```bash
pdfalive --help
pdfalive generate-toc --help
```

### generate-toc

Generate a clickable Table of Contents using PDF bookmarks. The tool extracts font and text features from the PDF and uses an LLM to intelligently identify chapter and section headings.

```bash
pdfalive generate-toc input.pdf output.pdf

# Or modify the file in place
pdfalive generate-toc --inplace input.pdf
```

**Choosing an LLM:**

By default, pdfalive uses the latest OpenAI model. Use any [LangChain-supported model](https://python.langchain.com/docs/integrations/chat/):

```bash
# Use Claude
pdfalive generate-toc --model-identifier 'claude-sonnet-4-5' input.pdf output.pdf

# Use a local model via Ollama
pdfalive generate-toc --model-identifier 'ollama/llama3' input.pdf output.pdf
```

Set the appropriate API key for your provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

**Scanned PDFs:**

OCR is enabled by default. Scanned documents without extractable text are automatically detected and processed:

```bash
# Default: OCR text layer discarded after TOC generation (preserves file size)
pdfalive generate-toc scanned.pdf output.pdf

# Include OCR text layer in output (makes PDF searchable)
pdfalive generate-toc --ocr-output scanned.pdf output.pdf

# Disable automatic OCR entirely
pdfalive generate-toc --no-ocr input.pdf output.pdf
```

**Postprocessing:**

For documents with a printed table of contents page, enable LLM postprocessing to refine results:

```bash
pdfalive generate-toc --postprocess input.pdf output.pdf
```

Postprocessing uses an additional LLM call to:
- Remove duplicate entries and fix typos
- Cross-reference against any printed TOC found in the document
- Add missing entries and correct page numbers

**Other options:**

| Option | Description |
|--------|-------------|
| `--inplace` | Modify the input file in place instead of creating a new output file |
| `--force` | Overwrite existing TOC if the PDF already has bookmarks |
| `--ocr-language` | Set OCR language (default: `eng`). Use [Tesseract language codes](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html) |
| `--request-delay` | Delay between LLM calls for rate limiting (default: 10s) |

### extract-text

Extract text from scanned PDFs using OCR and save to a new PDF with an embedded text layer:

```bash
pdfalive extract-text input.pdf output.pdf

# Or modify the file in place
pdfalive extract-text --inplace input.pdf
```

This creates a searchable/selectable PDF without generating a TOC.

**Options:**

| Option | Description |
|--------|-------------|
| `--inplace` | Modify the input file in place instead of creating a new output file |
| `--force` | Force OCR even if document already has text |
| `--ocr-language` | Set OCR language (default: `eng`) |
| `--ocr-dpi` | DPI resolution for OCR processing (default: 300) |

### rename

Intelligently rename files using LLM inference. Analyzes filenames and applies renaming rules based on natural language instructions.

```bash
pdfalive rename -q "Add 'REVIEWED_' prefix" *.pdf
```

**Custom naming formats:**

Specify exact formatting including special characters — the LLM respects brackets, parentheses, dashes, and other formatting:

```bash
pdfalive rename -q "[Author Last Name] - Title (Year).pdf" paper1.pdf paper2.pdf
```

**Reading paths from a file:**

When dealing with many files or long filenames that exceed command-line limits, use the `-f`/`--input-file` option to read paths from a text file (one per line):

```bash
# Generate a list of files to rename
find /path/to/docs -name "*.pdf" > files.txt

# Rename using the file list
pdfalive rename -q "Standardize filenames" -f files.txt
```

The input file supports comments (lines starting with `#`) and blank lines are ignored.

**Workflow:**

1. The tool analyzes each filename and generates rename suggestions
2. A preview table shows original names, proposed names, confidence scores, and reasoning
3. Confirm or cancel the operation (unless `-y` is used)
4. Files are renamed in place

**Automatic confirmation:**

```bash
pdfalive rename -q "Add sequential numbering prefix" -y *.pdf
```

**Options:**

| Option | Description |
|--------|-------------|
| `-f, --input-file` | Read input file paths from a text file (one per line) |
| `--model-identifier` | Choose which LLM to use (default: `gpt-5.2`) |
| `-y, --yes` | Automatically apply renames without confirmation |
| `--show-token-usage` | Display token usage statistics (default: enabled) |

## Configuration

pdfalive supports TOML configuration files for setting default options. This is useful for frequently-used settings like the `--query` argument for rename.

**Config file locations** (searched in order):
1. `pdfalive.toml` or `.pdfalive.toml` in the current directory
2. `pdfalive.toml` or `.pdfalive.toml` in your home directory
3. `~/.config/pdfalive/pdfalive.toml`

**Example `pdfalive.toml`:**

```toml
# Global settings (shared across commands)
[global]
model-identifier = "gpt-5.2"
show-token-usage = true

# Settings for generate-toc command
[generate-toc]
force = false
request-delay = 10.0
ocr = true
ocr-language = "eng"
ocr-dpi = 300
postprocess = false

# Settings for extract-text command
[extract-text]
ocr-language = "eng"
ocr-dpi = 300
force = false

# Settings for rename command
[rename]
query = "Rename to \"[Author Last Name] Book Title, Edition (Year).pdf\""
yes = false
```

**Using a specific config file:**

```bash
pdfalive --config /path/to/config.toml rename document.pdf
```

**Override hierarchy:**
1. Code defaults (lowest priority)
2. Config file values
3. CLI arguments (highest priority)

CLI arguments always override config file settings.

## Development

We use [uv](https://docs.astral.sh/uv/) to manage the project:

```bash
# Install dependencies
uv sync

# Install in editable mode
uv pip install -e .
```

**Code quality tools:**

| Tool | Purpose |
|------|---------|
| [ruff](https://docs.astral.sh/ruff/) | Formatting and linting |
| [mypy](https://mypy-lang.org/) | Static type checking |
| [pytest](https://docs.pytest.org/) | Unit testing |
| [pre-commit](https://pre-commit.com/) | Git hooks for quality checks |

```bash
# Run linting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run mypy pdfalive

# Run tests
uv run pytest
```

## License

pdfalive is distributed under the terms of the [MIT License](LICENSE).
