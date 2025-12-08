# Usage Guide: pdfalive

`pdfalive` is a Python package with a command-line interface for enhancing PDF files using LLMs.


## Installation

Install via pip:

	pip install pdfalive

Or run directly with [uvx](https://docs.astral.sh/uv/) (no installation needed):

	uvx pdfalive --help


## Commands

### generate-toc

Generate a clickable Table of Contents for any PDF. The tool analyzes font sizes, text patterns, and document structure to identify chapters and sections.

**Basic usage:**

	pdfalive generate-toc input.pdf output.pdf

**Using a different LLM:**

	# Use Claude instead of the default OpenAI model
	pdfalive generate-toc --model-identifier 'claude-sonnet-4-5' input.pdf output.pdf

	# Use a local model via Ollama
	pdfalive generate-toc --model-identifier 'ollama/llama3' input.pdf output.pdf

Don't forget to set the appropriate API key environment variable for your provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

**Working with scanned PDFs:**

For scanned documents without extractable text, OCR is performed automatically. You have several options:

	# Default: Include OCR text layer in output (larger file, but searchable)
	pdfalive generate-toc scanned.pdf output.pdf

	# Discard OCR text after TOC generation (keeps original file size)
	pdfalive generate-toc --no-ocr-output scanned.pdf output.pdf

	# Disable OCR entirely (only works if PDF already has text)
	pdfalive generate-toc --no-ocr document.pdf output.pdf

**OCR options:**

	# Use a different language for OCR (default: English)
	pdfalive generate-toc --ocr-language deu german_document.pdf output.pdf

	# Adjust OCR resolution (default: 300 DPI)
	pdfalive generate-toc --ocr-dpi 150 input.pdf output.pdf

**Other options:**

	# Overwrite existing bookmarks
	pdfalive generate-toc --force input.pdf output.pdf

	# Adjust rate limiting delay between LLM calls
	pdfalive generate-toc --request-delay 5 input.pdf output.pdf


### extract-text

Extract text from scanned PDFs using OCR, creating a searchable PDF with an embedded text layer.

**Basic usage:**

	pdfalive extract-text scanned.pdf searchable.pdf

**Options:**

	# Force OCR even if document already has text
	pdfalive extract-text --force input.pdf output.pdf

	# Use a different language
	pdfalive extract-text --ocr-language fra french_document.pdf output.pdf


## Tips

- For large documents, the tool automatically batches LLM requests to stay within context limits
- Rate limiting is built-in with automatic retry logic for API errors
- Use `--help` on any command for a full list of options
