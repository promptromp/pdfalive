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

OCR is enabled by default. Scanned documents without extractable text will be automatically detected and OCR will be performed to extract text before TOC generation.

	# Default behavior: OCR enabled, text layer discarded (preserves file size)
	pdfalive generate-toc scanned.pdf output.pdf

	# Include OCR text layer in output (makes PDF searchable)
	pdfalive generate-toc --ocr-output scanned.pdf output.pdf

	# Disable automatic OCR entirely
	pdfalive generate-toc --no-ocr input.pdf output.pdf

**OCR options:**

	# Use a different language for OCR (default: English)
	pdfalive generate-toc --ocr-language deu german_document.pdf output.pdf

	# Adjust OCR resolution (default: 300 DPI)
	pdfalive generate-toc --ocr-dpi 150 input.pdf output.pdf

**Postprocessing for improved quality:**

Enable LLM postprocessing to refine the generated TOC. This is especially useful for documents that have a printed table of contents page:

	# Enable postprocessing to clean up and improve the TOC
	pdfalive generate-toc --postprocess input.pdf output.pdf

Postprocessing performs an additional LLM call that:
- Removes duplicate entries and fixes typos
- Cross-references against any printed TOC found in the first pages
- Adds missing entries and corrects page numbers based on the printed TOC
- Ensures consistent hierarchy levels

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


### rename

Intelligently rename PDF files using LLM-powered inference. The tool analyzes filenames and applies your renaming instructions.

**Basic usage:**

	pdfalive rename -q "Add 'REVIEWED_' prefix" *.pdf

**Custom naming formats with special characters:**

	# Rename to format: [Author Last Name] - Title (Year).pdf
	pdfalive rename -q "[Author Last Name] - Title (Year).pdf" paper1.pdf paper2.pdf

	# Rename with curly braces: {Category}_Filename.pdf
	pdfalive rename -q "{Category}_Filename.pdf" *.pdf

The LLM respects your exact formatting, including brackets, parentheses, dashes, and other special characters.

**Options:**

	# Use a different LLM
	pdfalive rename -q "Standardize filenames" --model-identifier 'claude-sonnet-4-5' *.pdf

	# Skip confirmation prompt
	pdfalive rename -q "Add sequential numbering" -y *.pdf

**Workflow:**

1. The tool analyzes each filename and generates rename suggestions
2. A preview table shows original names, proposed new names, confidence scores, and reasoning
3. You confirm or cancel the operation (unless `-y` is used)
4. Files are renamed in place (same directory)


## Tips

- For large documents, the tool automatically batches LLM requests to stay within context limits
- Rate limiting is built-in with automatic retry logic for API errors
- Use `--help` on any command for a full list of options
