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

	# Or modify the file in place
	pdfalive generate-toc --inplace input.pdf

**Using a different LLM:**

	# Use Claude instead of the default OpenAI model
	pdfalive generate-toc --model-identifier 'claude-sonnet-5' input.pdf output.pdf

	# Use a local model via Ollama
	pdfalive generate-toc --model-identifier 'ollama/llama3' input.pdf output.pdf

Don't forget to set the appropriate API key environment variable for your provider (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

**Reasoning effort:**

Models with a reasoning-effort control accept it via the optional `--reasoning-effort`. Leave it out and each provider keeps its own default.

	# Spend less on reasoning for a straightforward document
	pdfalive generate-toc --reasoning-effort low input.pdf output.pdf

	# Spend more on a dense or badly structured one
	pdfalive generate-toc --reasoning-effort xhigh input.pdf output.pdf

Accepted levels vary by model, not just by provider: `gpt-6-sol` and `gpt-6-luna` take `none`, `low`, `medium`, `high` and `xhigh`; `gpt-6-astra` takes the same minus `none`; Anthropic models additionally offer `max`. The value is passed through unvalidated, so a model that rejects a level responds with an error naming the levels it accepts. Providers without the control (Ollama, for instance) ignore it.

**Working with scanned PDFs:**

OCR is enabled by default. Scanned documents without extractable text will be automatically detected and OCR will be performed to extract text before TOC generation.

	# Default behavior: OCR enabled, text layer discarded (preserves file size)
	pdfalive generate-toc scanned.pdf output.pdf

	# Include OCR text layer in output (makes PDF searchable)
	pdfalive generate-toc --ocr-output scanned.pdf output.pdf

	# Disable automatic OCR entirely
	pdfalive generate-toc --no-ocr input.pdf output.pdf

**Token usage:**

A summary of the LLM calls' token usage follows the result. Hide it with `--no-show-token-usage` (or `show-token-usage = false` in the config file):

	pdfalive generate-toc --no-show-token-usage input.pdf output.pdf

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

Printed TOC page numbers are mapped to PDF pages through the page numbers printed on the pages themselves (running headers/footers), so entries land on the right page even when the PDF omits or inserts pages relative to the printed book.

**Other options:**

	# Modify file in place
	pdfalive generate-toc --inplace input.pdf

	# Overwrite existing bookmarks
	pdfalive generate-toc --force input.pdf output.pdf

	# Adjust rate limiting delay between LLM calls
	pdfalive generate-toc --request-delay 5 input.pdf output.pdf


### extract-text

Extract text from scanned PDFs using OCR, creating a searchable PDF with an embedded text layer.

**Basic usage:**

	pdfalive extract-text scanned.pdf searchable.pdf

	# Or modify the file in place
	pdfalive extract-text --inplace scanned.pdf

**Options:**

	# Modify file in place
	pdfalive extract-text --inplace input.pdf

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

**Reading paths from a file:**

When dealing with many files or long filenames that exceed command-line limits, use the `-f`/`--input-file` option:

	# Generate a list of files to rename
	find /path/to/docs -name "*.pdf" > files.txt

	# Rename using the file list
	pdfalive rename -q "Standardize filenames" -f files.txt

The input file should contain one path per line. Lines starting with `#` are treated as comments and blank lines are ignored.

**Options:**

	# Use a different LLM
	pdfalive rename -q "Standardize filenames" --model-identifier 'claude-sonnet-5' *.pdf

	# Constrain reasoning effort (optional)
	pdfalive rename -q "Standardize filenames" --reasoning-effort low *.pdf

	# Skip confirmation prompt
	pdfalive rename -q "Add sequential numbering" -y *.pdf

	# Read paths from a file
	pdfalive rename -q "Add prefix" -f paths.txt

**Workflow:**

1. The tool analyzes each filename and generates rename suggestions
2. A preview table shows original names, proposed new names, confidence scores, and reasoning. A suggestion that keeps a file's current name is dropped silently. Suggestions that cannot be applied safely are skipped and listed with the reason:
   - a new name that is not a plain file name on this system: a path (`../name.pdf`, `sub/name.pdf`, `/abs/name.pdf`), `.` or `..`, or an empty name. On Windows that includes a drive (`C:name.pdf`, `C: Foo.pdf`, `X: The Unknown.pdf`) and a backslash; on macOS and Linux a colon or backslash is an ordinary character, so `X: The Unknown.pdf` is a plain name there
   - on Windows, a name Windows reserves or forbids: a device name with or without an extension (`CON`, `NUL`, `COM1`, `nul.txt`), a trailing dot or space, or a character such as `?`, `|` or `"` (checked with `ntpath.isreserved()`)
   - a new name that is hard to see, type or use: one with characters that act on the terminal or the displayed line (control characters such as a newline or tab, bidi controls such as the right-to-left override, line and paragraph separators), invisible marks (zero-width space, byte order mark, word joiner, left-to-right, right-to-left and Arabic letter marks; the zero-width joiner and non-joiner some scripts need are allowed) or an undecodable byte, one starting or ending with whitespace, or one starting with a dot when the file's current name does not (it would hide the file)
   - a suggestion that cannot be trusted to name its file: its file id names none of the files its request listed (ids count from 1 in each request), or the file name it repeats back is not that file's name as it was sent
   - differing suggestions for one file (all of them are skipped; the same suggestion repeated counts once)
   - for a file whose name was sent with escapes (see below), a new name containing a backslash: it would copy an escape into the name as literal text, which the preview shows exactly like the original character
   - two suggestions giving files in the same directory the same new name, compared without regard to letter case or Unicode normalization (both are skipped). The same new name in different directories is fine
   - a new name that is another input's current name, in any letter case, normalization or path spelling: swaps and chains of renames are not attempted, since their result would depend on the order they run in
3. You confirm or cancel the operation (unless `-y` is used)
4. Files are renamed in place (same directory). An existing file at the new name is never replaced: a case-only rename (`Report.pdf` to `report.pdf`) on a case-insensitive file system is the only rename onto a name that already exists, allowed only when the new name's exact spelling is not itself listed in the directory and it reaches the file's own entry, so a symlink or another file's hard link is never renamed onto. If any rename fails, the others are still attempted and the command exits with status 1
5. A token usage summary follows, whatever the outcome, even when the prompt is aborted or a later batch of LLM calls fails: it covers every call that completed (hide it with `--no-show-token-usage`)

With `-y`, a run in which any suggestion was skipped exits with status 1, whether or not the other renames were applied, since nobody saw the skipped list. A run that got no suggestions, or only suggestions keeping the current names, exits with status 0.

The files are sent to the LLM in requests that fit a token budget, including an estimate of the answer each file needs, and no request asks for more answer than current models can write. Each request lists its files as a JSON array (so a name cannot pass for instructions), numbered from 1, with their names: only file names are sent, never directories. A name with characters that act on the terminal, or with an undecodable byte, is sent in the escaped form pdfalive shows it in; other names, accented or in any script, are sent as they are. Each suggestion must give back both the id and the name, and suggestions are matched to files by id, so files that share a name in different directories can be renamed in one run. Inputs sharing a name are shown in the preview and the skipped list by their paths relative to the directory they have in common. A file given more than once, in any spelling (relative or absolute, through `..` or a symlinked directory, or in another letter case on a case-insensitive file system), is counted and renamed once. A symlink and the file it points to, and two hard links of one file, are separate inputs.

**Known limitation:** the check that the new name is free and the rename itself are two steps, so another program creating a file with that exact name in between would have it replaced. Python's standard library has no portable rename that refuses to replace an existing file.


### eval

Evaluate TOC generation quality against golden (ground truth) data. This is primarily a development tool: it lets you change feature-extraction heuristics, prompts, or correction logic and measure the impact on real documents before shipping.

Each evaluation case is a golden file at `evals/golden/<name>.json`:

	{
	  "name": "geometry",
	  "source_pdf": "data/My Book.pdf",
	  "postprocess": true,
	  "description": "Provenance notes for humans",
	  "entries": [
	    {"level": 1, "title": "Chapter 1: Distance and Angles", "page_number": 14},
	    ...
	  ]
	}

The `source_pdf` path is resolved relative to the parent of the evals directory (typically the repository root). The matching LLM cassette is stored at `evals/cassettes/<name>.json`.

**Record once, replay for free (VCR-style):**

	# One-time: run the live LLM and record its responses to a cassette
	pdfalive eval --mode record --case geometry

	# From then on: free, deterministic evaluation runs (no LLM calls)
	pdfalive eval --mode replay

Replay mode runs the full pipeline (feature extraction, batching, deterministic corrections, postprocess fixups) but serves LLM responses from the cassette. This makes it ideal for iterating on everything *around* the LLM calls. In strict replay (the default), a change that alters the prompts raises an error telling you to re-record; pass `--loose-replay` to replay by call order anyway (useful when a prompt change is intentional and you want a quick signal before re-recording).

**Reported metrics** (per case): precision / recall / F1 of matched titles (fuzzy matching tolerant to typos and punctuation), page accuracy (exact and within ±1 page), and hierarchy level accuracy. Missing and spurious entries are listed explicitly.

**CI / regression gating:**

	pdfalive eval --mode replay --min-f1 0.9 --min-page-accuracy 0.9

exits with code 1 if any case falls below the thresholds.

**Options:**

| Option | Description |
|--------|-------------|
| `--evals-dir` | Directory containing `golden/` and `cassettes/` (default: `evals`) |
| `--mode` | `replay` (default, free), `record` (live LLM, writes cassette), or `live` |
| `--case` | Run only the named case(s); may be repeated |
| `--model-identifier` | LLM for record/live modes (default: `gpt-5.6`) |
| `--reasoning-effort` | Reasoning effort for record/live modes (unset by default) |
| `--loose-replay` | Replay by call order even if prompts drifted since recording |
| `--request-delay` | Seconds between LLM calls (default: 0 in replay, pipeline default otherwise) |
| `--num-processes` | Parallel processes for feature extraction (default: CPU count - 1) |
| `--min-f1`, `--min-page-accuracy` | Fail (exit 1) if any case scores below threshold |

**Adding a new case:** generate a TOC with `generate-toc`, verify the bookmarks by hand (fix any errors — this is your ground truth), then save the verified entries as a golden file and record a cassette with `--mode record`.

**Scanned PDFs:** the eval runner does not perform OCR (it would dominate every run's wall-clock). Pre-OCR the document once with `pdfalive extract-text scanned.pdf ocr-copy.pdf` and point the golden file's `source_pdf` at the OCR'd copy.


## Configuration

pdfalive supports TOML configuration files for setting default CLI options. This is especially useful for frequently-used settings like the `--query` argument for rename.

**Config file locations** (searched in order):
1. `pdfalive.toml` or `.pdfalive.toml` in the current directory
2. `pdfalive.toml` or `.pdfalive.toml` in your home directory
3. `~/.config/pdfalive/pdfalive.toml`

**Example configuration file:**

	# pdfalive.toml

	# Global settings (shared across commands)
	[global]
	model-identifier = "gpt-5.6"
	reasoning-effort = "medium"   # optional; omit to use the provider's default
	show-token-usage = true      # false hides the token usage summary

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

	# Settings for eval command (development tool)
	[eval]
	evals-dir = "evals"
	mode = "replay"

**Using a specific config file:**

	pdfalive --config /path/to/config.toml rename document.pdf

**Override hierarchy:**
1. Code defaults (lowest priority)
2. Config file values
3. CLI arguments (highest priority)

CLI arguments always override config file settings.


## Tips

- For large documents, the tool automatically batches LLM requests to stay within context limits
- Rate limiting is built-in with automatic retry logic for API errors
- Use `--help` on any command for a full list of options
