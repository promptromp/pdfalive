# pdfalive

A Python library and set of CLI tools to bring PDF files alive with the magic of LLMs.


## CLI

### generate-toc

Automatically generate clickable Table of Contents (e.g. using PDF bookmarks) for a PDF file.

Example usage:

	uv run pdfalive generate-toc examples/example.pdf output.pdf --force


## Development

We use `uv` to manage the library. To install locally can run e.g. with:

	uv sync

and can then execute the CLI commands in the created local env, e.g.:

	uv run pdfalive generate-toc examples/example.pdf output.pdf --force
