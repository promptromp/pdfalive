# pdfalive

--------------------------------------------------------------------------------


[![CI](https://github.com/promptromp/pdfalive/actions/workflows/ci.yml/badge.svg)](https://github.com/promptromp/pdfalive/actions/workflows/ci.yml)
[![GitHub License](https://img.shields.io/github/license/promptromp/pdfalive)](https://github.com/promptromp/pdfalive/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/pdfalive)](https://pypi.org/project/pdfalive/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pdfalive)](https://pypi.org/project/pdfalive/)

A Python library and set of CLI tools to bring PDF files alive with the magic of LLMs.


## Usage

To use the CLIs described below, you can install the python package (`pip install pdfalive`), or run the cli directly using [uvx](https://docs.astral.sh/uv/guides/tools/):

	uvx pdfalive generate-toc input.pdf output.pdf

More detailed examples of the CLI sub-commands are provided below.
You can also use `--help` on the main command-line and any of the sub-commands to get an idea of the different options supported.

### generate-toc

Automatically generate clickable Table of Contents (e.g. using PDF bookmarks) for a PDF file by extracting features from the PDF and then calling an LLM to infer the pages and section names from these.

Example usage:

	pdfalive generate-toc input.pdf output.pdf

By default we use the latest Anthropic Claude Sonnet available, but you can change this by setting the model as part of invocation:

	pdfalive generate-toc --model-identifier 'claude-haiku-4-5' input.pdf output.pdf

Note that for using Anthropic models you'd want to set your api key via the environment variable `ANTHROPIC_API_KEY`. Similar mechanisms apply to OpenAI (`OPENAI_API_KEY`) and other vendors.


## Development

We use `uv` to manage the library. To install locally can run e.g. with:

	uv sync
	uv pip install -e .

We use `ruff` for formatting and linting, `mypy` for static type checking, and `pytest` for running unit-tests. We also use [pre-commit](https://pre-commit.com/) for ensuring high-quality commits.
