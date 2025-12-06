"""CLI entrypoints."""

import click
import pymupdf
from langchain.chat_models import init_chat_model

from pdfalive.processors.toc_generator import TOCGenerator


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--model-identifier", type=str, default="claude-sonnet-4-5-20250929", help="LLM model to use.")
@click.option("--force", is_flag=True, default=False, help="Force overwrite existing TOC if present.")
def generate_toc(input_file: str, output_file: str, model_identifier: str, force: bool) -> None:
    """Generate a table of contents for a PDF file."""
    click.echo(f"Generating TOC for {input_file} using model {model_identifier}...")

    doc = pymupdf.open(input_file)
    llm = init_chat_model(model=model_identifier)
    processor = TOCGenerator(doc=doc, llm=llm)

    processor.run(output_file=output_file, force=force)

    click.echo(f"All done. Saved modified PDF to {output_file}.")
