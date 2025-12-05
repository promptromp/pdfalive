"""CLI entrypoints."""
import click
import pymupdf


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def generate_toc(input_file: str, output_file: str) -> None:
    """Generate a table of contents for a PDF file."""
    doc = pymupdf.open(input_file)

    click.echo(f"Reading all pages in {input_file}...")
    for ix, page in enumerate(doc):
        text = page.get_text()
        print(f"page {ix}, text: {text[:15]}...")


    