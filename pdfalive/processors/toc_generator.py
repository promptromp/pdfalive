"""Table of Contents generator."""

import pymupdf
from langchain.chat_models.base import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from tqdm import tqdm

from pdfalive.models.prompts import TOC_GENERATOR_SYSTEM_PROMPT
from pdfalive.models.toc_entry import TOC


class TOCGenerator:
    """Class to generate table of contents for a PDF document."""

    def __init__(self, doc: pymupdf.Document, llm: BaseChatModel) -> None:
        self.doc = doc
        self.llm = llm

    def run(self, output_file: str, force: bool = False) -> None:
        """Generate the table of contents."""

        if self._check_for_existing_toc() and not force:
            # TODO: can also use any existing toc to guide LLM generation.
            raise ValueError("The document already has a Table of Contents. Use `--force` to overwrite.")

        features = self._extract_features(self.doc)
        toc = self._extract_toc(features)

        self.doc.set_toc(toc.to_list())
        self.doc.save(output_file)

    def _check_for_existing_toc(self) -> list:
        """Check if the document already has a TOC."""
        return self.doc.get_toc()

    def _extract_features(
        self,
        doc: pymupdf.Document,
        max_pages=None,
        max_blocks_per_page=3,
        max_lines_per_block=5,
        text_snippet_length=25,
    ) -> list:
        """Extract features from the document to generate TOC entries.

        Features are indexed by page, block, line, and span.
        They include attributes such as: font name, size, text length, and a text snippet.

        """

        # hierarchical structure of features detected in the page for (blocks, lines, spans)
        features: list[list] = []

        for ix, page in enumerate(
            tqdm(self.doc, desc="Processing pages for TOC generation", total=self.doc.page_count)
        ):
            page_number = ix + 1  # 1-indexed
            page_dict = page.get_text("dict")

            for block_ix, block in enumerate(page_dict["blocks"]):
                if block_ix >= max_blocks_per_page:
                    break

                features.append([])
                if block["type"] == 0:
                    # text block
                    for line_ix, line in enumerate(block["lines"]):
                        if line_ix >= max_lines_per_block:
                            break

                        features[-1].append([])

                        for span in line["spans"]:
                            features[-1][-1].append(
                                (
                                    page_number,
                                    span["font"],
                                    span["size"],
                                    len(span["text"]),
                                    span["text"][:text_snippet_length],
                                )
                            )

            if max_pages is not None and (ix + 1) >= max_pages:
                break

        return features

    def _extract_toc(self, features: list, max_depth=2) -> TOC:
        """Infer TOC entries from extracted features using the LLM."""
        messages = [
            SystemMessage(content=TOC_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
                Generate a table of contents based on the document features given below.
                Limit the TOC to a maximum depth of {max_depth} levels.
                \n\n
                ------------------------
                {features}

             """
            ),
        ]
        model = self.llm.with_structured_output(TOC)
        response = model.invoke(messages)

        return response
