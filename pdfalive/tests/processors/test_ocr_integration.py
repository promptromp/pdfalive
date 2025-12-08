"""Integration tests for OCR processor with real PDF files."""

from collections.abc import Generator
from importlib import resources
from pathlib import Path

import pymupdf
import pytest

from pdfalive.processors.ocr_detection import NoTextDetectionStrategy
from pdfalive.processors.ocr_processor import OCRProcessor
from pdfalive.tests import fixtures


@pytest.fixture
def example_pdf_path() -> Generator[Path]:
    """Return path to the example PDF fixture."""
    with resources.as_file(resources.files(fixtures) / "example.pdf") as path:
        if not path.exists():
            pytest.skip(f"Test fixture not found: {path}")
        yield path


class TestOCRIntegration:
    """Integration tests for OCR functionality with real PDFs."""

    def test_example_pdf_needs_ocr(self, example_pdf_path: Path):
        """Test that example PDF is detected as needing OCR."""
        doc = pymupdf.open(str(example_pdf_path))
        strategy = NoTextDetectionStrategy()

        needs_ocr = strategy.needs_ocr(doc)
        doc.close()

        # The example PDF should be a scanned document without text
        assert needs_ocr is True, "Expected example.pdf to need OCR (no extractable text)"

    def test_ocr_extracts_text_from_example_pdf(self, example_pdf_path: Path):
        """Test that OCR actually extracts text from the example PDF."""
        doc = pymupdf.open(str(example_pdf_path))

        # Verify no text before OCR
        strategy = NoTextDetectionStrategy()
        assert strategy.needs_ocr(doc) is True, "Document should have no text before OCR"

        # Perform OCR - returns a NEW document with OCR text layer
        processor = OCRProcessor(language="eng", dpi=150)  # Lower DPI for faster tests
        ocr_doc = processor.process_in_memory(doc, show_progress=False)
        doc.close()

        # Verify text is now extractable in the OCR'd document
        has_text_after = False
        for page in ocr_doc:
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                has_text_after = True
                                break

        ocr_doc.close()

        assert has_text_after, "OCR should have extracted text from the document"

    def test_ocr_text_persists_after_save(self, example_pdf_path: Path, tmp_path: Path):
        """Test that OCR text persists when document is saved and reopened."""
        output_path = tmp_path / "ocr_output.pdf"

        # Open and OCR the document
        doc = pymupdf.open(str(example_pdf_path))
        processor = OCRProcessor(language="eng", dpi=150)
        ocr_doc = processor.process_in_memory(doc, show_progress=False)
        doc.close()

        # Save to new file
        ocr_doc.save(str(output_path))
        ocr_doc.close()

        # Reopen and verify text is present
        reopened_doc = pymupdf.open(str(output_path))
        strategy = NoTextDetectionStrategy()

        needs_ocr_after = strategy.needs_ocr(reopened_doc)
        reopened_doc.close()

        assert needs_ocr_after is False, "Saved document should have extractable text"

    def test_document_has_text_after_process_in_memory(self, example_pdf_path: Path):
        """Test that process_in_memory returns a document with OCR text."""
        doc = pymupdf.open(str(example_pdf_path))

        # Check initial state
        initial_text = ""
        for page in doc:
            initial_text += page.get_text()

        # Perform OCR - returns a NEW document with OCR text layer
        processor = OCRProcessor(language="eng", dpi=150)
        ocr_doc = processor.process_in_memory(doc, show_progress=False)
        doc.close()

        # Check text in the OCR'd document
        final_text = ""
        for page in ocr_doc:
            final_text += page.get_text()

        ocr_doc.close()

        # The OCR'd document should have more text than the original
        assert len(final_text) > len(initial_text), (
            f"Expected text after OCR ({len(final_text)} chars) to be greater than before ({len(initial_text)} chars)"
        )
