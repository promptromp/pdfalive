"""Integration tests for OCR processor with real PDF files."""

from collections.abc import Generator
from importlib import resources
from pathlib import Path

import pymupdf
import pytest

from pdfalive.processors.ocr_detection import NoTextDetectionStrategy
from pdfalive.processors.ocr_processor import OCRProcessor
from pdfalive.processors.toc_generator import apply_toc_to_document
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


class TestNoOcrOutputFlag:
    """Tests for --no-ocr-output flag functionality."""

    def test_apply_toc_to_original_preserves_bookmarks(self, example_pdf_path: Path, tmp_path: Path):
        """Test that TOC can be applied to original document after extracting from OCR'd version."""
        output_path = tmp_path / "output_with_toc.pdf"

        # Open original document
        original_doc = pymupdf.open(str(example_pdf_path))
        original_size = len(original_doc.tobytes())

        # Create a sample TOC
        sample_toc = [[1, "Chapter 1", 1], [1, "Chapter 2", 1]]

        # Apply TOC to original and save
        apply_toc_to_document(original_doc, sample_toc, str(output_path))
        original_doc.close()

        # Reopen and verify TOC is present
        reopened = pymupdf.open(str(output_path))
        saved_toc = reopened.get_toc()
        saved_size = len(reopened.tobytes())
        reopened.close()

        assert saved_toc == sample_toc, "TOC should be preserved in saved document"
        # File size should be similar to original (not inflated by OCR data)
        # Allow 20% variance for metadata changes
        assert saved_size < original_size * 1.2, (
            f"Saved file ({saved_size} bytes) should be similar to original ({original_size} bytes)"
        )

    def test_no_ocr_output_preserves_original_structure(self, example_pdf_path: Path, tmp_path: Path):
        """Test that --no-ocr-output preserves original file structure (bookmarks only)."""
        output_without_ocr = tmp_path / "without_ocr.pdf"

        # Get original file size
        original_doc = pymupdf.open(str(example_pdf_path))
        original_size = len(original_doc.tobytes())

        # Create TOC and apply to original (simulating --no-ocr-output behavior)
        sample_toc = [[1, "Test Chapter", 1]]
        apply_toc_to_document(original_doc, sample_toc, str(output_without_ocr))
        original_doc.close()

        # Verify the output
        output_doc = pymupdf.open(str(output_without_ocr))
        output_size = len(output_doc.tobytes())
        output_toc = output_doc.get_toc()

        # Verify original still doesn't have extractable text (OCR was not persisted)
        strategy = NoTextDetectionStrategy()
        still_needs_ocr = strategy.needs_ocr(output_doc)
        output_doc.close()

        # Assertions
        assert output_toc == sample_toc, "TOC should be present in output"
        assert still_needs_ocr is True, "Output should still need OCR (OCR text not persisted)"
        # Output size should be similar to original (just added bookmarks, no OCR data)
        assert abs(output_size - original_size) < original_size * 0.1, (
            f"Output size ({output_size} bytes) should be close to original ({original_size} bytes)"
        )

    def test_original_doc_with_existing_text_preserved(self, tmp_path: Path):
        """Test that documents with existing text are not affected by --no-ocr-output."""
        # Create a simple PDF with text
        test_pdf = tmp_path / "with_text.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "This is existing text")
        doc.save(str(test_pdf))
        doc.close()

        # Open and check it has text
        doc = pymupdf.open(str(test_pdf))
        strategy = NoTextDetectionStrategy()
        needs_ocr = strategy.needs_ocr(doc)

        assert needs_ocr is False, "Document with text should not need OCR"

        # Verify text is extractable
        text = doc[0].get_text()
        doc.close()

        assert "existing text" in text.lower()
