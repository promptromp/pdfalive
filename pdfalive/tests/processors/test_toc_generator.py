"""Unit tests for TOC generator processor."""

from unittest.mock import MagicMock

import pytest

from pdfalive.models.toc import TOC, TOCEntry
from pdfalive.processors.toc_generator import TOCGenerator


@pytest.fixture
def mock_doc():
    """Create a mock PyMuPDF document."""
    doc = MagicMock()
    doc.page_count = 2
    doc.get_toc.return_value = []

    # Mock page iteration
    page1 = MagicMock()
    page1.get_text.return_value = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {"font": "Times-Bold", "size": 16, "text": "Chapter 1: Introduction"},
                        ]
                    }
                ],
            }
        ]
    }
    page2 = MagicMock()
    page2.get_text.return_value = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {"font": "Times-Bold", "size": 16, "text": "Chapter 2: Methods"},
                        ]
                    }
                ],
            }
        ]
    }
    doc.__iter__ = lambda self: iter([page1, page2])

    return doc


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MagicMock()


@pytest.fixture
def sample_toc_response():
    """Sample TOC response from LLM."""
    return TOC(
        entries=[
            TOCEntry(title="Chapter 1: Introduction", page_number=1, level=1, confidence=0.95),
            TOCEntry(title="Chapter 2: Methods", page_number=2, level=1, confidence=0.90),
        ]
    )


class TestTOCGenerator:
    """Tests for TOCGenerator processor."""

    def test_check_for_existing_toc_empty(self, mock_doc, mock_llm):
        """Test detection when no existing TOC."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        result = generator._check_for_existing_toc()

        assert result == []

    def test_check_for_existing_toc_present(self, mock_doc, mock_llm):
        """Test detection when TOC exists."""
        existing_toc = [[1, "Existing Chapter", 1]]
        mock_doc.get_toc.return_value = existing_toc
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        result = generator._check_for_existing_toc()

        assert result == existing_toc

    def test_extract_features(self, mock_doc, mock_llm):
        """Test feature extraction from document."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        features = generator._extract_features(mock_doc)

        assert len(features) > 0
        # Check that features contain expected TOCFeature structure
        first_span = features[0][0][0]
        assert first_span.page_number == 1
        assert first_span.font_name == "Times-Bold"
        assert first_span.font_size == 16

    def test_run_success(self, mock_doc, mock_llm, sample_toc_response, tmp_path):
        """Test successful TOC generation run."""
        output_file = tmp_path / "output.pdf"

        # Setup LLM mock to return structured TOC
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = sample_toc_response
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        generator.run(output_file=str(output_file))

        # Verify TOC was set on document
        mock_doc.set_toc.assert_called_once()
        toc_arg = mock_doc.set_toc.call_args[0][0]
        assert len(toc_arg) == 2
        assert toc_arg[0] == [1, "Chapter 1: Introduction", 1]

        # Verify document was saved
        mock_doc.save.assert_called_once_with(str(output_file))

    def test_run_raises_when_toc_exists_without_force(self, mock_doc, mock_llm):
        """Test that run raises error when TOC exists and force=False."""
        mock_doc.get_toc.return_value = [[1, "Existing", 1]]
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        with pytest.raises(ValueError, match="already has a Table of Contents"):
            generator.run(output_file="output.pdf", force=False)

    def test_run_overwrites_with_force(self, mock_doc, mock_llm, sample_toc_response, tmp_path):
        """Test that run overwrites existing TOC when force=True."""
        output_file = tmp_path / "output.pdf"
        mock_doc.get_toc.return_value = [[1, "Existing", 1]]

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = sample_toc_response
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        generator.run(output_file=str(output_file), force=True)

        # Should succeed and set new TOC
        mock_doc.set_toc.assert_called_once()
        mock_doc.save.assert_called_once()
