"""Unit tests for TOC generator processor."""

from unittest.mock import MagicMock

import pytest

from pdfalive.models.toc import TOC, TOCEntry, TOCFeature
from pdfalive.processors.toc_generator import TOCGenerator
from pdfalive.tokens import TokenUsage


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


class TestTOCGeneratorPagination:
    """Tests for TOCGenerator pagination functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MagicMock()

    @pytest.fixture
    def mock_doc(self):
        """Create a minimal mock document."""
        doc = MagicMock()
        doc.page_count = 0
        doc.get_toc.return_value = []
        doc.__iter__ = lambda self: iter([])
        return doc

    @pytest.fixture
    def sample_features(self):
        """Create sample features for multiple pages."""
        features = []
        for page_num in range(1, 101):  # 100 pages
            block_features = []
            for _ in range(3):  # 3 lines per block
                line_features = [
                    TOCFeature(
                        page_number=page_num,
                        font_name="Times-Bold",
                        font_size=16,
                        text_length=25,
                        text_snippet=f"Chapter {page_num}",
                    )
                ]
                block_features.append(line_features)
            features.append(block_features)
        return features

    def test_batch_features_single_batch(self, mock_doc, mock_llm):
        """Test that small feature sets result in a single batch."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        # Create a small feature set
        small_features = [
            [[TOCFeature(page_number=1, font_name="Bold", font_size=16, text_length=10, text_snippet="Ch 1")]]
        ]

        batches = list(generator._batch_features(small_features, max_tokens=10000))

        assert len(batches) == 1
        assert batches[0] == small_features

    def test_batch_features_multiple_batches(self, mock_doc, mock_llm, sample_features):
        """Test that large feature sets are split into multiple batches."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        # Use a small max_tokens to force multiple batches, no overlap for exact count
        batches = list(generator._batch_features(sample_features, max_tokens=500, overlap_blocks=0))

        assert len(batches) > 1
        # Verify all features are included across batches (no overlap = exact count)
        total_blocks = sum(len(batch) for batch in batches)
        assert total_blocks == len(sample_features)

    def test_batch_features_with_overlap(self, mock_doc, mock_llm, sample_features):
        """Test that batches include overlap when specified."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        batches = list(generator._batch_features(sample_features, max_tokens=500, overlap_blocks=2))

        # If we have multiple batches, later batches should start with overlapping blocks
        if len(batches) > 1:
            # The overlap should cause some duplication
            total_blocks = sum(len(batch) for batch in batches)
            # Total should be greater than original due to overlap
            assert total_blocks >= len(sample_features)

    def test_batch_features_preserves_structure(self, mock_doc, mock_llm):
        """Test that batching preserves the nested feature structure."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        # Create features with specific structure
        features = [
            [
                [
                    TOCFeature(page_number=1, font_name="Bold", font_size=16, text_length=10, text_snippet="Title 1"),
                    TOCFeature(page_number=1, font_name="Regular", font_size=12, text_length=50, text_snippet="Text"),
                ]
            ],
            [[TOCFeature(page_number=2, font_name="Bold", font_size=16, text_length=10, text_snippet="Title 2")]],
        ]

        batches = list(generator._batch_features(features, max_tokens=100000))

        # With large token limit, should be single batch with preserved structure
        assert len(batches) == 1
        assert len(batches[0]) == 2
        assert len(batches[0][0]) == 1  # One line in first block
        assert len(batches[0][0][0]) == 2  # Two features in first line

    def test_extract_toc_paginated_merges_results(self, mock_doc, mock_llm, sample_features):
        """Test that paginated extraction merges results from multiple batches."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        # Setup LLM to return different entries for each call
        call_count = [0]

        def mock_invoke(messages):
            call_count[0] += 1
            return TOC(
                entries=[
                    TOCEntry(
                        title=f"Chapter from batch {call_count[0]}",
                        page_number=call_count[0] * 10,
                        level=1,
                        confidence=0.9,
                    )
                ]
            )

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = mock_invoke
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Force multiple batches with small token limit, no delay for tests
        toc, usage = generator._extract_toc_paginated(
            sample_features, max_depth=2, max_tokens_per_batch=500, request_delay=0
        )

        # Should have entries from multiple batches, merged
        assert len(toc.entries) >= 1
        assert usage.llm_calls > 0

    def test_extract_toc_paginated_tracks_token_usage(self, mock_doc, mock_llm):
        """Test that token usage is tracked across paginated calls."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        # Create simple features
        features = [[[TOCFeature(page_number=1, font_name="Bold", font_size=16, text_length=10, text_snippet="Ch 1")]]]

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = TOC(
            entries=[TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.9)]
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm

        toc, usage = generator._extract_toc_paginated(features, max_depth=2, request_delay=0)

        assert isinstance(usage, TokenUsage)
        assert usage.llm_calls == 1
        # Token usage should be estimated (input) and recorded
        assert usage.input_tokens > 0

    def test_extract_toc_paginated_handles_duplicates(self, mock_doc, mock_llm):
        """Test that pagination correctly deduplicates overlapping entries."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        # Create overlapping features that will be in two batches
        features = []
        for i in range(10):
            feature = TOCFeature(
                page_number=i + 1, font_name="Bold", font_size=16, text_length=20, text_snippet=f"Ch {i + 1}"
            )
            features.append([[feature]])

        # LLM returns the same entry for overlapping batches
        def mock_invoke(messages):
            return TOC(
                entries=[
                    TOCEntry(title="Duplicate Chapter", page_number=5, level=1, confidence=0.9),
                    TOCEntry(title="Unique Entry", page_number=7, level=1, confidence=0.85),
                ]
            )

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = mock_invoke
        mock_llm.with_structured_output.return_value = mock_structured_llm

        toc, usage = generator._extract_toc_paginated(features, max_depth=2, max_tokens_per_batch=100, request_delay=0)

        # Despite multiple calls returning duplicates, they should be deduplicated
        titles = [e.title for e in toc.entries]
        assert titles.count("Duplicate Chapter") == 1  # Only one copy

    def test_extract_toc_paginated_uses_continuation_prompt(self, mock_doc, mock_llm, sample_features):
        """Test that continuation prompts are used for batches after the first."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        messages_received = []

        def mock_invoke(messages):
            messages_received.append(messages)
            return TOC(entries=[])

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = mock_invoke
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Force multiple batches, no delay for tests
        generator._extract_toc_paginated(sample_features, max_depth=2, max_tokens_per_batch=500, request_delay=0)

        # Should have multiple calls with different prompts
        if len(messages_received) > 1:
            # First call should use standard prompt
            first_system = messages_received[0][0].content
            # Subsequent calls should use continuation prompt
            second_system = messages_received[1][0].content
            assert "CONTINUATION" in second_system
            assert "CONTINUATION" not in first_system
