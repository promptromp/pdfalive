"""Unit tests for TOC generator processor."""

from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from pdfalive.models.toc import TOC, TOCEntry, TOCFeature
from pdfalive.processors.toc_generator import (
    TOCGenerator,
    _compute_body_font_profile,
    _extract_features_from_page_range,
    _is_bold_font,
    _is_heading_candidate,
)
from pdfalive.tokens import TokenUsage


@pytest.fixture
def mock_doc():
    """Create a mock PyMuPDF document."""
    doc = MagicMock()
    doc.page_count = 2
    doc.get_toc.return_value = []
    # Set name to None to force sequential processing (mocks can't be pickled for multiprocessing)
    doc.name = None

    page_height = 800.0

    # Mock page iteration
    page1 = MagicMock()
    page1.rect.height = page_height
    page1.get_text.return_value = {
        "height": page_height,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "font": "Times-Bold",
                                "size": 16,
                                "text": "Chapter 1: Introduction",
                                "bbox": (50, 100, 400, 120),
                                "flags": 16,
                            },
                        ]
                    }
                ],
            }
        ],
    }
    page2 = MagicMock()
    page2.rect.height = page_height
    page2.get_text.return_value = {
        "height": page_height,
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "font": "Times-Bold",
                                "size": 16,
                                "text": "Chapter 2: Methods",
                                "bbox": (50, 100, 400, 120),
                                "flags": 16,
                            },
                        ]
                    }
                ],
            }
        ],
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

        features = generator._extract_features(mock_doc, show_progress=False)

        assert len(features) > 0
        # Check that features contain expected TOCFeature structure
        first_span = features[0][0][0]
        assert first_span.page_number == 1
        assert first_span.font_name == "Times-Bold"
        assert first_span.font_size == 16

    def test_extract_features_sequential(self, mock_doc, mock_llm):
        """Test sequential feature extraction."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)

        features = generator._extract_features_sequential(mock_doc, show_progress=False)

        assert len(features) > 0
        first_span = features[0][0][0]
        assert first_span.page_number == 1
        assert first_span.font_name == "Times-Bold"

    def test_init_with_custom_num_processes(self, mock_doc, mock_llm):
        """Test TOCGenerator initialization with custom num_processes."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm, num_processes=4)

        assert generator.num_processes == 4

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


class TestFeatureExtractionMultiprocessing:
    """Tests for multiprocessing feature extraction and merge logic."""

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
        doc.name = None  # Force sequential processing for basic tests
        doc.__iter__ = lambda self: iter([])
        return doc

    def test_extract_features_from_page_range_single_process(self):
        """Test worker function with a single process handling all pages."""
        # Create mock document data
        mock_page_data = {
            "height": 800,
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {"spans": [{"font": "Times-Bold", "size": 18, "text": "Chapter Title"}]},
                        {"spans": [{"font": "Times-Roman", "size": 12, "text": "Body text"}]},
                    ],
                }
            ],
        }

        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = 3

            # Create mock pages
            mock_pages = []
            for _ in range(3):
                mock_page = MagicMock()
                mock_page.get_text.return_value = mock_page_data
                mock_pages.append(mock_page)

            mock_doc.__getitem__ = lambda self, idx: mock_pages[idx]
            mock_pymupdf.open.return_value = mock_doc

            # Single process handling all 3 pages
            args = (0, 1, "/fake/path.pdf", 3, 5, 25)
            start, end, features, _ = _extract_features_from_page_range(args)

            assert start == 0
            assert end == 3
            # Should have features from all 3 pages (1 block per page)
            assert len(features) == 3
            # Each block should have 2 lines
            assert len(features[0]) == 2
            # First line, first span should be the chapter title
            assert features[0][0][0].font_name == "Times-Bold"
            assert features[0][0][0].font_size == 18
            assert features[0][0][0].text_snippet == "Chapter Title"

    def test_extract_features_from_page_range_calculates_correct_ranges(self):
        """Test that page ranges are calculated correctly for multiple processes."""
        mock_page_data = {
            "height": 800,
            "blocks": [{"type": 0, "lines": [{"spans": [{"font": "Arial", "size": 12, "text": "Test"}]}]}],
        }

        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = 12

            mock_page = MagicMock()
            mock_page.get_text.return_value = mock_page_data
            mock_doc.__getitem__ = lambda self, idx: mock_page
            mock_pymupdf.open.return_value = mock_doc

            # Test with 4 processes on 12 pages
            # Process 0: pages 0-2 (3 pages)
            start0, end0, _, _ = _extract_features_from_page_range((0, 4, "/fake/path.pdf", 3, 5, 25))
            assert start0 == 0
            assert end0 == 3

            # Process 1: pages 3-5 (3 pages)
            start1, end1, _, _ = _extract_features_from_page_range((1, 4, "/fake/path.pdf", 3, 5, 25))
            assert start1 == 3
            assert end1 == 6

            # Process 2: pages 6-8 (3 pages)
            start2, end2, _, _ = _extract_features_from_page_range((2, 4, "/fake/path.pdf", 3, 5, 25))
            assert start2 == 6
            assert end2 == 9

            # Process 3 (last): pages 9-11 (gets remainder)
            start3, end3, _, _ = _extract_features_from_page_range((3, 4, "/fake/path.pdf", 3, 5, 25))
            assert start3 == 9
            assert end3 == 12

    @pytest.mark.parametrize(
        "num_pages,num_processes,expected_ranges",
        [
            (10, 2, [(0, 5), (5, 10)]),
            (10, 3, [(0, 3), (3, 6), (6, 10)]),
            (10, 4, [(0, 2), (2, 4), (4, 6), (6, 10)]),
            (7, 3, [(0, 2), (2, 4), (4, 7)]),
            (100, 5, [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]),
        ],
    )
    def test_page_range_distribution(self, num_pages, num_processes, expected_ranges):
        """Test that pages are distributed correctly across processes."""
        mock_page_data = {
            "height": 800,
            "blocks": [{"type": 0, "lines": [{"spans": [{"font": "Arial", "size": 12, "text": "X"}]}]}],
        }

        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = num_pages

            mock_page = MagicMock()
            mock_page.get_text.return_value = mock_page_data
            mock_doc.__getitem__ = lambda self, idx: mock_page
            mock_pymupdf.open.return_value = mock_doc

            actual_ranges = []
            for proc_idx in range(num_processes):
                args = (proc_idx, num_processes, "/fake/path.pdf", 3, 5, 25)
                start, end, _, _ = _extract_features_from_page_range(args)
                actual_ranges.append((start, end))

            assert actual_ranges == expected_ranges

    def test_merged_features_maintain_page_order(self):
        """Test that merged features from multiple processes maintain correct page order."""
        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = 9

            # Create pages with distinct content per page
            def create_page_data(page_idx):
                return {
                    "height": 800,
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [{"spans": [{"font": "Bold", "size": 16, "text": f"Page {page_idx + 1} Title"}]}],
                        }
                    ],
                }

            mock_pages = [MagicMock() for _ in range(9)]
            for i, page in enumerate(mock_pages):
                page.get_text.return_value = create_page_data(i)

            mock_doc.__getitem__ = lambda self, idx: mock_pages[idx]
            mock_pymupdf.open.return_value = mock_doc

            # Simulate 3 processes
            results = []
            for proc_idx in range(3):
                start, end, features, _ = _extract_features_from_page_range((proc_idx, 3, "/fake/path.pdf", 3, 5, 25))
                results.append((start, end, features))

            # Sort by start page (simulating what _extract_features_parallel does)
            results = sorted(results, key=lambda x: x[0])

            # Merge features
            all_features = []
            for _, _, features in results:
                all_features.extend(features)

            # Verify order: should have 9 blocks, each with page-specific content
            assert len(all_features) == 9
            for i, block in enumerate(all_features):
                page_num = block[0][0].page_number
                text = block[0][0].text_snippet
                assert page_num == i + 1, f"Expected page {i + 1}, got {page_num}"
                assert f"Page {i + 1}" in text, f"Expected 'Page {i + 1}' in text, got '{text}'"

    def test_merged_features_with_multiple_blocks_per_page(self):
        """Test merging when pages have multiple blocks."""
        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = 4

            def create_multi_block_page(page_idx):
                return {
                    "height": 800,
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [{"spans": [{"font": "Bold", "size": 18, "text": f"P{page_idx + 1} Block1"}]}],
                        },
                        {
                            "type": 0,
                            "lines": [{"spans": [{"font": "Regular", "size": 12, "text": f"P{page_idx + 1} Block2"}]}],
                        },
                    ],
                }

            mock_pages = [MagicMock() for _ in range(4)]
            for i, page in enumerate(mock_pages):
                page.get_text.return_value = create_multi_block_page(i)

            mock_doc.__getitem__ = lambda self, idx: mock_pages[idx]
            mock_pymupdf.open.return_value = mock_doc

            # Simulate 2 processes
            results = []
            for proc_idx in range(2):
                start, end, features, _ = _extract_features_from_page_range((proc_idx, 2, "/fake/path.pdf", 3, 5, 25))
                results.append((start, end, features))

            results = sorted(results, key=lambda x: x[0])

            all_features = []
            for _, _, features in results:
                all_features.extend(features)

            # 4 pages × 2 blocks = 8 blocks total
            assert len(all_features) == 8

            # Verify blocks are in order: P1B1, P1B2, P2B1, P2B2, P3B1, P3B2, P4B1, P4B2
            expected_order = [
                ("P1 Block1", 1),
                ("P1 Block2", 1),
                ("P2 Block1", 2),
                ("P2 Block2", 2),
                ("P3 Block1", 3),
                ("P3 Block2", 3),
                ("P4 Block1", 4),
                ("P4 Block2", 4),
            ]
            for i, (expected_text, expected_page) in enumerate(expected_order):
                actual_text = all_features[i][0][0].text_snippet
                actual_page = all_features[i][0][0].page_number
                assert expected_text in actual_text, f"Block {i}: expected '{expected_text}' in '{actual_text}'"
                assert actual_page == expected_page, f"Block {i}: expected page {expected_page}, got {actual_page}"

    def test_extract_features_respects_max_blocks_per_page(self):
        """Test that max_blocks_per_page limit is respected."""
        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = 1

            # Page with 5 blocks
            page_data = {
                "height": 800,
                "blocks": [
                    {"type": 0, "lines": [{"spans": [{"font": "Bold", "size": 12, "text": f"Block {i}"}]}]}
                    for i in range(5)
                ],
            }

            mock_page = MagicMock()
            mock_page.get_text.return_value = page_data
            mock_doc.__getitem__ = lambda self, idx: mock_page
            mock_pymupdf.open.return_value = mock_doc

            # Limit to 2 blocks per page
            _, _, features, remaining = _extract_features_from_page_range((0, 1, "/fake/path.pdf", 2, 5, 25))

            assert len(features) == 2
            assert "Block 0" in features[0][0][0].text_snippet
            assert "Block 1" in features[1][0][0].text_snippet
            # Remaining blocks should be buffered
            assert len(remaining) > 0

    def test_extract_features_respects_max_lines_per_block(self):
        """Test that max_lines_per_block limit is respected."""
        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = 1

            # Block with 5 lines
            page_data = {
                "height": 800,
                "blocks": [
                    {
                        "type": 0,
                        "lines": [{"spans": [{"font": "Bold", "size": 12, "text": f"Line {i}"}]} for i in range(5)],
                    }
                ],
            }

            mock_page = MagicMock()
            mock_page.get_text.return_value = page_data
            mock_doc.__getitem__ = lambda self, idx: mock_page
            mock_pymupdf.open.return_value = mock_doc

            # Limit to 3 lines per block
            _, _, features, _ = _extract_features_from_page_range((0, 1, "/fake/path.pdf", 3, 3, 25))

            assert len(features) == 1  # 1 block
            assert len(features[0]) == 3  # 3 lines
            assert "Line 0" in features[0][0][0].text_snippet
            assert "Line 2" in features[0][2][0].text_snippet

    def test_extract_features_respects_text_snippet_length(self):
        """Test that text_snippet_length limit is respected."""
        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = 1

            long_text = "This is a very long text that should be truncated"
            page_data = {
                "height": 800,
                "blocks": [{"type": 0, "lines": [{"spans": [{"font": "Bold", "size": 12, "text": long_text}]}]}],
            }

            mock_page = MagicMock()
            mock_page.get_text.return_value = page_data
            mock_doc.__getitem__ = lambda self, idx: mock_page
            mock_pymupdf.open.return_value = mock_doc

            # Limit snippet to 10 characters
            _, _, features, _ = _extract_features_from_page_range((0, 1, "/fake/path.pdf", 3, 5, 10))

            assert len(features[0][0][0].text_snippet) == 10
            assert features[0][0][0].text_snippet == "This is a "
            # But text_length should reflect full length
            assert features[0][0][0].text_length == len(long_text)

    def test_extract_features_skips_non_text_blocks(self):
        """Test that non-text blocks (type != 0) are skipped."""
        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            mock_doc = MagicMock()
            mock_doc.page_count = 1

            page_data = {
                "height": 800,
                "blocks": [
                    {"type": 0, "lines": [{"spans": [{"font": "Bold", "size": 12, "text": "Text block"}]}]},
                    {"type": 1, "image": "some_image_data"},  # Image block
                    {"type": 0, "lines": [{"spans": [{"font": "Bold", "size": 12, "text": "Another text"}]}]},
                ],
            }

            mock_page = MagicMock()
            mock_page.get_text.return_value = page_data
            mock_doc.__getitem__ = lambda self, idx: mock_page
            mock_pymupdf.open.return_value = mock_doc

            _, _, features, _ = _extract_features_from_page_range((0, 1, "/fake/path.pdf", 10, 5, 25))

            # Should have 3 blocks in features list, but image block will have empty lines
            assert len(features) == 3
            # First and third blocks should have content
            assert len(features[0]) == 1
            assert len(features[1]) == 0  # Image block - no lines
            assert len(features[2]) == 1

    def test_parallel_extraction_simulation_with_many_processes(self):
        """Simulate parallel extraction with many processes and verify merge correctness."""
        with patch("pdfalive.processors.toc_generator.pymupdf") as mock_pymupdf:
            num_pages = 50
            num_processes = 7  # Odd number to test uneven distribution

            mock_doc_obj = MagicMock()
            mock_doc_obj.page_count = num_pages

            def create_page(page_idx):
                mock_page = MagicMock()
                mock_page.get_text.return_value = {
                    "height": 800,
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [
                                {"spans": [{"font": "Bold", "size": 16, "text": f"Chapter {page_idx + 1}"}]},
                                {"spans": [{"font": "Regular", "size": 12, "text": f"Page {page_idx + 1}"}]},
                            ],
                        }
                    ],
                }
                return mock_page

            mock_pages = [create_page(i) for i in range(num_pages)]
            mock_doc_obj.__getitem__ = lambda self, idx: mock_pages[idx]
            mock_pymupdf.open.return_value = mock_doc_obj

            # Collect results from all "processes"
            results = []
            for proc_idx in range(num_processes):
                start, end, features, _ = _extract_features_from_page_range(
                    (proc_idx, num_processes, "/fake/path.pdf", 3, 5, 25)
                )
                results.append((start, end, features))

            # Verify no gaps or overlaps in page coverage
            results = sorted(results, key=lambda x: x[0])
            for i, (start, _, _) in enumerate(results):
                if i == 0:
                    assert start == 0, "First process should start at page 0"
                else:
                    prev_end = results[i - 1][1]
                    assert start == prev_end, f"Gap: process {i} starts at {start}, prev ended at {prev_end}"

            # Last process should end at num_pages
            assert results[-1][1] == num_pages, f"Last process should end at {num_pages}"

            # Merge and verify
            all_features = []
            for _, _, features in results:
                all_features.extend(features)

            # Should have exactly num_pages blocks (1 block per page)
            assert len(all_features) == num_pages

            # Verify page numbers are sequential
            page_numbers = [block[0][0].page_number for block in all_features]
            assert page_numbers == list(range(1, num_pages + 1)), "Page numbers should be sequential 1 to N"

            # Verify content matches expected pages
            for i, block in enumerate(all_features):
                expected_text = f"Chapter {i + 1}"
                actual_text = block[0][0].text_snippet
                assert expected_text in actual_text, f"Page {i + 1}: expected '{expected_text}' in '{actual_text}'"


class TestTOCGeneratorParallelExtraction:
    """Tests for parallel feature extraction with file-backed documents."""

    class DummyLLM:
        """Minimal LLM stub for testing."""

        def with_structured_output(self, schema):
            return None

    def test_tocgenerator_uses_parallel_for_file_backed_docs(self, tmp_path, monkeypatch):
        """Test that TOCGenerator uses parallel extraction for file-backed documents."""
        # Create a simple file-backed PDF with several blank pages
        input_pdf = tmp_path / "test.pdf"
        doc = pymupdf.open()
        for _ in range(4):
            doc.new_page()
        doc.save(str(input_pdf))
        doc.close()

        doc = pymupdf.open(str(input_pdf))

        called = {"parallel": False}

        def fake_parallel(self, *args, **kwargs):
            called["parallel"] = True
            return []

        generator = TOCGenerator(doc=doc, llm=self.DummyLLM(), num_processes=2)

        # Monkeypatch the parallel extractor
        monkeypatch.setattr(TOCGenerator, "_extract_features_parallel", fake_parallel)

        generator._extract_features(doc)

        assert called["parallel"], "TOCGenerator did not use parallel extraction for file-backed document"

        doc.close()


class TestTOCPostprocessing:
    """Tests for TOC postprocessing functionality."""

    @pytest.fixture
    def mock_doc(self):
        """Create a mock PyMuPDF document with multiple pages."""
        doc = MagicMock()
        doc.page_count = 10
        doc.get_toc.return_value = []
        doc.name = None

        # Mock pages with some containing "table of contents" text
        pages = []
        for i in range(10):
            page = MagicMock()
            if i == 1:  # Page 2 has a printed TOC
                # get_text("text") returns plain text string
                toc_text = (
                    "Table of Contents\n1. Introduction............1\n"
                    "2. Methods................15\n3. Results.................30"
                )
                page.get_text.side_effect = lambda arg, _i=i, _text=toc_text: _text if arg == "text" else {"blocks": []}
            else:
                page.get_text.side_effect = lambda arg, _i=i: (
                    f"Page {_i + 1} content" if arg == "text" else {"blocks": []}
                )
            pages.append(page)

        doc.__iter__ = lambda self: iter(pages)
        doc.__getitem__ = lambda self, idx: pages[idx]
        return doc

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MagicMock()

    @pytest.fixture
    def sample_toc_with_duplicates(self):
        """Sample TOC with duplicates and issues to be cleaned up."""
        return TOC(
            entries=[
                TOCEntry(title="Introduction", page_number=3, level=1, confidence=0.9),
                TOCEntry(title="Introduction", page_number=3, level=1, confidence=0.85),  # Duplicate
                TOCEntry(title="Methods", page_number=17, level=1, confidence=0.8),
                TOCEntry(title="Methdos", page_number=17, level=1, confidence=0.7),  # Typo duplicate
                TOCEntry(title="Results", page_number=32, level=1, confidence=0.9),
            ]
        )

    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        features = []
        for page_num in range(1, 11):
            block_features = [
                [
                    TOCFeature(
                        page_number=page_num,
                        font_name="Times-Bold",
                        font_size=16,
                        text_length=20,
                        text_snippet=f"Heading {page_num}",
                    )
                ]
            ]
            features.append(block_features)
        return features

    @pytest.fixture
    def cleaned_toc_response(self):
        """Expected cleaned TOC from postprocessing."""
        return TOC(
            entries=[
                TOCEntry(title="Introduction", page_number=3, level=1, confidence=0.95),
                TOCEntry(title="Methods", page_number=17, level=1, confidence=0.95),
                TOCEntry(title="Results", page_number=32, level=1, confidence=0.95),
            ]
        )

    def test_postprocess_toc_removes_duplicates(
        self, mock_doc, mock_llm, sample_toc_with_duplicates, sample_features, cleaned_toc_response
    ):
        """Test that postprocessing removes duplicate entries."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = cleaned_toc_response
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result, usage = generator._postprocess_toc(
            toc=sample_toc_with_duplicates,
            features=sample_features,
        )

        assert len(result.entries) == 3
        assert result.entries[0].title == "Introduction"
        assert result.entries[1].title == "Methods"
        assert result.entries[2].title == "Results"

    def test_postprocess_toc_returns_toc_structure(
        self, mock_doc, mock_llm, sample_toc_with_duplicates, sample_features, cleaned_toc_response
    ):
        """Test that postprocessing returns a TOC structure."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = cleaned_toc_response
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result, usage = generator._postprocess_toc(
            toc=sample_toc_with_duplicates,
            features=sample_features,
        )

        assert isinstance(result, TOC)
        assert all(isinstance(entry, TOCEntry) for entry in result.entries)

    def test_postprocess_toc_tracks_token_usage(
        self, mock_doc, mock_llm, sample_toc_with_duplicates, sample_features, cleaned_toc_response
    ):
        """Test that postprocessing tracks token usage."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = cleaned_toc_response
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result, usage = generator._postprocess_toc(
            toc=sample_toc_with_duplicates,
            features=sample_features,
        )

        assert usage.llm_calls == 1
        assert usage.input_tokens > 0

    def test_postprocess_toc_uses_document_text_for_reference_toc(
        self, mock_doc, mock_llm, sample_toc_with_duplicates, sample_features, cleaned_toc_response
    ):
        """Test that postprocessing extracts reference TOC from document pages."""
        messages_received = []

        def capture_messages(messages):
            messages_received.append(messages)
            return cleaned_toc_response

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = capture_messages
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        generator._postprocess_toc(
            toc=sample_toc_with_duplicates,
            features=sample_features,
        )

        # Verify that the LLM was called
        assert len(messages_received) == 1
        # The user message should contain context about the document
        user_message = messages_received[0][1].content
        assert "Introduction" in user_message or "generated TOC" in user_message.lower()

    def test_postprocess_toc_includes_existing_toc_in_prompt(
        self, mock_doc, mock_llm, sample_toc_with_duplicates, sample_features, cleaned_toc_response
    ):
        """Test that the existing generated TOC is included in the prompt."""
        messages_received = []

        def capture_messages(messages):
            messages_received.append(messages)
            return cleaned_toc_response

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = capture_messages
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        generator._postprocess_toc(
            toc=sample_toc_with_duplicates,
            features=sample_features,
        )

        # The user message should contain the generated TOC entries
        user_message = messages_received[0][1].content
        assert "Introduction" in user_message
        assert "Methods" in user_message
        assert "Results" in user_message

    def test_postprocess_toc_handles_empty_toc(self, mock_doc, mock_llm, sample_features):
        """Test that postprocessing handles empty TOC gracefully."""
        empty_toc = TOC(entries=[])
        empty_response = TOC(entries=[])

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = empty_response
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result, usage = generator._postprocess_toc(
            toc=empty_toc,
            features=sample_features,
        )

        assert isinstance(result, TOC)
        assert len(result.entries) == 0

    def test_postprocess_toc_preserves_valid_entries(self, mock_doc, mock_llm, sample_features):
        """Test that postprocessing preserves valid entries when no cleanup needed."""
        valid_toc = TOC(
            entries=[
                TOCEntry(title="Chapter 1", page_number=1, level=1, confidence=0.95),
                TOCEntry(title="Chapter 2", page_number=10, level=1, confidence=0.95),
            ]
        )

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = valid_toc
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result, usage = generator._postprocess_toc(
            toc=valid_toc,
            features=sample_features,
        )

        assert len(result.entries) == 2
        assert result.entries[0].title == "Chapter 1"
        assert result.entries[1].title == "Chapter 2"

    @pytest.mark.parametrize(
        "max_pages_to_scan",
        [5, 10, 20],
    )
    def test_postprocess_toc_respects_max_pages_for_reference_toc(
        self, mock_doc, mock_llm, sample_toc_with_duplicates, sample_features, cleaned_toc_response, max_pages_to_scan
    ):
        """Test that postprocessing respects the max pages limit for scanning reference TOC."""
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = cleaned_toc_response
        mock_llm.with_structured_output.return_value = mock_structured_llm

        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result, usage = generator._postprocess_toc(
            toc=sample_toc_with_duplicates,
            features=sample_features,
            max_pages_for_reference_toc=max_pages_to_scan,
        )

        # Should still return valid result
        assert isinstance(result, TOC)


class TestIsBoldFont:
    """Tests for _is_bold_font helper."""

    @pytest.mark.parametrize(
        "span,expected",
        [
            ({"font": "Times-Bold", "flags": 0}, True),  # bold in font name
            ({"font": "Times-Roman", "flags": 16}, True),  # bold via flags bit 4
            ({"font": "ArialBold", "flags": 16}, True),  # both
            ({"font": "Times-Roman", "flags": 0}, False),  # neither
            ({"font": "Helvetica", "flags": 4}, False),  # italic only
            ({}, False),  # missing keys
        ],
    )
    def test_is_bold_font(self, span, expected):
        """Test bold font detection from flags and font name."""
        assert _is_bold_font(span) == expected


class TestComputeBodyFontProfile:
    """Tests for _compute_body_font_profile helper."""

    def test_returns_most_common_font(self):
        """Test that the most frequently occurring font pair is returned."""
        body = TOCFeature(
            page_number=1,
            font_name="Times-Roman",
            font_size=12,
            text_length=100,
            text_snippet="body",
        )
        body2 = TOCFeature(
            page_number=1,
            font_name="Times-Roman",
            font_size=12,
            text_length=80,
            text_snippet="more",
        )
        heading = TOCFeature(
            page_number=2,
            font_name="Times-Bold",
            font_size=16,
            text_length=20,
            text_snippet="head",
        )
        body3 = TOCFeature(
            page_number=2,
            font_name="Times-Roman",
            font_size=12,
            text_length=90,
            text_snippet="text",
        )
        features = [
            [[body, body2]],
            [[heading, body3]],
        ]
        font_name, font_size = _compute_body_font_profile(features)
        assert font_name == "Times-Roman"
        assert font_size == 12

    def test_empty_features(self):
        """Test fallback for empty features."""
        font_name, font_size = _compute_body_font_profile([])
        assert font_name == ""
        assert font_size == 0.0


class TestIsHeadingCandidate:
    """Tests for _is_heading_candidate helper."""

    @pytest.fixture
    def body_font_name(self):
        return "Times-Roman"

    @pytest.fixture
    def body_font_size(self):
        return 12.0

    @pytest.mark.parametrize(
        "span,expected,description",
        [
            # Font size significantly larger than body
            (
                {"font": "Times-Bold", "size": 16, "text": "Chapter 1: Introduction", "flags": 16},
                True,
                "large bold heading",
            ),
            # Bold + same size as body
            (
                {"font": "Times-Bold", "size": 12, "text": "Section heading text", "flags": 16},
                True,
                "bold same-size heading",
            ),
            # Section numbering pattern
            (
                {"font": "Times-Roman", "size": 10, "text": "1.2 Subsection Title", "flags": 0},
                True,
                "section numbering",
            ),
            (
                {"font": "Times-Roman", "size": 10, "text": "Chapter 3 Overview", "flags": 0},
                True,
                "chapter numbering",
            ),
            (
                {"font": "Times-Roman", "size": 10, "text": "Appendix A Details", "flags": 0},
                True,
                "appendix numbering",
            ),
            # Body text - not a heading
            (
                {"font": "Times-Roman", "size": 12, "text": "This is regular body text content.", "flags": 0},
                False,
                "body text",
            ),
            # Too short
            ({"font": "Times-Bold", "size": 16, "text": "Ab", "flags": 16}, False, "too short"),
            # Too long (>200 chars)
            ({"font": "Times-Bold", "size": 16, "text": "x" * 201, "flags": 16}, False, "too long"),
            # Empty text
            ({"font": "Times-Bold", "size": 16, "text": "", "flags": 16}, False, "empty text"),
            # Small non-bold non-patterned text
            (
                {"font": "Times-Roman", "size": 10, "text": "Some random text", "flags": 0},
                False,
                "small regular text",
            ),
        ],
    )
    def test_heading_candidate_detection(self, span, expected, description, body_font_name, body_font_size):
        """Test heading candidate detection with various scenarios."""
        result = _is_heading_candidate(span, body_font_name, body_font_size)
        assert result == expected, f"Failed for: {description}"


class TestTOCFeatureStrFormat:
    """Tests for TOCFeature.__str__ format with new fields."""

    def test_str_without_new_fields(self):
        """Test backward compatibility: old format when new fields are None."""
        feature = TOCFeature(
            page_number=1, font_name="Times-Bold", font_size=16, text_length=45, text_snippet="Chapter 1"
        )
        result = str(feature)
        assert result == "(1, 'Times-Bold', 16.0, 45, 'Chapter 1')"

    def test_str_with_y_position_only(self):
        """Test format with y_position set."""
        feature = TOCFeature(
            page_number=1,
            font_name="Times-Bold",
            font_size=16,
            text_length=45,
            text_snippet="Chapter 1",
            y_position=0.12,
        )
        result = str(feature)
        assert result == "(1, 'Times-Bold', 16.0, 45, 'Chapter 1', y=0.12)"

    def test_str_with_both_fields(self):
        """Test format with both y_position and is_bold set."""
        feature = TOCFeature(
            page_number=1,
            font_name="Times-Bold",
            font_size=16,
            text_length=45,
            text_snippet="Chapter 1",
            y_position=0.45,
            is_bold=True,
        )
        result = str(feature)
        assert result == "(1, 'Times-Bold', 16.0, 45, 'Chapter 1', y=0.45, bold=True)"

    def test_str_with_is_bold_only(self):
        """Test format with only is_bold set."""
        feature = TOCFeature(
            page_number=1,
            font_name="Times-Bold",
            font_size=16,
            text_length=45,
            text_snippet="Chapter 1",
            is_bold=False,
        )
        result = str(feature)
        assert result == "(1, 'Times-Bold', 16.0, 45, 'Chapter 1', bold=False)"


class TestHeadingCandidateIntegration:
    """Integration test: heading candidate scanning picks up mid-page headings."""

    def test_mid_page_heading_found_in_features(self):
        """Test that a heading in block 4+ is detected and included in features."""
        doc = MagicMock()
        doc.page_count = 1
        doc.name = None

        page_height = 800.0
        page = MagicMock()
        page.rect.height = page_height

        body_span_1 = {
            "font": "Times-Roman",
            "size": 12,
            "text": "Body paragraph one",
            "bbox": (50, 50, 400, 65),
            "flags": 0,
        }
        body_span_2 = {
            "font": "Times-Roman",
            "size": 12,
            "text": "Body paragraph two",
            "bbox": (50, 100, 400, 115),
            "flags": 0,
        }
        body_span_3 = {
            "font": "Times-Roman",
            "size": 12,
            "text": "Body paragraph three",
            "bbox": (50, 200, 400, 215),
            "flags": 0,
        }
        heading_span = {
            "font": "Times-Bold",
            "size": 16,
            "text": "5.2 Convexity of Functions",
            "bbox": (50, 400, 400, 420),
            "flags": 16,
        }
        body_span_4 = {
            "font": "Times-Roman",
            "size": 12,
            "text": "More body text here",
            "bbox": (50, 450, 400, 465),
            "flags": 0,
        }

        page.get_text.return_value = {
            "height": page_height,
            "blocks": [
                {"type": 0, "lines": [{"spans": [body_span_1]}]},
                {"type": 0, "lines": [{"spans": [body_span_2]}]},
                {"type": 0, "lines": [{"spans": [body_span_3]}]},
                {"type": 0, "lines": [{"spans": [heading_span]}]},
                {"type": 0, "lines": [{"spans": [body_span_4]}]},
            ],
        }
        doc.__iter__ = lambda self: iter([page])

        mock_llm = MagicMock()
        generator = TOCGenerator(doc=doc, llm=mock_llm)

        features = generator._extract_features_sequential(doc, max_blocks_per_page=3, show_progress=False)

        # Collect all text snippets from features
        all_snippets = []
        for block in features:
            for line in block:
                for span in line:
                    if isinstance(span, TOCFeature):
                        all_snippets.append(span.text_snippet)

        # The mid-page heading should have been detected as a heading candidate
        assert any("5.2 Convexity" in s for s in all_snippets), (
            f"Mid-page heading not found in features. Got: {all_snippets}"
        )

    def test_y_position_is_set_on_features(self):
        """Test that y_position is populated when bbox data is available."""
        doc = MagicMock()
        doc.page_count = 1
        doc.name = None

        page_height = 800.0
        page = MagicMock()
        page.rect.height = page_height

        heading_span = {
            "font": "Times-Bold",
            "size": 16,
            "text": "Chapter 1: Introduction",
            "bbox": (50, 100, 400, 120),
            "flags": 16,
        }
        page.get_text.return_value = {
            "height": page_height,
            "blocks": [
                {"type": 0, "lines": [{"spans": [heading_span]}]},
            ],
        }
        doc.__iter__ = lambda self: iter([page])

        mock_llm = MagicMock()
        generator = TOCGenerator(doc=doc, llm=mock_llm)

        features = generator._extract_features_sequential(doc, show_progress=False)

        first_span = features[0][0][0]
        assert first_span.y_position == round(100 / 800, 2)
        assert first_span.is_bold is True
