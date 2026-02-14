"""Unit tests for TOC generator processor."""

from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from pdfalive.models.toc import TOC, TOCEntry, TOCFeature
from pdfalive.processors.toc_generator import (
    _LETTERSPACED_PATTERN,
    _SECTION_NUMBER_PATTERN,
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


class TestRomanNumeralAndLetterspacedPatterns:
    """Tests for Roman numeral and letter-spaced heading detection."""

    @pytest.fixture
    def body_font_name(self):
        return "Times-Roman"

    @pytest.fixture
    def body_font_size(self):
        return 12.0

    @pytest.mark.parametrize(
        "text,should_match",
        [
            # Roman numerals that should match _SECTION_NUMBER_PATTERN
            ("I THE SAMPLE SPACE", True),
            ("II ELEMENTS OF COMBINATORIAL ANALYSIS", True),
            ("III FLUCTUATIONS IN COIN TOSSING", True),
            ("IV COMBINATION OF EVENTS", True),
            ("V CONDITIONAL PROBABILITY", True),
            ("IX THE BERNOULLI SCHEME", True),
            ("XIV RANDOM VARIABLES", True),
            ("XVII THE EXPONENTIAL", True),
            ("IV. Some Section", True),
            # Letter-spaced text that should match _LETTERSPACED_PATTERN
            ("C H A P T E R  I", True),
            ("P R E F A C E", True),
            ("C H A P T E R  XIV", True),
            # Arabic that should still match
            ("1. The Empirical Background", True),
            ("1.2 Subsection Title", True),
            ("Chapter 3 Overview", True),
            ("Appendix A Details", True),
            # "I " matches Roman numeral pattern — accepted trade-off, LLM filters it
            ("I went to the store", True),
            # Should NOT match
            ("Some random text", False),
            ("The quick brown fox", False),
        ],
    )
    def test_section_pattern_matches(self, text, should_match):
        """Test that _SECTION_NUMBER_PATTERN and _LETTERSPACED_PATTERN match expected text."""
        matches_section = bool(_SECTION_NUMBER_PATTERN.match(text))
        matches_letterspaced = bool(_LETTERSPACED_PATTERN.match(text))
        result = matches_section or matches_letterspaced
        assert result == should_match, f"Text '{text}' expected match={should_match}, got {result}"

    @pytest.mark.parametrize(
        "text,expected",
        [
            # Roman numeral headings detected as heading candidates via pattern
            ("I THE SAMPLE SPACE", True),
            ("XIV RANDOM VARIABLES", True),
            ("IV. Some Section Title", True),
            # Letter-spaced headings detected as heading candidates
            ("C H A P T E R  I", True),
            ("P R E F A C E", True),
            # "I went..." is short-ish prose starting with "I " — the pattern matches
            # the Roman numeral "I " but the LLM will filter this false positive.
            # We accept this trade-off for better recall on Roman numeral chapters.
            ("I went to the store", True),
            # Regular body text - no match
            ("Some random text here", False),
        ],
    )
    def test_heading_candidate_roman_and_letterspaced(self, text, expected, body_font_name, body_font_size):
        """Test _is_heading_candidate with Roman numerals and letter-spaced text."""
        span = {"font": "Times-Roman", "size": 10, "text": text, "flags": 0}
        result = _is_heading_candidate(span, body_font_name, body_font_size)
        assert result == expected, f"Text '{text}' expected heading={expected}, got {result}"

    def test_mid_page_roman_numeral_heading_found(self):
        """Integration: Roman numeral heading in block 4+ is detected and included."""
        doc = MagicMock()
        doc.page_count = 1
        doc.name = None

        page_height = 800.0
        page = MagicMock()
        page.rect.height = page_height

        body_span = {
            "font": "Times-Roman",
            "size": 12,
            "text": "Body paragraph text",
            "bbox": (50, 50, 400, 65),
            "flags": 0,
        }
        roman_heading = {
            "font": "Times-Bold",
            "size": 12,
            "text": "III FLUCTUATIONS IN COIN TOSSING",
            "bbox": (50, 400, 400, 420),
            "flags": 0,
        }

        page.get_text.return_value = {
            "height": page_height,
            "blocks": [
                {"type": 0, "lines": [{"spans": [body_span]}]},
                {"type": 0, "lines": [{"spans": [body_span]}]},
                {"type": 0, "lines": [{"spans": [body_span]}]},
                # Beyond max_blocks_per_page=3, must be detected by Phase 2
                {"type": 0, "lines": [{"spans": [roman_heading]}]},
            ],
        }
        doc.__iter__ = lambda self: iter([page])

        mock_llm = MagicMock()
        generator = TOCGenerator(doc=doc, llm=mock_llm)
        features = generator._extract_features_sequential(doc, max_blocks_per_page=3, show_progress=False)

        all_snippets = []
        for block in features:
            for line in block:
                for span in line:
                    if isinstance(span, TOCFeature):
                        all_snippets.append(span.text_snippet)

        assert any("III FLUCTUATIONS" in s for s in all_snippets), (
            f"Roman numeral heading not found in features. Got: {all_snippets}"
        )

    def test_mid_page_letterspaced_heading_found(self):
        """Integration: letter-spaced heading in block 4+ is detected and included."""
        doc = MagicMock()
        doc.page_count = 1
        doc.name = None

        page_height = 800.0
        page = MagicMock()
        page.rect.height = page_height

        body_span = {
            "font": "Times-Roman",
            "size": 12,
            "text": "Body paragraph text",
            "bbox": (50, 50, 400, 65),
            "flags": 0,
        }
        letterspaced_heading = {
            "font": "Times-Roman",
            "size": 12,
            "text": "C H A P T E R  V I I I",
            "bbox": (50, 400, 400, 420),
            "flags": 0,
        }

        page.get_text.return_value = {
            "height": page_height,
            "blocks": [
                {"type": 0, "lines": [{"spans": [body_span]}]},
                {"type": 0, "lines": [{"spans": [body_span]}]},
                {"type": 0, "lines": [{"spans": [body_span]}]},
                {"type": 0, "lines": [{"spans": [letterspaced_heading]}]},
            ],
        }
        doc.__iter__ = lambda self: iter([page])

        mock_llm = MagicMock()
        generator = TOCGenerator(doc=doc, llm=mock_llm)
        features = generator._extract_features_sequential(doc, max_blocks_per_page=3, show_progress=False)

        all_snippets = []
        for block in features:
            for line in block:
                for span in line:
                    if isinstance(span, TOCFeature):
                        all_snippets.append(span.text_snippet)

        assert any("C H A P T E R" in s for s in all_snippets), (
            f"Letter-spaced heading not found in features. Got: {all_snippets}"
        )


class TestSummarizeFeaturesForPostprocessing:
    """Tests for even-sampling feature summary."""

    @pytest.fixture
    def mock_doc(self):
        doc = MagicMock()
        doc.page_count = 0
        doc.get_toc.return_value = []
        doc.name = None
        doc.__iter__ = lambda self: iter([])
        return doc

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    def _make_features(self, num_pages: int, spans_per_page: int = 1) -> list:
        """Helper to create features spanning many pages."""
        features = []
        for page_num in range(1, num_pages + 1):
            block = [
                [
                    TOCFeature(
                        page_number=page_num,
                        font_name="Times-Roman",
                        font_size=12,
                        text_length=20,
                        text_snippet=f"Page {page_num} text",
                    )
                    for _ in range(spans_per_page)
                ]
            ]
            features.append(block)
        return features

    def test_small_document_returns_all_spans(self, mock_doc, mock_llm):
        """When total spans <= max_entries, all are included."""
        features = self._make_features(10)
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result = generator._summarize_features_for_postprocessing(features, max_entries=50)

        assert result.count("\n") == 9  # 10 lines, 9 newlines
        assert "Page 1" in result
        assert "Page 10" in result

    def test_large_document_samples_evenly(self, mock_doc, mock_llm):
        """When total spans > max_entries, sampling covers the full document."""
        features = self._make_features(500)
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result = generator._summarize_features_for_postprocessing(features, max_entries=50)

        lines = result.strip().split("\n")
        assert len(lines) == 50

        # Extract page numbers from summary lines
        page_numbers = []
        for line in lines:
            # Line format: "Page N ..."
            page_num = int(line.split()[1])
            page_numbers.append(page_num)

        # First sample should be from early pages, last from late pages
        assert page_numbers[0] == 1
        assert page_numbers[-1] >= 490  # Near the end of the 500-page document

        # Samples should be roughly evenly spaced (~10 pages apart for 500/50)
        for i in range(1, len(page_numbers)):
            gap = page_numbers[i] - page_numbers[i - 1]
            assert gap >= 1, "Samples should be monotonically increasing"

    def test_empty_features(self, mock_doc, mock_llm):
        """Empty features returns placeholder text."""
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        result = generator._summarize_features_for_postprocessing([], max_entries=50)
        assert result == "(No features available)"

    def test_default_max_entries_is_150(self, mock_doc, mock_llm):
        """Verify the default max_entries is 150."""
        features = self._make_features(200)
        generator = TOCGenerator(doc=mock_doc, llm=mock_llm)
        # Call with default — should cap at 150
        result = generator._summarize_features_for_postprocessing(features)
        lines = result.strip().split("\n")
        assert len(lines) == 150


class TestDetectPageOffsetNote:
    """Tests for _detect_page_offset_note."""

    @pytest.fixture
    def generator(self, mock_doc, mock_llm):
        return TOCGenerator(doc=mock_doc, llm=mock_llm)

    def test_empty_toc_returns_empty(self, generator):
        """No entries means no offset note."""
        toc = TOC(entries=[])
        assert generator._detect_page_offset_note(toc, "some text") == ""

    def test_no_chapter_past_page_5_returns_empty(self, generator):
        """If all level-1 entries are in first 5 pages, no offset detected."""
        toc = TOC(
            entries=[
                TOCEntry(title="Preface", page_number=3, level=1, confidence=0.9),
            ]
        )
        assert generator._detect_page_offset_note(toc, "some text") == ""

    def test_small_offset_returns_empty(self, generator):
        """Offset of 2 or less is not flagged (minimal front matter)."""
        toc = TOC(
            entries=[
                TOCEntry(title="Chapter 1", page_number=3, level=1, confidence=0.9),
            ]
        )
        # offset = 3 - 1 = 2, which is < 3
        assert generator._detect_page_offset_note(toc, "some text") == ""

    def test_significant_offset_generates_warning(self, generator):
        """Offset of 15 generates a warning with correct page reference."""
        toc = TOC(
            entries=[
                TOCEntry(title="Table of Contents", page_number=3, level=1, confidence=0.9),
                TOCEntry(title="Introduction", page_number=16, level=1, confidence=0.95),
            ]
        )
        note = generator._detect_page_offset_note(toc, "some text")
        assert "Note" in note
        assert "15" in note  # estimated offset
        assert "PDF page 16" in note

    def test_skips_early_level1_entries(self, generator):
        """Entries in first 5 pages (like preface/TOC) are skipped for offset detection."""
        toc = TOC(
            entries=[
                TOCEntry(title="Preface", page_number=4, level=1, confidence=0.9),
                TOCEntry(title="Chapter I", page_number=22, level=1, confidence=0.95),
            ]
        )
        note = generator._detect_page_offset_note(toc, "some text")
        assert "Note" in note
        assert "21" in note  # offset = 22 - 1
        assert "PDF page 22" in note


class TestDetectFrontMatterOffset:
    """Tests for _detect_front_matter_offset."""

    @pytest.fixture
    def generator(self, mock_doc, mock_llm):
        return TOCGenerator(doc=mock_doc, llm=mock_llm)

    def _make_toc(self, entries_data: list[tuple[str, int, int]]) -> TOC:
        """Helper: create TOC from (title, page, level) tuples."""
        return TOC(entries=[TOCEntry(title=t, page_number=p, level=lvl, confidence=0.9) for t, p, lvl in entries_data])

    def test_empty_toc_returns_zero(self, generator):
        """Empty TOC yields no offset."""
        assert generator._detect_front_matter_offset(TOC(entries=[])) == 0

    def test_no_level1_past_page_5_returns_zero(self, generator):
        """Level-1 entries only in first 5 pages means no significant front matter."""
        toc = self._make_toc([("Preface", 3, 1)])
        assert generator._detect_front_matter_offset(toc) == 0

    def test_first_level1_past_page_5_returns_offset(self, generator):
        """First level-1 entry past page 5 determines the offset."""
        toc = self._make_toc(
            [
                ("Table of Contents", 3, 1),
                ("Introduction", 16, 1),
                ("Chapter II", 41, 1),
            ]
        )
        assert generator._detect_front_matter_offset(toc) == 15  # 16 - 1

    def test_skips_level2_entries(self, generator):
        """Level-2 entries are ignored for offset detection."""
        toc = self._make_toc(
            [
                ("Subsection 1.1", 20, 2),
                ("Chapter I", 30, 1),
            ]
        )
        assert generator._detect_front_matter_offset(toc) == 29  # 30 - 1

    @pytest.mark.parametrize(
        "page_number,expected_offset",
        [
            (6, 5),
            (10, 9),
            (22, 21),
        ],
    )
    def test_offset_equals_page_minus_one(self, generator, page_number, expected_offset):
        """Offset is always first qualifying page_number - 1."""
        toc = self._make_toc([("Chapter I", page_number, 1)])
        assert generator._detect_front_matter_offset(toc) == expected_offset


class TestPageContainsHeading:
    """Tests for _page_contains_heading."""

    @pytest.fixture
    def mock_doc_with_pages(self):
        """Create a mock document with configurable page text."""
        doc = MagicMock()
        doc.page_count = 100
        doc.get_toc.return_value = []
        doc.name = None

        # Default page text - can be customized per test
        page_texts = {}

        def get_page(idx):
            page = MagicMock()
            text = page_texts.get(idx, f"Page {idx + 1} body text content")
            page.get_text.return_value = text
            return page

        doc.__getitem__ = lambda self, idx: get_page(idx)
        doc._page_texts = page_texts  # Expose for test configuration
        return doc

    @pytest.fixture
    def generator(self, mock_doc_with_pages, mock_llm):
        return TOCGenerator(doc=mock_doc_with_pages, llm=mock_llm)

    def test_finds_exact_heading_on_page(self, generator):
        """Heading text found on the exact page returns True."""
        generator.doc._page_texts[15] = "Some text\nIntroduction to Probability\nMore text"
        assert generator._page_contains_heading(16, "Introduction to Probability") is True

    def test_case_insensitive_match(self, generator):
        """Matching is case-insensitive."""
        generator.doc._page_texts[15] = "CHAPTER I: THE SAMPLE SPACE"
        assert generator._page_contains_heading(16, "Chapter I: The Sample Space") is True

    def test_whitespace_normalized(self, generator):
        """Extra whitespace in page text or title is collapsed."""
        generator.doc._page_texts[15] = "Chapter  I:  The   Sample Space"
        assert generator._page_contains_heading(16, "Chapter I: The Sample Space") is True

    def test_returns_false_for_missing_heading(self, generator):
        """Returns False when heading text is not on the page or neighbors."""
        generator.doc._page_texts[15] = "Completely unrelated text about nothing"
        generator.doc._page_texts[14] = "Also unrelated"
        generator.doc._page_texts[16] = "Still unrelated"
        assert generator._page_contains_heading(16, "Introduction to Probability") is False

    def test_returns_false_for_short_search_text(self, generator):
        """Titles shorter than 3 chars after normalization are rejected."""
        generator.doc._page_texts[0] = "AB is here"
        assert generator._page_contains_heading(1, "AB") is False

    def test_strips_number_prefix(self, generator):
        """Section number prefixes like '1. ' are stripped before searching."""
        generator.doc._page_texts[15] = "The Empirical Background of probability"
        assert generator._page_contains_heading(16, "1. The Empirical Background") is True

    def test_searches_within_window(self, generator):
        """Heading on an adjacent page (within window) is found."""
        generator.doc._page_texts[16] = "Introduction to Methods"
        assert generator._page_contains_heading(16, "Introduction to Methods", window=1) is True

    def test_clamps_to_document_bounds(self, generator):
        """Search window is clamped to valid page range."""
        generator.doc._page_texts[0] = "Preface to the Book"
        # page_number=1, window=1 → search pages 0..1 (clamped, not -1..1)
        assert generator._page_contains_heading(1, "Preface to the Book") is True


class TestResolvePageNumber:
    """Tests for _resolve_page_number."""

    @pytest.fixture
    def mock_doc_with_pages(self):
        """Create a mock document with configurable page text."""
        doc = MagicMock()
        doc.page_count = 600
        doc.get_toc.return_value = []
        doc.name = None

        page_texts = {}

        def get_page(idx):
            page = MagicMock()
            text = page_texts.get(idx, "")
            page.get_text.return_value = text
            return page

        doc.__getitem__ = lambda self, idx: get_page(idx)
        doc._page_texts = page_texts
        return doc

    @pytest.fixture
    def generator(self, mock_doc_with_pages, mock_llm):
        return TOCGenerator(doc=mock_doc_with_pages, llm=mock_llm)

    def test_returns_corrected_page_when_heading_found_at_corrected(self, generator):
        """When heading is found at offset-corrected page, returns corrected page."""
        offset = 15
        # Heading "Methods" at printed page 10 → should be PDF page 25
        generator.doc._page_texts[24] = "Methods and Materials"
        result = generator._resolve_page_number("Methods", 10, offset)
        assert result == 25  # 10 + 15

    def test_returns_original_page_when_heading_found_at_original(self, generator):
        """When heading only found at original page (already correct PDF page), keeps it."""
        offset = 15
        # Heading at original page 30 (already a PDF page number)
        generator.doc._page_texts[29] = "Results and Discussion"
        result = generator._resolve_page_number("Results", 30, offset)
        assert result == 30

    def test_defaults_to_corrected_when_not_found_anywhere(self, generator):
        """When heading not found at either page, defaults to offset-corrected."""
        offset = 15
        result = generator._resolve_page_number("Unknown Chapter", 10, offset)
        assert result == 25  # 10 + 15 (default)

    def test_keeps_original_when_corrected_out_of_range(self, generator):
        """When corrected page exceeds document length, keeps original."""
        generator.doc.page_count = 100
        result = generator._resolve_page_number("Appendix", 90, 15)
        assert result == 90  # 90 + 15 = 105 > 100

    def test_prefers_corrected_over_original_when_both_match(self, generator):
        """When heading found at both pages, prefers corrected (checked first)."""
        offset = 15
        generator.doc._page_texts[24] = "Methods in Science"
        generator.doc._page_texts[9] = "Methods in Science"
        result = generator._resolve_page_number("Methods", 10, offset)
        assert result == 25  # corrected is checked first


class TestCorrectPostprocessedPageNumbers:
    """Tests for _correct_postprocessed_page_numbers."""

    @pytest.fixture
    def mock_doc_with_pages(self):
        """Create a mock document with configurable page text for verification."""
        doc = MagicMock()
        doc.page_count = 600
        doc.get_toc.return_value = []
        doc.name = None

        page_texts = {}

        def get_page(idx):
            page = MagicMock()
            text = page_texts.get(idx, "")
            page.get_text.return_value = text
            return page

        doc.__getitem__ = lambda self, idx: get_page(idx)
        doc._page_texts = page_texts
        return doc

    @pytest.fixture
    def generator(self, mock_doc_with_pages, mock_llm):
        return TOCGenerator(doc=mock_doc_with_pages, llm=mock_llm)

    def _make_toc(self, entries_data: list[tuple[str, int, int]]) -> TOC:
        """Helper: create TOC from (title, page, level) tuples."""
        return TOC(entries=[TOCEntry(title=t, page_number=p, level=lvl, confidence=0.9) for t, p, lvl in entries_data])

    def test_no_correction_when_pages_match(self, generator):
        """No correction applied when original and refined page numbers agree."""
        original = self._make_toc(
            [
                ("Introduction", 16, 1),
                ("Chapter I", 22, 1),
                ("Chapter II", 41, 1),
            ]
        )
        refined = self._make_toc(
            [
                ("Introduction", 16, 1),
                ("Chapter I", 22, 1),
                ("Chapter II", 41, 1),
            ]
        )
        result = generator._correct_postprocessed_page_numbers(original, refined)
        for orig, res in zip(original.entries, result.entries, strict=True):
            assert orig.page_number == res.page_number

    def test_restores_original_pages_for_matched_entries(self, generator):
        """Matched entries get their original PDF page numbers restored."""
        original = self._make_toc(
            [
                ("Introduction", 16, 1),
                ("Chapter I", 22, 1),
                ("Chapter II", 41, 1),
            ]
        )
        # Postprocessor changed page numbers for existing entries
        refined = self._make_toc(
            [
                ("Introduction", 1, 1),
                ("Chapter I", 7, 1),
                ("Chapter II", 26, 1),
            ]
        )
        result = generator._correct_postprocessed_page_numbers(original, refined)
        assert result.entries[0].page_number == 16
        assert result.entries[1].page_number == 22
        assert result.entries[2].page_number == 41

    def test_applies_offset_to_new_entries(self, generator):
        """New entries from the postprocessor get the front matter offset applied."""
        # Set up page text so _resolve_page_number finds headings at corrected pages
        generator.doc._page_texts[24] = "Subsection 1.1 content"  # PDF page 25
        generator.doc._page_texts[34] = "Subsection 1.2 content"  # PDF page 35

        original = self._make_toc(
            [
                ("Introduction", 16, 1),
                ("Chapter I", 22, 1),
                ("Chapter II", 41, 1),
            ]
        )
        # Postprocessor keeps existing entries and adds new ones with printed page numbers
        refined = self._make_toc(
            [
                ("Introduction", 16, 1),
                ("Chapter I", 22, 1),
                ("Subsection 1.1", 10, 2),  # printed page → needs +15
                ("Subsection 1.2", 20, 2),  # printed page → needs +15
                ("Chapter II", 41, 1),
            ]
        )
        result = generator._correct_postprocessed_page_numbers(original, refined)
        assert result.entries[0].page_number == 16  # matched, restored
        assert result.entries[1].page_number == 22  # matched, restored
        assert result.entries[2].page_number == 25  # new, offset applied (10 + 15)
        assert result.entries[3].page_number == 35  # new, offset applied (20 + 15)
        assert result.entries[4].page_number == 41  # matched, restored

    def test_empty_refined_toc_returned_as_is(self, generator):
        """Empty refined TOC is returned unchanged."""
        empty = TOC(entries=[])
        original = self._make_toc([("Chapter I", 22, 1)])

        assert generator._correct_postprocessed_page_numbers(original, empty) == empty

    def test_empty_original_with_nonempty_refined(self, generator):
        """When original is empty, refined entries are kept (no offset detected)."""
        empty = TOC(entries=[])
        refined = self._make_toc([("Chapter I", 22, 1)])

        result = generator._correct_postprocessed_page_numbers(empty, refined)
        assert result.entries[0].page_number == 22  # no offset, kept as-is

    def test_no_offset_applied_when_front_matter_small(self, generator):
        """New entries are kept as-is when front matter offset < 3."""
        original = self._make_toc(
            [
                ("Chapter 1", 4, 1),  # offset = 4 - 1 = 3... but page <= 5, so skipped
            ]
        )
        refined = self._make_toc(
            [
                ("Chapter 1", 4, 1),
                ("Section 1.1", 10, 2),  # new entry
            ]
        )
        result = generator._correct_postprocessed_page_numbers(original, refined)
        # offset = 0 (no level-1 entry past page 5), so new entry kept as-is
        assert result.entries[1].page_number == 10

    def test_new_entry_keeps_original_when_already_correct_pdf_page(self, generator):
        """New entry with correct PDF page number is verified and kept."""
        # Set heading text at PDF page 30 (the original page), not at corrected page 45
        generator.doc._page_texts[29] = "Special Section content here"

        original = self._make_toc(
            [
                ("Chapter I", 16, 1),
            ]
        )
        refined = self._make_toc(
            [
                ("Chapter I", 16, 1),
                ("Special Section", 30, 2),  # already a correct PDF page
            ]
        )
        result = generator._correct_postprocessed_page_numbers(original, refined)
        assert result.entries[1].page_number == 30  # kept, found at original page

    def test_title_matching_is_case_insensitive(self, generator):
        """Title matching between original and refined is case-insensitive."""
        original = self._make_toc(
            [
                ("Introduction", 16, 1),
                ("CHAPTER I", 22, 1),
            ]
        )
        refined = self._make_toc(
            [
                ("introduction", 1, 1),
                ("Chapter I", 7, 1),
            ]
        )
        result = generator._correct_postprocessed_page_numbers(original, refined)
        assert result.entries[0].page_number == 16
        assert result.entries[1].page_number == 22
