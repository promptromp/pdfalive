"""Unit tests for token counting utilities."""

from io import StringIO

import pytest
from rich.console import Console

from pdfalive.tokens import (
    DEFAULT_ENCODING,
    TokenUsage,
    estimate_features_tokens,
    estimate_tokens,
    get_encoding,
)


class TestEstimateTokens:
    """Tests for token counting functions."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", 0),
            ("hello", 1),
            ("a" * 100, 13),
        ],
    )
    def test_estimate_tokens_exact(self, text, expected):
        """Test token counting returns exact tiktoken values."""
        assert estimate_tokens(text) == expected

    def test_estimate_tokens_scales_with_length(self):
        """Test that token count increases with text length."""
        short = estimate_tokens("a" * 100)
        long = estimate_tokens("a" * 1000)
        assert long > short

    def test_uses_tiktoken_encoding(self):
        """Test that the encoding singleton is loaded from tiktoken."""
        enc = get_encoding()
        assert enc.name == DEFAULT_ENCODING
        # Verify it returns the same singleton on repeated calls
        assert get_encoding() is enc

    def test_compact_format_tokenization(self):
        """Test that compact pipe-delimited format is tokenized accurately."""
        compact_line = "F0|14|Chapter 1|.12|B\n"
        tokens = estimate_tokens(compact_line)
        # tiktoken gives exact count (no ratio guesswork)
        assert tokens > 0
        # 100 lines should be roughly 100x one line (within tokenizer rounding)
        tokens_100 = estimate_tokens(compact_line * 100)
        assert 90 * tokens <= tokens_100 <= 110 * tokens


class TestEstimateFeaturesTokens:
    """Tests for feature-based token estimation."""

    def test_estimate_empty_features(self):
        """Test estimation with empty features."""
        result = estimate_features_tokens([])
        assert result >= 0

    def test_estimate_features_increases_with_content(self):
        """Test that more features means more tokens."""
        small = estimate_features_tokens([[["feature1"]]])
        large = estimate_features_tokens([[["feature1"]], [["feature2"]], [["feature3"]]])
        assert large > small


class TestTokenUsage:
    """Tests for TokenUsage tracking class."""

    @pytest.fixture
    def usage(self):
        """Create a fresh TokenUsage instance."""
        return TokenUsage()

    def test_initial_state(self, usage):
        """Test that TokenUsage starts with zero counts."""
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.llm_calls == 0
        assert usage.total_tokens == 0

    def test_add_single_call(self, usage):
        """Test adding a single LLM call."""
        usage.add_call(input_tokens=100, output_tokens=50, description="test call")

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.llm_calls == 1
        assert usage.total_tokens == 150

    def test_add_multiple_calls(self, usage):
        """Test accumulating multiple LLM calls."""
        usage.add_call(input_tokens=100, output_tokens=50, description="call 1")
        usage.add_call(input_tokens=200, output_tokens=75, description="call 2")
        usage.add_call(input_tokens=150, output_tokens=60, description="call 3")

        assert usage.input_tokens == 450
        assert usage.output_tokens == 185
        assert usage.llm_calls == 3
        assert usage.total_tokens == 635

    def test_summary_format(self, usage):
        """Test that summary returns a readable string."""
        usage.add_call(input_tokens=1000, output_tokens=500, description="test")

        summary = usage.summary()

        assert "Token Usage Summary" in summary
        assert "LLM calls: 1" in summary
        assert "Input tokens: 1,000" in summary
        assert "Output tokens: 500" in summary
        assert "Total tokens: 1,500" in summary

    def test_call_details_recorded(self, usage):
        """Test that call details are recorded."""
        usage.add_call(input_tokens=100, output_tokens=50, description="first call")
        usage.add_call(input_tokens=200, output_tokens=75, description="second call")

        assert len(usage._call_details) == 2
        assert usage._call_details[0]["description"] == "first call"
        assert usage._call_details[1]["description"] == "second call"

    def test_add_token_usage_instances(self):
        """Test combining two TokenUsage instances with + operator."""
        usage1 = TokenUsage()
        usage1.add_call(input_tokens=100, output_tokens=50, description="call 1")
        usage1.add_call(input_tokens=200, output_tokens=75, description="call 2")

        usage2 = TokenUsage()
        usage2.add_call(input_tokens=150, output_tokens=60, description="call 3")

        combined = usage1 + usage2

        assert combined.input_tokens == 450
        assert combined.output_tokens == 185
        assert combined.llm_calls == 3
        assert combined.total_tokens == 635

    def test_add_token_usage_preserves_call_details(self):
        """Test that combining TokenUsage instances preserves call details."""
        usage1 = TokenUsage()
        usage1.add_call(input_tokens=100, output_tokens=50, description="first")

        usage2 = TokenUsage()
        usage2.add_call(input_tokens=200, output_tokens=75, description="second")

        combined = usage1 + usage2

        assert len(combined._call_details) == 2
        assert combined._call_details[0]["description"] == "first"
        assert combined._call_details[1]["description"] == "second"
        # Call numbers should be renumbered
        assert combined._call_details[0]["call_number"] == 1
        assert combined._call_details[1]["call_number"] == 2

    def test_add_empty_token_usage(self, usage):
        """Test combining with empty TokenUsage."""
        usage.add_call(input_tokens=100, output_tokens=50, description="test")

        empty = TokenUsage()
        combined = usage + empty

        assert combined.input_tokens == 100
        assert combined.output_tokens == 50
        assert combined.llm_calls == 1

    def test_print_summary_output(self, usage):
        """Test that print_summary outputs formatted token usage."""
        usage.add_call(input_tokens=1000, output_tokens=500, description="test")

        # Capture console output without ANSI codes for easier assertion
        string_io = StringIO()
        console = Console(file=string_io, force_terminal=False, no_color=True)

        usage.print_summary(console)

        output = string_io.getvalue()

        assert "Token Usage:" in output
        assert "LLM calls:" in output
        assert "1" in output  # llm_calls
        assert "Input tokens:" in output
        assert "1,000" in output  # input_tokens formatted
        assert "Output tokens:" in output
        assert "500" in output  # output_tokens
        assert "Total tokens:" in output
        assert "1,500" in output  # total_tokens formatted
        # Token counts are now accurate (not estimated)
        assert "(estimated)" not in output

    def test_print_summary_creates_console_if_none(self, usage):
        """Test that print_summary works without passing a console."""
        usage.add_call(input_tokens=100, output_tokens=50, description="test")

        # Should not raise an error
        usage.print_summary()

    def test_print_summary_with_zero_usage(self):
        """Test print_summary with zero token usage."""
        usage = TokenUsage()

        string_io = StringIO()
        console = Console(file=string_io, force_terminal=False, no_color=True)

        usage.print_summary(console)

        output = string_io.getvalue()

        assert "LLM calls:" in output
        assert "0" in output
