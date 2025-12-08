"""Unit tests for token counting utilities."""

import pytest

from pdfalive.tokens import (
    TokenUsage,
    estimate_features_tokens,
    estimate_tokens,
)


class TestEstimateTokens:
    """Tests for token estimation functions."""

    @pytest.mark.parametrize(
        "text,expected_min,expected_max",
        [
            ("", 0, 0),
            ("hello", 1, 2),  # 5 chars / 3 = 1.67 -> 1
            ("a" * 100, 30, 40),  # 100 chars / 3 = 33
        ],
    )
    def test_estimate_tokens_ranges(self, text, expected_min, expected_max):
        """Test token estimation returns reasonable values."""
        result = estimate_tokens(text)
        assert expected_min <= result <= expected_max

    def test_estimate_tokens_scales_linearly(self):
        """Test that token count scales roughly linearly with text length."""
        short = estimate_tokens("a" * 99)  # Use 99 to get clean division by 3
        long = estimate_tokens("a" * 990)
        assert long == short * 10


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
