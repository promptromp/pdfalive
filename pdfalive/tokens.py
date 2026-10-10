"""Token counting utilities using tiktoken for accurate tokenizer-based counts."""

from dataclasses import dataclass, field

import tiktoken
from rich.text import Text

from pdfalive.console import ConsoleType, TextPart, create_console, safe_text


# Default tiktoken encoding — o200k_base (GPT-4o/5) produces near-identical counts
# to cl100k_base for our data format, so it works as a universal default.
DEFAULT_ENCODING = "o200k_base"

_encoding: tiktoken.Encoding | None = None


def get_encoding() -> tiktoken.Encoding:
    """Return the lazily-loaded tiktoken encoding singleton."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
    return _encoding


@dataclass
class TokenUsage:
    """Tracks token usage across multiple LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    _call_details: list[dict] = field(default_factory=list)

    def add_call(self, input_tokens: int, output_tokens: int, description: str = "") -> None:
        """Record a single LLM call's token usage."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_calls += 1
        self._call_details.append(
            {
                "call_number": self.llm_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "description": description,
            }
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Combine two TokenUsage instances."""
        combined = TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
        )
        # Renumber call details when combining
        combined._call_details = list(self._call_details)
        for detail in other._call_details:
            new_detail = detail.copy()
            new_detail["call_number"] = len(combined._call_details) + 1
            combined._call_details.append(new_detail)
        return combined

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all calls."""
        return self.input_tokens + self.output_tokens

    def _summary_fields(self) -> list[tuple[str, str]]:
        """(label, formatted count) pairs shared by the plain and styled summaries."""
        return [
            ("LLM calls", f"{self.llm_calls}"),
            ("Input tokens", f"{self.input_tokens:,}"),
            ("Output tokens", f"{self.output_tokens:,}"),
            ("Total tokens", f"{self.total_tokens:,}"),
        ]

    def summary(self) -> str:
        """Return a human-readable summary of token usage."""
        lines = ["Token Usage Summary:", *(f"  {label}: {value}" for label, value in self._summary_fields())]
        return "\n".join(lines)

    def summary_text(self) -> Text:
        """Return the token usage summary styled for printing on a Rich console."""
        parts: list[TextPart] = [("Token Usage:", "bold")]
        for label, value in self._summary_fields():
            parts += [f"\n  {label}: ", (value, "cyan")]
        return safe_text(*parts, multiline=True)

    def print_summary(self, console: ConsoleType | None = None) -> None:
        """Print the token usage summary, set off by a blank line.

        Args:
            console: Console to print on; one from ``create_console()`` when omitted.
        """
        if console is None:
            console = create_console()
        console.print()
        console.print(self.summary_text())


def estimate_tokens(text: str) -> int:
    """Count the number of tokens in a text string using tiktoken.

    Uses the o200k_base encoding which provides accurate counts for both
    English prose and structured/pipe-delimited content.
    """
    return len(get_encoding().encode(text))


def estimate_features_tokens(features: list) -> int:
    """Estimate tokens needed to represent features in the prompt."""
    return estimate_tokens(str(features))
