"""Token counting and estimation utilities."""

from dataclasses import dataclass, field

from rich.console import Console


# Rough estimate for token counting
# For structured data with special characters, punctuation, and numbers,
# the ratio is closer to 3 chars per token (more conservative than 4)
CHARS_PER_TOKEN = 3


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

    def summary(self) -> str:
        """Return a human-readable summary of token usage."""
        lines = [
            "Token Usage Summary:",
            f"  LLM calls: {self.llm_calls}",
            f"  Input tokens: {self.input_tokens:,}",
            f"  Output tokens: {self.output_tokens:,}",
            f"  Total tokens: {self.total_tokens:,}",
        ]
        return "\n".join(lines)

    def print_summary(self, console: Console | None = None) -> None:
        """Print a formatted token usage summary to the console.

        Args:
            console: Rich Console instance to use. If None, creates a new one.
        """
        if console is None:
            console = Console()

        console.print()
        console.print("[bold]Token Usage:[/bold]")
        console.print(f"  LLM calls: [cyan]{self.llm_calls}[/cyan]")
        console.print(f"  Input tokens: [cyan]{self.input_tokens:,}[/cyan] (estimated)")
        console.print(f"  Output tokens: [cyan]{self.output_tokens:,}[/cyan] (estimated)")
        console.print(f"  Total tokens: [cyan]{self.total_tokens:,}[/cyan]")


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    This is a rough estimate based on character count.
    For more accurate counts, use a tokenizer specific to the model.
    """
    return len(text) // CHARS_PER_TOKEN


def estimate_features_tokens(features: list) -> int:
    """Estimate tokens needed to represent features in the prompt."""
    return estimate_tokens(str(features))
