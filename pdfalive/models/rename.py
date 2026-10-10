"""Rename operation data models."""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class RenameError:
    """Represents a failed rename operation."""

    source: Path
    target: Path
    error: str


@dataclass
class ApplyRenamesResult:
    """Result of applying rename operations."""

    successful: list[tuple[Path, Path]] = field(default_factory=list)
    failed: list[RenameError] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        """Number of successful renames."""
        return len(self.successful)

    @property
    def failure_count(self) -> int:
        """Number of failed renames."""
        return len(self.failed)

    @property
    def all_succeeded(self) -> bool:
        """True if all renames succeeded."""
        return len(self.failed) == 0


class RenameOp(BaseModel):
    """A single file rename operation; immutable, so the resolved renames that hold one are hashable."""

    model_config = ConfigDict(frozen=True)

    input_filename: str = Field(description="Original filename (without directory path)")
    output_filename: str = Field(description="New filename (without directory path)")
    confidence: float = Field(
        description="Confidence score for this rename operation (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Brief explanation of why this rename was suggested",
        default="",
    )

    def __str__(self) -> str:
        return f"RenameOp('{self.input_filename}' -> '{self.output_filename}', confidence={self.confidence})"


@dataclass(frozen=True)
class PlannedRename:
    """A rename suggestion resolved to the file it renames and its new path."""

    op: RenameOp
    source: Path
    target: Path


@dataclass(frozen=True)
class RejectedRename:
    """A rename suggestion that will not be applied, and why."""

    op: RenameOp
    reason: str


@dataclass
class ResolvedRenames:
    """Rename suggestions resolved against the input files: the renames to apply and the ones rejected."""

    planned: list[PlannedRename] = field(default_factory=list)
    rejected: list[RejectedRename] = field(default_factory=list)

    @property
    def source_target_pairs(self) -> list[tuple[Path, Path]]:
        """The planned renames as (source, target) paths, as ``RenameProcessor.apply_renames`` takes them."""
        return [(rename.source, rename.target) for rename in self.planned]


class RenameResult(BaseModel):
    """Result of a rename operation containing multiple file renames."""

    operations: list[RenameOp] = Field(
        description="List of rename operations to perform",
        default_factory=list,
    )

    def __len__(self) -> int:
        return len(self.operations)

    def merge(self, other: "RenameResult") -> "RenameResult":
        """Merge another RenameResult into this one, handling duplicates.

        When operations have the same input_filename, the operation from `self`
        (the earlier batch) is preferred.

        Args:
            other: Another RenameResult to merge with this one.

        Returns:
            A new RenameResult containing operations from both, with duplicates removed.
        """
        # Use input_filename as key for deduplication
        # Prefer operations from self (earlier batch)
        seen: dict[str, RenameOp] = {}

        for op in self.operations:
            if op.input_filename not in seen:
                seen[op.input_filename] = op

        for op in other.operations:
            if op.input_filename not in seen:
                seen[op.input_filename] = op

        return RenameResult(operations=list(seen.values()))
