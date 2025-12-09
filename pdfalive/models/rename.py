"""Rename operation data models."""

from pydantic import BaseModel, Field


class RenameOp(BaseModel):
    """A single file rename operation."""

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


class RenameResult(BaseModel):
    """Result of a rename operation containing multiple file renames."""

    operations: list[RenameOp] = Field(
        description="List of rename operations to perform",
        default_factory=list,
    )

    def __len__(self) -> int:
        return len(self.operations)
