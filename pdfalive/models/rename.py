"""Rename operation data models."""

import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pdfalive.paths import name_key, unique_input_paths


# Ids of the files in each request start here: every request numbers its files from one.
FIRST_FILE_ID = 1


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

    file_id: int = Field(description="Id of the file to rename, exactly as given in this request's list of files")
    input_filename: str = Field(description="Name of the file to rename, exactly as given with its id")
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
        return (
            f"RenameOp(#{self.file_id} '{self.input_filename}' -> '{self.output_filename}', "
            f"confidence={self.confidence})"
        )


class RenameInputs:
    """The files to rename, in input order, each file once.

    Construction enforces that: a file given more than once, in any spelling,
    is listed once (pdfalive.paths.unique_input_paths). Files sharing a name in
    different directories are separate inputs; suggestions name files by their
    id in a request (RenameBatch), not by name.
    """

    __slots__ = ("_paths",)

    def __init__(self, paths: Iterable[Path]) -> None:
        self._paths = tuple(unique_input_paths(paths))

    @property
    def paths(self) -> tuple[Path, ...]:
        return self._paths

    def __len__(self) -> int:
        return len(self._paths)

    def display_names(self) -> dict[Path, str]:
        """How to show each input so that inputs can be told apart.

        An input is shown by its file name, unless another input's name agrees
        with it by name key (pdfalive.paths.name_key); those are shown by their
        resolved paths relative to the directory they have in common.
        """
        sharing_a_name: dict[str, list[Path]] = defaultdict(list)
        for path in self._paths:
            sharing_a_name[name_key(path.name)].append(path)

        names: dict[Path, str] = {}
        for group in sharing_a_name.values():
            if len(group) == 1:
                names[group[0]] = group[0].name
                continue
            resolved = [path.parent.resolve() / path.name for path in group]
            try:
                common = os.path.commonpath([path.parent for path in resolved])
            except ValueError:  # paths on different Windows drives share no directory
                names.update(zip(group, map(str, resolved), strict=True))
                continue
            names.update((path, os.path.relpath(full, common)) for path, full in zip(group, resolved, strict=True))
        return names


@dataclass(frozen=True)
class RenameBatch:
    """One LLM request's rename suggestions, with the files that request listed.

    The request numbered its files from FIRST_FILE_ID, so a suggestion's file_id
    is a position in files: positional numbering by the model is right by
    construction, and an id cannot name a file another request listed.
    """

    files: tuple[Path, ...]
    operations: tuple[RenameOp, ...]

    def path(self, file_id: int) -> Path | None:
        """The file with this id in the request, or None if the request listed no such file."""
        index = file_id - FIRST_FILE_ID
        return self.files[index] if 0 <= index < len(self.files) else None


@dataclass(frozen=True)
class PlannedRename:
    """A rename suggestion resolved to the file it renames and its new path."""

    op: RenameOp
    source: Path
    target: Path


@dataclass(frozen=True)
class RejectedRename:
    """A rename suggestion that will not be applied, and why.

    ``source`` is the input the suggestion was for, or None when its file id names none
    of the files its request listed.
    """

    op: RenameOp
    reason: str
    source: Path | None


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
