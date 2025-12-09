"""File rename processor using LLM inference."""

from pathlib import Path
from typing import cast

from langchain.chat_models.base import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage

from pdfalive.models.rename import RenameOp, RenameResult
from pdfalive.prompts import RENAME_SYSTEM_PROMPT


class RenameProcessor:
    """Processor for intelligent file renaming using LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialize the rename processor.

        Args:
            llm: LangChain chat model for rename inference.
        """
        self.llm = llm

    def _extract_filenames(self, paths: list[Path]) -> list[str]:
        """Extract filenames from full paths.

        Args:
            paths: List of file paths.

        Returns:
            List of filenames (without directory components).
        """
        return [path.name for path in paths]

    def _build_path_mapping(self, paths: list[Path]) -> dict[str, Path]:
        """Build a mapping from filename to full path.

        Args:
            paths: List of file paths.

        Returns:
            Dictionary mapping filename to full path.

        Raises:
            ValueError: If duplicate filenames are found.
        """
        mapping: dict[str, Path] = {}
        for path in paths:
            filename = path.name
            if filename in mapping:
                raise ValueError(
                    f"Found duplicate filename '{filename}' in different directories. "
                    "All input files must have unique filenames."
                )
            mapping[filename] = path
        return mapping

    def generate_renames(self, paths: list[Path], query: str) -> RenameResult:
        """Generate rename suggestions using LLM.

        Args:
            paths: List of file paths to rename.
            query: User's renaming instruction/query.

        Returns:
            RenameResult containing suggested rename operations.
        """
        if not paths:
            return RenameResult(operations=[])

        filenames = self._extract_filenames(paths)

        # Build user prompt with filenames and query
        filenames_list = "\n".join(f"- {filename}" for filename in filenames)
        user_content = f"""Please rename the following files according to the instruction below.

## Files to rename:
{filenames_list}

## Renaming instruction:
{query}
"""

        messages = [
            SystemMessage(content=RENAME_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        model = self.llm.with_structured_output(RenameResult)
        response = model.invoke(messages)

        return cast(RenameResult, response)

    def _resolve_full_paths(
        self,
        operations: list[RenameOp],
        original_paths: list[Path],
    ) -> list[tuple[Path, Path]]:
        """Resolve rename operations to full source/target paths.

        Args:
            operations: List of rename operations with filenames only.
            original_paths: Original list of input paths.

        Returns:
            List of (source_path, target_path) tuples.
        """
        path_mapping = self._build_path_mapping(original_paths)
        resolved: list[tuple[Path, Path]] = []

        for op in operations:
            if op.input_filename not in path_mapping:
                # Skip operations for files that don't exist in our input
                continue

            source_path = path_mapping[op.input_filename]
            target_path = source_path.parent / op.output_filename
            resolved.append((source_path, target_path))

        return resolved

    def apply_renames(self, renames: list[tuple[Path, Path]]) -> None:
        """Apply rename operations to files.

        Args:
            renames: List of (source_path, target_path) tuples.

        Raises:
            FileNotFoundError: If source file doesn't exist.
            FileExistsError: If target file already exists.
        """
        # First validate all operations
        for source, target in renames:
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            if target.exists():
                raise FileExistsError(f"Target file already exists: {target}")

        # Apply renames
        for source, target in renames:
            source.rename(target)
