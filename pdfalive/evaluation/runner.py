"""Evaluation runner: execute the TOC pipeline on golden cases and score the output.

A case is defined entirely by a golden file at ``<evals_dir>/golden/<name>.json``:

    {
      "name": "geometry",
      "source_pdf": "data/My Book.pdf",      // resolved relative to evals_dir's parent
      "postprocess": true,                    // pipeline configuration for this case
      "description": "...",                   // optional, for humans
      "entries": [{"title": ..., "page_number": ..., "level": ...}, ...]
    }

The matching cassette lives at ``<evals_dir>/cassettes/<name>.json``. Modes:

- ``replay``: serve recorded LLM responses from the cassette (free, deterministic).
- ``record``: call the real LLM and write the cassette as a side effect.
- ``live``: call the real LLM without touching the cassette.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pymupdf
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel, Field

from pdfalive.evaluation.cassette import RecordingChatModel, ReplayChatModel
from pdfalive.evaluation.metrics import EvalReport, GoldenEntry, evaluate_toc
from pdfalive.models.toc import TOC, TOCEntry
from pdfalive.processors.toc_generator import DEFAULT_REQUEST_DELAY_SECONDS, TOCGenerator


EvalMode = Literal["replay", "record", "live"]

_GOLDEN_SUBDIR = "golden"
_CASSETTE_SUBDIR = "cassettes"

# Replay never hits the network, so inter-call delays are pointless.
_REPLAY_REQUEST_DELAY_SECONDS = 0.0


class GoldenFile(BaseModel):
    """Schema of a golden case file."""

    name: str = Field(description="Case name; must match the file stem")
    source_pdf: str = Field(description="PDF path, absolute or relative to the evals dir's parent")
    postprocess: bool = Field(default=True, description="Whether the pipeline runs with postprocessing")
    description: str = Field(default="", description="Human-readable provenance notes")
    entries: list[GoldenEntry] = Field(description="Ground-truth TOC entries")


@dataclass(frozen=True)
class EvalCase:
    """A resolved evaluation case: golden data, source PDF, and cassette location."""

    name: str
    pdf_path: Path
    golden_path: Path
    cassette_path: Path
    postprocess: bool


def load_golden_file(golden_path: Path) -> GoldenFile:
    """Load and validate a golden case file."""
    return GoldenFile.model_validate_json(golden_path.read_text())


def discover_cases(evals_dir: Path) -> list[EvalCase]:
    """Discover evaluation cases from golden files under ``<evals_dir>/golden/``.

    Args:
        evals_dir: Root of the evals data directory (containing golden/ and cassettes/).

    Returns:
        Cases sorted by name. Source PDF paths are resolved relative to the
        parent of evals_dir (typically the repository root).
    """
    evals_dir = Path(evals_dir)
    golden_dir = evals_dir / _GOLDEN_SUBDIR
    if not golden_dir.is_dir():
        return []

    cases = []
    for golden_path in sorted(golden_dir.glob("*.json")):
        golden = load_golden_file(golden_path)
        source_pdf = Path(golden.source_pdf)
        pdf_path = source_pdf if source_pdf.is_absolute() else evals_dir.parent / source_pdf
        cases.append(
            EvalCase(
                name=golden.name,
                pdf_path=pdf_path,
                golden_path=golden_path,
                cassette_path=evals_dir / _CASSETTE_SUBDIR / f"{golden.name}.json",
                postprocess=golden.postprocess,
            )
        )
    return cases


def _build_llm(case: EvalCase, mode: EvalMode, model_identifier: str, strict_replay: bool) -> BaseChatModel:
    """Build the chat model for the requested evaluation mode."""
    if mode == "replay":
        return cast(BaseChatModel, ReplayChatModel(cassette_path=case.cassette_path, strict=strict_replay))
    if mode == "record":
        real_llm = init_chat_model(model=model_identifier)
        return cast(BaseChatModel, RecordingChatModel(llm=real_llm, cassette_path=case.cassette_path))
    if mode == "live":
        return init_chat_model(model=model_identifier)
    raise ValueError(f"Unknown evaluation mode: {mode!r}")


def run_eval_case(
    case: EvalCase,
    mode: EvalMode,
    model_identifier: str = "gpt-5.5",
    strict_replay: bool = True,
    request_delay: float | None = None,
    num_processes: int | None = None,
) -> EvalReport:
    """Run the TOC generation pipeline on a case's PDF and score it against golden data.

    Args:
        case: The evaluation case to run.
        mode: "replay" (cassette, free), "record" (live + write cassette), or "live".
        model_identifier: LLM to use in record/live modes.
        strict_replay: In replay mode, fail on prompt drift (recommended).
        request_delay: Seconds between LLM calls; defaults to 0 for replay and
            the pipeline default for record/live.
        num_processes: Feature-extraction parallelism (passed to TOCGenerator).

    Returns:
        The EvalReport scoring the generated TOC against the golden entries.
    """
    golden = load_golden_file(case.golden_path)
    llm = _build_llm(case, mode, model_identifier, strict_replay)

    if request_delay is None:
        request_delay = _REPLAY_REQUEST_DELAY_SECONDS if mode == "replay" else DEFAULT_REQUEST_DELAY_SECONDS

    doc = pymupdf.open(case.pdf_path)
    try:
        generator = TOCGenerator(doc=doc, llm=llm, num_processes=num_processes)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as output:
            generator.run(
                output_file=output.name,
                force=True,
                request_delay=request_delay,
                postprocess=case.postprocess,
            )
        generated = TOC(entries=[TOCEntry.from_list(item) for item in doc.get_toc()])
    finally:
        doc.close()

    return evaluate_toc(golden.entries, generated)
