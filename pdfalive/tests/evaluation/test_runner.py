"""Tests for the evaluation runner and case discovery."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pymupdf
import pytest
from click.testing import CliRunner
from langchain.chat_models.base import BaseChatModel

from pdfalive.cli import cli
from pdfalive.evaluation.cassette import RecordingChatModel
from pdfalive.evaluation.runner import EvalCase, discover_cases, run_eval_case
from pdfalive.models.toc import TOC
from pdfalive.processors.toc_generator import TOCGenerator


CASE_NAME = "book"
PDF_FILENAME = "book.pdf"

BODY_FONT = "helv"
HEADING_FONT = "hebo"
BODY_FONT_SIZE = 11
HEADING_FONT_SIZE = 18

# (title, page_number, level) for the synthetic book's golden TOC.
GOLDEN_SPECS = [
    ("Chapter 1: Foundations", 1, 1),
    ("Chapter 2: Advanced Topics", 2, 1),
    ("Index", 3, 1),
]


@pytest.fixture
def synthetic_pdf_path(tmp_path: Path) -> Path:
    """A small PDF with one heading per page plus body text."""
    doc = pymupdf.open()
    for title, _, _ in GOLDEN_SPECS:
        page = doc.new_page()
        page.insert_text((72, 100), title, fontname=HEADING_FONT, fontsize=HEADING_FONT_SIZE)
        for line_ix in range(5):
            page.insert_text(
                (72, 150 + 20 * line_ix),
                f"Body text line {line_ix} for context.",
                fontname=BODY_FONT,
                fontsize=BODY_FONT_SIZE,
            )
    pdf_path = tmp_path / PDF_FILENAME
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def evals_dir(tmp_path: Path, synthetic_pdf_path: Path) -> Path:
    """An evals directory with a golden file for the synthetic book."""
    evals = tmp_path / "evals"
    (evals / "golden").mkdir(parents=True)
    (evals / "cassettes").mkdir(parents=True)
    golden = {
        "name": CASE_NAME,
        "source_pdf": synthetic_pdf_path.name,  # resolved relative to evals_dir parent
        "postprocess": False,
        "entries": [{"title": t, "page_number": p, "level": lvl} for t, p, lvl in GOLDEN_SPECS],
    }
    (evals / "golden" / f"{CASE_NAME}.json").write_text(json.dumps(golden))
    return evals


@pytest.fixture
def mock_llm(make_toc: Callable[..., TOC]) -> MagicMock:
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = make_toc(*GOLDEN_SPECS)
    llm.with_structured_output.return_value = structured
    return llm


@pytest.fixture
def recorded_evals_dir(evals_dir: Path, synthetic_pdf_path: Path, mock_llm: MagicMock, tmp_path: Path) -> Path:
    """Evals dir with a cassette recorded through the real TOCGenerator pipeline."""
    cassette_path = evals_dir / "cassettes" / f"{CASE_NAME}.json"
    recorder = RecordingChatModel(llm=mock_llm, cassette_path=cassette_path)
    doc = pymupdf.open(synthetic_pdf_path)
    generator = TOCGenerator(doc=doc, llm=cast(BaseChatModel, recorder), num_processes=1)
    generator.run(output_file=str(tmp_path / "recorded_output.pdf"), force=True, request_delay=0)
    doc.close()
    return evals_dir


class TestDiscoverCases:
    def test_finds_cases_from_golden_files(self, evals_dir: Path, synthetic_pdf_path: Path) -> None:
        cases = discover_cases(evals_dir)

        assert [case.name for case in cases] == [CASE_NAME]
        case = cases[0]
        assert case.pdf_path == synthetic_pdf_path
        assert case.golden_path == evals_dir / "golden" / f"{CASE_NAME}.json"
        assert case.cassette_path == evals_dir / "cassettes" / f"{CASE_NAME}.json"
        assert case.postprocess is False

    def test_empty_evals_dir_yields_no_cases(self, tmp_path: Path) -> None:
        (tmp_path / "golden").mkdir()

        assert discover_cases(tmp_path) == []


class TestRunEvalCase:
    @pytest.fixture
    def case(self, recorded_evals_dir: Path) -> EvalCase:
        return discover_cases(recorded_evals_dir)[0]

    def test_replay_scores_perfectly_against_matching_golden(self, case: EvalCase) -> None:
        report = run_eval_case(case, mode="replay", num_processes=1)

        assert report.f1 == 1.0
        assert report.page_accuracy() == 1.0
        assert report.level_accuracy == 1.0

    def test_replay_makes_no_real_llm_calls(self, case: EvalCase, mock_llm: MagicMock) -> None:
        calls_before = mock_llm.with_structured_output.return_value.invoke.call_count

        run_eval_case(case, mode="replay", num_processes=1)

        assert mock_llm.with_structured_output.return_value.invoke.call_count == calls_before

    def test_replay_detects_missing_golden_entries(self, case: EvalCase, recorded_evals_dir: Path) -> None:
        golden = json.loads(case.golden_path.read_text())
        golden["entries"].append({"title": "Appendix A: Extra Material", "page_number": 3, "level": 1})
        case.golden_path.write_text(json.dumps(golden))
        case = discover_cases(recorded_evals_dir)[0]  # golden data is parsed at discovery time

        report = run_eval_case(case, mode="replay", num_processes=1)

        assert report.recall == pytest.approx(3 / 4)
        assert [entry.title for entry in report.missing] == ["Appendix A: Extra Material"]

    def test_unknown_mode_raises(self, case: EvalCase) -> None:
        with pytest.raises(ValueError, match="mode"):
            run_eval_case(case, mode="invalid", num_processes=1)  # type: ignore[arg-type]


class TestEvalCommand:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def eval_args(self, evals_dir: Path, *extra: str) -> list[str]:
        return ["eval", "--evals-dir", str(evals_dir), "--mode", "replay", "--num-processes", "1", *extra]

    def test_replay_reports_metrics_for_all_cases(self, runner: CliRunner, recorded_evals_dir: Path) -> None:
        result = runner.invoke(cli, self.eval_args(recorded_evals_dir))

        assert result.exit_code == 0, result.output
        assert CASE_NAME in result.output
        assert "1.00" in result.output  # perfect scores rendered in the table

    def test_failing_threshold_sets_exit_code(self, runner: CliRunner, recorded_evals_dir: Path) -> None:
        golden_path = recorded_evals_dir / "golden" / f"{CASE_NAME}.json"
        golden = json.loads(golden_path.read_text())
        golden["entries"].append({"title": "Appendix A: Extra Material", "page_number": 3, "level": 1})
        golden_path.write_text(json.dumps(golden))

        result = runner.invoke(cli, self.eval_args(recorded_evals_dir, "--min-f1", "0.99"))

        assert result.exit_code == 1
        assert "Appendix A: Extra Material" in result.output  # missing entries are listed

    def test_passing_threshold_keeps_exit_code_zero(self, runner: CliRunner, recorded_evals_dir: Path) -> None:
        result = runner.invoke(cli, self.eval_args(recorded_evals_dir, "--min-f1", "0.99"))

        assert result.exit_code == 0, result.output

    def test_case_filter_selects_subset(self, runner: CliRunner, recorded_evals_dir: Path) -> None:
        result = runner.invoke(cli, self.eval_args(recorded_evals_dir, "--case", CASE_NAME))

        assert result.exit_code == 0, result.output

    def test_unknown_case_name_errors(self, runner: CliRunner, recorded_evals_dir: Path) -> None:
        result = runner.invoke(cli, self.eval_args(recorded_evals_dir, "--case", "nonexistent"))

        assert result.exit_code != 0
        assert "nonexistent" in result.output


class TestMissingSourcePdf:
    def test_missing_pdf_raises_clear_error_naming_the_case(self, evals_dir: Path, synthetic_pdf_path: Path) -> None:
        synthetic_pdf_path.unlink()
        case = discover_cases(evals_dir)[0]

        with pytest.raises(FileNotFoundError, match=CASE_NAME):
            run_eval_case(case, mode="replay", num_processes=1)
