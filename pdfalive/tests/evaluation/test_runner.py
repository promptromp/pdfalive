"""Tests for the evaluation runner and case discovery."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from click.testing import CliRunner
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel

from pdfalive.cli import _FIRST_PASS_TABLE_TITLE, cli
from pdfalive.evaluation.cassette import Cassette, RecordingChatModel
from pdfalive.evaluation.runner import EvalCase, discover_cases, run_eval_case
from pdfalive.models.toc import TOC, PostprocessedTOC
from pdfalive.processors.toc_generator import TOCGenerator


CASE_NAME = "book"
PDF_FILENAME = "book.pdf"

BODY_FONT = "helv"
HEADING_FONT = "hebo"
BODY_FONT_SIZE = 11
HEADING_FONT_SIZE = 18

RECORDING_MODEL = "test-model"

# (title, page_number, level) for the synthetic book's golden TOC.
GOLDEN_SPECS = [
    ("Chapter 1: Foundations", 1, 1),
    ("Chapter 2: Advanced Topics", 2, 1),
    ("Index", 3, 1),
]
# What the first pass of a postprocessing case finds: everything but the last golden entry,
# which the postprocessing step then restores.
FIRST_PASS_SPECS = GOLDEN_SPECS[:-1]
# An F1 threshold between that first pass's score (it misses one entry) and a perfect score.
MIN_F1_THRESHOLD = 0.99


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
def mock_llm(make_toc: Callable[..., TOC], make_structured_answer: Callable[..., dict[str, Any]]) -> MagicMock:
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.return_value = make_structured_answer(make_toc(*GOLDEN_SPECS))
    llm.with_structured_output.return_value = structured
    return llm


@pytest.fixture
def recorded_evals_dir(evals_dir: Path, synthetic_pdf_path: Path, mock_llm: MagicMock, tmp_path: Path) -> Path:
    """Evals dir with a cassette recorded through the real TOCGenerator pipeline."""
    cassette_path = evals_dir / "cassettes" / f"{CASE_NAME}.json"
    recorder = RecordingChatModel(llm=mock_llm, model_identifier=RECORDING_MODEL, cassette_path=cassette_path)
    doc = pymupdf.open(synthetic_pdf_path)
    try:
        generator = TOCGenerator(doc=doc, llm=cast(BaseChatModel, recorder), num_processes=1)
        generator.run(output_file=str(tmp_path / "recorded_output.pdf"), force=True, request_delay=0)
    finally:
        doc.close()
    return evals_dir


@pytest.fixture
def refining_llm(make_toc: Callable[..., TOC], make_structured_answer: Callable[..., dict[str, Any]]) -> MagicMock:
    """An LLM whose first pass misses the last golden entry and whose postprocessing restores it."""
    refined = PostprocessedTOC.model_validate(make_toc(*GOLDEN_SPECS), from_attributes=True)
    answers: dict[type[BaseModel], dict[str, Any]] = {
        TOC: make_structured_answer(make_toc(*FIRST_PASS_SPECS)),
        PostprocessedTOC: make_structured_answer(refined),
    }

    def structured_model(schema: type[BaseModel], include_raw: bool) -> MagicMock:
        model = MagicMock()
        model.invoke.return_value = answers[schema]
        return model

    llm = MagicMock()
    llm.with_structured_output.side_effect = structured_model
    return llm


@pytest.fixture
def postprocess_evals_dir(evals_dir: Path, synthetic_pdf_path: Path, refining_llm: MagicMock) -> Path:
    """Evals dir whose case runs with postprocessing, with a cassette recorded through the real pipeline."""
    golden_path = evals_dir / "golden" / f"{CASE_NAME}.json"
    golden_path.write_text(json.dumps(json.loads(golden_path.read_text()) | {"postprocess": True}))
    recorder = RecordingChatModel(
        llm=refining_llm, model_identifier=RECORDING_MODEL, cassette_path=evals_dir / "cassettes" / f"{CASE_NAME}.json"
    )
    doc = pymupdf.open(synthetic_pdf_path)
    try:
        TOCGenerator(doc=doc, llm=cast(BaseChatModel, recorder), num_processes=1).generate(
            request_delay=0, postprocess=True
        )
    finally:
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
        report = run_eval_case(case, mode="replay", num_processes=1).report

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

        report = run_eval_case(case, mode="replay", num_processes=1).report

        assert report.recall == pytest.approx(3 / 4)
        assert [entry.title for entry in report.missing] == ["Appendix A: Extra Material"]

    def test_record_mode_names_the_model_in_the_cassette(self, case: EvalCase, mock_llm: MagicMock) -> None:
        with patch("pdfalive.evaluation.runner.build_chat_model", return_value=mock_llm) as build_chat_model:
            run_eval_case(case, mode="record", model_identifier=RECORDING_MODEL, request_delay=0, num_processes=1)

        build_chat_model.assert_called_once_with(RECORDING_MODEL, None)
        assert Cassette.load(case.cassette_path).model_identifier == RECORDING_MODEL

    def test_unknown_mode_raises(self, case: EvalCase) -> None:
        with pytest.raises(ValueError, match="mode"):
            run_eval_case(case, mode="invalid", num_processes=1)  # type: ignore[arg-type]

    def test_without_postprocessing_there_is_no_separate_first_pass_report(self, case: EvalCase) -> None:
        """The written TOC is then the first pass's, so it is scored once."""
        result = run_eval_case(case, mode="replay", num_processes=1)

        assert result.first_pass_report is None

    def test_postprocessing_case_also_scores_the_first_pass(self, postprocess_evals_dir: Path) -> None:
        """One run scores both the written TOC and the TOC a run without postprocessing writes."""
        case = discover_cases(postprocess_evals_dir)[0]

        result = run_eval_case(case, mode="replay", num_processes=1)

        assert result.report.recall == 1.0
        assert result.first_pass_report is not None
        assert result.first_pass_report.recall == pytest.approx(len(FIRST_PASS_SPECS) / len(GOLDEN_SPECS))
        assert [entry.title for entry in result.first_pass_report.missing] == [GOLDEN_SPECS[-1][0]]


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

        result = runner.invoke(cli, self.eval_args(recorded_evals_dir, "--min-f1", str(MIN_F1_THRESHOLD)))

        assert result.exit_code == 1
        assert "Appendix A: Extra Material" in result.output  # missing entries are listed

    def test_passing_threshold_keeps_exit_code_zero(self, runner: CliRunner, recorded_evals_dir: Path) -> None:
        result = runner.invoke(cli, self.eval_args(recorded_evals_dir, "--min-f1", str(MIN_F1_THRESHOLD)))

        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize(
        ("evals_dir_fixture", "shows_first_pass_table"),
        [("recorded_evals_dir", False), ("postprocess_evals_dir", True)],
    )
    def test_first_pass_table_only_for_postprocessing_cases(
        self, runner: CliRunner, request: pytest.FixtureRequest, evals_dir_fixture: str, shows_first_pass_table: bool
    ) -> None:
        result = runner.invoke(cli, self.eval_args(request.getfixturevalue(evals_dir_fixture)))

        assert result.exit_code == 0, result.output
        assert (_FIRST_PASS_TABLE_TITLE in result.output) == shows_first_pass_table

    def test_thresholds_gate_on_the_written_toc(self, runner: CliRunner, postprocess_evals_dir: Path) -> None:
        """The first pass misses an entry, but the written (postprocessed) TOC passes the threshold."""
        result = runner.invoke(cli, self.eval_args(postprocess_evals_dir, "--min-f1", str(MIN_F1_THRESHOLD)))

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
