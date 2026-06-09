"""Tests for the VCR-style LLM cassette (record/replay)."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain.messages import HumanMessage, SystemMessage

from pdfalive.evaluation.cassette import (
    Cassette,
    CassetteMissError,
    RecordingChatModel,
    ReplayChatModel,
)
from pdfalive.models.toc import TOC


SYSTEM_PROMPT = "You are an expert TOC generator."
USER_PROMPT_BATCH_1 = "Generate a TOC for batch 1."
USER_PROMPT_BATCH_2 = "Generate a TOC for batch 2."


def make_messages(user_content: str) -> list:
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]


@pytest.fixture
def cassette_path(tmp_path: Path) -> Path:
    return tmp_path / "cassette.json"


@pytest.fixture
def toc_batch_1(make_toc: Callable[..., TOC]) -> TOC:
    return make_toc(("Chapter 1: Distance and Angles", 14, 1), ("1, §1. Lines", 14, 2))


@pytest.fixture
def toc_batch_2(make_toc: Callable[..., TOC]) -> TOC:
    return make_toc(("Chapter 2: Coordinates", 78, 1))


@pytest.fixture
def mock_llm(toc_batch_1: TOC, toc_batch_2: TOC) -> MagicMock:
    llm = MagicMock()
    structured = MagicMock()
    structured.invoke.side_effect = [toc_batch_1, toc_batch_2]
    llm.with_structured_output.return_value = structured
    return llm


class TestRecording:
    def test_delegates_to_wrapped_llm_and_returns_response(
        self, mock_llm: MagicMock, cassette_path: Path, toc_batch_1: TOC
    ) -> None:
        recorder = RecordingChatModel(llm=mock_llm, cassette_path=cassette_path)
        model = recorder.with_structured_output(TOC)

        response = model.invoke(make_messages(USER_PROMPT_BATCH_1))

        assert response == toc_batch_1
        mock_llm.with_structured_output.assert_called_once_with(TOC)

    def test_persists_each_call_to_cassette_file(
        self, mock_llm: MagicMock, cassette_path: Path, toc_batch_1: TOC, toc_batch_2: TOC
    ) -> None:
        model = RecordingChatModel(llm=mock_llm, cassette_path=cassette_path).with_structured_output(TOC)

        model.invoke(make_messages(USER_PROMPT_BATCH_1))
        assert len(Cassette.load(cassette_path).entries) == 1  # saved incrementally

        model.invoke(make_messages(USER_PROMPT_BATCH_2))
        cassette = Cassette.load(cassette_path)
        assert len(cassette.entries) == 2
        assert cassette.entries[0].response == toc_batch_1.model_dump()
        assert cassette.entries[1].response == toc_batch_2.model_dump()
        assert cassette.entries[0].schema_name == "TOC"


class TestReplay:
    @pytest.fixture
    def recorded_cassette_path(self, mock_llm: MagicMock, cassette_path: Path) -> Path:
        model = RecordingChatModel(llm=mock_llm, cassette_path=cassette_path).with_structured_output(TOC)
        model.invoke(make_messages(USER_PROMPT_BATCH_1))
        model.invoke(make_messages(USER_PROMPT_BATCH_2))
        return cassette_path

    def test_replays_recorded_responses_in_order(
        self, recorded_cassette_path: Path, toc_batch_1: TOC, toc_batch_2: TOC
    ) -> None:
        model = ReplayChatModel(cassette_path=recorded_cassette_path).with_structured_output(TOC)

        first = model.invoke(make_messages(USER_PROMPT_BATCH_1))
        second = model.invoke(make_messages(USER_PROMPT_BATCH_2))

        assert isinstance(first, TOC)
        assert first == toc_batch_1
        assert second == toc_batch_2

    def test_strict_replay_raises_on_request_mismatch(self, recorded_cassette_path: Path) -> None:
        model = ReplayChatModel(cassette_path=recorded_cassette_path).with_structured_output(TOC)

        with pytest.raises(CassetteMissError, match="hash"):
            model.invoke(make_messages("A prompt that was never recorded."))

    def test_loose_replay_returns_next_response_on_mismatch(
        self, recorded_cassette_path: Path, toc_batch_1: TOC
    ) -> None:
        model = ReplayChatModel(cassette_path=recorded_cassette_path, strict=False).with_structured_output(TOC)

        response = model.invoke(make_messages("A prompt that was never recorded."))

        assert response == toc_batch_1

    def test_raises_when_cassette_is_exhausted(self, recorded_cassette_path: Path) -> None:
        model = ReplayChatModel(cassette_path=recorded_cassette_path).with_structured_output(TOC)
        model.invoke(make_messages(USER_PROMPT_BATCH_1))
        model.invoke(make_messages(USER_PROMPT_BATCH_2))

        with pytest.raises(CassetteMissError, match="exhausted"):
            model.invoke(make_messages(USER_PROMPT_BATCH_1))

    def test_missing_cassette_file_raises_with_guidance(self, tmp_path: Path) -> None:
        with pytest.raises(CassetteMissError, match="record"):
            ReplayChatModel(cassette_path=tmp_path / "does_not_exist.json")
