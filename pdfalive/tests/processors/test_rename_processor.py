"""Unit tests for RenameProcessor."""

import os
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from pdfalive.models.rename import PlannedRename, RejectedRename, RenameOp, RenameResult, ResolvedRenames
from pdfalive.processors import rename_processor as rename_processor_module
from pdfalive.processors.rename_processor import (
    PROMPT_OVERHEAD_TOKENS,
    REJECTED_DIRECTORY_NAME,
    REJECTED_DUPLICATE_TARGET,
    REJECTED_EMPTY_NAME,
    REJECTED_HIDDEN_NAME,
    REJECTED_PATH,
    REJECTED_RESERVED_NAME,
    REJECTED_SURROUNDING_WHITESPACE,
    REJECTED_TARGET_IS_INPUT,
    REJECTED_UNKNOWN_INPUT,
    REJECTED_UNSAFE_CHARACTER,
    DuplicateInputNameError,
    RenameProcessor,
    _collision_problem,
    _target_name_problem,
    build_input_mapping,
)
from pdfalive.tokens import TokenUsage


# Batch label passed to _invoke_with_retry for its progress messages.
_BATCH_DESCRIPTION = "rename request"

QUERY = "Rename files"
NEW_NAME = "renamed.pdf"
OTHER_NEW_NAME = "also renamed.pdf"
SOURCE = Path("/docs/input.pdf")
OTHER = Path("/docs/other.pdf")
# The inputs the sample suggestions are for.
SAMPLE_PATHS = (Path("/docs/old_file.pdf"), Path("/docs/another_file.pdf"))
# Bytes of a file that must survive a rejected rename onto it.
PRECIOUS_CONTENT = b"the only copy"
# Long names, and a token limit with room for less than one: a batch always takes at
# least one file, so each of these gets a batch of its own.
LONG_NAMES = tuple(f"very_long_filename_to_force_multiple_batches_file{n}.pdf" for n in range(1, 4))
ONE_FILE_PER_BATCH = PROMPT_OVERHEAD_TOKENS + 1
SMALL_BATCH_TOKENS = 500 + PROMPT_OVERHEAD_TOKENS
LARGE_BATCH_TOKENS = 10000


def _inputs(*paths: Path) -> dict[str, Path]:
    return build_input_mapping(paths)


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MagicMock()


@pytest.fixture
def structured_llm(mock_llm: MagicMock) -> MagicMock:
    """The structured-output runnable the processor invokes."""
    return mock_llm.with_structured_output.return_value


@pytest.fixture
def sample_rename_result(make_rename_op: Callable[..., RenameOp]) -> RenameResult:
    """Sample rename result from LLM, for the two SAMPLE_PATHS."""
    return RenameResult(
        operations=[
            make_rename_op(SAMPLE_PATHS[0].name, "New File Name.pdf"),
            make_rename_op(SAMPLE_PATHS[1].name, "Another New Name.pdf"),
        ]
    )


@pytest.fixture
def resolve(mock_llm: MagicMock) -> Callable[..., ResolvedRenames]:
    """Resolve suggestions against inputs (SOURCE and OTHER by default)."""

    def _resolve(*operations: RenameOp, paths: tuple[Path, ...] = (SOURCE, OTHER)) -> ResolvedRenames:
        return RenameProcessor(llm=mock_llm).resolve_renames(list(operations), _inputs(*paths))

    return _resolve


def _user_message(structured_llm: MagicMock, call_index: int = 0) -> str:
    return structured_llm.invoke.call_args_list[call_index].args[0][-1].content


def _system_message(structured_llm: MagicMock, call_index: int) -> str:
    return structured_llm.invoke.call_args_list[call_index].args[0][0].content


class TestBuildInputMapping:
    def test_maps_each_name_key_to_its_path_in_input_order(self) -> None:
        paths = (Path("/z/Zebra.pdf"), Path("/a/apple.pdf"), Path("relative/path/mango.pdf"))

        mapping = _inputs(*paths)

        assert list(mapping.values()) == list(paths)
        assert mapping["zebra.pdf"] == paths[0]

    @pytest.mark.parametrize(
        "other_name", ["input.pdf", "INPUT.pdf", unicodedata.normalize("NFD", "input.pdf")], ids=["same", "case", "nfd"]
    )
    def test_different_files_sharing_a_name_are_an_error(self, other_name: str) -> None:
        """Suggestions are matched by name key, so names that agree by it cannot be told apart."""
        with pytest.raises(DuplicateInputNameError, match="duplicate"):
            _inputs(Path("/first/input.pdf"), Path("/second") / other_name)

    def test_the_same_file_in_other_spellings_is_listed_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        source = tmp_path / SOURCE.name
        source.touch()

        assert list(_inputs(source, Path(source.name), tmp_path / ".." / tmp_path.name / source.name).values()) == [
            source
        ]


class TestGenerateRenames:
    def test_calls_the_llm_and_returns_its_suggestions(self, mock_llm, structured_llm, sample_rename_result):
        structured_llm.invoke.return_value = sample_rename_result

        result = RenameProcessor(llm=mock_llm).generate_renames(_inputs(*SAMPLE_PATHS), QUERY, usage=TokenUsage())

        mock_llm.with_structured_output.assert_called_once_with(RenameResult)
        structured_llm.invoke.assert_called_once()
        assert result == sample_rename_result

    def test_the_prompt_holds_the_query_and_file_names_but_no_directories(
        self, mock_llm, structured_llm, sample_rename_result
    ):
        structured_llm.invoke.return_value = sample_rename_result

        RenameProcessor(llm=mock_llm).generate_renames(_inputs(*SAMPLE_PATHS), QUERY, usage=TokenUsage())

        message = _user_message(structured_llm)
        assert QUERY in message
        assert all(f"- {path.name}" in message for path in SAMPLE_PATHS)
        assert str(SAMPLE_PATHS[0].parent) not in message

    def test_each_call_is_recorded_in_the_callers_usage(self, mock_llm, structured_llm, sample_rename_result):
        structured_llm.invoke.return_value = sample_rename_result
        usage = TokenUsage()

        RenameProcessor(llm=mock_llm).generate_renames(_inputs(*SAMPLE_PATHS), QUERY, usage=usage)

        assert usage.llm_calls == 1
        assert usage.input_tokens > 0

    def test_no_inputs_make_no_calls(self, mock_llm, structured_llm):
        usage = TokenUsage()

        result = RenameProcessor(llm=mock_llm).generate_renames({}, QUERY, usage=usage)

        assert result.operations == []
        assert usage.llm_calls == 0
        structured_llm.invoke.assert_not_called()

    def test_retry_warning_renders_exception_message_verbatim(
        self, mock_llm, sample_rename_result, recording_console, monkeypatch, instant_retries, markup_like_text
    ):
        monkeypatch.setattr(rename_processor_module, "console", recording_console)
        error = RuntimeError(markup_like_text)
        model = MagicMock()
        model.invoke.side_effect = [error, sample_rename_result]

        RenameProcessor(llm=mock_llm)._invoke_with_retry(
            model, messages=[], batch_description=_BATCH_DESCRIPTION, input_tokens=0
        )

        assert f"({type(error).__name__}: {markup_like_text})" in recording_console.export_text()

    def test_retry_survives_a_non_string_exception_message(
        self,
        mock_llm,
        sample_rename_result,
        recording_console,
        monkeypatch,
        instant_retries,
        non_string_message_error,
        non_string_error_message,
    ):
        """An error whose ``message`` is not a str is reported, and the retry goes on to succeed."""
        monkeypatch.setattr(rename_processor_module, "console", recording_console)
        model = MagicMock()
        model.invoke.side_effect = [non_string_message_error, sample_rename_result]

        result = RenameProcessor(llm=mock_llm)._invoke_with_retry(
            model, messages=[], batch_description=_BATCH_DESCRIPTION, input_tokens=0
        )

        assert result == sample_rename_result
        expected = f"({type(non_string_message_error).__name__}: {non_string_error_message})"
        assert expected in recording_console.export_text()


class TestBatching:
    def test_batch_filenames_single_batch(self, mock_llm):
        filenames = ["file1.pdf", "file2.pdf", "file3.pdf"]

        assert list(RenameProcessor(llm=mock_llm)._batch_filenames(filenames, max_tokens=LARGE_BATCH_TOKENS)) == [
            filenames
        ]

    def test_batch_filenames_splits_in_order(self, mock_llm):
        filenames = [f"very_long_filename_number_{i:04d}_with_lots_of_text.pdf" for i in range(100)]

        batches = list(RenameProcessor(llm=mock_llm)._batch_filenames(filenames, max_tokens=SMALL_BATCH_TOKENS))

        assert len(batches) > 1
        assert [name for batch in batches for name in batch] == filenames

    def test_batch_filenames_empty_list(self, mock_llm):
        assert list(RenameProcessor(llm=mock_llm)._batch_filenames([])) == [[]]

    def test_batched_results_are_merged(self, mock_llm, structured_llm, make_rename_op):
        answers = [RenameResult(operations=[make_rename_op(name, f"{name} {NEW_NAME}")]) for name in LONG_NAMES]
        structured_llm.invoke.side_effect = answers
        usage = TokenUsage()

        result = RenameProcessor(llm=mock_llm).generate_renames(
            _inputs(*(Path("/docs") / name for name in LONG_NAMES)),
            QUERY,
            usage=usage,
            max_tokens_per_batch=ONE_FILE_PER_BATCH,
            request_delay=0,
        )

        assert result.operations == [op for answer in answers for op in answer.operations]
        assert usage.llm_calls == len(LONG_NAMES)

    def test_later_batches_use_the_continuation_prompt(self, mock_llm, structured_llm):
        structured_llm.invoke.return_value = RenameResult(operations=[])

        RenameProcessor(llm=mock_llm).generate_renames(
            _inputs(*(Path("/docs") / name for name in LONG_NAMES)),
            QUERY,
            usage=TokenUsage(),
            max_tokens_per_batch=ONE_FILE_PER_BATCH,
            request_delay=0,
        )

        assert "CONTINUATION" not in _system_message(structured_llm, call_index=0)
        assert "CONTINUATION" in _system_message(structured_llm, call_index=1)

    @pytest.mark.usefixtures("instant_retries")
    def test_completed_batches_stay_recorded_when_a_later_one_fails(
        self, mock_llm, structured_llm, sample_rename_result
    ) -> None:
        """The caller's TokenUsage keeps what was billed, so the CLI can still report it."""
        completed_batches = len(LONG_NAMES) - 1
        responses = iter([sample_rename_result] * completed_batches)

        def invoke(messages):
            response = next(responses, None)
            if response is None:
                raise RuntimeError("the last batch fails on every attempt")
            return response

        structured_llm.invoke.side_effect = invoke
        usage = TokenUsage()

        with pytest.raises(RuntimeError):
            RenameProcessor(llm=mock_llm).generate_renames(
                _inputs(*(Path("/docs") / name for name in LONG_NAMES)),
                QUERY,
                usage=usage,
                max_tokens_per_batch=ONE_FILE_PER_BATCH,
                request_delay=0,
            )

        assert usage.llm_calls == completed_batches


class TestTargetNameRules:
    """What a plain file name is, per OS flavor, and which names are hard to see, type or use."""

    @pytest.mark.parametrize(
        ("output_filename", "reason"),
        [
            ("", REJECTED_EMPTY_NAME),
            ("   ", REJECTED_EMPTY_NAME),
            (" leading.pdf", REJECTED_SURROUNDING_WHITESPACE),
            ("trailing.pdf ", REJECTED_SURROUNDING_WHITESPACE),
            (".", REJECTED_DIRECTORY_NAME),
            ("..", REJECTED_DIRECTORY_NAME),
            ("../escape.pdf", REJECTED_PATH),
            ("sub/dir.pdf", REJECTED_PATH),
            ("/etc/absolute.pdf", REJECTED_PATH),
            ("trailing slash/", REJECTED_PATH),
            ("nul\x00.pdf", REJECTED_UNSAFE_CHARACTER),
            ("escape\x1b[8m.pdf", REJECTED_UNSAFE_CHARACTER),
            ("line\nbreak.pdf", REJECTED_UNSAFE_CHARACTER),
            ("tab\there.pdf", REJECTED_UNSAFE_CHARACTER),
            ("delete\x7f.pdf", REJECTED_UNSAFE_CHARACTER),
            ("c1\x9b2K.pdf", REJECTED_UNSAFE_CHARACTER),
            ("invoice\u202efdp.exe", REJECTED_UNSAFE_CHARACTER),
            ("line\u2028separator.pdf", REJECTED_UNSAFE_CHARACTER),
            ("undecodable \udce9.pdf", REJECTED_UNSAFE_CHARACTER),
            ("zero\u200bwidth space.pdf", REJECTED_UNSAFE_CHARACTER),
            ("\ufeffbyte order mark.pdf", REJECTED_UNSAFE_CHARACTER),
            ("word\u2060joiner.pdf", REJECTED_UNSAFE_CHARACTER),
            ("left\u200eto right mark.pdf", REJECTED_UNSAFE_CHARACTER),
            ("right\u200fto left mark.pdf", REJECTED_UNSAFE_CHARACTER),
            ("arabic\u061cletter mark.pdf", REJECTED_UNSAFE_CHARACTER),
            (".hidden.pdf", REJECTED_HIDDEN_NAME),
        ],
        ids=[
            "empty",
            "blank",
            "leading-whitespace",
            "trailing-whitespace",
            "current-dir",
            "parent-dir",
            "parent",
            "subdirectory",
            "absolute",
            "trailing-slash",
            "nul",
            "escape",
            "newline",
            "tab",
            "delete",
            "c1",
            "rlo",
            "line-separator",
            "lone-surrogate",
            "zero-width-space",
            "byte-order-mark",
            "word-joiner",
            "left-to-right-mark",
            "right-to-left-mark",
            "arabic-letter-mark",
            "hidden",
        ],
    )
    @pytest.mark.parametrize("path_flavor", [PurePosixPath, PureWindowsPath], ids=["posix", "windows"])
    def test_names_rejected_on_every_os(self, output_filename: str, reason: str, path_flavor: type[PurePath]) -> None:
        assert _target_name_problem(output_filename, SOURCE, path_flavor) == reason

    @pytest.mark.parametrize(
        "output_filename",
        [
            "sub\\escape.pdf",
            "C:\\absolute.pdf",
            "C:drive-relative.pdf",
            "C:",
            "C: Foo.pdf",
            "X: The Unknown.pdf",
            "\\\\server\\share\\x.pdf",
        ],
        ids=["subdirectory", "absolute", "drive-relative", "bare-drive", "drive-then-space", "colon-title", "unc"],
    )
    def test_names_that_are_paths_on_windows_only(self, output_filename: str) -> None:
        """Path semantics follow the running OS: these are plain (if odd) names on POSIX."""
        assert _target_name_problem(output_filename, SOURCE, PureWindowsPath) == REJECTED_PATH
        assert _target_name_problem(output_filename, SOURCE, PurePosixPath) is None

    @pytest.mark.parametrize(
        "output_filename",
        ["CON", "nul.txt", "COM1.pdf", "lpt9", "trailing dot.", "question?.pdf", "pipe|.pdf", 'quote".pdf'],
        ids=["con", "nul-with-extension", "com1", "lpt9-lowercase", "trailing-dot", "question", "pipe", "quote"],
    )
    def test_names_windows_reserves_or_forbids(self, output_filename: str) -> None:
        """ntpath.isreserved() decides on Windows; on POSIX these are ordinary names."""
        assert _target_name_problem(output_filename, SOURCE, PureWindowsPath) == REJECTED_RESERVED_NAME
        assert _target_name_problem(output_filename, SOURCE, PurePosixPath) is None

    @pytest.mark.parametrize(
        "output_filename",
        [
            "New Name.pdf",
            "[van Author] Some Title.pdf",
            "with (parens), commas & dots.v2.pdf",
            "CONTRACT.pdf",  # a reserved name only as the whole stem
            "soft\u00adhyphen.pdf",  # format characters legitimate in titles are kept
            "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645.pdf",  # Persian, with a zero-width non-joiner
        ],
        ids=["plain", "brackets", "punctuation", "reserved-prefix", "soft-hyphen", "zwnj"],
    )
    @pytest.mark.parametrize("path_flavor", [PurePosixPath, PureWindowsPath], ids=["posix", "windows"])
    def test_plain_names_are_allowed_on_every_os(self, output_filename: str, path_flavor: type[PurePath]) -> None:
        assert _target_name_problem(output_filename, SOURCE, path_flavor) is None

    def test_a_hidden_file_may_keep_its_leading_dot(self) -> None:
        assert _target_name_problem(".hidden renamed.pdf", Path("/docs/.hidden.pdf")) is None


class TestResolveRenames:
    """Tests for validating rename suggestions against the input files."""

    def test_plans_each_rename_beside_its_source(self, resolve, make_rename_op: Callable[..., RenameOp]) -> None:
        paths = (Path("/docs/old_file.pdf"), Path("/data/another.pdf"))
        ops = [make_rename_op(paths[0].name, NEW_NAME), make_rename_op(paths[1].name, OTHER_NEW_NAME)]

        resolved = resolve(*ops, paths=paths)

        assert resolved.source_target_pairs == [
            (paths[0], paths[0].parent / NEW_NAME),
            (paths[1], paths[1].parent / OTHER_NEW_NAME),
        ]

    def test_a_rejected_name_is_reported_with_its_reason(
        self, resolve, make_rename_op: Callable[..., RenameOp]
    ) -> None:
        op = make_rename_op(SOURCE.name, "../escape.pdf")

        resolved = resolve(op)

        assert resolved.planned == []
        assert resolved.rejected == [RejectedRename(op=op, reason=REJECTED_PATH)]

    def test_suggestions_are_matched_by_name_key(self, resolve, make_rename_op: Callable[..., RenameOp]) -> None:
        """An echo in another case or normalization still names its file."""
        op = make_rename_op(SOURCE.name.upper(), NEW_NAME)

        assert resolve(op).source_target_pairs == [(SOURCE, SOURCE.parent / NEW_NAME)]

    def test_suggestions_for_unknown_files_are_rejected(self, resolve, make_rename_op: Callable[..., RenameOp]) -> None:
        op = make_rename_op("not-an-input.pdf", NEW_NAME)

        resolved = resolve(op)

        assert resolved.planned == []
        assert resolved.rejected == [RejectedRename(op=op, reason=REJECTED_UNKNOWN_INPUT)]

    def test_an_unchanged_name_is_dropped_before_validation(
        self, resolve, make_rename_op: Callable[..., RenameOp]
    ) -> None:
        """Regression: keeping a current name that the rules reject (here, trailing whitespace) was reported."""
        source = Path("/docs/odd name.pdf ")
        op = make_rename_op(source.name, source.name)

        assert resolve(op, paths=(source,)) == ResolvedRenames()

    @pytest.mark.parametrize("other_new_name", [NEW_NAME, NEW_NAME.upper()], ids=["same", "differing-in-case"])
    def test_two_suggestions_for_the_same_new_name_are_both_rejected(
        self, resolve, make_rename_op: Callable[..., RenameOp], other_new_name: str
    ) -> None:
        """Which file would win depended on the order the renames were applied in."""
        ops = [make_rename_op(SOURCE.name, NEW_NAME), make_rename_op(OTHER.name, other_new_name)]

        resolved = resolve(*ops)

        assert resolved.planned == []
        assert resolved.rejected == [RejectedRename(op=op, reason=REJECTED_DUPLICATE_TARGET) for op in ops]

    @pytest.mark.parametrize("arrangement", ["swap", "chain", "other-case"])
    def test_a_new_name_taken_by_another_input_is_rejected(
        self, resolve, make_rename_op: Callable[..., RenameOp], arrangement: str
    ) -> None:
        """Swaps and chains would depend on the order renames are applied in, so they are not attempted."""
        into_other = make_rename_op(SOURCE.name, OTHER.name.upper() if arrangement == "other-case" else OTHER.name)
        other_op = make_rename_op(OTHER.name, SOURCE.name if arrangement == "swap" else NEW_NAME)

        resolved = resolve(into_other, other_op)

        assert RejectedRename(op=into_other, reason=REJECTED_TARGET_IS_INPUT) in resolved.rejected
        assert all(rename.target not in {SOURCE, OTHER} for rename in resolved.planned)

    def test_a_case_only_rename_is_planned(self, resolve, make_rename_op: Callable[..., RenameOp]) -> None:
        op = make_rename_op(SOURCE.name, SOURCE.name.upper())

        assert resolve(op, paths=(SOURCE,)).planned == [
            PlannedRename(op=op, source=SOURCE, target=SOURCE.with_name(op.output_filename))
        ]

    def test_an_input_sharing_the_targets_key_is_another_input_even_if_the_source_does_too(self) -> None:
        """Regression: the source's own key exempted the target, so on a case-sensitive volume a rename
        Foo.pdf -> foo.pdf passed as a case-only rename while foo.pdf was another input.

        build_input_mapping() keeps such inputs from meeting here (they share a name key), so the
        rule is checked directly; matching by file id lets them meet.
        """
        key = "/docs/foo.pdf"

        assert _collision_problem(key, key, Counter({key: 1}), Counter({key: 2})) == REJECTED_TARGET_IS_INPUT
        assert _collision_problem(key, key, Counter({key: 1}), Counter({key: 1})) is None

    @pytest.mark.parametrize(
        "spell_directory",
        [lambda directory: directory, lambda directory: directory.with_name(directory.name.swapcase())],
        ids=["absolute", "directory-in-another-case"],
    )
    def test_collisions_compare_entries_not_spellings(
        self,
        resolve,
        make_rename_op: Callable[..., RenameOp],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        spell_directory: Callable[[Path], Path],
    ) -> None:
        """A relative input, and another input in the same directory spelled differently: the target is taken."""
        monkeypatch.chdir(tmp_path)
        relative, other = Path(SOURCE.name), spell_directory(tmp_path) / OTHER.name
        op = make_rename_op(relative.name, other.name)

        assert resolve(op, paths=(relative, other)).rejected == [RejectedRename(op=op, reason=REJECTED_TARGET_IS_INPUT)]

    def test_resolved_renames_are_hashable_values(self, make_rename_op: Callable[..., RenameOp]) -> None:
        """PlannedRename and RejectedRename are frozen dataclasses, so the suggestion they hold is immutable too."""
        op = make_rename_op(SOURCE.name, NEW_NAME)
        planned = PlannedRename(op=op, source=SOURCE, target=SOURCE.parent / NEW_NAME)
        rejected = RejectedRename(op=op, reason=REJECTED_UNKNOWN_INPUT)

        assert len({planned, planned, rejected, rejected}) == 2
        with pytest.raises(ValidationError):
            op.output_filename = OTHER.name  # type: ignore[misc]


class TestApplyRenames:
    def test_renames_files(self, mock_llm, tmp_path: Path) -> None:
        sources = [tmp_path / "file1.pdf", tmp_path / "file2.pdf"]
        targets = [tmp_path / NEW_NAME, tmp_path / OTHER_NEW_NAME]
        for source in sources:
            source.touch()

        result = RenameProcessor(llm=mock_llm).apply_renames(list(zip(sources, targets, strict=True)))

        assert result.all_succeeded
        assert sorted(tmp_path.iterdir()) == sorted(targets)

    def test_a_missing_source_is_reported_and_the_rest_continue(self, mock_llm, tmp_path: Path) -> None:
        missing, present = tmp_path / "missing.pdf", tmp_path / "present.pdf"
        present.touch()

        result = RenameProcessor(llm=mock_llm).apply_renames(
            [(missing, tmp_path / NEW_NAME), (present, tmp_path / OTHER_NEW_NAME)]
        )

        assert result.failure_count == 1
        assert "Source file not found" in result.failed[0].error
        assert result.success_count == 1
        assert (tmp_path / OTHER_NEW_NAME).exists()

    def test_an_existing_target_is_reported(self, mock_llm, tmp_path: Path) -> None:
        source, target = tmp_path / "source.pdf", tmp_path / NEW_NAME
        source.touch()
        target.write_bytes(PRECIOUS_CONTENT)

        result = RenameProcessor(llm=mock_llm).apply_renames([(source, target)])

        assert result.failure_count == 1
        error = result.failed[0].error
        assert "Target file already exists" in error and str(source) in error and str(target) in error
        assert target.read_bytes() == PRECIOUS_CONTENT

    @pytest.mark.parametrize("hard_linked", [False, True], ids=["single-link", "hard-linked"])
    def test_case_only_rename(self, mock_llm, tmp_path: Path, hard_linked: bool) -> None:
        """Regression: on a case-insensitive file system the target "exists" (it is the source), which failed.

        A file with other hard links is case-renamed too: the directory listing, not the
        link count, tells its own entry from another.
        """
        source = tmp_path / "Case.pdf"
        source.touch()
        if hard_linked:
            os.link(source, tmp_path / "elsewhere.pdf")
        target = tmp_path / "case.pdf"
        if not target.exists():
            pytest.skip("case-sensitive file system: a case-only rename is an ordinary rename here")

        result = RenameProcessor(llm=mock_llm).apply_renames([(source, target)])

        assert result.all_succeeded
        assert target.name in os.listdir(tmp_path)
        assert source.name not in os.listdir(tmp_path)

    def test_a_symlink_is_not_renamed_onto_the_file_it_points_to(self, mock_llm, tmp_path: Path) -> None:
        """Regression (data loss): the case-only exemption followed the symlink and let it overwrite its target."""
        real = tmp_path / "real.pdf"
        real.write_bytes(PRECIOUS_CONTENT)
        link = tmp_path / "link.pdf"
        link.symlink_to(real)

        result = RenameProcessor(llm=mock_llm).apply_renames([(link, real)])

        assert result.failure_count == 1
        assert not real.is_symlink()
        assert real.read_bytes() == PRECIOUS_CONTENT
        assert link.is_symlink()

    def test_a_hard_link_is_not_renamed_onto_another_name_of_its_file(self, mock_llm, tmp_path: Path) -> None:
        source = tmp_path / "source.pdf"
        source.write_bytes(PRECIOUS_CONTENT)
        other_name = tmp_path / "other name.pdf"
        os.link(source, other_name)

        result = RenameProcessor(llm=mock_llm).apply_renames([(source, other_name)])

        assert result.failure_count == 1
        assert source.exists()
        assert other_name.read_bytes() == PRECIOUS_CONTENT

    def test_a_dangling_symlink_at_the_target_is_a_collision(self, mock_llm, tmp_path: Path) -> None:
        source = tmp_path / "source.pdf"
        source.touch()
        dangling = tmp_path / "dangling.pdf"
        dangling.symlink_to(tmp_path / "missing.pdf")

        result = RenameProcessor(llm=mock_llm).apply_renames([(source, dangling)])

        assert result.failure_count == 1
        assert dangling.is_symlink()

    def test_an_unexpected_error_on_one_file_does_not_stop_the_others(
        self, mock_llm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failing, fine = tmp_path / "failing.pdf", tmp_path / "fine.pdf"
        failing.touch()
        fine.touch()
        original_rename = Path.rename

        def rename(path: Path, target: Path) -> Path:
            if path == failing:
                raise RuntimeError("unexpected")
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", rename)

        result = RenameProcessor(llm=mock_llm).apply_renames(
            [(failing, failing.with_name(NEW_NAME)), (fine, fine.with_name(OTHER_NEW_NAME))]
        )

        assert result.failure_count == 1
        assert result.success_count == 1
        assert "unexpected" in result.failed[0].error


class TestIntegration:
    def test_full_workflow(self, mock_llm, structured_llm, make_rename_op, tmp_path):
        sources = [tmp_path / path.name for path in SAMPLE_PATHS]
        for source in sources:
            source.touch()
        structured_llm.invoke.return_value = RenameResult(
            operations=[make_rename_op(sources[0].name, NEW_NAME), make_rename_op(sources[1].name, OTHER_NEW_NAME)]
        )
        processor = RenameProcessor(llm=mock_llm)
        inputs = _inputs(*sources)
        usage = TokenUsage()

        result = processor.generate_renames(inputs, QUERY, usage=usage)
        processor.apply_renames(processor.resolve_renames(result.operations, inputs).source_target_pairs)

        assert sorted(tmp_path.iterdir()) == sorted([tmp_path / NEW_NAME, tmp_path / OTHER_NEW_NAME])
        assert usage.llm_calls == 1
