"""Tests for comparing paths by the directory entries they name."""

import os
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pytest

from pdfalive.paths import entry_key, is_case_only_rename, name_key, unique_input_paths


# What the source file holds, to tell it from other files.
SOURCE_CONTENT = b"the source"


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A real file, one directory down."""
    path = tmp_path / "folder" / "input.pdf"
    path.parent.mkdir()
    path.write_bytes(SOURCE_CONTENT)
    return path


@pytest.fixture
def linked_folder(source: Path, tmp_path: Path) -> Path:
    """A symlink to the source's directory, giving the source another spelling."""
    link = tmp_path / "linked folder"
    link.symlink_to(source.parent, target_is_directory=True)
    return link


@pytest.fixture
def spellings(source: Path, linked_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The source spelled absolute, relative, through "..", and through a symlinked directory."""
    monkeypatch.chdir(tmp_path)
    return [
        source,
        source.relative_to(tmp_path),
        source.parent / ".." / source.parent.name / source.name,
        linked_folder / source.name,
    ]


class TestEntryKey:
    def test_spellings_of_one_entry_agree(self, spellings: list[Path]) -> None:
        assert len({entry_key(path) for path in spellings}) == 1

    def test_letter_case_is_ignored_in_the_name_and_the_directory(self, source: Path) -> None:
        other_case = source.parent.with_name(source.parent.name.upper()) / source.name.upper()

        assert entry_key(other_case) == entry_key(source)

    def test_a_symlinked_file_is_its_own_entry(self, source: Path) -> None:
        link = source.with_name("link.pdf")
        link.symlink_to(source)

        assert entry_key(link) != entry_key(source)


def _hard_link(source: Path, link: Path) -> Path:
    os.link(source, link)
    return link


def _symlink(link: Path, target: Path) -> Path:
    link.symlink_to(target)
    return link


def _directory(path: Path) -> Path:
    path.mkdir()
    return path


class TestUniqueInputPaths:
    """A file given more than once is one input; distinct directory entries stay distinct."""

    def test_different_spellings_of_one_file_are_one_input(self, source: Path, spellings: list[Path]) -> None:
        assert unique_input_paths(spellings) == [source]

    @pytest.mark.parametrize(
        "make_other",
        [
            lambda source, tmp_path: _hard_link(source, source.with_name("hard link.pdf")),
            lambda source, tmp_path: _symlink(source.with_name("link.pdf"), source),
            lambda source, tmp_path: _hard_link(source, _directory(tmp_path / "elsewhere") / source.name),
        ],
        ids=["hard-link", "symlink-to-it", "hard-link-with-its-name-elsewhere"],
    )
    def test_another_entry_for_the_same_file_is_a_separate_input(
        self, source: Path, tmp_path: Path, make_other: Callable[[Path, Path], Path]
    ) -> None:
        other = make_other(source, tmp_path)

        assert unique_input_paths([source, other]) == [source, other]

    def test_names_differing_in_case(self, source: Path) -> None:
        """One file on a case-insensitive file system, two on a case-sensitive one."""
        other_case = source.with_name(source.name.upper())
        if other_case.exists():
            expected = [source]
        else:
            other_case.touch()
            expected = [source, other_case]

        assert unique_input_paths([source, other_case]) == expected

    def test_a_directory_spelled_in_another_case(self, source: Path) -> None:
        """resolve() keeps a directory's spelled case, so the comparison must casefold it too."""
        other_case = source.parent.with_name(source.parent.name.upper()) / source.name
        expected = [source] if other_case.exists() else [source, other_case]

        assert unique_input_paths([source, other_case]) == expected

    def test_hard_links_with_one_name_in_different_directories_are_separate_inputs(
        self, source: Path, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "elsewhere" / source.name
        elsewhere.parent.mkdir()
        os.link(source, elsewhere)

        assert unique_input_paths([source, elsewhere]) == [source, elsewhere]


class TestIsCaseOnlyRename:
    """Each case runs on one kind of file system and is skipped on the other."""

    @pytest.fixture
    def other_case(self, source: Path) -> Path:
        return source.with_name(source.name.upper())

    @pytest.mark.parametrize("hard_linked", [False, True], ids=["single-link", "hard-linked"])
    def test_the_same_entry_in_another_case(
        self, source: Path, other_case: Path, tmp_path: Path, hard_linked: bool
    ) -> None:
        """The other case's exact name is not listed, so it reaches the source's own entry, hard links or not."""
        if not other_case.exists():
            pytest.skip("case-sensitive file system: the other case is another entry")
        if hard_linked:
            os.link(source, tmp_path / "elsewhere.pdf")

        assert is_case_only_rename(source, other_case)

    def test_the_same_entry_in_another_normalization(self, tmp_path: Path) -> None:
        composed = tmp_path / unicodedata.normalize("NFC", "café.pdf")
        composed.touch()
        decomposed = composed.with_name(unicodedata.normalize("NFD", composed.name))
        if decomposed.name in os.listdir(tmp_path) or not decomposed.exists():
            pytest.skip("normalization-sensitive file system: the other form is another name")

        assert is_case_only_rename(composed, decomposed)

    def test_a_symlink_onto_the_file_it_points_to(self, source: Path, other_case: Path) -> None:
        """Its exact name is listed: an entry of its own, so no exemption."""
        if other_case.exists():
            pytest.skip("case-insensitive file system: no second entry can have the other case")
        other_case.symlink_to(source)

        assert not is_case_only_rename(other_case, source)

    def test_a_hard_link_in_another_case_is_another_entry(self, source: Path, other_case: Path) -> None:
        """One inode at both names, but two listed entries: rename() would do nothing yet report success."""
        if other_case.exists():
            pytest.skip("case-insensitive file system: no second entry can have the other case")
        os.link(source, other_case)

        assert not is_case_only_rename(source, other_case)

    def test_a_different_name(self, source: Path) -> None:
        sibling = source.with_name("sibling.pdf")
        sibling.touch()

        assert not is_case_only_rename(source, sibling)


class TestNameKey:
    @pytest.mark.parametrize(
        "variant",
        [unicodedata.normalize("NFD", "Café.pdf"), "CAFÉ.PDF", unicodedata.normalize("NFD", "café.PDF")],
        ids=["decomposed", "upper-case", "both"],
    )
    def test_case_and_normalization_variants_agree(self, variant: str) -> None:
        assert name_key(variant) == name_key(unicodedata.normalize("NFC", "café.pdf"))

    def test_different_names_differ(self) -> None:
        assert name_key("cafe.pdf") != name_key("café.pdf")
