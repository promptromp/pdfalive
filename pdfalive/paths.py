"""Comparing file paths by the directory entries they name, not by how they are spelled.

"./a.pdf", "/abs/a.pdf", "dir/../a.pdf" and, on a case-insensitive file
system, "A.pdf" can all name one entry, while a symlink or a hard link is a
separate entry for the same file. Renaming must tell these apart: to list a
file given twice once, to detect collisions, and never to rename one entry
over another.
"""

import os
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Final


# Composed and decomposed accents ("é" as one code point or two) are the same name to
# APFS and to people, so names are compared in one normalization form.
_NAME_NORMALIZATION_FORM: Final = "NFC"


def name_key(name: str) -> str:
    """The key two names are compared by: the name in NFC, casefolded.

    Names whose keys agree are treated as one name: by input deduplication,
    collision checks, display grouping and suggestion matching. That is right
    on case- and normalization-insensitive file systems (APFS, NTFS) and on
    case-sensitive ones at worst a needless rejection.
    """
    return unicodedata.normalize(_NAME_NORMALIZATION_FORM, name).casefold()


def entry_key(path: Path) -> str:
    """The directory entry a path names, for comparing spellings: its resolved directory and name, by name_key().

    Resolving makes "./a.pdf", "/abs/a.pdf" and "dir/../a.pdf" agree, and also
    follows a symlinked directory; the final component is not resolved, so a
    symlinked file is its own entry. The whole path goes through name_key(),
    since resolve() keeps a directory's spelled case.
    """
    return name_key(str(path.parent.resolve() / path.name))


# Identifies an input: the file (device and inode, from os.lstat) and its directory entry.
type _InputIdentity = tuple[int, int, str]


def _input_identity(path: Path) -> _InputIdentity | str:
    """What makes two input paths the same input: the same file, at the same directory entry.

    The file comes from os.lstat, which does not follow a final symlink, so a
    symlink is a separate input from the file it points to. Requiring the same
    entry too keeps a hard link (another name for the file) a separate input,
    and keeps Foo.pdf and foo.pdf apart on a case-sensitive volume, where they
    are different files. A path that cannot be stat'ed is identified by its
    entry alone.
    """
    try:
        status = os.lstat(path)
    except OSError:
        return entry_key(path)
    return status.st_dev, status.st_ino, entry_key(path)


def unique_input_paths(paths: Iterable[Path]) -> list[Path]:
    """The input paths, in order, with a file given more than once (in any spelling) listed once."""
    seen: set[_InputIdentity | str] = set()
    unique: list[Path] = []
    for path in paths:
        identity = _input_identity(path)
        if identity not in seen:
            seen.add(identity)
            unique.append(path)
    return unique


def is_case_only_rename(source: Path, target: Path) -> bool:
    """Whether the existing target is source's own directory entry, reached by a name differing only in case.

    That is a case-only (or normalization-only) rename on a case-insensitive
    file system, where the target already "exists". target is a name in
    source's directory. The names must agree by name_key(), and the target's
    exact name must be missing from the directory listing: a listed name is an
    entry of its own (on a case-sensitive volume, "INPUT.PDF" can be a hard
    link or another file beside "input.pdf"), so renaming onto it would
    replace it or do nothing. os.lstat must then find the source's own entry
    there, never following symlinks, so a symlink is not renamed onto the
    file it points to.
    """
    if name_key(source.name) != name_key(target.name):
        return False
    if target.name in os.listdir(target.parent):
        return False
    return os.path.samestat(os.lstat(source), os.lstat(target))
