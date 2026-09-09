#!/usr/bin/env python3

import argparse
import fnmatch
import os
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from .excluded_patterns import EXCLUDED_PATTERNS


class ExcludeMatcher:
    """Compiled representation of exclusion patterns."""

    patterns: frozenset[str]
    exact_files: frozenset[str]
    exact_dirs: frozenset[str]
    glob_files: tuple[str, ...]
    glob_dirs: tuple[str, ...]

    def __init__(
        self,
        patterns: set[str],
    ) -> None:
        self.patterns = frozenset(patterns)
        (
            self.exact_files,
            self.exact_dirs,
            self.glob_files,
            self.glob_dirs,
        ) = self._compile(patterns)

    def matches_file(self, name: str) -> bool:
        if name in self.exact_files:
            return True

        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.glob_files)

    def matches_dir(self, name: str) -> bool:
        if name in self.exact_dirs:
            return True

        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.glob_dirs)

    def _has_glob(self, pattern: str) -> bool:
        """Return True if pattern contains fnmatch metacharacters."""
        return any(char in pattern for char in "*?[")

    def _compile(
        self,
        patterns: set[str],
    ) -> tuple[
        frozenset[str],
        frozenset[str],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        exact_files: set[str] = set()
        exact_dirs: set[str] = set()
        glob_files: list[str] = []
        glob_dirs: list[str] = []

        for raw_pattern in patterns:
            is_dir = raw_pattern.endswith("/")
            pattern = raw_pattern[:-1] if is_dir else raw_pattern

            if not pattern:
                raise ValueError("Empty exclusion pattern")

            if "/" in pattern or "\\" in pattern:
                raise ValueError(
                    f"Exclusion patterns must match basenames only: {raw_pattern!r}"
                )

            if self._has_glob(pattern):
                target = glob_dirs if is_dir else glob_files
                target.append(pattern)
            else:
                target = exact_dirs if is_dir else exact_files
                target.add(pattern)

        # Stable ordering makes debugging and profiling deterministic.
        glob_files.sort()
        glob_dirs.sort()

        return (
            frozenset(exact_files),
            frozenset(exact_dirs),
            tuple(glob_files),
            tuple(glob_dirs),
        )


@dataclass(frozen=True)
class ArchiveEntry:
    path: Path
    relative: Path
    kind: Literal["file", "dir", "symlink"]


def collect_entries(
    source: Path,
    exclude_matcher: ExcludeMatcher,
    *,
    ignored_paths: frozenset[Path] = frozenset(),
) -> Iterator[ArchiveEntry]:
    """Yield files, directories, and symbolic links.

    Symbolic links are never followed. This includes directory symlinks,
    links whose targets are outside the source tree, and broken links.
    """

    for root, dirs, files in os.walk(source, followlinks=False):
        root_path = Path(root)

        # Sorting gives deterministic traversal/archive ordering.
        dirs.sort()
        files.sort()

        kept_dirs: list[str] = []

        for name in dirs:
            path = root_path / name

            if path in ignored_paths:
                continue

            if exclude_matcher.matches_dir(name):
                continue

            # os.walk puts symlinks to directories in dirs even when
            # followlinks=False. Emit the link itself, but do not allow
            # os.walk to descend through it.
            if path.is_symlink():
                yield ArchiveEntry(
                    path=path,
                    relative=path.relative_to(source),
                    kind="symlink",
                )
                continue

            kept_dirs.append(name)

        # Prune before os.walk descends into excluded directories or
        # directory symlinks.
        dirs[:] = kept_dirs

        # Add every retained real directory explicitly so that empty
        # directories survive a backup/restore cycle.
        if root_path != source and root_path not in ignored_paths:
            yield ArchiveEntry(
                path=root_path,
                relative=root_path.relative_to(source),
                kind="dir",
            )

        for name in files:
            path = root_path / name

            if path in ignored_paths:
                continue

            if exclude_matcher.matches_file(name):
                continue

            # is_symlink() does not require the target to exist, so broken
            # symbolic links are preserved as entries as well.
            kind = "symlink" if path.is_symlink() else "file"

            yield ArchiveEntry(
                path=path,
                relative=path.relative_to(source),
                kind=kind,
            )


def create_zip(
    source: Path,
    output: Path,
    exclude_matcher: ExcludeMatcher,
    *,
    ignored_paths: frozenset[Path],
) -> list[ArchiveEntry]:
    """Create a ZIP archive.

    Symbolic links are intentionally omitted. ZIP output is intended
    primarily for transferring regular files/directories to platforms
    where symbolic-link semantics may not be available.

    Returns the symbolic links that were skipped.
    """

    skipped_symlinks: list[ArchiveEntry] = []

    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for entry in collect_entries(
            source,
            exclude_matcher,
            ignored_paths=ignored_paths,
        ):
            if entry.kind == "symlink":
                skipped_symlinks.append(entry)
                print(f"{entry.relative} [symlink: skipped]")
                continue

            print(entry.relative)

            archive_name = entry.relative.as_posix()

            if entry.kind == "dir":
                # A trailing slash represents a directory entry in ZIP.
                archive.write(
                    entry.path,
                    archive_name.rstrip("/") + "/",
                )
            else:
                archive.write(
                    entry.path,
                    archive_name,
                )

    return skipped_symlinks


def create_tar_gz(
    source: Path,
    output: Path,
    exclude_matcher: ExcludeMatcher,
    *,
    ignored_paths: frozenset[Path],
) -> None:
    """Create a tar.gz archive preserving symbolic links.

    Symbolic links are stored as links and are never followed. The target
    does not need to exist or reside inside the source tree.
    """

    with tarfile.open(
        output,
        mode="w:gz",
        dereference=False,
    ) as archive:
        for entry in collect_entries(
            source,
            exclude_matcher,
            ignored_paths=ignored_paths,
        ):
            if entry.kind == "symlink":
                try:
                    target = os.readlink(entry.path)
                except OSError:
                    target = "?"
                print(f"{entry.relative} -> {target} [symlink]")
            else:
                print(entry.relative)

            # recursive=False is important because collect_entries()
            # already performs filtering and explicitly emits directories.
            #
            # dereference=False on the TarFile ensures symbolic links are
            # archived as links rather than as their targets.
            archive.add(
                entry.path,
                arcname=entry.relative.as_posix(),
                recursive=False,
            )


@dataclass(frozen=True)
class ArgsResults:
    source: Path
    output: Path | None
    format: Literal["zip", "tar.gz"]
    force: bool


def parse_args() -> ArgsResults:
    parser = argparse.ArgumentParser(
        description="Create a source backup archive.",
        epilog=(
            "Symbolic links are preserved in tar.gz archives without "
            "following their targets. ZIP archives omit symbolic links."
        ),
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Directory to back up",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output archive path",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=("zip", "tar.gz"),
        default="tar.gz",
        help=(
            "Archive format (default: tar.gz). "
            "tar.gz preserves symbolic links; zip omits them."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output archive",
    )

    args = parser.parse_args()

    return ArgsResults(
        source=args.source,
        output=args.output,
        format=args.format,
        force=args.force,
    )


def main() -> None:
    args = parse_args()

    source = args.source.expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Not a directory: {source}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.output is not None:
        output = args.output.expanduser().resolve()
    else:
        extension = ".zip" if args.format == "zip" else ".tar.gz"
        output = (Path.cwd() / f"{source.name}-{timestamp}{extension}").resolve()

    expected_suffix = ".zip" if args.format == "zip" else ".tar.gz"

    if not output.name.endswith(expected_suffix):
        raise SystemExit(f"Output filename must end with {expected_suffix}: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output}\nUse --force to overwrite.")

    print(f"Source : {source}")
    print(f"Output : {output}")
    print(f"Format : {args.format}")
    print()

    exclude_matcher = ExcludeMatcher(EXCLUDED_PATTERNS)

    # Build the archive in the destination directory and atomically replace
    # the final path only after successful completion.
    #
    # delete=False is intentional: the archive libraries open the path
    # themselves, and we need os.replace() after the temporary file closes.
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(fd)

    tmp_output = Path(temp_name).resolve()

    # Never archive the destination or the temporary archive itself if they
    # happen to live inside the source tree.
    ignored_paths = frozenset(
        {
            output,
            tmp_output,
        }
    )

    skipped_symlinks: list[ArchiveEntry] = []

    try:
        if args.format == "zip":
            skipped_symlinks = create_zip(
                source,
                tmp_output,
                exclude_matcher,
                ignored_paths=ignored_paths,
            )
        else:
            create_tar_gz(
                source,
                tmp_output,
                exclude_matcher,
                ignored_paths=ignored_paths,
            )

        os.replace(tmp_output, output)

    except BaseException:
        try:
            tmp_output.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    print()
    print(f"Created: {output}")

    if skipped_symlinks:
        print()
        print(
            f"Warning: {len(skipped_symlinks)} symbolic link(s) "
            "were not included in the ZIP archive."
        )
        print("Use --format tar.gz to preserve symbolic links.")


if __name__ == "__main__":
    main()
