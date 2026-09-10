#!/usr/bin/env python3

import argparse
import fnmatch
import os
import re
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

from excluded_patterns import EXCLUDED_PATTERNS


class ExcludeMatcher:
    """Compiled representation of exclusion patterns.

    Pattern syntax:

    - ``name``:
      Match a file basename at any depth.

    - ``name/``:
      Match a directory basename at any depth.

    - ``path/name``:
      Match a file path relative to the backup source.

    - ``path/name/``:
      Match a directory path relative to the backup source.

    - ``*``, ``?``, and ``[...]``:
      Glob metacharacters are supported. In path patterns, glob
      metacharacters never match the path separator ``/``.

    - ``target :: marker [+ marker ...]``:
      Match ``target`` only when its parent directory contains entries for
      all specified markers. Each marker must be an exact basename.

    Path patterns always use ``/`` as the separator, regardless of the
    operating system.

    Leading ``/``, empty path components, ``.`` components, ``..``
    components, and backslashes are not allowed.

    Exclusions are monotonic: once a directory is excluded, none of its
    descendants can be included again.
    """

    patterns: frozenset[str]
    source: Path

    exact_files: tuple[tuple[str, tuple[str, ...]], ...]
    exact_dirs: tuple[tuple[str, tuple[str, ...]], ...]
    glob_files: tuple[tuple[str, tuple[str, ...]], ...]
    glob_dirs: tuple[tuple[str, tuple[str, ...]], ...]

    exact_file_paths: tuple[tuple[str, tuple[str, ...]], ...]
    exact_dir_paths: tuple[tuple[str, tuple[str, ...]], ...]
    glob_file_paths: tuple[
        tuple[tuple[re.Pattern[str], ...], tuple[str, ...]],
        ...,
    ]
    glob_dir_paths: tuple[
        tuple[tuple[re.Pattern[str], ...], tuple[str, ...]],
        ...,
    ]

    def __init__(
        self,
        patterns: set[str],
        source: Path,
    ) -> None:
        if not source.is_absolute():
            raise ValueError(f"Source must be an absolute path: {source}")

        self.patterns = frozenset(patterns)
        self.source = source

        exact_files: list[tuple[str, tuple[str, ...]]] = []
        exact_dirs: list[tuple[str, tuple[str, ...]]] = []
        glob_files: list[tuple[str, tuple[str, ...]]] = []
        glob_dirs: list[tuple[str, tuple[str, ...]]] = []

        exact_file_paths: list[tuple[str, tuple[str, ...]]] = []
        exact_dir_paths: list[tuple[str, tuple[str, ...]]] = []
        glob_file_paths: list[tuple[tuple[re.Pattern[str], ...], tuple[str, ...]]] = []
        glob_dir_paths: list[tuple[tuple[re.Pattern[str], ...], tuple[str, ...]]] = []

        for raw_pattern in sorted(patterns):
            markers, raw_target = self._parse_marker_pattern(raw_pattern)

            (
                pattern,
                is_dir,
                is_path,
                is_glob,
            ) = self._parse_pattern(raw_target)

            if is_path:
                if is_glob:
                    compiled = tuple(
                        re.compile(fnmatch.translate(component))
                        for component in pattern.split("/")
                    )
                    target = glob_dir_paths if is_dir else glob_file_paths
                    target.append((compiled, markers))
                else:
                    target = exact_dir_paths if is_dir else exact_file_paths
                    target.append((pattern, markers))
            else:
                if is_glob:
                    target = glob_dirs if is_dir else glob_files
                    target.append((pattern, markers))
                else:
                    target = exact_dirs if is_dir else exact_files
                    target.append((pattern, markers))

        self.exact_files = tuple(exact_files)
        self.exact_dirs = tuple(exact_dirs)
        self.glob_files = tuple(glob_files)
        self.glob_dirs = tuple(glob_dirs)

        self.exact_file_paths = tuple(exact_file_paths)
        self.exact_dir_paths = tuple(exact_dir_paths)
        self.glob_file_paths = tuple(glob_file_paths)
        self.glob_dir_paths = tuple(glob_dir_paths)

    def matches_file(self, relative: Path | str) -> bool:
        relative_path, relative_string, name = self._normalize_candidate(relative)

        if self._matches_exact(name, relative_path, self.exact_files):
            return True

        if self._matches_exact(
            relative_string,
            relative_path,
            self.exact_file_paths,
        ):
            return True

        if self._matches_glob(name, relative_path, self.glob_files):
            return True

        return self._matches_path_glob(
            relative_string,
            relative_path,
            self.glob_file_paths,
        )

    def matches_dir(self, relative: Path | str) -> bool:
        relative_path, relative_string, name = self._normalize_candidate(relative)

        if self._matches_exact(name, relative_path, self.exact_dirs):
            return True

        if self._matches_exact(
            relative_string,
            relative_path,
            self.exact_dir_paths,
        ):
            return True

        if self._matches_glob(name, relative_path, self.glob_dirs):
            return True

        return self._matches_path_glob(
            relative_string,
            relative_path,
            self.glob_dir_paths,
        )

    def _matches_exact(
        self,
        candidate: str,
        relative: Path,
        patterns: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> bool:
        return any(
            candidate == pattern and self._matches_marker(relative, markers)
            for pattern, markers in patterns
        )

    def _matches_glob(
        self,
        candidate: str,
        relative: Path,
        patterns: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> bool:
        return any(
            fnmatch.fnmatchcase(candidate, pattern)
            and self._matches_marker(relative, markers)
            for pattern, markers in patterns
        )

    def _matches_path_glob(
        self,
        relative: str,
        relative_path: Path,
        patterns: tuple[
            tuple[tuple[re.Pattern[str], ...], tuple[str, ...]],
            ...,
        ],
    ) -> bool:
        components = relative.split("/")

        return any(
            len(pattern) == len(components)
            and all(
                component_pattern.fullmatch(component)
                for component_pattern, component in zip(pattern, components)
            )
            and self._matches_marker(relative_path, markers)
            for pattern, markers in patterns
        )

    def _matches_marker(self, relative: Path, markers: tuple[str, ...]) -> bool:
        parent = self.source / relative.parent
        return all((parent / marker).exists() for marker in markers)

    @staticmethod
    def _normalize_candidate(
        relative: Path | str,
    ) -> tuple[Path, str, str]:
        path = Path(relative)

        if path.is_absolute():
            raise ValueError(f"Expected a relative path, got: {relative!r}")

        relative_string = path.as_posix()

        components = relative_string.split("/")

        if any(component in ("", ".", "..") for component in components):
            raise ValueError(
                "Relative paths must not contain empty, '.', or '..' "
                f"components: {relative!r}"
            )

        return path, relative_string, path.name

    @classmethod
    def _parse_pattern(
        cls,
        raw_pattern: str,
    ) -> tuple[str, bool, bool, bool]:
        if not raw_pattern:
            raise ValueError("Empty exclusion pattern")

        if "\\" in raw_pattern:
            raise ValueError(
                f"Path separators in exclusion patterns must be '/': {raw_pattern!r}"
            )

        if "**" in raw_pattern:
            raise ValueError(f"Recursive glob '**' is not supported: {raw_pattern!r}")

        is_dir = raw_pattern.endswith("/")
        pattern = raw_pattern[:-1] if is_dir else raw_pattern

        if not pattern:
            raise ValueError(f"Invalid exclusion pattern: {raw_pattern!r}")

        if pattern.startswith("/"):
            raise ValueError(
                f"Exclusion patterns must not start with '/': {raw_pattern!r}"
            )

        components = pattern.split("/")

        if any(component in ("", ".", "..") for component in components):
            raise ValueError(
                "Exclusion patterns must not contain empty, '.', or '..' "
                f"path components: {raw_pattern!r}"
            )

        is_path = len(components) > 1
        is_glob = cls._has_glob(pattern)

        return pattern, is_dir, is_path, is_glob

    @classmethod
    def _parse_marker_pattern(cls, raw_pattern: str) -> tuple[tuple[str, ...], str]:
        if "::" not in raw_pattern:
            return (), raw_pattern

        parts = raw_pattern.split("::")

        if len(parts) != 2:
            raise ValueError(f"Invalid marker exclusion pattern: {raw_pattern!r}")

        target, raw_markers = (part.strip() for part in parts)
        markers = tuple(marker.strip() for marker in raw_markers.split("+"))

        for marker in markers:
            if (
                not marker
                or marker in (".", "..")
                or "/" in marker
                or "\\" in marker
                or cls._has_glob(marker)
            ):
                raise ValueError(f"Marker must be an exact basename: {marker!r}")

        if not target:
            raise ValueError(f"Marker target must not be empty: {raw_pattern!r}")

        return markers, target

    @staticmethod
    def _has_glob(pattern: str) -> bool:
        """Return True if pattern contains supported glob metacharacters."""
        return any(char in pattern for char in "*?[")


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
            relative = path.relative_to(source)

            if path in ignored_paths:
                continue

            if exclude_matcher.matches_dir(relative):
                continue

            # os.walk puts symlinks to directories in dirs even when
            # followlinks=False. Emit the link itself, but do not allow
            # os.walk to descend through it.
            if path.is_symlink():
                yield ArchiveEntry(
                    path=path,
                    relative=relative,
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
            relative = path.relative_to(source)

            if path in ignored_paths:
                continue

            if exclude_matcher.matches_file(relative):
                continue

            # is_symlink() does not require the target to exist, so broken
            # symbolic links are preserved as entries as well.
            kind = "symlink" if path.is_symlink() else "file"

            yield ArchiveEntry(
                path=path,
                relative=relative,
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

    exclude_matcher = ExcludeMatcher(EXCLUDED_PATTERNS, source=source)

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
