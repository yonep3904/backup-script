import fnmatch
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

# backup.py uses a package-relative import. Give the standalone repository a
# package name so the tests work with plain `python -m unittest`.
PACKAGE_NAME = "backup_script_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(Path(__file__).parent)]
sys.modules.setdefault(PACKAGE_NAME, package)
backup = importlib.import_module(f"{PACKAGE_NAME}.backup")

ExcludeMatcher = backup.ExcludeMatcher


class ExcludeMatcherTest(unittest.TestCase):
    def matcher(self, patterns: set[str], *, source: Path | None = None):
        return ExcludeMatcher(patterns, source=source or Path.cwd())

    def test_basename_and_path_patterns_have_fnmatch_semantics(self) -> None:
        patterns = (
            "[^].txt",
            "[^a].txt",
            "[!a].txt",
            "[^^].txt",
            "[a-z].txt",
            "[0-9].txt",
            "[z-a].txt",
            "[a--].txt",
            "[abc.txt",
        )
        names = (
            "^.txt",
            "a.txt",
            "b.txt",
            "z.txt",
            "0.txt",
            "7.txt",
            "-.txt",
            "[abc.txt",
        )

        for pattern in patterns:
            with self.subTest(pattern=pattern):
                basename_matcher = self.matcher({pattern})
                path_matcher = self.matcher({f"foo/{pattern}"})

                for name in names:
                    expected = fnmatch.fnmatchcase(name, pattern)
                    with self.subTest(name=name):
                        self.assertEqual(
                            basename_matcher.matches_file(f"elsewhere/{name}"),
                            expected,
                        )
                        self.assertEqual(
                            path_matcher.matches_file(f"foo/{name}"),
                            expected,
                        )

    def test_path_globs_do_not_cross_separator_boundaries(self) -> None:
        one_level = self.matcher({"foo/*.txt"})
        two_levels = self.matcher({"foo/*/*.txt"})

        self.assertTrue(one_level.matches_file("foo/file.txt"))
        self.assertFalse(one_level.matches_file("foo/bar/file.txt"))
        self.assertFalse(two_levels.matches_file("foo/file.txt"))
        self.assertTrue(two_levels.matches_file("foo/bar/file.txt"))
        self.assertFalse(two_levels.matches_file("foo/bar/baz/file.txt"))

    def test_invalid_patterns_are_rejected(self) -> None:
        patterns = (
            "/foo",
            "foo//bar",
            "foo/./bar",
            "foo/../bar",
            r"foo\bar",
            "foo/**/bar",
            "**.txt",
            "foo**bar",
        )

        for pattern in patterns:
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    self.matcher({pattern})

    def test_parent_components_in_candidates_are_rejected(self) -> None:
        matcher = self.matcher({"*.txt", "foo/*.txt"})

        for relative in ("../file.txt", "foo/../file.txt", Path("foo/../file.txt")):
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    matcher.matches_file(relative)
                with self.assertRaises(ValueError):
                    matcher.matches_dir(relative)

    def test_marker_matches_target_beside_marker_only(self) -> None:
        with self.subTest(marker="Cargo.toml"):
            with tempfile.TemporaryDirectory() as temp:
                source = Path(temp)
                (source / "crate").mkdir()
                (source / "crate" / "Cargo.toml").touch()
                matcher = self.matcher(
                    {"Cargo.toml :: target/"},
                    source=source,
                )

                self.assertTrue(matcher.matches_dir("crate/target"))
                self.assertFalse(matcher.matches_dir("other/target"))
                self.assertFalse(matcher.matches_file("crate/target"))

        with self.subTest(marker="pyproject.toml"):
            with tempfile.TemporaryDirectory() as temp:
                source = Path(temp)
                (source / "pyproject.toml").touch()
                matcher = self.matcher(
                    {"pyproject.toml :: .venv/"},
                    source=source,
                )

                self.assertTrue(matcher.matches_dir(".venv"))
                self.assertFalse(matcher.matches_dir("nested/.venv"))

    def test_marker_target_supports_path_globs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "projects" / "app").mkdir(parents=True)
            (source / "projects" / "app" / "pyproject.toml").touch()
            matcher = self.matcher(
                {"pyproject.toml :: projects/*/.venv/"},
                source=source,
            )

            self.assertTrue(matcher.matches_dir("projects/app/.venv"))
            self.assertFalse(matcher.matches_dir("projects/other/.venv"))

    def test_all_markers_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            matcher = self.matcher(
                {"pyproject.toml + uv.lock :: .venv/"},
                source=source,
            )

            self.assertFalse(matcher.matches_dir(".venv"))

            (source / "pyproject.toml").touch()
            self.assertFalse(matcher.matches_dir(".venv"))

            (source / "uv.lock").touch()
            self.assertTrue(matcher.matches_dir(".venv"))

    def test_patterns_without_markers_use_an_empty_tuple(self) -> None:
        matcher = self.matcher({"file.txt", "cache/"})

        self.assertEqual(matcher.exact_files, (("file.txt", ()),))
        self.assertEqual(matcher.exact_dirs, (("cache", ()),))

    def test_marker_must_be_an_exact_basename(self) -> None:
        patterns = (
            "*.toml :: target/",
            "config/Cargo.toml :: target/",
            r"config\Cargo.toml :: target/",
            ". :: target/",
            "Cargo.toml ::",
            "Cargo.toml :: marker :: target/",
            "+ uv.lock :: .venv/",
            "pyproject.toml + :: .venv/",
            "pyproject.toml ++ uv.lock :: .venv/",
        )

        for pattern in patterns:
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    self.matcher({pattern})


if __name__ == "__main__":
    unittest.main()
