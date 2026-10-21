import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

from parameterized import parameterized

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCRIPT_PATH = os.path.join(REPO_ROOT, "ci", "update_citation.py")
CITATION_PATH = os.path.join(REPO_ROOT, "CITATION.cff")

# the script is a standalone CI helper outside the package, so load it by path
_SPEC = importlib.util.spec_from_file_location("update_citation", SCRIPT_PATH)
update_citation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(update_citation)

# a top-level version line plus two decoys that must stay untouched: a
# version-like substring inside a quoted scalar and a nested version field
CFF_TEXT = (
    "cff-version: 1.2.0\n"
    'message: "If you use ritme, cite it.\\nversion: 0.0.0"\n'
    'title: "ritme"\n'
    "# set on release\n"
    "version: 1.2.8\n"
    "preferred-citation:\n"
    "  type: article\n"
    "  version: 9.9.9\n"
    "  year: 2025\n"
)
CFF_TEXT_UPDATED = CFF_TEXT.replace("version: 1.2.8", "version: 1.4.5")


class TestVersionFromTag(unittest.TestCase):
    @parameterized.expand(
        [
            ("v1.4.5", "1.4.5"),
            ("1.4.5", "1.4.5"),
            ("v10.0.12", "10.0.12"),
        ]
    )
    def test_returns_version_without_tag_prefix(self, tag, expected):
        self.assertEqual(update_citation.version_from_tag(tag), expected)

    @parameterized.expand(
        [
            ("v1.4",),
            ("v1.4.5rc1",),
            ("v1.4.5.1",),
            ("vv1.4.5",),
            ("v1.4.5\n",),
            ("release-1",),
            ("v",),
            ("",),
        ]
    )
    def test_rejects_tag_not_of_form_vx_y_z(self, tag):
        with self.assertRaisesRegex(ValueError, "X.Y.Z"):
            update_citation.version_from_tag(tag)


class TestSetVersion(unittest.TestCase):
    def test_replaces_only_top_level_version_line(self):
        self.assertEqual(
            update_citation.set_version(CFF_TEXT, "1.4.5"), CFF_TEXT_UPDATED
        )

    def test_is_idempotent(self):
        self.assertEqual(
            update_citation.set_version(CFF_TEXT_UPDATED, "1.4.5"), CFF_TEXT_UPDATED
        )

    def test_raises_without_top_level_version_field(self):
        text = CFF_TEXT.replace("version: 1.2.8\n", "")
        with self.assertRaisesRegex(ValueError, "version"):
            update_citation.set_version(text, "1.4.5")

    def test_repo_citation_file_has_top_level_version_field(self):
        with open(CITATION_PATH) as f:
            text = f.read()
        self.assertIn("\nversion: 0.0.0\n", update_citation.set_version(text, "0.0.0"))


class TestCli(unittest.TestCase):
    def setUp(self):
        super().setUp()
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.cff_path = os.path.join(tmp_dir.name, "CITATION.cff")
        with open(self.cff_path, "w") as f:
            f.write(CFF_TEXT)

    def _run_script(self, tag):
        return subprocess.run(
            [sys.executable, SCRIPT_PATH, self.cff_path, tag],
            capture_output=True,
            text=True,
        )

    def _read_cff(self):
        with open(self.cff_path) as f:
            return f.read()

    def test_rewrites_version_from_release_tag(self):
        result = self._run_script("v1.4.5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._read_cff(), CFF_TEXT_UPDATED)

    def test_leaves_file_untouched_on_malformed_tag(self):
        result = self._run_script("v1.4")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("X.Y.Z", result.stderr)
        self.assertEqual(self._read_cff(), CFF_TEXT)
