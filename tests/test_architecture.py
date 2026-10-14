"""ARCHITECTURE.md and the folder layout stay in step.

Fails when a code file is missing from the module map, when the map names a
file that does not exist, when a code folder is missing from paths.CODE_DIRS
or the document, or when Python code appears outside the code folders (for
example back in data/ or at the repo root).
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (puts every code folder on sys.path)

ROOT = paths.ROOT
DOC = os.path.join(ROOT, "ARCHITECTURE.md")
ROOT_CODE = {"paths.py"}                  # the only code allowed at the top level
SKIP_DIRS = {".git", ".claude", ".github", "__pycache__", "public", "node_modules",
             ".venv", "venv", "env", "docs", "tests"}


def doc_text():
    with open(DOC, encoding="utf-8") as f:
        return f.read()


def code_files():
    """Every .py file in the code folders, as folder/name.py."""
    out = set()
    for d in paths.CODE_DIRS:
        for fn in os.listdir(os.path.join(ROOT, d)):
            if fn.endswith(".py"):
                out.add(f"{d}/{fn}")
    return out


class TestArchitectureDoc(unittest.TestCase):
    def test_every_code_file_is_in_the_module_map(self):
        text = doc_text()
        missing = sorted(f for f in code_files() | ROOT_CODE if f"`{f}`" not in text)
        self.assertEqual(missing, [], "add these to the module map in ARCHITECTURE.md")

    def test_every_mapped_file_exists(self):
        named = set(re.findall(r"`((?:[a-z_]+/)+[A-Za-z0-9_.-]+\.(?:py|html|json|yml|jpg|png))`", doc_text()))
        gone = sorted(p for p in named
                      if not p.startswith(("data/", "public/", "out/"))   # data/ and outputs: generated or listed by name
                      and not os.path.exists(os.path.join(ROOT, p)))
        self.assertEqual(gone, [], "ARCHITECTURE.md names files that no longer exist")

    def test_every_code_folder_is_documented(self):
        text = doc_text()
        for d in paths.CODE_DIRS:
            self.assertTrue(os.path.isdir(os.path.join(ROOT, d)), d)
            self.assertIn(f"### `{d}/`", text, f"ARCHITECTURE.md needs a module-map section for {d}/")

    def test_change_log_has_entries(self):
        text = doc_text()
        self.assertIn("## Change log", text)
        log = text.split("## Change log", 1)[1]
        self.assertRegex(log, r"- \*\*\d{4}-\d{2}-\d{2}\*\*", "the change log needs dated entries")


class TestLayout(unittest.TestCase):
    def test_no_code_outside_the_code_folders(self):
        allowed = set(paths.CODE_DIRS)
        stray = []
        for fn in os.listdir(ROOT):
            if fn.endswith(".py") and fn not in ROOT_CODE:
                stray.append(fn)
        for top in os.listdir(ROOT):
            p = os.path.join(ROOT, top)
            if not os.path.isdir(p) or top in allowed or top in SKIP_DIRS:
                continue
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                stray += [os.path.relpath(os.path.join(dirpath, f), ROOT).replace(os.sep, "/")
                          for f in filenames if f.endswith(".py")]
        self.assertEqual(sorted(stray), [], "code belongs in one of paths.CODE_DIRS (see ARCHITECTURE.md)")

    def test_module_names_are_unique_across_folders(self):
        seen = {}
        for f in sorted(code_files()):
            seen.setdefault(os.path.basename(f), []).append(f)
        dupes = {k: v for k, v in seen.items() if len(v) > 1}
        self.assertEqual(dupes, {}, "folders share one import namespace; rename one of these")

    def test_pyright_sees_every_code_folder(self):
        import json
        with open(os.path.join(ROOT, "pyrightconfig.json"), encoding="utf-8") as f:
            extra = set(json.load(f)["extraPaths"])
        self.assertEqual(sorted(set(paths.CODE_DIRS) - extra), [])


if __name__ == "__main__":
    unittest.main()
