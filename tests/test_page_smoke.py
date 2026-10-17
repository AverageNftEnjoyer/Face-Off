"""checks/page_smoke.py on SYNTHETIC pages: a good page passes; a missing file,
null data, an unfilled placeholder and a script that does not parse all fail."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401

import page_smoke as PS  # noqa: E402

PAD = "x" * (PS.MIN_BYTES + 10)


def page(script, data="{}"):
    return (f"<html><body><!-- {PAD} --><script>const DATA = {data};\nconst FIXTURES = [];\nconst RESULTS = [];\n"
            f"const META = {{}};\nconst PAGE_BUNDLE_VERSION = 2;\n{script}</script></body></html>")


class PageSmokeTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def write(self, html):
        p = os.path.join(self.dir, "index.html")
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
        return p

    def test_good_page_passes(self):
        self.assertEqual(PS.check(self.write(page("function ok() { return 1; }"))), [])

    def test_missing_and_tiny(self):
        self.assertTrue(PS.check(os.path.join(self.dir, "nope.html")))
        self.assertTrue(any("KB" in e for e in PS.check(self.write("<script>const DATA = {};</script>"))))

    def test_null_data_and_placeholder(self):
        self.assertTrue(any("null" in e for e in PS.check(self.write(page("", data="null")))))
        self.assertTrue(any("placeholder" in e for e in PS.check(self.write(page("const NOTES = /*__NOTES__*/null;")))))

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_script_that_does_not_parse_fails(self):
        errs = PS.check(self.write(page("const w = 1;\nconst w = 2;")))
        self.assertTrue(any("does not parse" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
