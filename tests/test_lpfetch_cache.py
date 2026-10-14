"""lpfetch.query: an unchanged re-fetched page must not rewrite its cache file."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

import lpfetch  # noqa: E402


class _Resp:
    headers = {}

    def __init__(self, obj):
        self._b = json.dumps(obj).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class UnchangedPageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.saved = (lpfetch.CACHE, lpfetch.RUN_STARTED, lpfetch.MIN_GAP, lpfetch._PAGE_INDEX)
        # the page index is module state other tests build; this test's temp
        # records must not be added to it
        lpfetch.CACHE, lpfetch.MIN_GAP, lpfetch._PAGE_INDEX = self.tmp, 0, None
        lpfetch.RUN_STARTED = "2999-01-01T00:00:00Z"      # every cached copy is stale this run
        lpfetch._CONFIRMED.clear()
        lpfetch.NETWORK_REQUESTS[0] = 0

    def tearDown(self):
        lpfetch.CACHE, lpfetch.RUN_STARTED, lpfetch.MIN_GAP, lpfetch._PAGE_INDEX = self.saved
        lpfetch._CONFIRMED.clear()

    def _query(self, body):
        with mock.patch.object(lpfetch.urllib.request, "urlopen", return_value=_Resp(body)) as u:
            out = lpfetch.query({"action": "query", "titles": "X"}, refresh=True)
        return out, u.call_count

    def _file(self):
        (name,) = os.listdir(self.tmp)
        return os.path.join(self.tmp, name)

    def test_same_content_leaves_the_file_byte_identical(self):
        self._query({"query": {"pages": 1}})
        path = self._file()
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["fetched"] = "2000-01-01T00:00:00Z"        # an old stamp, as in a committed cache
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        before = open(path, "rb").read()
        out, n = self._query({"query": {"pages": 1}})
        self.assertEqual(out, {"query": {"pages": 1}})
        self.assertEqual(n, 1)
        self.assertEqual(open(path, "rb").read(), before)

    def test_changed_content_is_written_with_a_new_stamp(self):
        self._query({"query": {"pages": 1}})
        out, _ = self._query({"query": {"pages": 2}})
        with open(self._file(), encoding="utf-8") as f:
            rec = json.load(f)
        self.assertEqual(rec["response"], {"query": {"pages": 2}})
        self.assertGreater(rec["fetched"], "2000")

    def test_confirmed_page_is_not_fetched_twice_in_one_run(self):
        self._query({"query": {"pages": 1}})
        path = self._file()
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["fetched"] = "2000-01-01T00:00:00Z"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        self._query({"query": {"pages": 1}})                    # confirms, no write
        _, n = self._query({"query": {"pages": 1}})
        self.assertEqual(n, 0)                                   # served from cache


if __name__ == "__main__":
    unittest.main()
