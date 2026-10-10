"""Tests for the refresh pipeline's stale-page handling (data/lpfetch.py,
scripts/daily_refresh.py). Everything is SYNTHETIC: a temporary cache and a
fake network; nothing touches Liquipedia or data/."""
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "data", "scripts", "viewer"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import lpfetch  # noqa: E402
import daily_refresh as dr  # noqa: E402
import validate_freshness as vf  # noqa: E402


class TestStaleServed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (lpfetch.CACHE, lpfetch.MIN_GAP, list(lpfetch.RETRY_BACKOFF), lpfetch._THROTTLED[0],
                      lpfetch.RUN_STARTED)
        lpfetch.CACHE = self.tmp.name
        lpfetch.MIN_GAP = 0
        lpfetch.RETRY_BACKOFF[:] = [0, 0]
        lpfetch._THROTTLED[0] = False
        lpfetch.STALE_SERVED.clear()

    def tearDown(self):
        lpfetch.CACHE, lpfetch.MIN_GAP, backoff, lpfetch._THROTTLED[0], lpfetch.RUN_STARTED = self.saved
        lpfetch.RETRY_BACKOFF[:] = backoff
        lpfetch.STALE_SERVED.clear()
        self.tmp.cleanup()

    def _cache(self, params, response):
        p = dict(params, format="json")
        k, _ = lpfetch._key(p)
        with open(os.path.join(lpfetch.CACHE, k + ".json"), "w", encoding="utf-8") as f:
            json.dump({"url": "x", "fetched": "2000-01-01T00:00:00Z", "response": response}, f)

    def test_throttled_refresh_is_reported_not_hidden(self):
        params = {"action": "query", "prop": "revisions", "titles": "Alpha Cup|Alpha Cup/Playoffs"}
        self._cache(params, {"old": True})
        err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, io.BytesIO(b""))
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            got = lpfetch.query(params, refresh=True)
        self.assertEqual(got, {"old": True})              # the old copy is still returned...
        self.assertEqual(lpfetch.STALE_SERVED, ["Alpha Cup|Alpha Cup/Playoffs"])   # ...but on record

    def test_fresh_fetch_is_not_reported(self):
        params = {"action": "query", "prop": "revisions", "titles": "Alpha Cup"}
        self._cache(params, {"old": True})
        body = json.dumps({"new": True}).encode()

        class R(io.BytesIO):
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        with mock.patch.object(urllib.request, "urlopen", return_value=R(body)):
            got = lpfetch.query(params, refresh=True)
        self.assertEqual(got, {"new": True})
        self.assertEqual(lpfetch.STALE_SERVED, [])

    def test_stale_live_picks_only_live_pages(self):
        lpfetch.STALE_SERVED[:] = ["Alpha Cup|Alpha Cup/Playoffs", "Some Team"]
        self.assertEqual(dr.stale_live({"Alpha Cup", "Alpha Cup/Playoffs", "Beta Cup"}),
                         ["Alpha Cup", "Alpha Cup/Playoffs"])

    def test_status_file_written_only_on_change(self):
        with tempfile.TemporaryDirectory() as d:
            saved = dr.STATUS_FILE
            dr.STATUS_FILE = os.path.join(d, "status.json")
            try:
                dr.write_status([], set())
                first = os.path.getmtime(dr.STATUS_FILE)
                os.utime(dr.STATUS_FILE, (first - 100, first - 100))
                dr.write_status([], set())                        # unchanged: no rewrite, so no commit
                self.assertEqual(os.path.getmtime(dr.STATUS_FILE), first - 100)
                dr.write_status(["Alpha Cup"], {"Alpha Cup"})
                with open(dr.STATUS_FILE, encoding="utf-8") as f:
                    self.assertEqual(json.load(f), {"ok": False, "stale_pages": ["Alpha Cup"]})
            finally:
                dr.STATUS_FILE = saved


class TestValidate(unittest.TestCase):
    def test_live_page_behind_is_detected(self):
        old = {"Alpha Cup": ("2026-01-01T00:00:00Z", "{{Match|finished=true}}"),
               "Beta Cup": ("2026-01-01T00:00:00Z", "same")}
        new = {"Alpha Cup": "{{Match|finished=true}}{{Match|finished=true}}", "Beta Cup": "same"}
        with mock.patch.object(lpfetch, "cached_pages", return_value=old),                 mock.patch.object(lpfetch, "wikitext", return_value=new):
            self.assertEqual(vf.live_pages_behind({"Alpha Cup", "Beta Cup"}), ["Alpha Cup"])

    def test_site_hash_is_read_from_the_page(self):
        class R(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        page = b'<script>const DATA = {"as_of":"2026-10-10","data_hash":"0123456789ab","events":[]}</script>'
        with mock.patch.object(urllib.request, "urlopen", return_value=R(page)):
            self.assertEqual(vf.site_hash("http://x/"), "0123456789ab")
        with mock.patch.object(urllib.request, "urlopen", return_value=R(b"<html>old build</html>")):
            self.assertEqual(vf.site_hash("http://x/"), "")


if __name__ == "__main__":
    unittest.main()
