"""Tests for the refresh pipeline's stale-page handling (data/lpfetch.py,
scripts/daily_refresh.py). Everything is SYNTHETIC: a temporary cache and a
fake network; nothing touches Liquipedia or data/."""
import email.message
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
        err = urllib.error.HTTPError("u", 429, "Too Many Requests", email.message.Message(), io.BytesIO(b""))
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

            def __exit__(self, *a) -> None:
                return None
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
                    st = json.load(f)
                self.assertEqual((st["ok"], st["stale_pages"]), (False, ["Alpha Cup"]))
                self.assertRegex(st["stale_since"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
            finally:
                dr.STATUS_FILE = saved


class TestOutageClock(unittest.TestCase):
    """A refusal by Liquipedia must not email you every 10 minutes: the status file
    remembers WHEN pages first went stale and the run turns red only after a while."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = dr.STATUS_FILE
        dr.STATUS_FILE = os.path.join(self.tmp.name, "status.json")

    def tearDown(self):
        dr.STATUS_FILE = self.saved
        self.tmp.cleanup()

    def _write(self, obj):
        with open(dr.STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def _read(self):
        with open(dr.STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)

    def test_stale_since_is_kept_while_the_outage_lasts(self):
        self._write({"ok": False, "stale_pages": ["Alpha Cup"], "stale_since": "2026-10-10T20:00:00Z"})
        before = os.path.getmtime(dr.STATUS_FILE)
        dr.write_status(["Alpha Cup"], {"Alpha Cup"})
        self.assertEqual(self._read()["stale_since"], "2026-10-10T20:00:00Z")
        self.assertEqual(os.path.getmtime(dr.STATUS_FILE), before)     # unchanged: nothing to commit

    def test_outage_end_clears_the_clock(self):
        self._write({"ok": False, "stale_pages": ["Alpha Cup"], "stale_since": "2026-10-10T20:00:00Z"})
        dr.write_status([], set())
        self.assertEqual(self._read(), {"ok": True, "stale_pages": [], "stale_since": None})

    def test_check_status_waits_before_failing(self):
        from datetime import datetime, timezone
        self._write({"ok": False, "stale_pages": ["Alpha Cup"], "stale_since": "2026-10-10T20:00:00Z"})
        at = lambda h, m: datetime(2026, 10, 10, h, m, tzinfo=timezone.utc)   # noqa: E731
        with mock.patch("builtins.print"):
            self.assertEqual(dr.check_status(30, now=at(20, 10)), 0)    # 10 min: warning only
            self.assertEqual(dr.check_status(30, now=at(20, 45)), 1)    # 45 min: red
            self.assertEqual(dr.check_status(0, now=at(20, 1)), 1)      # the daily run fails at once

    def test_check_status_ok_and_missing_file(self):
        with mock.patch("builtins.print"):
            self.assertEqual(dr.check_status(30), 0)                    # no file: nothing known to be stale
            self._write({"ok": True, "stale_pages": [], "stale_since": None})
            self.assertEqual(dr.check_status(0), 0)

    def test_live_refresh_does_not_wait_between_tries(self):
        # one pass, no sleeping loop: the next scheduled run is the retry
        calls = []
        lpfetch.STALE_SERVED[:] = []

        def collect():
            calls.append(1)
            lpfetch.STALE_SERVED.append("Alpha Cup")
        with mock.patch("time.sleep") as sleep:
            stale = dr.refresh_live({"Alpha Cup"}, collect, pauses=())
        self.assertEqual((stale, len(calls)), (["Alpha Cup"], 1))
        sleep.assert_not_called()


class TestMatchWindows(unittest.TestCase):
    """Liquipedia is only asked while a match is due, under way or overdue."""

    def _ev(self, *matches):
        return {"title": "Alpha Cup", "start": "2026-10-01", "end": "2026-10-20",
                "matches": [dict({"finished": False, "when": None, "day": None}, **m) for m in matches]}

    def _at(self, h, m=0, day=10):
        from datetime import datetime, timezone
        return datetime(2026, 10, day, h, m, tzinfo=timezone.utc)

    def test_quiet_hours_do_not_ask(self):
        ev = self._ev({"when": "2026-10-10T18:00:00+00:00", "day": "2026-10-10"})
        self.assertFalse(dr.match_window_active(ev, self._at(3)))        # 3 a.m., match tonight
        self.assertFalse(dr.match_window_active(ev, self._at(12)))       # long before the lead time

    def test_window_opens_before_and_closes_after(self):
        ev = self._ev({"when": "2026-10-10T18:00:00+00:00", "day": "2026-10-10"})
        self.assertFalse(dr.match_window_active(ev, self._at(17, 20)))   # 40 min before
        self.assertTrue(dr.match_window_active(ev, self._at(17, 35)))    # inside the 30 min lead
        self.assertTrue(dr.match_window_active(ev, self._at(20, 30)))    # under way
        self.assertTrue(dr.match_window_active(ev, self._at(21, 59)))    # still overdue, within 4 h
        self.assertFalse(dr.match_window_active(ev, self._at(22, 5)))    # given up

    def test_finished_matches_never_count(self):
        ev = self._ev({"when": "2026-10-10T18:00:00+00:00", "day": "2026-10-10", "finished": True})
        self.assertFalse(dr.match_window_active(ev, self._at(18, 30)))

    def test_time_zone_offsets_are_respected(self):
        ev = self._ev({"when": "2026-10-10T20:00:00+02:00", "day": "2026-10-10"})   # 18:00 UTC
        self.assertTrue(dr.match_window_active(ev, self._at(18, 10)))
        self.assertFalse(dr.match_window_active(ev, self._at(23)))

    def test_match_with_a_date_but_no_time_counts_all_that_day(self):
        ev = self._ev({"day": "2026-10-10"})
        self.assertTrue(dr.match_window_active(ev, self._at(3)))
        self.assertFalse(dr.match_window_active(ev, self._at(3, day=11)))

    def test_live_titles_only_returns_tournaments_in_a_window(self):
        from datetime import date
        busy = self._ev({"when": "2026-10-10T18:00:00+00:00", "day": "2026-10-10"})
        idle = dict(self._ev({"when": "2026-10-10T23:00:00+00:00", "day": "2026-10-10"}), title="Beta Cup")
        with mock.patch.object(dr.E, "discover", return_value=[busy, idle]):
            got = dr.live_titles(date(2026, 10, 10), now=self._at(18, 10))
            self.assertIn("Alpha Cup", got)
            self.assertNotIn("Beta Cup", got)
            self.assertEqual(dr.live_titles(date(2026, 10, 10), now=self._at(3)), set())
            forced = dr.live_titles(date(2026, 10, 10), now=self._at(3), force=True)
            self.assertTrue({"Alpha Cup", "Beta Cup"} <= forced)


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

            def __exit__(self, *a) -> None:
                return None
        page = b'<script>const DATA = {"as_of":"2026-10-10","data_hash":"0123456789ab","events":[]}</script>'
        with mock.patch.object(urllib.request, "urlopen", return_value=R(page)):
            self.assertEqual(vf.site_hash("http://x/"), "0123456789ab")
        with mock.patch.object(urllib.request, "urlopen", return_value=R(b"<html>old build</html>")):
            self.assertEqual(vf.site_hash("http://x/"), "")


class TestRefreshReliability(unittest.TestCase):
    """2026-10-14: the daily run's bulk refresh got rate-limited and the live page,
    fetched in the same batch, was refused; the watchdog then failed on a page the
    next live refresh fixed. The live pages now go first, alone, and one refusal
    heals instead of failing."""

    def test_daily_refresh_fetches_live_pages_first_and_alone(self):
        import collect_liquipedia
        seen = []

        class Stop(Exception):
            pass

        saved = set(lpfetch.FRESH_TITLES)
        lpfetch.FRESH_TITLES.clear()
        try:
            with mock.patch.object(dr, "stale_titles", return_value=({"Bulk Cup", "Old Cup"}, {"Some Team"}, ())),                     mock.patch.object(dr, "discover", return_value=([], [])),                     mock.patch.object(dr, "live_titles", return_value={"Live Cup"}),                     mock.patch.object(collect_liquipedia, "main", side_effect=lambda: seen.append(set(lpfetch.FRESH_TITLES))),                     mock.patch.object(dr, "write_status", side_effect=Stop):
                with self.assertRaises(Stop):
                    dr.main([])
        finally:
            lpfetch.FRESH_TITLES.clear()
            lpfetch.FRESH_TITLES |= saved
        self.assertEqual(seen[0], {"Live Cup"})                       # live page alone, first
        self.assertTrue({"Bulk Cup", "Old Cup", "Some Team"} <= seen[1])   # then the bulk pass

    def _validate(self, minutes_ago):
        from datetime import datetime, timedelta, timezone
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "data"))
        since = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(os.path.join(tmp, "data", "refresh_status.json"), "w", encoding="utf-8") as f:
            json.dump({"ok": False, "stale_pages": ["Live Cup"], "stale_since": since}, f)
        out = os.path.join(tmp, "gh_out")
        with mock.patch.object(vf, "ROOT", tmp), mock.patch.object(vf, "site_hash", return_value="h"),                 mock.patch.object(vf, "local_hash", return_value="h"),                 mock.patch.dict(os.environ, {"GITHUB_OUTPUT": out}):
            code = vf.main(["--no-liquipedia", "--no-valve", "--site", "http://x/"])
        return code, open(out, encoding="utf-8").read()

    def test_a_fresh_refusal_heals_without_failing(self):
        code, out = self._validate(5)
        self.assertEqual(code, 0)
        self.assertIn("needs_refresh=true", out)

    def test_a_refusal_that_lasts_fails(self):
        code, out = self._validate(45)
        self.assertEqual(code, 1)
        self.assertIn("needs_refresh=true", out)


if __name__ == "__main__":
    unittest.main()
