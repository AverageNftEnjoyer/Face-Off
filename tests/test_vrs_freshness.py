"""Valve standings freshness: newest-snapshot selection, no-change leaves the
file byte-identical, a failure keeps the last good data and is flagged, and the
freshness gate. Synthetic only: a temporary data directory, no network."""
import json
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "data", "scripts", "viewer"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import vrs  # noqa: E402
import data_gates as dg  # noqa: E402
import validate_freshness as vf  # noqa: E402

TABLE = "| Standing | Points | Team | Roster |\n|--|--|--|--|\n| 1 | 2000 | Alpha | a, b, c, d, e |\n| 2 | 1900 | Beta | f, g, h, i, j |\n"
MONTHLY = [f"live/2026/standings_global_2026_{m:02d}_{d:02d}.md"
           for m, d in ((1, 5), (2, 2), (3, 2), (4, 6), (5, 4), (6, 1), (7, 6), (8, 3), (9, 7), (10, 5))]


def paths_with(*extra):
    return MONTHLY + list(extra)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (vrs.HERE, vrs.CACHE)
        vrs.HERE = self.tmp.name
        vrs.CACHE = os.path.join(self.tmp.name, "cache")
        self.calls = []

    def tearDown(self):
        vrs.HERE, vrs.CACHE = self.saved
        self.tmp.cleanup()

    def get(self, url):
        self.calls.append(url)
        return TABLE

    def p(self, name):
        return os.path.join(self.tmp.name, name)

    def read(self, name):
        with open(self.p(name), "rb") as f:
            return f.read()


class TestSelection(unittest.TestCase):
    def test_newest_by_date_not_list_order(self):
        paths = ["live/2026/standings_global_2026_10_05.md", "live/2026/standings_europe_2026_10_05.md",
                 "live/2025/standings_global_2025_12_01.md", "live/2026/standings_global_2026_09_07.md"]
        d, files = vrs.newest_published(paths)
        self.assertEqual(d, "2026-10-05")
        self.assertEqual(set(files), {"global", "europe"})

    def test_incomplete_or_invalid_dates_ignored(self):
        paths = ["live/2026/standings_europe_2026_11_02.md",      # no global file yet
                 "live/2026/standings_global_2026_13_45.md",      # not a date
                 "live/2026/standings_global_2026_10_05.md"]
        self.assertEqual(vrs.newest_published(paths)[0], "2026-10-05")
        with self.assertRaises(ValueError):
            vrs.newest_published(["README.md"])

    def test_cadence_measured_from_dates(self):
        c = vrs.cadence(vrs.published(MONTHLY))
        self.assertEqual((c["typical"], c["max"], c["stale_after"]), (28, 35, 42))
        self.assertEqual(vrs.cadence(["2026-01-05"])["stale_after"], vrs.DEFAULT_STALE_AFTER)


class TestRefresh(Base):
    def test_picks_newest_and_writes_status(self):
        st = vrs.refresh(get=self.get, paths=paths_with("live/2026/standings_global_2026_11_02.md"))
        self.assertTrue(st["ok"])
        self.assertEqual(json.loads(self.read("vrs.json"))["date"], "2026-11-02")
        self.assertEqual(st["published"], "2026-11-02")
        self.assertEqual(len(self.calls), 1)       # one file

    def test_unchanged_refresh_leaves_files_identical(self):
        vrs.refresh(get=self.get, paths=MONTHLY)
        a, s = self.read("vrs.json"), self.read("vrs_status.json")
        m = (os.stat(self.p("vrs.json")).st_mtime_ns, os.stat(self.p("vrs_status.json")).st_mtime_ns)
        # a different tree sha for the same standings must not rewrite the file
        vrs.refresh(get=self.get, paths=MONTHLY, now=datetime(2030, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(a, self.read("vrs.json"))
        self.assertEqual(s, self.read("vrs_status.json"))
        self.assertEqual(m, (os.stat(self.p("vrs.json")).st_mtime_ns, os.stat(self.p("vrs_status.json")).st_mtime_ns))

    def test_failure_keeps_last_good_and_flags(self):
        vrs.refresh(get=self.get, paths=MONTHLY)
        good = self.read("vrs.json")

        def boom(url):
            raise OSError("rate limited")
        st = vrs.refresh(get=boom, paths=paths_with("live/2026/standings_global_2026_11_02.md"),
                         now=datetime(2026, 11, 3, tzinfo=timezone.utc))
        self.assertFalse(st["ok"])
        self.assertIn("rate limited", st["error"])
        self.assertEqual(good, self.read("vrs.json"))
        saved = json.loads(self.read("vrs_status.json"))
        self.assertFalse(saved["ok"])
        self.assertEqual(saved["stored"], "2026-10-05")
        self.assertEqual(saved["failing_since"], "2026-11-03T00:00:00Z")
        # the same failure later keeps the original failing_since
        st2 = vrs.refresh(get=boom, paths=paths_with("live/2026/standings_global_2026_11_02.md"),
                          now=datetime(2026, 11, 4, tzinfo=timezone.utc))
        self.assertEqual(st2["failing_since"], "2026-11-03T00:00:00Z")

    def test_empty_parse_is_a_failure_and_not_cached(self):
        vrs.refresh(get=self.get, paths=MONTHLY)
        good = self.read("vrs.json")
        st = vrs.refresh(get=lambda u: "no table here", paths=paths_with("live/2026/standings_global_2026_11_02.md"))
        self.assertFalse(st["ok"])
        self.assertEqual(good, self.read("vrs.json"))
        self.assertFalse(os.path.exists(os.path.join(vrs.CACHE, "standings_global_2026_11_02.md")))

    def test_older_than_stored_is_refused(self):
        vrs.refresh(get=self.get, paths=MONTHLY)
        good = self.read("vrs.json")
        st = vrs.refresh(get=self.get, paths=MONTHLY[:5])
        self.assertFalse(st["ok"])
        self.assertEqual(good, self.read("vrs.json"))


class TestGate(Base):
    def write(self, stored, status):
        with open(self.p("vrs.json"), "w") as f:
            json.dump({"date": stored}, f)
        if status is not None:
            with open(self.p("vrs_status.json"), "w") as f:
                json.dump(status, f)

    def gate(self, today):
        return dg.gate_vrs_freshness(today, self.p("vrs.json"), self.p("vrs_status.json"))

    def test_current_passes(self):
        self.write("2026-10-05", {"ok": True, "published": "2026-10-05", "stale_after_days": 42})
        self.assertEqual(self.gate(date(2026, 10, 14))["problems"], [])

    def test_behind_valve_fails(self):
        self.write("2026-10-05", {"ok": True, "published": "2026-11-02", "stale_after_days": 42})
        g = self.gate(date(2026, 11, 3))
        self.assertTrue(any("Valve has published 2026-11-02" in p for p in g["problems"]))

    def test_past_cadence_fails_even_if_valve_is_overdue(self):
        self.write("2026-10-05", {"ok": True, "published": "2026-10-05", "stale_after_days": 42})
        g = self.gate(date(2026, 11, 20))
        self.assertTrue(any("Valve itself is overdue" in p for p in g["problems"]))

    def test_fetch_failure_alone_warns_not_blocks(self):
        self.write("2026-10-05", {"ok": False, "error": "OSError: x", "published": "2026-10-05",
                                  "stale_after_days": 42})
        g = self.gate(date(2026, 10, 20))
        self.assertEqual(g["problems"], [])
        self.assertEqual(len(g["warnings"]), 1)

    def test_missing_vrs_fails(self):
        self.assertTrue(self.gate(date(2026, 10, 14))["problems"])

    def test_assess_failure_is_reported_by_check(self):
        probs = vrs.assess("2026-10-05", {"ok": False, "error": "boom"}, date(2026, 10, 14))
        self.assertEqual(len(probs), 1)


class TestValidateFreshness(Base):
    def test_live_tree_newer_than_stored(self):
        vf.ROOT = self.tmp.name
        os.makedirs(os.path.join(self.tmp.name, "data"))
        with open(os.path.join(self.tmp.name, "data", "vrs.json"), "w") as f:
            json.dump({"date": "2026-10-05"}, f)
        try:
            tr = {"tree": [{"path": p} for p in paths_with("live/2026/standings_global_2026_11_02.md")]}
            bad = vf.vrs_problems(tr, date(2026, 11, 3))
            self.assertTrue(any("2026-11-02" in p for p in bad))
            ok = vf.vrs_problems({"tree": [{"path": p} for p in MONTHLY]}, date(2026, 10, 14))
            self.assertEqual(ok, [])
        finally:
            vf.ROOT = ROOT


if __name__ == "__main__":
    unittest.main()
