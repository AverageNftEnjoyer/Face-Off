"""Tests for the track-record data the viewer carries (viewer/build_viewer.py):
the Elo-only baseline on every pre-match call and the track_payload block.
Team, event and map names here are test data.

Run from the repo root:  python -m unittest tests.test_track_record
"""
import os
import sys
import unittest
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "viewer"))

import backtest as bt  # noqa: E402
import build_viewer as bv  # noqa: E402
import events as E  # noqa: E402
import walkforward as wf  # noqa: E402

MAPS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]


def series(day, a, b, winner, i):
    """One finished BO3 series: the winner takes two maps."""
    loser = b if winner == a else a
    return {"id": i, "date": day, "team_a": a, "team_b": b, "winner": winner, "best_of": 3,
            "event": "Test Cup", "tier": "S",
            "maps": [{"map": MAPS[i % 7], "winner": winner}, {"map": MAPS[(i + 1) % 7], "winner": loser},
                     {"map": MAPS[(i + 2) % 7], "winner": winner}]}


def history():
    ms, i = [], 0
    for d in range(1, 21):
        for a, b, w in (("TA", "TB", "TA"), ("TB", "TC", "TB"), ("TC", "TA", "TA")):
            ms.append(series(f"2026-01-{d:02d}", a, b, w, i))
            i += 1
    return ms


def report(after, models=("shipped", "elo_only")):
    m = {"n": 10, "accuracy": 0.6, "brier": 0.21, "log_loss": 0.61, "ece10_fav": 0.02, "fav_buckets": []}
    return {"clean_window_start_after": after, "oof_range": ["2025-01-01", "2026-02-01"],
            "clean": {k: dict(m) for k in models}}


class TestEloBaseline(unittest.TestCase):
    def setUp(self):
        bv.VD_JOBS.clear()

    def tearDown(self):
        bv.VD_JOBS.clear()

    def test_match_call_carries_point_in_time_elo_expectation(self):
        day = date(2026, 1, 21)
        tl = bv.Timeline(history() + [series("2026-01-21", "TA", "TB", "TB", 999)])
        h = tl.at(day)
        out = bv.predict(h, "TA", "TB", day, elo=True)
        # the baseline is backtest's elo_only number from the same pre-match state ...
        _, meta = h.features({"team_a": "TA", "team_b": "TB", "date": day.isoformat(), "event_title": None})
        self.assertEqual(out["q"], meta["p_elo"])
        self.assertEqual(out["q"], bt.elo_expected(h.feature_elo("TA"), h.feature_elo("TB")))
        # ... which never sees the day's own result: the same call without that
        # series in the data is identical
        bv.VD_JOBS.clear()
        clean = bv.predict(bv.Timeline(history()).at(day), "TA", "TB", day, elo=True)
        self.assertEqual(out["q"], clean["q"])
        self.assertEqual(out["p"], clean["p"])
        # and agrees with the rounded pre-match Elo the page also shows
        ea, eb = out["e"]
        self.assertAlmostEqual(out["q"], bt.elo_expected(ea, eb), delta=0.002)

    def test_grid_predictions_do_not_carry_it(self):
        day = date(2026, 1, 21)
        h = bv.Timeline(history()).at(day)
        out = bv.predict(h, "TA", "TB", day)
        self.assertNotIn("q", out)
        self.assertNotIn("e", out)


class TestTrackPayload(unittest.TestCase):
    def test_buckets_and_rule_come_from_walkforward(self):
        t = bv.track_payload(wf.SHIPPED_FIT_END, wf.FAV_BUCKETS, wf.CAL_MAX_GAP, wf.CAL_MIN_N, None)
        self.assertEqual(t["fit_end"], wf.SHIPPED_FIT_END)
        self.assertEqual(len(t["buckets"]), len(wf.FAV_BUCKETS))
        self.assertEqual([b[0] for b in t["buckets"]], [lo for lo, _ in wf.FAV_BUCKETS])
        self.assertTrue(all(hi <= 1.0 for _, hi in t["buckets"]))     # open top edge clipped for display
        self.assertEqual((t["gap"], t["min_n"]), (wf.CAL_MAX_GAP, wf.CAL_MIN_N))
        self.assertIsNone(t["backtest"])

    def test_backtest_block_only_for_matching_fit_date(self):
        ok = bv.track_payload("2025-10-08", [(0.5, 1.0)], 0.1, 30, report("2025-10-08"))
        self.assertEqual(ok["backtest"]["after"], "2025-10-08")
        self.assertEqual(ok["backtest"]["to"], "2026-02-01")
        self.assertEqual(sorted(ok["backtest"]["models"]), ["elo_only", "shipped"])
        self.assertEqual(sorted(ok["backtest"]["models"]["shipped"]), ["accuracy", "brier", "log_loss", "n"])
        stale = bv.track_payload("2025-12-01", [(0.5, 1.0)], 0.1, 30, report("2025-10-08"))
        self.assertIsNone(stale["backtest"])
        partial = bv.track_payload("2025-10-08", [(0.5, 1.0)], 0.1, 30, report("2025-10-08", ("shipped",)))
        self.assertIsNone(partial["backtest"])


class TestMatchIds(unittest.TestCase):
    """The page keys every call by match id: two events must never share one."""

    @staticmethod
    def event(slug_, start, n):
        p = slug_[:E.ID_PREFIX_LEN]
        return {"slug": slug_, "start": start, "matches": [{"id": f"{p}-{i}"} for i in range(1, n + 1)]}

    def test_clashing_prefixes_get_full_slugs_others_untouched(self):
        long = "x" * E.ID_PREFIX_LEN
        a, b = self.event(long + "-one", "2026-01-01", 3), self.event(long + "-two", "2027-01-01", 2)
        c = self.event("short-event", "2026-03-01", 2)
        E.dedupe_match_ids([a, b, c])
        ids = [m["id"] for e in (a, b, c) for m in e["matches"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([m["id"] for m in a["matches"]], [f"{long}-one-{i}" for i in (1, 2, 3)])
        self.assertEqual([m["id"] for m in c["matches"]], ["short-event-1", "short-event-2"])

    def test_identical_slugs_fall_back_to_start_date(self):
        a, b = self.event("same-name", "2026-01-01", 2), self.event("same-name", "2027-01-01", 2)
        E.dedupe_match_ids([a, b])
        self.assertEqual([m["id"] for m in b["matches"]], ["same-name-2027-01-01-1", "same-name-2027-01-01-2"])
        self.assertEqual(len({m["id"] for e in (a, b) for m in e["matches"]}), 4)

    def test_no_clash_no_change(self):
        a, b = self.event("alpha", "2026-01-01", 2), self.event("beta", "2026-01-01", 2)
        E.dedupe_match_ids([a, b])
        self.assertEqual([m["id"] for m in a["matches"] + b["matches"]], ["alpha-1", "alpha-2", "beta-1", "beta-2"])


if __name__ == "__main__":
    unittest.main()
