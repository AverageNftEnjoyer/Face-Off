"""Unit tests for roster_flags.py (display-only roster caution flags).

All fixtures are SYNTHETIC (made-up teams and players); nothing is written to data/.
"""
import os
import sys
import unittest
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "viewer"))
import lineup_features as LF  # noqa: E402
import predictor as pr  # noqa: E402
import roster_flags as RF  # noqa: E402

OLD = ["o1", "o2", "o3", "o4", "o5"]
TWO_NEW = ["o1", "o2", "o3", "n1", "n2"]
ONE_NEW = ["o1", "o2", "o3", "o4", "n1"]
START = date(2025, 3, 1)


def day(i):
    return (START + timedelta(days=i)).isoformat()


def season(fives):
    """SYNTHETIC: Alpha plays Bravo on day i with fives[i] (index 0 = day 0)."""
    matches, lineups = [], {}
    for i, f in enumerate(fives):
        matches.append({"date": day(i), "team_a": "Alpha", "team_b": "Bravo"})
        if f is not None:
            lineups[f"{day(i)}|Alpha|Bravo"] = {"a": f, "b": ["b1", "b2", "b3", "b4", "b5"]}
    return matches, lineups


def flagger(matches, lineups, overrides=(), events=None):
    return RF.Flagger(LF.team_history(matches, lineups, list(overrides)), events=events or {},
                      played=RF.series_dates(matches))


class Thresholds(unittest.TestCase):
    def test_named_and_reused(self):
        """No magic numbers: the window and look-back reuse existing constants."""
        self.assertEqual(RF.RECENT_SERIES, pr.CONFIG["veto_roster_recover"])
        self.assertEqual(RF.CORE_N, LF.CORE_N)
        self.assertEqual(RF.MAX_DAYS, LF.CONT_DAYS)
        self.assertGreaterEqual(RF.MIN_NEW, 1)


class LineupChange(unittest.TestCase):
    def setUp(self):
        # 12 series with the old five, then the change on day 12
        self.fives = [OLD] * 12 + [TWO_NEW] * 4

    def test_point_in_time(self):
        """The change played on day 12 is invisible to day 12 and seen from day 13."""
        f = flagger(*season(self.fives))
        self.assertEqual(f.flags("Alpha", day(12)), [])
        got = f.flags("Alpha", day(13))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["k"], "change")
        self.assertEqual(got[0]["since"], day(12))
        self.assertIn("2 new players in the last 1 series", got[0]["r"])

    def test_hand_entered_move_before_any_series(self):
        """SYNTHETIC: a roster_change override dated after the last series is
        labelled as an override and says no series has been played yet; it is
        first read the day after its `from` date."""
        ms, lu = season([OLD] * 12)
        ov = [{"team": "Alpha", "from": day(13), "until": None, "players": TWO_NEW, "kind": "roster_change"}]
        f = flagger(ms, lu, ov)
        self.assertEqual(f.flags("Alpha", day(13)), [])
        got = f.flags("Alpha", day(14))
        self.assertEqual([(g["k"], g["src"]) for g in got], [("change", "override")])
        self.assertIn("no series played yet", got[0]["r"])

    def test_future_rows_cannot_change_a_day(self):
        """Scramble check: anything dated >= D (more series, a later reversal,
        a different five) leaves the flags for D unchanged."""
        base = flagger(*season(self.fives)).flags("Alpha", day(14))
        for tail in ([OLD] * 6, [ONE_NEW] * 6, [None] * 6):
            ms, lu = season(self.fives + tail)
            self.assertEqual(flagger(ms, lu).flags("Alpha", day(14)), base)
        ms, lu = season(self.fives[:14] + [OLD] * 2)   # day 14 and 15 rewritten
        self.assertEqual(flagger(ms, lu).flags("Alpha", day(14)), base)

    def test_single_new_player_below_threshold(self):
        got = flagger(*season([OLD] * 12 + [ONE_NEW] * 3)).flags("Alpha", day(14))
        self.assertEqual(bool(got), RF.MIN_NEW <= 1)

    def test_expires_after_recent_series(self):
        n = RF.RECENT_SERIES
        f = flagger(*season([OLD] * 12 + [TWO_NEW] * (n + 2)))
        self.assertTrue(f.flags("Alpha", day(12 + n)))       # n series with the new five
        self.assertEqual(f.flags("Alpha", day(12 + n + 1)), [])

    def test_series_without_published_lineup_still_count(self):
        """Lineups lag: series with no lineup row still age the change out."""
        n = RF.RECENT_SERIES
        fives = [OLD] * 12 + [TWO_NEW] + [None] * (n + 1)
        f = flagger(*season(fives))
        got = f.flags("Alpha", day(14))
        self.assertTrue(got)
        self.assertEqual(got[0]["thru"], day(12))            # newest lineup used
        self.assertEqual(f.flags("Alpha", day(12 + n + 1)), [])

    def test_expires_after_max_days(self):
        ms, lu = season([OLD] * 12 + [TWO_NEW])
        f = flagger(ms, lu)
        self.assertTrue(f.flags("Alpha", day(13)))
        self.assertEqual(f.flags("Alpha", day(12 + RF.MAX_DAYS + 1)), [])

    def test_too_little_history_is_not_a_change(self):
        f = flagger(*season([OLD] * 2 + [TWO_NEW] * 2))
        self.assertEqual(f.flags("Alpha", day(4)), [])

    def test_display_names(self):
        ms, lu = season(self.fives)
        f = RF.Flagger(LF.team_history(ms, lu, []), events={}, played=RF.series_dates(ms),
                       display={"n1": "NewOne", "n2": "NewTwo"})
        self.assertIn("NewOne, NewTwo", f.flags("Alpha", day(13))[0]["r"])


class EventStandins(unittest.TestCase):
    EV = {"Alpha": {"Cup/2025": {"standin": True, "missing_igl": True, "players": [["S9", "o5"]]}}}

    def setUp(self):
        # event starts day 12; the stand-in s9 first appears in a lineup on day 13
        self.fives = [OLD] * 13 + [["o1", "o2", "o3", "o4", "s9"]] * 3
        self.f = flagger(*season(self.fives), events=self.EV)

    def test_unplayed_series_uses_the_list(self):
        got = self.f.flags("Alpha", day(12), "Cup/2025", finished=False, event_start=day(12))
        self.assertEqual([g["r"] for g in got], ["Stand-in: S9 for o5"])
        self.assertEqual(got[0]["igl"], 1)
        # a stage page of the event is covered too (backtest.roster_flags rule)
        self.assertTrue(self.f.flags("Alpha", day(12), "Cup/2025/Playoffs"))
        self.assertEqual(self.f.flags("Alpha", day(12), "Other/2025"), [])

    def test_finished_series_needs_a_dated_lineup_before_its_day(self):
        """Undated list: a played series is flagged only once a lineup dated
        before its day (and inside the event) shows the stand-in."""
        kw = {"finished": True, "event_start": day(12)}
        self.assertEqual(self.f.flags("Alpha", day(12), "Cup/2025", **kw), [])
        self.assertEqual(self.f.flags("Alpha", day(13), "Cup/2025", **kw), [])   # his first series
        got = self.f.flags("Alpha", day(14), "Cup/2025", **kw)
        self.assertEqual([g["k"] for g in got], ["standin"])
        # a lineup from before the event does not confirm it
        self.assertEqual(self.f.flags("Alpha", day(14), "Cup/2025", finished=True, event_start=day(14)), [])

    def test_no_duplicate_change_line_for_a_listed_standin(self):
        got = self.f.flags("Alpha", day(15), "Cup/2025", finished=False, event_start=day(12))
        self.assertEqual(len(got), 1)


class Overrides(unittest.TestCase):
    def test_stand_in_override_point_in_time(self):
        ms, lu = season([OLD] * 16)
        ov = {"team": "Alpha", "from": day(13), "until": None, "kind": "stand_in", "note": "",
              "players": ["o1", "o2", "o3", "o4", "s7"]}
        f = flagger(ms, lu, [ov])
        self.assertEqual(f.flags("Alpha", day(13)), [])          # dated F: first read by D > F
        got = f.flags("Alpha", day(14))
        self.assertEqual([(g["k"], g["src"]) for g in got], [("standin", "override")])
        self.assertIn("s7", got[0]["r"])


class DisplayOnly(unittest.TestCase):
    def test_engine_never_reads_flags(self):
        """The flags are attached after the predictions; no model module imports them."""
        for name in ("predictor.py", "backtest.py", "veto.py", "lineup_features.py"):
            with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
                src = fh.read()
                self.assertNotIn("import roster_flags", src, name)
                self.assertNotIn("from roster_flags", src, name)

    def test_bundle_field_is_optional(self):
        import build_bundle as bb
        self.assertEqual(bb.roster_flags({}), {})
        rf = {"a": [{"k": "change", "r": "x"}], "b": []}
        self.assertEqual(bb.roster_flags({"rf": rf}), {"roster_flags": rf})


if __name__ == "__main__":
    unittest.main()
