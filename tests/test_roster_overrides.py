"""Unit tests for manual roster overrides in lineup_features.py.

All fixtures are SYNTHETIC (made-up teams and players); nothing is written to data/.
"""
import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import lineup_features as LF  # noqa: E402
import roster_overrides_check as chk  # noqa: E402

OLD = ["o1", "o2", "o3", "o4", "o5"]
NEW = ["n1", "n2", "n3", "n4", "n5"]
SUB = ["o1", "o2", "o3", "o4", "s9"]


def day(i):
    return f"2025-03-{i:02d}"


def fixture(n=12):
    """SYNTHETIC: team Alpha plays Bravo each day 1..n with a fixed old five."""
    matches, lineups = [], {}
    for i in range(1, n + 1):
        matches.append({"date": day(i), "team_a": "Alpha", "team_b": "Bravo"})
        lineups[f"{day(i)}|Alpha|Bravo"] = {"a": OLD, "b": ["b1", "b2", "b3", "b4", "b5"]}
    return matches, lineups


def ov(frm, players, kind="roster_change", until=None):
    return {"team": "Alpha", "from": frm, "players": players, "kind": kind,
            "until": until, "note": ""}


def feats(matches, lineups, ovs, d, team="Alpha"):
    return LF.features(LF.team_history(matches, lineups, ovs).get(team, []), d)


class DateCutoff(unittest.TestCase):
    def test_applies_from_date_forward_only(self):
        m, l = fixture()
        h = LF.team_history(m, l, [ov(day(6), NEW)])
        rows = dict(h["Alpha"])
        for i in range(1, 6):
            self.assertEqual(tuple(rows[day(i)]), tuple(OLD), day(i))
        for i in range(6, 13):
            self.assertEqual(tuple(rows[day(i)]), tuple(NEW), day(i))

    def test_override_wins_over_valve_only_for_that_team(self):
        m, l = fixture()
        h = LF.team_history(m, l, [ov(day(1), NEW)])
        self.assertEqual({tuple(f) for _, f in h["Bravo"]}, {("b1", "b2", "b3", "b4", "b5")})
        self.assertEqual({tuple(f) for _, f in h["Alpha"]}, {tuple(NEW)})

    def test_until_ends_override(self):
        m, l = fixture()
        rows = dict(LF.team_history(m, l, [ov(day(4), SUB, "stand_in", day(7))])["Alpha"])
        self.assertEqual(tuple(rows[day(5)]), tuple(SUB))
        self.assertEqual(tuple(rows[day(7)]), tuple(OLD))

    def test_move_known_before_team_plays(self):
        m, l = fixture(3)
        h = LF.team_history(m, l, [ov("2025-03-20", NEW)])
        self.assertEqual(tuple(h["Alpha"][-1][1]), tuple(NEW))
        self.assertEqual(h["Alpha"][-1][0], "2025-03-20")
        self.assertEqual(feats(m, l, [ov("2025-03-20", NEW)], "2025-03-21")["standin"], 1)

    def test_roster_change_vs_stand_in_flags(self):
        m, l = fixture()
        si = feats(m, l, [ov(day(8), SUB, "stand_in")], day(9))
        self.assertEqual(si["standin"], 1)
        self.assertAlmostEqual(si["continuity"], 0.8)
        base = feats(m, l, [], day(9))
        self.assertEqual((base["standin"], base["continuity"]), (0, 1.0))
        # permanent change: later, once the new five is the regular five, no stand-in flag
        rc = feats(m, l, [ov(day(2), NEW)], day(12))
        self.assertEqual(rc["standin"], 0)

    def test_empty_overrides_change_nothing(self):
        m, l = fixture()
        self.assertEqual(LF.team_history(m, l, []), LF.team_history(m, l, []))
        for d in (day(3), day(9)):
            self.assertEqual(feats(m, l, [], d), LF.features([(x, tuple(f)) for x, f in
                             [(mm["date"], l[f"{mm['date']}|Alpha|Bravo"]["a"]) for mm in m]], d))

    def test_parse_rejects_malformed(self):
        good = {"team": "Alpha", "from": day(1), "players": OLD, "kind": "stand_in"}
        self.assertIsNotNone(LF.parse_override(good))
        for bad in ({**good, "kind": "x"}, {**good, "from": "2025-13-01"},
                    {**good, "players": OLD[:4]}, {**good, "players": OLD[:4] + ["O1"]},
                    {**good, "until": day(1)}, {k: v for k, v in good.items() if k != "team"}):
            self.assertIsNone(LF.parse_override(bad), bad)

    def test_players_normalised_lowercase(self):
        o = LF.parse_override({"team": "Alpha", "from": day(1), "kind": "stand_in",
                               "players": [" O1", "O2", "o3", "o4", "o5"]})
        assert o is not None
        self.assertEqual(o["players"][0], "o1")


class Leakage(unittest.TestCase):
    def test_override_at_or_after_D_cannot_change_day_D(self):
        m, l = fixture()
        for i in range(2, 13):
            D = day(i)
            clean = feats(m, l, [], D)
            for later in (D, day(min(i + 1, 28)), "2025-04-15"):
                for kind, five in (("roster_change", NEW), ("stand_in", SUB)):
                    self.assertEqual(feats(m, l, [ov(later, five, kind)], D), clean,
                                     f"D={D} override={later} {kind}")

    def test_canary_override_before_D_is_visible(self):
        m, l = fixture()
        D = day(9)
        self.assertNotEqual(feats(m, l, [ov(day(8), SUB, "stand_in")], D), feats(m, l, [], D))

    def test_history_rows_before_D_unchanged_by_later_override(self):
        m, l = fixture()
        D = day(7)
        a = [r for r in LF.team_history(m, copy.deepcopy(l), [])["Alpha"] if r[0] < D]
        b = [r for r in LF.team_history(m, copy.deepcopy(l), [ov(D, NEW)])["Alpha"] if r[0] < D]
        self.assertEqual(a, b)


class CheckScript(unittest.TestCase):
    PLAYERS = set(OLD + NEW + ["s9"])
    TEAMS = {"Alpha", "Bravo"}

    def test_lists_misspelled_players_and_teams(self):
        raw = [{"team": "Alpah", "from": day(1), "players": ["o1", "o2", "o3", "o4", "s99"],
                "kind": "stand_in"}]
        errors, unknown = chk.check(raw, self.PLAYERS, self.TEAMS)
        self.assertEqual(errors, [])
        text = "\n".join(unknown)
        self.assertIn("unknown team 'Alpah'", text)
        self.assertIn("Alpha", text)
        self.assertIn("unknown player 's99'", text)
        self.assertIn("s9", text)

    def test_clean_file_and_structural_errors(self):
        good = {"team": "Alpha", "from": day(1), "players": NEW, "kind": "roster_change"}
        self.assertEqual(chk.check([good], self.PLAYERS, self.TEAMS), ([], []))
        errors, _ = chk.check([good, dict(good), {"team": "Alpha"}, {**good, "kind": "bogus"}],
                              self.PLAYERS, self.TEAMS)
        self.assertEqual(len(errors), 3)
        self.assertTrue(chk.check({}, set(), set())[0])

    def test_committed_file_is_valid_list(self):
        self.assertIsInstance(LF.load_overrides(), list)
        import json
        with open(LF.OVERRIDES_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual(chk.check(raw, set(), set())[0], [])


if __name__ == "__main__":
    unittest.main()
