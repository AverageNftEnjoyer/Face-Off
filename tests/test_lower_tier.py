"""Lower-tier results (data/matches_lower.json): history order and weight
(backtest.py), team-page labelling (viewer/build_viewer.py), the daily refresh
step (scripts/lower_refresh.py) and its failure handling. Everything is
SYNTHETIC: made-up teams, temporary files, no network."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import date
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "data", "scripts", "viewer"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import backtest as bt  # noqa: E402
import build_viewer as bv  # noqa: E402
import daily_refresh as dr  # noqa: E402
import data_gates as G  # noqa: E402
import lower_refresh as LR  # noqa: E402


def sa(i, d, a, b, w=None, bo=3):
    return {"id": i, "match_id": f"{i:012x}", "date": d, "event": "Alpha Cup", "team_a": a, "team_b": b,
            "winner": w or a, "best_of": bo, "maps": [], "event_page": "Alpha Cup", "tier": "S"}


def lo(i, d, a, b, w=None, tier="B", kind="main", bo=3):
    return {"id": ("L", i), "match_id": f"{1000 + i:012x}", "date": d, "event": "Beta Open", "team_a": a,
            "team_b": b, "winner": w or a, "best_of": bo, "maps": [], "event_page": "Beta Open",
            "tier": tier, "kind": kind}


class WeightCase(unittest.TestCase):
    def setUp(self):
        self.saved = (bt.LOWER_TIER_ELO_WEIGHT, bt.LOWER_TIER_MODE)

    def tearDown(self):
        bt.LOWER_TIER_ELO_WEIGHT, bt.LOWER_TIER_MODE = self.saved


class TestHistory(WeightCase):
    def season(self):
        top = [sa(0, "2026-01-01", "Ant", "Bee"), sa(1, "2026-01-02", "Cat", "Dog"),
               sa(2, "2026-01-05", "Ant", "Cat"), sa(3, "2026-01-06", "Bee", "Dog")]
        low = [lo(0, "2026-01-03", "Ant", "Eel"), lo(1, "2026-01-05", "Ant", "Fox"),
               lo(2, "2026-01-05", "Eel", "Fox")]
        return top, low

    def test_weight_scales_the_elo_step(self):
        bt.LOWER_TIER_ELO_WEIGHT, bt.LOWER_TIER_MODE = 0.25, "elo"
        h = bt.History()
        self.assertTrue(h.add_lower(lo(0, "2026-01-03", "Ant", "Eel", bo=1)))
        na, nb = bt.elo_update(bt.ELO_INIT, bt.ELO_INIT, True, 1, k=bt.ELO_K * 0.25)
        self.assertAlmostEqual(h.elo["Ant"], na)
        self.assertAlmostEqual(h.elo["Eel"], nb)
        self.assertAlmostEqual(h.elo["Ant"] - bt.ELO_INIT, 0.25 * bt.ELO_K * bt.ELO_K_BY_BO[1] * 0.5)

    def test_elo_only_nothing_else_moves(self):
        bt.LOWER_TIER_ELO_WEIGHT = 0.25
        h = bt.History()
        h.add_lower(lo(0, "2026-01-03", "Ant", "Eel"))
        self.assertEqual(h.n_prior("Ant"), 0)                 # prior S/A counts stay S/A-only
        self.assertEqual(h.vseries, [])                       # veto evidence untouched
        self.assertIsNone(h.first_day)                        # newcomer clock untouched
        f = h.team_feats("Ant", date(2026, 1, 10))
        self.assertNotIn("form30", f)
        self.assertNotIn("form5", f)
        self.assertEqual(h.lower_n["Ant"], 1)

    def test_weight_zero_is_off(self):
        bt.LOWER_TIER_ELO_WEIGHT = 0.0
        h = bt.History()
        self.assertFalse(h.add_lower(lo(0, "2026-01-03", "Ant", "Eel")))
        self.assertNotIn("Ant", h.elo)
        top, low = self.season()
        strip = lambda rows: [(r["input"], r["meta"]) for r in rows]  # noqa: E731
        self.assertEqual(strip(bt.build_dataset(top, low)), strip(bt.build_dataset(top)))
        self.assertEqual(bt.load_lower(), [])                 # off: nothing loaded for the history

    def test_linked_mode_needs_a_team_with_s_a_history(self):
        bt.LOWER_TIER_ELO_WEIGHT, bt.LOWER_TIER_MODE = 0.25, "linked"
        h = bt.History()
        self.assertFalse(h.add_lower(lo(0, "2026-01-03", "Eel", "Fox")))
        h.add(sa(0, "2026-01-01", "Ant", "Bee"))
        self.assertTrue(h.add_lower(lo(1, "2026-01-03", "Ant", "Eel")))
        self.assertNotIn("Eel", h.games)                      # the check must not create entries

    def test_day_order_same_day_lower_is_not_seen(self):
        bt.LOWER_TIER_ELO_WEIGHT = 0.25
        top, low = self.season()
        rows = {r["match"]["id"]: r for r in bt.build_dataset(top, low)}
        base = {r["match"]["id"]: r for r in bt.build_dataset(top)}
        # 2026-01-05 Ant vs Cat: Ant's lower series of 01-03 is in, its 01-05 one is not
        h = bt.History()
        h.add(top[0])
        h.add(top[1])
        h.add_lower(low[0])
        self.assertAlmostEqual(rows[2]["meta"]["elo_a"], h.elo["Ant"])
        self.assertNotEqual(rows[2]["meta"]["elo_a"], base[2]["meta"]["elo_a"])
        # the same-day lower series (01-05) only reaches the next day
        moved = [dict(m, winner=m["team_b"]) if m["date"] == "2026-01-05" else m for m in low]
        again = {r["match"]["id"]: r for r in bt.build_dataset(top, moved)}
        self.assertEqual(again[2]["meta"], rows[2]["meta"])
        self.assertEqual(again[2]["input"], rows[2]["input"])

    def test_day_groups_put_s_a_first(self):
        top, low = self.season()
        days = bt.day_groups(top, low)
        self.assertEqual([d for d, _, _ in days], ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05",
                                                   "2026-01-06"])
        d5 = dict((d, (s, l_)) for d, s, l_ in days)["2026-01-05"]
        self.assertEqual([m["id"] for m in d5[0]], [2])
        self.assertEqual([m["id"] for m in d5[1]], [("L", 1), ("L", 2)])

    def test_replay_matches_build_dataset(self):
        bt.LOWER_TIER_ELO_WEIGHT = 0.25
        top, low = self.season()
        h = bt.Replay(top, low).until(date(2026, 1, 6))
        rows = {r["match"]["id"]: r for r in bt.build_dataset(top, low)}
        self.assertAlmostEqual(h.elo["Bee"], rows[3]["meta"]["elo_a"])
        self.assertAlmostEqual(h.elo["Dog"], rows[3]["meta"]["elo_b"])
        steps = []
        bt.Replay(top, low, on_step=lambda hh, m, is_low: steps.append((m["id"], is_low))).all()
        self.assertEqual(steps, [(0, False), (1, False), (("L", 0), True), (2, False), (("L", 1), True),
                                 (("L", 2), True), (3, False)])


class TestReadLower(unittest.TestCase):
    def test_malformed_records_skipped_and_missing_file_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "l.json")
            recs = [lo(0, "2026-01-03", "Ant", "Eel"), {"date": "bad", "team_a": "A", "team_b": "B", "winner": "A"},
                    dict(lo(1, "2026-01-02", "Ant", "Fox"), winner="Nobody"), lo(2, "2026-01-01", "Cat", "Eel")]
            with open(p, "w", encoding="utf-8") as f:
                json.dump(recs, f)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                got = bt.read_lower(p)
                none = bt.read_lower(os.path.join(d, "missing.json"))
            self.assertEqual([m["id"] for m in got], [("L", 3), ("L", 0)])    # date order, bad ones out
            self.assertIn("2 malformed", err.getvalue())
            self.assertEqual(none, [])
            self.assertIn("WARNING", err.getvalue())


class TestDisplay(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(bt.lower_label({"tier": "B", "kind": "main"}), "B-tier")
        self.assertEqual(bt.lower_label({"tier": "A", "kind": "qualifier"}), "A-tier qualifier")
        self.assertEqual(bt.lower_label({"tier": "", "kind": "main"}), "Lower tier")

    def test_team_results_labelled_newest_first_before_cutoff(self):
        low = [lo(0, "2026-01-03", "Ant", "Eel"), lo(1, "2026-01-04", "Fox", "Ant", w="Ant", kind="qualifier"),
               lo(2, "2026-01-09", "Ant", "Gnu")]
        low[0]["maps"] = [{"map": "Nuke", "winner": "Ant"}, {"map": "Anubis", "winner": "Eel"},
                          {"map": "Mirage", "winner": "Ant"}]
        low[1]["maps"] = [{"map": "Nuke", "winner": "Ant"}, {"map": "Mirage", "winner": "Ant"}]
        res = bv.lower_results(low, date(2026, 1, 9), 10)
        self.assertEqual([r["date"] for r in res["Ant"]], ["2026-01-04", "2026-01-03"])   # 01-09 is not before
        self.assertEqual(res["Ant"][0], {"date": "2026-01-04", "opp": "Fox", "won": True, "score": "2-0",
                                         "event": "Beta Open", "tier": "B-tier qualifier"})
        self.assertEqual(res["Fox"][0]["score"], "0-2")
        self.assertEqual(res["Eel"][0]["score"], "1-2")
        self.assertEqual(res["Eel"][0]["tier"], "B-tier")
        self.assertEqual(len(bv.lower_results(low, date(2026, 2, 1), 1)["Ant"]), 1)

    def test_template_labels_lower_rows_and_never_lists_them_as_events(self):
        with open(os.path.join(ROOT, "viewer", "template.html"), encoding="utf-8") as f:
            tpl = f.read()
        self.assertIn("function teamResults(t)", tpl)
        self.assertIn('class="chip plain lt"', tpl)
        # lower-tier rows are plain text: no event link is built from them
        body = tpl[tpl.index("function teamResults(t)"):tpl.index("function lowerNote()")]
        self.assertNotIn("#e-", body)
        for brand in ("Liquipedia", "HLTV"):
            self.assertNotIn(brand, tpl[tpl.index("function teamResults(t)"):tpl.index("function spark(")])


class TestRefreshStep(unittest.TestCase):
    def test_refresh_failure_is_a_warning(self):
        for exc in (RuntimeError("boom"), SystemExit("offline: 3 team codes not in cache")):
            def bad():
                raise exc
            err, out = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                got = dr.refresh_lower(run=bad)
            self.assertIsNone(got)
            self.assertIn("WARNING lower tier: refresh skipped", err.getvalue())

    def test_refresh_warnings_are_printed_not_raised(self):
        err, out = io.StringIO(), io.StringIO()
        rep = {"requests": 3, "by_purpose": {}, "warnings": ["budget reached"], "series": 5}
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            got = dr.refresh_lower(run=lambda: rep)
        self.assertIs(got, rep)
        self.assertIn("WARNING lower tier: budget reached", err.getvalue())
        self.assertIn("Liquipedia requests 3", out.getvalue())

    def _run(self, records, old):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "matches_lower.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(old, f)
            import collect_liquipedia as C
            with mock.patch.object(LR, "discover_events", return_value=([], [])), \
                    mock.patch.object(LR, "discover_qualifiers", return_value=[]), \
                    mock.patch.object(LR, "refresh_live", return_value=([], [], [])), \
                    mock.patch.object(LR, "write_titles", return_value=0), \
                    mock.patch.object(LR.lpfetch, "_THROTTLED", [False]), \
                    mock.patch.object(C, "build_lower", return_value=records), \
                    contextlib.redirect_stdout(io.StringIO()):
                rep = LR.run(today=date(2026, 1, 10), out_path=out)
            with open(out, encoding="utf-8") as f:
                return rep, json.load(f)

    @staticmethod
    def plain(m):
        return {k: v for k, v in m.items() if k != "id"}

    def test_valid_rebuild_is_written(self):
        old = [self.plain(lo(0, "2026-01-03", "Ant", "Eel"))]
        new = old + [self.plain(lo(1, "2026-01-04", "Ant", "Fox"))]
        rep, got = self._run(new, old)
        self.assertEqual(got, new)
        self.assertTrue(rep.get("written"))
        self.assertEqual(rep["added_series"], 1)
        self.assertEqual(rep["warnings"], [])

    def test_bad_schema_keeps_the_old_file(self):
        old = [self.plain(lo(0, "2026-01-03", "Ant", "Eel"))]
        bad = old + [dict(self.plain(lo(1, "2026-01-04", "Ant", "Fox")), winner="Nobody")]
        rep, got = self._run(bad, old)
        self.assertEqual(got, old)
        self.assertIn("schema check", rep["warnings"][0])

    def test_collapse_keeps_the_old_file(self):
        old = [self.plain(lo(i, "2026-01-03", f"T{i}", "Eel")) for i in range(100)]
        rep, got = self._run(old[:50], old)
        self.assertEqual(got, old)
        self.assertIn("unchanged", rep["warnings"][0])

    def test_only_lower_tiers_are_events(self):
        years = ("2026", "2027")
        self.assertTrue(LR.lower_main_ok(("B", "", "2026-03-01", "2026-03-20"), years))
        self.assertFalse(LR.lower_main_ok(("A", "", "2026-03-01", "2026-03-20"), years))
        self.assertFalse(LR.lower_main_ok(("B", "Qualifier", "2026-03-01", "2026-03-20"), years))
        self.assertFalse(LR.lower_main_ok(("B", "", "2025-03-01", "2025-03-20"), years))
        self.assertFalse(LR.lower_main_ok(("B", "", "2026-01-01", "2026-12-01"), years))   # > LOWER_MAX_DAYS

    def test_gate_reports_never_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "l.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump([dict(self.plain(lo(0, "2026-01-03", "Ant", "Eel")), kind="weird")], f)
            rep = G.gate_lower(p)
            self.assertTrue(any("unknown kind" in w for w in rep["warnings"]))
            self.assertIn("note", G.gate_lower(os.path.join(d, "none.json")))


if __name__ == "__main__":
    unittest.main()
