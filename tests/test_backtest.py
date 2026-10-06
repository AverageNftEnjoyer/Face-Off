"""Unit tests for backtest.py.

All fixtures built in this file are SYNTHETIC (made-up teams "Alpha", "Bravo",
...). They exist only to test the code paths and are never written to data/.
"""
import math
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import backtest as bt  # noqa: E402


def synth(date, a, b, winner, maps=None, bo=3, i=0):
    """SYNTHETIC match record (not real data)."""
    return {"id": i, "date": date, "event": "SYNTHETIC", "team_a": a, "team_b": b,
            "winner": winner, "best_of": bo,
            "maps": maps if maps is not None else [{"map": "Mirage", "winner": winner},
                                                   {"map": "Nuke", "winner": winner}],
            "source": "synthetic-test-fixture"}


def synthetic_season():
    """SYNTHETIC: 4 fake teams, round-robin-ish results over ~40 days."""
    teams = ["Alpha", "Bravo", "Charlie", "Delta"]
    rng = random.Random(7)
    ms, i = [], 0
    for day in range(1, 41):
        a, b = rng.sample(teams, 2)
        w = a if rng.random() < 0.6 else b
        ms.append(synth(f"2025-01-{day:02d}" if day <= 31 else f"2025-02-{day - 31:02d}", a, b, w, i=i))
        i += 1
    # two series on the same day to test same-day isolation
    ms.append(synth("2025-02-09", "Alpha", "Bravo", "Alpha", i=i))
    return ms


class TestNoLeakage(unittest.TestCase):
    def test_features_ignore_same_day_and_future(self):
        ms = synthetic_season()
        full = {r["match"]["id"]: r for r in bt.build_dataset(ms)}
        for target in ms:
            truncated = [m for m in ms if m["date"] < target["date"]] + [target]
            trunc = {r["match"]["id"]: r for r in bt.build_dataset(truncated)}
            self.assertEqual(full[target["id"]]["input"], trunc[target["id"]]["input"],
                             f"features for {target['id']} depend on same-day/later series")
            self.assertEqual(full[target["id"]]["meta"], trunc[target["id"]]["meta"])

    def test_flipping_future_results_changes_nothing(self):
        ms = synthetic_season()
        cut = "2025-01-20"
        flipped = [dict(m, winner=(m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]))
                   if m["date"] >= cut else m for m in ms]
        f1 = {r["match"]["id"]: r["input"] for r in bt.build_dataset(ms)}
        f2 = {r["match"]["id"]: r["input"] for r in bt.build_dataset(flipped)}
        for m in ms:
            if m["date"] <= cut:   # features of the cut day itself must not see that day either
                self.assertEqual(f1[m["id"]], f2[m["id"]])

    def test_first_match_has_no_history(self):
        ms = [synth("2025-01-01", "Alpha", "Bravo", "Alpha"),
              synth("2025-01-02", "Alpha", "Bravo", "Bravo", i=1)]
        rows = bt.build_dataset(ms)
        r0 = rows[0]
        self.assertEqual(r0["input"]["rating_a"], 1.0)
        self.assertEqual(r0["input"]["h2h"]["meetings"], 0)
        self.assertNotIn("form30_a", r0["input"])
        self.assertEqual(r0["meta"]["hist_a"], 0)
        r1 = rows[1]
        self.assertEqual(r1["input"]["h2h"], {"a_wins": 1, "b_wins": 0, "meetings": 1})
        self.assertEqual(r1["input"]["form30_a"], 1.0)
        self.assertEqual(r1["input"]["maps_a"], {"Mirage": [1.0, 1], "Nuke": [1.0, 1]})

    def test_windows(self):
        ms = [synth("2025-01-01", "Alpha", "Bravo", "Alpha"),
              synth("2025-03-01", "Alpha", "Bravo", "Bravo", i=1),
              synth("2025-03-15", "Alpha", "Bravo", "Alpha", i=2)]
        r = bt.build_dataset(ms)[2]["input"]
        # 30-day window only sees the 2025-03-01 series
        self.assertEqual(r["n30_a"], 1)
        self.assertEqual(r["form30_a"], 0.0)
        # 90-day map window sees both earlier series (73 and 14 days back)
        self.assertEqual(r["maps_a"]["Mirage"], [0.5, 2])
        self.assertEqual(r["form5_a"], 0.5)


class TestElo(unittest.TestCase):
    def test_expected(self):
        self.assertAlmostEqual(bt.elo_expected(1500, 1500), 0.5)
        self.assertAlmostEqual(bt.elo_expected(1900, 1500), 1 / (1 + 10 ** -1))

    def test_update_hand_computed(self):
        a, b = bt.elo_update(1500, 1500, True, best_of=3, k=32)
        self.assertAlmostEqual(a, 1516.0)
        self.assertAlmostEqual(b, 1484.0)
        # upset: 1400 beats 1600; E = 1/(1+10^0.5) = 0.240253
        a, b = bt.elo_update(1400, 1600, True, best_of=3, k=32)
        self.assertAlmostEqual(a, 1400 + 32 * (1 - 0.2402530733520421), places=6)
        self.assertAlmostEqual(a + b, 3000.0)
        # BO1 uses a smaller step
        a1, _ = bt.elo_update(1500, 1500, True, best_of=1, k=32)
        self.assertAlmostEqual(a1, 1500 + 32 * bt.ELO_K_BY_BO[1] * 0.5)

    def test_rating_map(self):
        self.assertAlmostEqual(bt.elo_to_rating(1500), 1.0)
        self.assertAlmostEqual(bt.elo_to_rating(1500 + bt.ELO_PER_RATING * 0.05), 1.05)


class TestSplit(unittest.TestCase):
    def test_chronological_and_day_safe(self):
        rows = [{"match": {"date": d}} for d in
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-03",
                 "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08"]]
        tr, te = bt.chrono_split(rows, 0.4)    # k=4 lands inside 01-03 -> moves to 5
        self.assertEqual(len(tr), 5)
        self.assertTrue(max(r["match"]["date"] for r in tr) < min(r["match"]["date"] for r in te))
        self.assertEqual(len(tr) + len(te), len(rows))


class TestTemperature(unittest.TestCase):
    def test_recovers_known_temperature(self):
        rng = random.Random(2024)
        true_t = 0.6
        xs = [rng.uniform(-4, 4) for _ in range(20000)]
        ys = [1 if rng.random() < bt.sigmoid(true_t * x) else 0 for x in xs]
        t = bt.fit_temperature(xs, ys)
        self.assertAlmostEqual(t, true_t, delta=0.05)


class TestMetrics(unittest.TestCase):
    def test_brier_logloss(self):
        ps, ys = [0.8, 0.3, 0.5], [1, 1, 0]
        self.assertAlmostEqual(bt.brier(ps, ys), (0.04 + 0.49 + 0.25) / 3)
        self.assertAlmostEqual(bt.log_loss(ps, ys),
                               -(math.log(0.8) + math.log(0.3) + math.log(0.5)) / 3)

    def test_accuracy_ties(self):
        self.assertAlmostEqual(bt.accuracy([0.9, 0.2, 0.5], [1, 1, 0]), 1.5 / 3)

    def test_ece(self):
        # bin 0.1-0.2: preds .15,.15 actual 0,1 -> |.15-.5|=.35 ; bin 0.9-1.0: .95 actual 1 -> .05
        ps, ys = [0.15, 0.15, 0.95], [0, 1, 1]
        self.assertAlmostEqual(bt.ece(ps, ys), (2 / 3) * 0.35 + (1 / 3) * 0.05)
        # p=1.0 must land in the top bin, not overflow
        self.assertEqual(bt.reliability_table([1.0], [1])[-1]["n"], 1)

    def test_wilson(self):
        lo, hi = bt.wilson(50, 100)
        # hand-computed: 0.5 +- 1.96*sqrt(.25/100+ 1.96^2/40000)/(1+1.96^2/100)
        z = 1.959964
        den = 1 + z * z / 100
        h = z * math.sqrt(0.25 / 100 + z * z / 40000) / den
        self.assertAlmostEqual(lo, 0.5 - h)
        self.assertAlmostEqual(hi, 0.5 + h)
        self.assertAlmostEqual(lo, 0.4038, places=3)

    def test_bootstrap_seeded(self):
        ps, ys = [0.6, 0.4, 0.7, 0.2], [1, 0, 0, 0]
        self.assertEqual(bt.bootstrap_ci(bt.brier, ps, ys, n_boot=200, seed=1),
                         bt.bootstrap_ci(bt.brier, ps, ys, n_boot=200, seed=1))


class TestCollectorFinishedTypos(unittest.TestCase):
    """SYNTHETIC wikitext snippets (made-up teams "alpha"/"bravo"), modelled on
    editor typos in the real cache: finished=t / trur / trie / y / reyw."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT, "data"))
        import collect_liquipedia
        cls.c = collect_liquipedia

    def test_typo_flags_count_as_finished(self):
        for fin in ("true", "t", "trur", "trie", "y", "reyw", "TRUE"):
            snip = "{{Map|map=Mirage|finished=%s|t1t=4|t1ct=3|t2t=8|t2ct=5}}" % fin
            self.assertEqual(self.c.parse_map(snip), {"map": "Mirage", "w": 2}, fin)

    def test_unplayed_maps_still_dropped(self):
        self.assertIsNone(self.c.parse_map("{{Map|map=Anubis|finished=skip}}"))
        self.assertIsNone(self.c.parse_map("{{Map|map=Anubis|finished=}}"))
        self.assertIsNone(self.c.parse_map("{{Map|map=Anubis|finished=false|t1t=1|t1ct=0|t2t=0|t2ct=0}}"))
        # a typo flag alone is not enough: a winner needs round scores or winner=
        self.assertIsNone(self.c.parse_map("{{Map|map=Anubis|finished=t}}"))
        self.assertEqual(self.c.parse_map("{{Map|map=Dust II|finished=trie|winner=1}}"),
                         {"map": "Dust2", "w": 1})

    def test_series_keeps_typo_map(self):
        # SYNTHETIC match: map2 has finished=t (typo) -> must give a 2-1, not 2-0
        text = """{{Match
 |opponent1={{TeamOpponent|alpha}}|opponent2={{TeamOpponent|bravo}}
 |date=August 1, 2025 - 12:00 {{Abbr/CEST}} |finished=true
 |map1={{Map|map=Train|finished=true|t1t=7|t1ct=6|t2t=5|t2ct=3}}
 |map2={{Map|map=Mirage|finished=t|t1t=4|t1ct=3|t2t=8|t2ct=5}}
 |map3={{Map|map=Inferno|finished=true|t1t=8|t1ct=5|t2t=4|t2ct=4}}
 }}"""
        out = self.c.parse_page("SYNTHETIC", text)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["winner"], 1)
        self.assertEqual([m["w"] for m in out[0]["maps"]], [1, 2, 1])

    def test_showmatch_section_skipped(self):
        text = """==Results==
{{Match|opponent1={{TeamOpponent|alpha}}|opponent2={{TeamOpponent|bravo}}|date=June 1, 2024|finished=true
 |map1={{Map|map=Nuke|finished=true|t1t=7|t1ct=6|t2t=5|t2ct=3}}}}
=={{Stage|Showmatch}}==
{{Match|opponent1={{TeamOpponent|legends}}|opponent2={{TeamOpponent|stars}}|date=June 2, 2024|finished=y
 |map1={{Map|map=Nuke|finished=y|t1t=7|t1ct=6|t2t=5|t2ct=3}}}}
"""
        out = self.c.parse_page("SYNTHETIC", text)
        self.assertEqual([(m["c1"], m["c2"]) for m in out], [("alpha", "bravo")])


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data", "matches.json")), "no data/matches.json")
class TestRealDataSmoke(unittest.TestCase):
    def test_dataset_well_formed(self):
        ms = bt.load_matches()
        self.assertGreater(len(ms), 300)
        for m in ms:
            self.assertIn(m["winner"], (m["team_a"], m["team_b"]))
            self.assertTrue(m["source"].startswith("liquipedia:"))
            if m["best_of"] == 3 and m["maps"]:
                self.assertIn(bt.actual_scoreline(m), ("2-0", "2-1", "1-2", "0-2"))
        dates = [m["date"] for m in ms]
        self.assertEqual(dates, sorted(dates))

    def test_pipeline_runs(self):
        rows = bt.build_dataset(bt.load_matches())
        ev, _ = bt.select_eval(rows)
        self.assertGreater(len(ev), 300)
        tr, te = bt.chrono_split(ev)
        self.assertLess(tr[-1]["match"]["date"], te[0]["match"]["date"])
        new, old, errs = bt.load_engines()
        if old is not None:
            r = old.predict_match(dict(te[0]["input"]))
            self.assertTrue(0 <= r["p_a"] <= 1)
        if new is not None:
            r = new.predict_match(dict(te[0]["input"]))
            self.assertTrue(0 <= r["p_a"] <= 1)


if __name__ == "__main__":
    unittest.main()
