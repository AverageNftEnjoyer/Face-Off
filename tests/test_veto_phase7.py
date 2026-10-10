"""Tests for veto phase 7: recency-decayed evidence, soft avoid score, roster-change
discount (CONFIG veto_decay_halflife / veto_soft_avoid / veto_roster_discount, all
off by default). Synthetic teams and maps only.

Run from the repo root:  python -m unittest tests.test_veto_phase7 -v
"""
import copy
import math
import os
import sys
import unittest
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor  # noqa: E402
import veto  # noqa: E402
from predictor import CONFIG  # noqa: E402

POOL = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
D0 = date(2025, 1, 1)


def series(i, day, a, b, maps):
    return {"id": i, "date": (D0 + timedelta(days=day)).isoformat(), "team_a": a, "team_b": b, "best_of": 3,
            "winner": a, "tier": "main", "maps": [{"map": m, "winner": a} for m in maps]}


class Cfg:
    def setUp(self):
        self._saved = copy.deepcopy(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self._saved)


class TestDefaultsOff(Cfg, unittest.TestCase):
    def test_flags_default_off(self):
        self.assertIsNone(CONFIG["veto_decay_halflife"])
        self.assertFalse(CONFIG["veto_soft_avoid"])
        self.assertFalse(CONFIG["veto_roster_discount"])
        self.assertFalse(CONFIG["veto_use_roster"])

    def test_evidence_unchanged_when_off(self):
        h = bt.History()
        h.add(series(1, 0, "A", "B", ["m1", "m2"]))
        ev = h.veto_evidence("A", D0 + timedelta(days=5), POOL)
        self.assertEqual(sorted(ev), ["maps", "n"])


class TestDecay(Cfg, unittest.TestCase):
    def test_weights_are_exponential(self):
        h = bt.History()
        h.add(series(1, 0, "A", "B", ["m1", "m2", "m3"]))
        h.add(series(2, 60, "A", "C", ["m2", "m3", "m4"]))
        CONFIG["veto_decay_halflife"] = 60
        dt = D0 + timedelta(days=120)
        ev = h.veto_evidence("A", dt, POOL)
        w = ev["wmaps"]
        self.assertAlmostEqual(w["m1"][0], 0.25)          # 120 days old = two half-lives
        self.assertAlmostEqual(w["m2"][0], 0.25 + 0.5)
        self.assertAlmostEqual(w["m4"][0], 0.5)
        self.assertEqual(w["m1"][4], 1)
        self.assertEqual(ev["maps"]["m1"][0], 0)               # baseline window evidence untouched (flat 90 days)

    def test_horizon_and_point_in_time(self):
        h = bt.History()
        h.add(series(1, 0, "A", "B", ["m1", "m2"]))
        CONFIG["veto_decay_halflife"] = 90
        CONFIG["veto_decay_horizon"] = 100
        self.assertEqual(h.veto_evidence("A", D0 + timedelta(days=150), POOL)["wmaps"]["m1"][0], 0.0)
        # a series on the day itself is invisible
        self.assertEqual(h.veto_evidence("A", D0, POOL)["wmaps"]["m1"][0], 0.0)

    def test_no_decay_weights_one_equals_baseline_counts(self):
        h = bt.History()
        for i in range(4):
            h.add(series(i, i * 3, "A", "B", ["m1", "m2"]))
        CONFIG["veto_soft_avoid"] = True
        ev = h.veto_evidence("A", D0 + timedelta(days=20), POOL)
        for mp in POOL:
            self.assertEqual(ev["wmaps"][mp][0], ev["maps"][mp][0])
            self.assertEqual(ev["wmaps"][mp][1], ev["maps"][mp][1])
            self.assertAlmostEqual(ev["wmaps"][mp][2], ev["maps"][mp][2])

    def test_weighted_exclusion(self):
        ev = {"maps": {"m1": [0, 10, -3.0], "m2": [1, 10, -3.0]},
              "wmaps": {"m1": [0.0, 8.0, -3.0, 2.0, 0], "m2": [0.2, 8.0, -3.0, 2.0, 1]}}
        alpha = 0.05
        self.assertEqual(veto.exclusion_set(ev, ["m1", "m2"], alpha), {"m1"})
        # a play whose weight fell under EXCL_PLAY_EPS no longer protects its map
        self.assertEqual(veto.exclusion_set(ev, ["m1", "m2"], alpha, weighted=True), {"m1", "m2"})
        ev["wmaps"]["m2"][0] = 0.6
        self.assertEqual(veto.exclusion_set(ev, ["m1", "m2"], alpha, weighted=True), {"m1"})


class TestSoftAvoid(Cfg, unittest.TestCase):
    EV = {"wmaps": {"m1": [0.0, 10.0, -2.0, 3.0, 0], "m2": [6.0, 10.0, -2.0, 3.0, 6], "m3": [3.0, 10.0, -2.0, 3.0, 3]}}

    def test_sign_and_shrinkage(self):
        CONFIG["veto_habit_k_ban"] = 2.0
        s = veto.avoid_scores(self.EV, ["m1", "m2", "m3"], CONFIG)
        self.assertGreater(s["ban1"]["m1"], 0)
        self.assertLess(s["ban1"]["m2"], 0)
        self.assertEqual(s["ban1"]["m3"], 0.0)
        # pick scores are shrunk 4x harder than ban scores
        self.assertEqual(veto.shrink_k(CONFIG), {"ban1": 2.0, "ban2": 2.0, "pick": 8.0})
        self.assertLess(abs(s["pick"]["m1"]), abs(s["ban1"]["m1"]))
        self.assertEqual(s["ban1"]["m1"], s["ban2"]["m1"])

    def test_default_shrinkage_is_habit_k(self):
        self.assertEqual(set(veto.shrink_k(CONFIG).values()), {CONFIG["veto_habit_k"]})

    def test_no_evidence_is_zero(self):
        s = veto.avoid_scores({}, POOL, CONFIG)
        self.assertTrue(all(v == 0.0 for k in s.values() for v in k.values()))

    def _ctx(self):
        CONFIG["veto_soft_avoid"] = True
        CONFIG["veto_habit_k_ban"] = 1.0
        ev = {"n": 8, "maps": {mp: [1, 8, -1.0] for mp in POOL}, "wmaps": {mp: [3.0, 8.0, -2.0, 3.0, 3] for mp in POOL}}
        ev["wmaps"]["m1"] = [0.0, 8.0, -2.0, 3.0, 0]
        m = {"team_a": "A", "team_b": "B", "rating_a": 1.0, "rating_b": 1.0, "map_pool": list(POOL),
             "maps_a": {}, "maps_b": {}, "veto_ev_a": copy.deepcopy(ev), "veto_ev_b": copy.deepcopy(ev)}
        return m

    def test_soft_replaces_the_mask_and_moves_the_policy(self):
        m = self._ctx()
        ctx = predictor.veto_context(m, 3)
        self.assertEqual(ctx.masked, {"a": [], "b": []})      # no binary exclusion
        self.assertTrue(ctx.has_avoid)
        w = {"lookahead": False, "pi": 0.5, "temp": 1.0}
        t0 = veto.Tree(ctx, w, "a")
        w1 = dict(w, zeta_pick=1.0, zeta_ban1=1.0, zeta_ban2=1.0)
        t1 = veto.Tree(ctx, w1, "a")
        for t, kind in ((0, "ban1"),):
            p0 = {i: p for i, p, _ in t0.node(t, ctx.full, ())[1]}
            p1 = {i: p for i, p, _ in t1.node(t, ctx.full, ())[1]}
            self.assertGreater(p1[0], p0[0])                  # m1 (index 0) is banned more
        pick_step = next(t for t, (r, k) in enumerate(ctx.steps) if k == "pick")
        st = (pick_step, ctx.full & ~1, ())
        # uniform-ish state: the avoided map is picked less once its weight is on
        a = {i: p for i, p, _ in t0.node(*st)[1]}
        b = {i: p for i, p, _ in t1.node(*st)[1]}
        self.assertTrue(all(p > 0 for p in b.values()))
        full = (pick_step, ctx.full, ())
        a = {i: p for i, p, _ in t0.node(*full)[1]}
        b = {i: p for i, p, _ in t1.node(*full)[1]}
        self.assertLess(b[0], a[0])

    def test_zero_weights_are_bit_identical_to_no_avoid(self):
        m = self._ctx()
        ctx = predictor.veto_context(m, 3)
        w = {"lookahead": True, "pi": 0.4, "temp": 1.0, "alpha_ban1": 1.0, "gamma_ban1": 2.0, "alpha_ban2": 1.0,
             "gamma_ban2": 1.0, "alpha_pick": 1.0, "gamma_pick": 1.0}
        d0 = veto.distribution(ctx, w)[0]
        d1 = veto.distribution(ctx, dict(w, zeta_pick=0.0, zeta_ban1=0.0, zeta_ban2=0.0))[0]
        self.assertEqual(d0["outcomes"], d1["outcomes"])
        obs = ["m2", "m3", "m4"]
        self.assertEqual(veto.loglik(ctx, w, obs)[0],
                         veto.loglik(ctx, dict(w, zeta_pick=0.0, zeta_ban1=0.0), obs)[0])
        self.assertNotEqual(veto.loglik(ctx, w, obs)[0], veto.loglik(ctx, dict(w, zeta_pick=1.0), obs)[0])

    def test_flag_without_weighted_evidence_raises(self):
        CONFIG["veto_soft_avoid"] = True
        m = {"team_a": "A", "team_b": "B", "rating_a": 1.0, "rating_b": 1.0, "map_pool": list(POOL),
             "maps_a": {}, "maps_b": {}, "veto_ev_a": {"n": 0, "maps": {}}, "veto_ev_b": {"n": 0, "maps": {}}}
        with self.assertRaises(ValueError):
            predictor.veto_context(m, 3)
        # the plain context (used to label past vetoes) ignores the options
        predictor.veto_context(m, 3, legacy=True)


class TestRosterDiscount(Cfg, unittest.TestCase):
    def hist(self, fives, start=0, gap=5):
        """History with one team's series; fives[i] is its lineup in series i."""
        h = bt.History()
        h.lineups = {}
        for i, five in enumerate(fives):
            m = series(i, start + i * gap, "A", "B", ["m1", "m2"])
            h.lineups[f"{m['date']}|A|B"] = {"a": list(five), "b": ["x1", "x2", "x3", "x4", "x5"]}
            h.add(m)
        return h

    F0 = ["p1", "p2", "p3", "p4", "p5"]
    F2 = ["p1", "p2", "p3", "q4", "q5"]
    F1 = ["p1", "p2", "p3", "p4", "q5"]

    def test_factor_floor_then_recovers(self):
        fives = [self.F0] * 4 + [self.F2] * 12
        h = self.hist(fives)
        out = []
        for k in range(1, 12):
            dt = D0 + timedelta(days=(3 + k) * 5 + 1)         # after the k-th series with the new five
            ev, stale = h.roster_events("A", dt, 0.4, 8)
            self.assertFalse(stale)
            out.append(ev[0][1] if ev else 1.0)
        self.assertAlmostEqual(out[0], 0.4)                    # right after a 2-player change
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(out, out[1:])))
        self.assertEqual(out[8], 1.0)                          # full weight from the 9th series
        self.assertEqual(out[10], 1.0)

    def test_one_player_change_is_milder(self):
        h = self.hist([self.F0] * 3 + [self.F1] * 2)
        ev, _ = h.roster_events("A", D0 + timedelta(days=100), 0.4, 8)
        self.assertEqual(len(ev), 1)
        self.assertGreater(ev[0][1], 0.4 + 0.2)               # 1 of 2+ players: half the drop, minus recovery
        self.assertLess(ev[0][1], 1.0)

    def test_stable_roster_no_events(self):
        h = self.hist([self.F0] * 6)
        self.assertEqual(h.roster_events("A", D0 + timedelta(days=100), 0.4, 8), ([], False))

    def test_stale_lineup_skips_discount(self):
        h = self.hist([self.F0] * 3 + [self.F2] * 2)
        # a later series without lineup data makes the last lineup stale
        h.add(series(99, 60, "A", "C", ["m1", "m2"]))
        ev, stale = h.roster_events("A", D0 + timedelta(days=100), 0.4, 8)
        self.assertTrue(stale)
        self.assertEqual(ev, [])
        self.assertEqual(h.roster_events("Z", D0 + timedelta(days=100), 0.4, 8), ([], True))

    def test_only_older_records_are_discounted_and_evidence_uses_it(self):
        h = self.hist([self.F0] * 4 + [self.F2])
        CONFIG["veto_roster_discount"] = True
        CONFIG["veto_roster_floor"] = 0.4
        dt = D0 + timedelta(days=60)
        ev = h.veto_evidence("A", dt, POOL)
        # 5 series each playing m1: four old (weight 0.4) and one new (weight 1)
        self.assertAlmostEqual(ev["wmaps"]["m1"][0], 4 * 0.4 + 1.0)
        self.assertEqual(ev["wmaps"]["m1"][4], 5)
        self.assertFalse(ev["roster_stale"])

    def test_point_in_time(self):
        a = self.hist([self.F0] * 3 + [self.F2] * 2)
        dt = D0 + timedelta(days=22)                          # between series 4 (day 20) and a later one
        e1 = a.roster_events("A", dt, 0.4, 8)
        b = self.hist([self.F0] * 3 + [self.F2] * 2)
        later = series(50, 30, "A", "C", ["m1", "m2"])        # lineups after dt change nothing before dt
        b.lineups[f"{later['date']}|A|C"] = {"a": ["z1", "z2", "z3", "z4", "z5"], "b": self.F0}
        b.add(later)
        self.assertEqual(b.roster_events("A", dt, 0.4, 8), e1)

    def test_stale_warning_reaches_the_veto_output(self):
        CONFIG["veto_roster_discount"] = True
        ev = {"n": 3, "maps": {mp: [1, 3, -1.0] for mp in POOL}, "wmaps": {mp: [1.0, 3.0, -1.0, 1.0, 1] for mp in POOL},
              "roster_stale": True}
        m = {"team_a": "A", "team_b": "B", "rating_a": 1.0, "rating_b": 1.0, "map_pool": list(POOL),
             "maps_a": {}, "maps_b": {}, "veto_ev_a": ev, "veto_ev_b": dict(ev, roster_stale=False)}
        ctx = predictor.veto_context(m, 3)
        self.assertIn("lineup_stale_a", ctx.warnings)
        self.assertNotIn("lineup_stale_b", ctx.warnings)
        self.assertFalse(CONFIG["veto_use_roster"])


if __name__ == "__main__":
    unittest.main()
