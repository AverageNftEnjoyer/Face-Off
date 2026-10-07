"""Unit tests for predictor.py (stdlib unittest).

Run from the repo root:  python -m unittest discover -s tests -v
"""
import copy
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predictor  # noqa: E402
from predictor import CONFIG, display_percent, predict_match, series_probs, total_logodds  # noqa: E402

# Mechanism tests pin the reasoned reference weights: they check HOW the
# machinery behaves (veto scaling, shrinkage, warnings, tiers), which needs
# every factor switched on. The shipped weights are fitted (fit_weights.py)
# and can legitimately set a factor to 0, which would make these checks vacuous.
REFERENCE_WEIGHTS = {"w_base": 0.8, "w_form30": 0.7, "w_form5": 0.25, "w_h2h": 0.3,
                     "w_veto": 0.75, "temperature": 0.909,
                     "rel_high_width_pp": 34.4, "rel_med_width_pp": 37.5}


class ReferenceWeights:
    def setUp(self):
        self._saved = copy.deepcopy(CONFIG)
        CONFIG.update(REFERENCE_WEIGHTS)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self._saved)

POOL = CONFIG["map_pool"]


def base_match(**kw):
    """A complete, plausible match with SYMMETRIC (no) map data."""
    m = {
        "team_a": "Alpha", "team_b": "Bravo",
        "rating_a": 1.04, "rating_b": 1.00,
        "form30_a": 0.60, "form30_b": 0.50, "n30_a": 14, "n30_b": 12,
        "form5_a": 0.60, "form5_b": 0.40,
        "h2h": {"a_wins": 3, "b_wins": 2, "meetings": 5},
        "stakes_a": "elimination", "stakes_b": "qualification",
        "roster_a": {}, "roster_b": {"standin": True},
        "volatility_a": 0.3, "volatility_b": 0.4,
    }
    m.update(kw)
    return m


def swap(m):
    """Mirror a match: B becomes A."""
    pairs = [("team_a", "team_b"), ("rating_a", "rating_b"), ("form30_a", "form30_b"),
             ("n30_a", "n30_b"), ("form5_a", "form5_b"), ("maps_a", "maps_b"),
             ("permaban_a", "permaban_b"), ("roster_a", "roster_b"),
             ("stakes_a", "stakes_b"), ("volatility_a", "volatility_b"),
             ("opp_rating30_a", "opp_rating30_b")]
    out = copy.deepcopy(m)
    for a, b in pairs:
        va, vb = m.get(a), m.get(b)
        out.pop(a, None)
        out.pop(b, None)
        if vb is not None:
            out[a] = vb
        if va is not None:
            out[b] = va
    if "h2h" in m:
        h = m["h2h"]
        out["h2h"] = {"a_wins": h.get("b_wins", 0), "b_wins": h.get("a_wins", 0)}
    out.pop("market_price_a", None)
    return out


def series_a(r):
    s = r["series_probs_exact"]
    return s["p_2_0"] + s["p_2_1"]


def factor(r, name):
    return next(f for f in r["factor_breakdown"] if f["factor"] == name)


def all_agree_match():
    """Spirit-vs-1WIN-strength gap where every factor with data points to A.
    Maps: A uniformly 5pp better everywhere -> pool-relative veto signal is 0
    (a single spike map would be banned by B, leaving A slightly BELOW its own
    pool average on the rest, which is a legitimate negative veto signal)."""
    maps_a = {mp: [0.55, 10] for mp in POOL}
    maps_b = {mp: [0.50, 10] for mp in POOL}
    return {
        "team_a": "Spirit", "team_b": "1WIN",
        "rating_a": 1.11, "rating_b": 1.00,
        "form30_a": 0.80, "form30_b": 0.50, "n30_a": 13, "n30_b": 12,
        "form5_a": 0.90, "form5_b": 0.40,
        "h2h": {"a_wins": 3, "b_wins": 0, "meetings": 3},
        "maps_a": maps_a, "maps_b": maps_b,
        "stakes_a": "elimination", "stakes_b": "none",
        "roster_a": {}, "roster_b": {"standin": True},
        "volatility_a": 0.3, "volatility_b": 0.3,
    }


class TestCoherence(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for r in predictor.day4_backtest() + [predict_match(base_match())]:
            self.assertAlmostEqual(r["p_a"] + r["p_b"], 1.0, places=6)

    def test_series_probs_sum_and_match_p_a(self):
        for r in predictor.day4_backtest() + [predict_match(base_match()), predict_match(all_agree_match())]:
            s = r["series_probs_exact"]
            self.assertAlmostEqual(sum(s.values()), 1.0, places=9)
            # H1: series P(A) implied by the scoreline probs equals the final p_a
            self.assertAlmostEqual(series_a(r), r["p_a_exact"], places=6)
            # scoreline is the argmax of the four
            a, b = r["match"].split(" vs ")
            labels = {"p_2_0": f"{a} 2-0", "p_2_1": f"{a} 2-1",
                      "p_1_2": f"{b} 2-1", "p_0_2": f"{b} 2-0"}
            self.assertEqual(r["scoreline"], labels[max(s, key=s.get)])

    def test_iid_favourite_scoreline_is_2_0(self):
        # With no map structure, maps are iid: P(2-0)=q^2 > P(2-1)=2q^2(1-q) for q>0.5
        r = predict_match({"rating_a": 1.03, "rating_b": 1.00})
        self.assertGreater(r["p_a"], 0.5)
        self.assertEqual(r["scoreline"], "A 2-0")

    def test_total_logodds_and_temperature(self):
        m = base_match()
        r = predict_match(m)
        self.assertAlmostEqual(r["total_logodds"], total_logodds(m), places=12)
        self.assertAlmostEqual(r["p_a_exact"],
                               predictor.sigmoid(CONFIG["temperature"] * total_logodds(m)), places=12)

    def test_temperature_zero_gives_half(self):
        old = CONFIG["temperature"]
        try:
            CONFIG["temperature"] = 0.0
            r = predict_match(all_agree_match())
            self.assertAlmostEqual(r["p_a_exact"], 0.5, places=12)
            self.assertAlmostEqual(series_a(r), 0.5, places=6)
        finally:
            CONFIG["temperature"] = old


class TestSymmetryMonotonicity(unittest.TestCase):
    def test_swap_symmetry_strict(self):
        m = base_match()
        r1, r2 = predict_match(m), predict_match(swap(m))
        self.assertAlmostEqual(r1["p_a_exact"], 1.0 - r2["p_a_exact"], places=9)

    def test_swap_symmetry_with_symmetric_maps(self):
        # mirrored map data: A's rates for B and vice versa
        maps_a = {"Dust2": [0.6, 10], "Mirage": [0.4, 10], "Nuke": [0.5, 10]}
        maps_b = {"Dust2": [0.4, 10], "Mirage": [0.6, 10], "Nuke": [0.5, 10]}
        m = base_match(maps_a=maps_a, maps_b=maps_b)
        r1, r2 = predict_match(m), predict_match(swap(m))
        self.assertAlmostEqual(r1["p_a_exact"], 1.0 - r2["p_a_exact"], delta=0.02)

    def test_swap_symmetry_backtest_approx(self):
        # approximate symmetry on asymmetric real-ish inputs (ban order may differ)
        m = all_agree_match()
        r1, r2 = predict_match(m), predict_match(swap(m))
        self.assertAlmostEqual(r1["p_a_exact"], 1.0 - r2["p_a_exact"], delta=0.03)

    def test_monotone_in_rating_gap(self):
        prev = -1
        for ra in [0.90 + 0.01 * i for i in range(31)]:
            p = predict_match(base_match(rating_a=ra))["p_a_exact"]
            self.assertGreaterEqual(p, prev - 1e-12, f"non-monotone at rating_a={ra}")
            prev = p

    def test_monotone_in_form(self):
        for key in ("form30_a", "form5_a"):
            prev = -1
            for i in range(21):
                p = predict_match(base_match(**{key: i / 20}))["p_a_exact"]
                self.assertGreaterEqual(p, prev - 1e-12, f"{key} non-monotone at {i / 20}")
                prev = p


class TestEdgeCases(unittest.TestCase):
    def test_empty_input(self):
        r = predict_match({})
        self.assertEqual(r["p_a"], 0.5)
        self.assertAlmostEqual(r["p_a_exact"], 0.5, places=12)
        self.assertNotIn("FACTOR SPLIT", " ".join(r["warnings"]))

    def test_extreme_inputs_finite(self):
        m = {
            "rating_a": 5.0, "rating_b": -3.0, "form30_a": 7, "form30_b": -2,
            "n30_a": 1e9, "n30_b": -5, "form5_a": 1.0, "form5_b": 0.0,
            "h2h": {"a_wins": 1e6, "b_wins": 0, "meetings": 3},
            "maps_a": {mp: [1.0, 1e6] for mp in POOL}, "maps_b": {mp: [0.0, 1e6] for mp in POOL},
            "volatility_a": 9, "volatility_b": -1, "stakes_a": "title", "stakes_b": "none",
            "roster_b": {"standin": True, "missing_igl": True},
        }
        for mm in (m, swap(m), {"rating_a": float("inf"), "volatility_a": float("nan")}):
            r = predict_match(mm)
            for v in [r["p_a_exact"], *r["confidence_interval"], r["total_logodds"]]:
                self.assertTrue(math.isfinite(v))
            self.assertGreater(r["p_a_exact"], 0.0)
            self.assertLess(r["p_a_exact"], 1.0)

    def test_sigmoid_overflow_safe(self):
        self.assertEqual(predictor.sigmoid(1e6), 1.0)
        self.assertEqual(predictor.sigmoid(-1e6), 0.0)

    def test_signal_caps_bound_each_factor(self):
        m = {"rating_a": 2.0, "rating_b": 0.5, "form30_a": 1.0, "form30_b": 0.0,
             "n30_a": 1000, "n30_b": 1000, "rating_b2": 0}
        r = predict_match(m)
        cap = CONFIG["signal_cap"]
        self.assertLessEqual(abs(factor(r, "base_strength")["delta_logodds"]), cap * CONFIG["w_base"] + 1e-9)
        self.assertLessEqual(abs(factor(r, "form_30d")["delta_logodds"]), cap * CONFIG["w_form30"] + 5e-4)

    def test_stakes_levels(self):
        r = predict_match({"stakes_a": "title", "stakes_b": "qualification"})
        self.assertAlmostEqual(factor(r, "stakes")["delta_logodds"],
                               round(math.log(CONFIG["stakes_mult"]), 3), places=3)


class TestFactors(ReferenceWeights, unittest.TestCase):
    def test_uniform_cross_map_edge_no_veto_signal(self):
        m = {"rating_a": 1.05, "rating_b": 1.0,
             "maps_a": {mp: [0.65, 12] for mp in POOL},
             "maps_b": {mp: [0.45, 9] for mp in POOL}}
        r = predict_match(m)
        self.assertAlmostEqual(factor(r, "map_veto")["delta_logodds"], 0.0, places=9)

    def test_missing_map_is_no_data(self):
        # one-sided data on a single map, nothing else: overall == that map -> 0 edge
        r = predict_match({"maps_a": {"Dust2": [0.9, 20]}})
        self.assertAlmostEqual(factor(r, "map_veto")["delta_logodds"], 0.0, places=9)

    def test_all_agree_no_split_and_below_old_97(self):
        r = predict_match(all_agree_match())
        print(f"\n  [info] all-agree case p_a = {r['p_a_exact']:.3f} "
              f"(old engine on Spirit vs 1WIN: 0.967); reliability {r['reliability']}")
        self.assertFalse(any("FACTOR SPLIT" in w for w in r["warnings"]))
        self.assertEqual(r["conflict"], 0.0)
        self.assertLess(r["p_a_exact"], 0.93)
        self.assertGreater(r["p_a_exact"], 0.75)

    def test_spirit_backtest_no_longer_97(self):
        spirit = predictor.day4_backtest()[1]
        print(f"\n  [info] Spirit vs 1WIN backtest p_a = {spirit['p_a']}")
        self.assertLess(spirit["p_a"], 0.90)
        self.assertGreater(spirit["p_a"], 0.5)
        self.assertFalse(any("FACTOR SPLIT" in w for w in spirit["warnings"]))

    def test_genuine_split_flagged(self):
        r = predict_match(base_match(form30_a=0.25, form30_b=0.75, roster_b={},
                                     h2h={}, stakes_a="none", stakes_b="none",
                                     form5_a=0.25, form5_b=0.75, rating_a=1.06))
        self.assertTrue(any("FACTOR SPLIT" in w for w in r["warnings"]), r["warnings"])

    def test_roster_continuity(self):
        # H3: A has a stand-in; B's penalty 0.9999 must be ~identical to B full strength
        p_none = predict_match(base_match(roster_a={"standin": True}, roster_b={}))["p_a_exact"]
        saved = CONFIG["no_igl_penalty"]
        try:
            CONFIG["no_igl_penalty"] = 0.9999
            p_tiny = predict_match(base_match(roster_a={"standin": True},
                                              roster_b={"missing_igl": True}))["p_a_exact"]
        finally:
            CONFIG["no_igl_penalty"] = saved
        print(f"\n  [info] roster continuity: B none {p_none:.4f} vs B x0.9999 {p_tiny:.4f}")
        self.assertLess(abs(p_tiny - p_none), 0.01)
        self.assertLess(abs(p_tiny - p_none), 0.001)

    def test_roster_in_logodds_and_not_double_counted(self):
        r = predict_match({"roster_a": {"standin": True, "missing_igl": True}})
        d = factor(r, "roster")["delta_logodds"]
        self.assertAlmostEqual(d, round(math.log(CONFIG["standin_igl_penalty"]), 3), places=3)
        self.assertGreater(CONFIG["standin_igl_penalty"],
                           CONFIG["standin_penalty"] * CONFIG["no_igl_penalty"])
        self.assertLess(r["p_a"], 0.5)

    def test_marginals_sign_consistent(self):
        for r in predictor.day4_backtest() + [predict_match(base_match()), predict_match(all_agree_match())]:
            for f in r["factor_breakdown"]:
                if abs(f["delta_logodds"]) > 1e-3:
                    self.assertEqual(f["marginal_pp"] >= 0, f["delta_logodds"] > 0,
                                     f"{r['match']} {f['factor']}")
                    self.assertTrue(f["marginal_pp"] == 0 or
                                    (f["marginal_pp"] > 0) == (f["delta_logodds"] > 0))

    def test_marginal_matches_final_p(self):
        m = base_match()
        r = predict_match(m)
        k = 1.0 / r["uncertainty_shrink"]
        raw = r["raw_logodds"]
        f = factor(r, "roster")
        p_without = predictor.sigmoid(CONFIG["temperature"] * (raw - f["delta_logodds"]) / k)
        self.assertAlmostEqual(f["marginal_pp"], (r["p_a_exact"] - p_without) * 100, delta=0.1)

    def test_opp_rating_adjustment(self):
        # A's 30d form came against much tougher opponents -> credited more
        p0 = predict_match(base_match())["p_a_exact"]
        p1 = predict_match(base_match(opp_rating30_a=1.05, opp_rating30_b=0.98))["p_a_exact"]
        self.assertGreater(p1, p0)


class TestUncertainty(ReferenceWeights, unittest.TestCase):
    def test_reliability_levels_reachable(self):
        # Tiers are cut on band width (pp), fitted on the backtest TRAIN split.
        maps = {mp: [0.5, 30] for mp in POOL}
        high = {"rating_a": 1.03, "rating_b": 1.00, "form30_a": 0.58, "form30_b": 0.50,
                "n30_a": 30, "n30_b": 30, "form5_a": 0.6, "form5_b": 0.6,
                "h2h": {"a_wins": 4, "b_wins": 4}, "maps_a": maps, "maps_b": maps,
                "volatility_a": 0.1, "volatility_b": 0.1}
        medium = dict(high, n30_a=12, n30_b=12, volatility_a=0.5, volatility_b=0.5,
                      h2h={"a_wins": 1, "b_wins": 1})
        low = dict(medium, volatility_a=0.9, volatility_b=0.8)
        rs = [predict_match(x) for x in (high, medium, low)]
        self.assertEqual([r["reliability"] for r in rs], ["HIGH", "MEDIUM", "LOW"],
                         [r["interval_width_pp"] for r in rs])
        # tier is a pure function of the reported width
        for r in rs:
            w = r["interval_width_pp"]
            exp = ("HIGH" if w <= CONFIG["rel_high_width_pp"] else
                   "MEDIUM" if w <= CONFIG["rel_med_width_pp"] else "LOW")
            self.assertEqual(r["reliability"], exp)

    def test_band_contains_p_and_is_asymmetric(self):
        for r in predictor.day4_backtest() + [predict_match(all_agree_match())]:
            lo, hi = r["confidence_interval"]
            self.assertLessEqual(lo, r["p_a"])
            self.assertGreaterEqual(hi, r["p_a"])
            self.assertIn("uncertainty band", r["band_kind"])
        r = predict_match(all_agree_match())
        lo, hi = r["confidence_interval"]
        p = r["p_a_exact"]
        self.assertGreater(p - lo, hi - p)  # more room below than above near 1

    def test_volatility_pulls_toward_half(self):
        m = base_match()
        prev = None
        for v in [0.0, 0.3, 0.6, 1.0]:
            p = predict_match(dict(m, volatility_a=v, volatility_b=v))["p_a_exact"]
            if prev is not None:
                self.assertLess(abs(p - 0.5), abs(prev - 0.5))
            prev = p

    def test_volatility_widens_band(self):
        m = base_match()
        w1 = predict_match(dict(m, volatility_a=0.1, volatility_b=0.1))["interval_width_pp"]
        w2 = predict_match(dict(m, volatility_a=0.9, volatility_b=0.9))["interval_width_pp"]
        self.assertGreater(w2, w1)


class TestVetoPool(unittest.TestCase):
    def test_five_map_pool(self):
        pool = ["Dust2", "Mirage", "Inferno", "Nuke", "Ancient"]
        m = base_match(map_pool=pool,
                       maps_a={"Dust2": [0.7, 10], "Mirage": [0.4, 10], "Nuke": [0.5, 10]},
                       maps_b={"Dust2": [0.4, 10], "Mirage": [0.6, 10]})
        r = predict_match(m)
        self.assertEqual(len(r["veto_maps"]), 3)
        self.assertEqual(len(set(r["veto_maps"])), 3)
        self.assertTrue(set(r["veto_maps"]) <= set(pool))
        self.assertAlmostEqual(sum(r["series_probs_exact"].values()), 1.0, places=9)

    def test_config_pool_five_and_three(self):
        old = CONFIG["map_pool"]
        try:
            CONFIG["map_pool"] = ["Dust2", "Mirage", "Inferno", "Nuke", "Ancient"]
            r = predict_match(base_match())
            self.assertEqual(len(r["veto_maps"]), 3)
            CONFIG["map_pool"] = ["Dust2", "Mirage", "Inferno"]
            r = predict_match(base_match())
            self.assertEqual(sorted(r["veto_maps"]), sorted(CONFIG["map_pool"]))
        finally:
            CONFIG["map_pool"] = old

    def test_large_pool(self):
        pool = POOL + ["Train", "Overpass", "Vertigo"]
        r = predict_match(base_match(map_pool=pool))
        self.assertEqual(len(r["veto_maps"]), 3)

    def test_small_pool_raises(self):
        with self.assertRaises(ValueError):
            predict_match(base_match(map_pool=["Dust2", "Mirage"]))

    def test_permaban_honored(self):
        r = predict_match(base_match(permaban_a="Nuke"))
        self.assertIn("A bans Nuke (permaban)", r["veto_log"])
        self.assertNotIn("Nuke", r["veto_maps"])

    def test_series_probs_helper(self):
        s = series_probs(0.6, 0.6, 0.6)
        self.assertAlmostEqual(s[0], 0.36)
        self.assertAlmostEqual(s[1], 2 * 0.36 * 0.4)
        self.assertGreater(s[0], s[1])

    def test_display_percent_largest_remainder(self):
        # Naive round-half-up of 41.6/29.5/16.3/12.6 is 42/30/16/13 = 101.
        whole = display_percent([0.416, 0.295, 0.163, 0.126], places=0)
        self.assertEqual(whole, [42.0, 29.0, 16.0, 13.0])
        self.assertEqual(sum(whole), 100.0)
        # 1/3 + 1/3 + 1/3 printed to 2 decimals is 33.33 three times = 99.99.
        thirds = display_percent([1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(thirds, [33.34, 33.33, 33.33])
        self.assertEqual(sum(thirds), 100.0)
        pair = display_percent([0.504, 0.496])
        self.assertEqual(pair, [50.40, 49.60])
        self.assertEqual(sum(pair), 100.0)

    def test_map_win_chances_keep_veto_shape_when_series_weight_is_off(self):
        # w_veto is 0, so the veto must not move the series winner. The three
        # map win chances still follow the simulated veto, then shift together
        # so the scoreline still implies p_a.
        self.assertEqual(CONFIG["w_veto"], 0.0)
        r = predict_match(base_match(
            map_pool=["Dust2", "Mirage", "Inferno"],
            maps_a={"Dust2": [0.75, 30], "Mirage": [0.35, 30], "Inferno": [0.50, 30]},
            maps_b={"Dust2": [0.40, 30], "Mirage": [0.70, 30], "Inferno": [0.50, 30]},
        ))
        probs = r["map_probs_exact"]
        self.assertEqual(len(set(round(p, 4) for p in probs)), 3)
        self.assertGreater(max(probs) - min(probs), 0.02)
        veto = next(f for f in r["factor_breakdown"] if f["factor"] == "map_veto")
        self.assertEqual(veto["delta_logodds"], 0.0)
        self.assertAlmostEqual(series_a(r), r["p_a_exact"], places=6)

    def test_reported_probability_sets_sum_to_100(self):
        r = predict_match(base_match())
        shown = display_percent([r["series_probs_exact"][k]
                                 for k in ("p_2_0", "p_2_1", "p_1_2", "p_0_2")])
        self.assertEqual(sum(shown), 100.0)
        self.assertEqual([round(v * 100, 2) for v in
                          (r["series_probs"]["p_2_0"], r["series_probs"]["p_2_1"],
                           r["series_probs"]["p_1_2"], r["series_probs"]["p_0_2"])],
                         shown)
        pair = display_percent([r["p_a_exact"], 1 - r["p_a_exact"]])
        self.assertEqual(sum(pair), 100.0)
        self.assertAlmostEqual(r["p_a"] + r["p_b"], 1.0, places=9)
        txt = predictor.format_result(r)
        # the four scoreline labels in the report are the same largest-remainder set
        self.assertIn(f"{shown[0]:.2f}%", txt)
        self.assertIn(f"{shown[3]:.2f}%", txt)


def _rand_match(rng):
    """Seeded random but plausible match, incl. opp keys and fractional n."""
    pool = list(POOL)
    ns = [0.2, 0.5, 0.77, 1, 3, 8, 15, 40]
    m = {
        "rating_a": rng.uniform(0.85, 1.20), "rating_b": rng.uniform(0.85, 1.20),
        "form30_a": rng.random(), "form30_b": rng.random(),
        "n30_a": rng.choice(ns + [rng.uniform(0, 30)]),
        "n30_b": rng.choice(ns + [rng.uniform(0, 30)]),
        "form5_a": rng.random(), "form5_b": rng.random(),
        "h2h": {"a_wins": rng.randint(0, 8), "b_wins": rng.randint(0, 8)},
        "volatility_a": rng.random(), "volatility_b": rng.random(),
        "maps_a": {mp: [rng.random(), rng.randint(0, 25)] for mp in pool if rng.random() < 0.85},
        "maps_b": {mp: [rng.random(), rng.randint(0, 25)] for mp in pool if rng.random() < 0.85},
        "stakes_a": rng.choice(["none", "qualification", "elimination", "title"]),
        "stakes_b": rng.choice(["none", "qualification", "elimination", "title"]),
        "roster_a": {"standin": rng.random() < 0.2, "missing_igl": rng.random() < 0.1},
        "roster_b": {"standin": rng.random() < 0.2, "missing_igl": rng.random() < 0.1},
    }
    if rng.random() < 0.7:
        m["opp_rating30_a"] = rng.uniform(0.85, 1.15)
        m["opp_rating30_b"] = rng.uniform(0.85, 1.15)
    if rng.random() < 0.3:
        m["permaban_a"] = rng.choice(pool)
    return m


class TestBO5(unittest.TestCase):
    """BO5 output: new fields only; the BO3 fields keep their meaning."""

    def test_general_function_matches_bo3_formula(self):
        for ps in ([0.6, 0.6, 0.6], [0.71, 0.47, 0.53], [0.2, 0.9, 0.5]):
            for x, y in zip(predictor.series_scorelines(ps), series_probs(*ps)):
                self.assertAlmostEqual(x, y, places=12)

    def test_bo5_sums_and_series_probability(self):
        for kw in ({}, {"rating_a": 1.08}, {"rating_b": 1.12}):
            r = predict_match(base_match(**kw))
            ex = [r["series_probs_bo5_exact"][k] for k in predictor.BO5_KEYS]
            self.assertAlmostEqual(sum(ex), 1.0, places=12)
            # BO5 win chance shown equals the engine's p_a
            self.assertAlmostEqual(sum(ex[:3]), r["p_a_exact"], places=9)
            shown = [r["series_probs_bo5"][k] for k in predictor.BO5_KEYS]
            self.assertEqual(round(sum(shown) * 100, 6), 100.0)
            self.assertEqual(shown, [v / 100 for v in display_percent(ex)])

    def test_three_one_is_modal_below_two_thirds(self):
        # flat per-map q: P(3-1)/P(3-0) = 3(1-q), so 3-1 wins exactly when q < 2/3
        for q, modal in ((0.55, 1), (0.65, 1), (0.68, 0), (0.8, 0)):
            s = predictor.series_scorelines([q] * 5)
            self.assertEqual(max(range(3), key=lambda k: s[k]), modal, q)

    def test_bo3_fields_unchanged(self):
        r = predict_match(base_match(rating_a=1.05))
        self.assertEqual(sorted(r["series_probs"]), ["p_0_2", "p_1_2", "p_2_0", "p_2_1"])
        s = r["series_probs_exact"]
        self.assertAlmostEqual(s["p_2_0"] + s["p_2_1"], r["p_a_exact"], places=6)


def _steps(log):
    """'A bans X (permaban)' -> ('A', 'bans', 'X'); 'decider: X' -> (None, 'decider', 'X')."""
    out = []
    for s in log:
        if s.startswith("decider: "):
            out.append((None, "decider", s[9:]))
        else:
            side, verb, mp = s.split(" ", 2)
            out.append((side, verb, mp.replace(" (permaban)", "")))
    return out


GP_MAPS_A = {"Dust2": [0.70, 20], "Mirage": [0.35, 20], "Inferno": [0.55, 15], "Nuke": [0.60, 12],
             "Ancient": [0.45, 18], "Anubis": [0.50, 10], "Cache": [0.40, 9]}
GP_MAPS_B = {"Dust2": [0.40, 15], "Mirage": [0.72, 22], "Inferno": [0.50, 10], "Nuke": [0.45, 14],
             "Ancient": [0.60, 16], "Anubis": [0.55, 11], "Cache": [0.50, 8]}


class TestVetoFormats(unittest.TestCase):
    """BO1 / BO3 / BO5 vetoes. BO3 output must be exactly what it was."""

    # BO3 vetoes and map chances before the format generalisation (2026-10-07)
    BO3_PINNED = {
        "Alpha vs Bravo": (['A bans Dust2', 'B bans Mirage', 'A picks Inferno', 'B picks Nuke',
                            'A bans Ancient', 'B bans Anubis', 'decider: Cache'],
                           [0.6159227007, 0.6159227007, 0.6159227007]),
        "G2 vs PARIVISION": (['A bans Ancient', 'B bans Inferno (permaban)', 'A picks Mirage', 'B picks Dust2',
                              'A bans Cache', 'B bans Anubis', 'decider: Nuke'],
                             [0.5857183963, 0.4669758861, 0.5801946722]),
        "Spirit vs 1WIN": (['A bans Inferno (permaban)', 'B bans Anubis', 'A picks Nuke', 'B picks Dust2',
                            'A bans Cache', 'B bans Ancient', 'decider: Mirage'],
                           [0.746669971, 0.6653128506, 0.7249867221]),
        "FURIA vs Aurora": (['A bans Anubis (permaban)', 'B bans Nuke', 'A picks Inferno', 'B picks Ancient',
                             'A bans Dust2', 'B bans Cache', 'decider: Mirage'],
                            [0.6139269432, 0.5687414646, 0.5911389186]),
        "Legacy vs M80": (['A bans Mirage', 'B bans Dust2', 'A picks Inferno', 'B picks Nuke',
                           'A bans Anubis', 'B bans Ancient', 'decider: Cache'],
                          [0.5424434821, 0.5196995793, 0.541436567]),
    }

    def test_bo3_unchanged_base_and_day4(self):
        # pinned values were taken with map_shape 0.75 before the veto refactor;
        # run at that setting so the check still guards the refactor itself
        saved = CONFIG["map_shape"]
        CONFIG["map_shape"] = 0.75
        try:
            rs = [predict_match(base_match())] + predictor.day4_backtest()
        finally:
            CONFIG["map_shape"] = saved
        for r in rs:
            log, probs = self.BO3_PINNED[r["match"]]
            self.assertEqual(r["veto_log"], log)
            for x, y in zip(r["map_probs_exact"], probs):
                self.assertAlmostEqual(x, y, places=9)
        # the explicit and the default best_of give the same veto
        mm =base_match(maps_a=GP_MAPS_A, maps_b=GP_MAPS_B, permaban_b="Cache")
        self.assertEqual(predictor.simulate_veto(mm), predictor.simulate_veto(mm, best_of=3))

    def test_bo1_six_bans_and_decider(self):
        for kw in ({}, {"maps_a": GP_MAPS_A, "maps_b": GP_MAPS_B}, {"permaban_a": "Nuke"}):
            r = predict_match(base_match(**kw))
            v = r["veto_bo1"]
            st = _steps(v["veto_log"])
            self.assertEqual([(s[0], s[1]) for s in st],
                             [("A", "bans"), ("B", "bans")] * 3 + [(None, "decider")])
            self.assertEqual(len(v["maps"]), 1)
            self.assertEqual(v["maps"][0], st[-1][2])
            self.assertEqual(sorted(s[2] for s in st), sorted(POOL))     # every map once
            self.assertEqual(v["map_probs_exact"], [r["p_a_exact"]])     # series = map
        r = predict_match(base_match(permaban_a="Nuke"))
        self.assertEqual(r["veto_bo1"]["veto_log"][0], "A bans Nuke (permaban)")

    def test_bo5_two_bans_four_picks_and_decider(self):
        for kw in ({}, {"maps_a": GP_MAPS_A, "maps_b": GP_MAPS_B}, {"permaban_b": "Dust2"}):
            r = predict_match(base_match(**kw))
            v = r["veto_bo5"]
            st = _steps(v["veto_log"])
            self.assertEqual([(s[0], s[1]) for s in st],
                             [("A", "bans"), ("B", "bans"), ("A", "picks"), ("B", "picks"),
                              ("A", "picks"), ("B", "picks"), (None, "decider")])
            self.assertEqual(v["maps"], [s[2] for s in st[2:]])          # pick order, decider last
            self.assertEqual(len(set(v["maps"])), 5)
            self.assertEqual(len(v["map_probs_exact"]), 5)

    def test_bo5_picks_follow_map_edges(self):
        # A picks its strongest relative maps, B its own; each bans the other's best
        v = predictor.simulate_veto(base_match(maps_a=GP_MAPS_A, maps_b=GP_MAPS_B), 5)
        self.assertEqual(v["veto_log"][0], "A bans Mirage")   # B's biggest edge
        self.assertEqual(v["veto_log"][1], "B bans Dust2")    # A's biggest edge
        self.assertEqual(v["veto_log"][2], "A picks Nuke")
        edge_a = [v["map_logits"][0], v["map_logits"][2]]
        edge_b = [v["map_logits"][1], v["map_logits"][3]]
        self.assertGreater(min(edge_a), max(edge_b))

    def test_bo5_map_probs_imply_p_a_and_display(self):
        for kw in ({}, {"maps_a": GP_MAPS_A, "maps_b": GP_MAPS_B}, {"rating_b": 1.12},
                   {"maps_a": GP_MAPS_A, "maps_b": GP_MAPS_B, "rating_a": 1.10}):
            r = predict_match(base_match(**kw))
            p5 = r["veto_bo5"]["map_probs_exact"]
            sc = predictor.series_scorelines(p5)
            self.assertAlmostEqual(sum(sc[:3]), r["p_a_exact"], places=9)
            ex = [r["series_probs_bo5_exact"][k] for k in predictor.BO5_KEYS]
            for x, y in zip(sc, ex):
                self.assertAlmostEqual(x, y, places=12)
            shown = display_percent(ex)
            self.assertEqual(round(sum(shown), 6), 100.0)
            self.assertEqual([r["series_probs_bo5"][k] for k in predictor.BO5_KEYS],
                             [v / 100 for v in shown])
            self.assertEqual(r["map_prob_bo5_exact"], predictor.flat_map_prob(r["p_a_exact"], 5))

    def test_bo5_shape_uses_map_shape(self):
        m = base_match(maps_a=GP_MAPS_A, maps_b=GP_MAPS_B)
        saved = CONFIG["map_shape"]
        try:
            CONFIG["map_shape"] = 0.0
            flat = predict_match(m)
            CONFIG["map_shape"] = 2.0
            wide = predict_match(m)
        finally:
            CONFIG["map_shape"] = saved
        p0 = flat["veto_bo5"]["map_probs_exact"]
        self.assertLess(max(p0) - min(p0), 1e-9)
        self.assertAlmostEqual(p0[0], flat["map_prob_bo5_exact"], places=9)
        p2 = wide["veto_bo5"]["map_probs_exact"]
        self.assertGreater(max(p2) - min(p2), 0.1)
        self.assertEqual(flat["p_a_exact"], wide["p_a_exact"])   # shape never moves p_a

    def test_underdog_pick_can_make_2_1_modal(self):
        # flat q > 0.5 always makes 2-0 beat 2-1; a map-shaped veto can flip it
        self.assertGreater(0.6 ** 2, 2 * 0.6 ** 2 * 0.4)
        s = series_probs(0.80, 0.35, 0.60)          # A huge on its pick, B on its own
        self.assertGreater(s[1], s[0])

    def test_pool_sizes(self):
        # 5 maps: BO5 has no bans; BO1 bans 4. 3 maps: no BO5 veto, flat BO5 scorelines.
        pool5 = ["Dust2", "Mirage", "Inferno", "Nuke", "Ancient"]
        r = predict_match(base_match(map_pool=pool5))
        self.assertEqual([s[1] for s in _steps(r["veto_bo5"]["veto_log"])], ["picks"] * 4 + ["decider"])
        self.assertEqual([s[1] for s in _steps(r["veto_bo1"]["veto_log"])], ["bans"] * 4 + ["decider"])
        r = predict_match(base_match(map_pool=["Dust2", "Mirage", "Inferno"]))
        self.assertIsNone(r["veto_bo5"])
        q = r["map_prob_bo5_exact"]
        for x, y in zip(predictor.series_scorelines([q] * 5),
                        [r["series_probs_bo5_exact"][k] for k in predictor.BO5_KEYS]):
            self.assertAlmostEqual(x, y, places=12)
        self.assertEqual(len(r["veto_bo1"]["maps"]), 1)
        big = POOL + ["Train", "Overpass", "Vertigo"]
        r = predict_match(base_match(map_pool=big))
        self.assertEqual(len(r["veto_bo5"]["maps"]), 5)
        self.assertEqual(len(r["veto_bo1"]["veto_log"]), len(big))
        with self.assertRaises(ValueError):
            predictor.simulate_veto(base_match(), best_of=7)
        with self.assertRaises(ValueError):
            predictor.simulate_veto(base_match(map_pool=["Dust2", "Mirage", "Inferno", "Nuke"]), best_of=5)

    def test_series_win_matches_scorelines(self):
        for ps in ([0.6] * 5, [0.7, 0.4, 0.55, 0.3, 0.9], [0.2, 0.9, 0.5], [0.42]):
            need = (len(ps) + 1) // 2
            self.assertAlmostEqual(predictor._series_win(ps),
                                   sum(predictor.series_scorelines(ps)[:need]), places=12)

    def test_existing_schema_unchanged(self):
        r = predict_match(base_match())
        for k in ("match", "p_a", "p_b", "pick", "scoreline", "scoreline_prob", "confidence_interval",
                  "band_kind", "interval_width_pp", "reliability", "logit_sd", "raw_logodds",
                  "total_logodds", "uncertainty_shrink", "temperature", "conflict", "factor_breakdown",
                  "veto_log", "veto_maps", "map_probs", "map_probs_exact", "veto_only_series_p_a",
                  "veto_only_map_probs", "map_logit_shift", "series_probs", "series_probs_exact",
                  "series_probs_bo5", "series_probs_bo5_exact", "map_prob_bo5_exact", "p_a_exact",
                  "market_edge_pp", "market_edge_note", "warnings", "volatility"):
            self.assertIn(k, r)
        self.assertEqual(len(r["veto_maps"]), 3)
        self.assertEqual(len(r["map_probs_exact"]), 3)
        self.assertEqual(sorted(r["series_probs_bo5_exact"]), sorted(predictor.BO5_KEYS))
        self.assertEqual(sorted(r["veto_bo1"]), ["map_probs_exact", "maps", "veto_log"])
        self.assertEqual(sorted(r["veto_bo5"]), ["map_logit_shift", "map_probs", "map_probs_exact",
                                                 "maps", "veto_log"])


class TestRandomizedMonotonicity(unittest.TestCase):
    """p_a must be non-decreasing in A's rating, 30d form and last-5 form for
    every input, including opp-rating adjustment and fractional sample sizes."""

    def _sweep(self, key, grid, n_cases=150, seed=0):
        import random
        rng = random.Random(seed)
        for case in range(n_cases):
            m = _rand_match(rng)
            prev, prev_v = -1.0, None
            for v in grid:
                p = predict_match(dict(m, **{key: v}))["p_a_exact"]
                self.assertGreaterEqual(p, prev - 1e-12,
                                        f"{key} non-monotone {prev_v}->{v} in case {case}: {m}")
                prev, prev_v = p, v

    def test_rating_random(self):
        self._sweep("rating_a", [0.80 + 0.01 * i for i in range(46)], seed=1)

    def test_form30_random(self):
        self._sweep("form30_a", [i / 20 for i in range(21)], seed=2)

    def test_form5_random(self):
        self._sweep("form5_a", [i / 20 for i in range(21)], seed=3)

    def test_reviewer_repro_opp_rating(self):
        m = {"rating_b": 1.0, "form30_a": .6, "form30_b": .5, "n30_a": 15, "n30_b": 15,
             "opp_rating30_a": 1.05, "opp_rating30_b": 1.0}
        ps = [predict_match(dict(m, rating_a=1.10 + 0.005 * i))["p_a_exact"] for i in range(11)]
        for a, b in zip(ps, ps[1:]):
            self.assertGreaterEqual(b, a - 1e-12)

    def test_reviewer_repro_fractional_n(self):
        m = {"form30_b": .5, "n30_a": .4, "n30_b": .4, "form5_a": .5, "form5_b": .5}
        ps = [predict_match(dict(m, form30_a=i / 10))["p_a_exact"] for i in range(11)]
        for a, b in zip(ps, ps[1:]):
            self.assertGreaterEqual(b, a - 1e-12)


class TestFuzz(unittest.TestCase):
    JUNK = [None, "", "abc", "0.6", "nan", "inf", [], [0.6], {}, {"x": 1}, True, False,
            float("nan"), float("inf"), float("-inf"), 1e308, -1e308, -5, 0, 0.5, 7, 1e6, 3.2]

    def test_fuzz_2000(self):
        import random
        rng = random.Random(20261006)
        keys = ["rating_a", "rating_b", "form30_a", "form30_b", "n30_a", "n30_b",
                "opp_rating30_a", "opp_rating30_b", "form5_a", "form5_b", "volatility_a",
                "volatility_b", "market_price_a", "stakes_a", "stakes_b", "permaban_a",
                "permaban_b", "team_a", "team_b"]
        J = self.JUNK
        for i in range(2000):
            m = _rand_match(rng) if rng.random() < 0.5 else {}
            for k in keys:
                if rng.random() < 0.35:
                    m[k] = rng.choice(J)
            if rng.random() < 0.3:
                m["h2h"] = rng.choice([rng.choice(J), {"a_wins": rng.choice(J), "b_wins": rng.choice(J),
                                                       "meetings": rng.choice(J)}])
            for side in ("maps_a", "maps_b"):
                if rng.random() < 0.3:
                    m[side] = rng.choice([rng.choice(J),
                                          {mp: rng.choice([[rng.choice(J), rng.choice(J)], rng.choice(J)])
                                           for mp in POOL}])
            for side in ("roster_a", "roster_b"):
                if rng.random() < 0.2:
                    m[side] = rng.choice([rng.choice(J), {"standin": rng.choice(J),
                                                          "missing_igl": rng.choice(J)}])
            if rng.random() < 0.1:
                m["map_pool"] = rng.choice([7, "abc", None, [], {}, list(POOL), POOL[:4],
                                            [1, 2, 3, "Dust2"] + list(POOL), ["Dust2", 5]])
            try:
                r = predict_match(m)
            except ValueError:
                # only allowed failure: an explicit pool list with < 3 valid maps
                pool = m.get("map_pool")
                self.assertTrue(isinstance(pool, (list, tuple)) and
                                len({x for x in pool if isinstance(x, str)}) < 3, m)
                continue
            p = r["p_a_exact"]
            self.assertTrue(math.isfinite(p) and 0.0 < p < 1.0, (i, m, p))
            s = r["series_probs_exact"]
            self.assertTrue(all(math.isfinite(v) for v in s.values()), (i, m))
            self.assertAlmostEqual(sum(s.values()), 1.0, places=9)
            self.assertTrue(all(math.isfinite(v) for v in r["confidence_interval"]))
            if r["market_edge_pp"] is not None:
                self.assertTrue(math.isfinite(r["market_edge_pp"]))
            predictor.format_result(r)


class TestReviewerInputValidation(unittest.TestCase):
    def test_market_price_validation(self):
        self.assertIsNone(predict_match({"market_price_a": [0.6]})["market_edge_pp"])
        self.assertIsNone(predict_match({"market_price_a": float("nan")})["market_edge_pp"])
        self.assertEqual(predict_match({"market_price_a": "0.6"})["market_edge_pp"], -10.0)
        self.assertIsNone(predict_match({"market_price_a": 5})["market_edge_pp"])  # out of range -> ignored
        self.assertIsNone(predict_match({"market_price_a": -0.1})["market_edge_pp"])
        # a list with no map names falls back to the default pool instead of raising
        self.assertAlmostEqual(predict_match({"map_pool": [1, 2, 3]})["p_a"], 0.5)

    def test_huge_counts_finite(self):
        for m in ({"h2h": {"a_wins": 1e308, "b_wins": 1e308}},
                  {"form30_a": .6, "form30_b": .4, "n30_a": 1e308, "n30_b": 1e308},
                  {"maps_a": {mp: [0.5 + 0.05 * i, 1e308] for i, mp in enumerate(POOL)}},
                  {"rating_a": 1e308, "rating_b": -1e308,
                   "opp_rating30_a": 1e308, "opp_rating30_b": -1e308}):
            p = predict_match(m)["p_a_exact"]
            self.assertTrue(math.isfinite(p) and 0 < p < 1, m)

    def test_map_pool_junk_falls_back(self):
        default = predict_match({})["veto_maps"]
        for junk in (7, "abc", {"Dust2": 1}, None):
            r = predict_match({"map_pool": junk})
            self.assertEqual(r["veto_maps"], default)
            self.assertTrue(set(r["veto_maps"]) <= set(POOL))

    def test_roster_string_flags(self):
        self.assertEqual(predict_match({"roster_a": {"standin": "false"}})["p_a_exact"], 0.5)
        self.assertEqual(predict_match({"roster_a": {"standin": "no", "missing_igl": "0"}})["p_a_exact"], 0.5)
        self.assertEqual(predict_match({"roster_a": {"standin": None, "missing_igl": []}})["p_a_exact"], 0.5)
        self.assertLess(predict_match({"roster_a": {"standin": "true"}})["p_a_exact"], 0.5)
        self.assertLess(predict_match({"roster_a": {"missing_igl": 1}})["p_a_exact"], 0.5)


class TestBehaviourGuards(ReferenceWeights, unittest.TestCase):
    """Behavioural tests, each aimed at a specific way the model could regress."""

    def test_form5_does_not_recount_30d_edge(self):
        # Last-5 that merely repeats the 30-day form is not new information:
        # adding it must barely move p (form5 is measured against the 30d form
        # the model already credited, not against the rating expectation).
        m = {"form30_a": 0.7, "form30_b": 0.3, "n30_a": 30, "n30_b": 30}
        p0 = predict_match(m)["p_a_exact"]
        p1 = predict_match(dict(m, form5_a=0.7, form5_b=0.3))["p_a_exact"]
        self.assertLess(abs(p1 - p0), 0.012)
        # ...whereas a genuine swing beyond the 30d form does move it
        p2 = predict_match(dict(m, form5_a=1.0, form5_b=0.0))["p_a_exact"]
        self.assertGreater(p2 - p0, 0.01)

    def test_form30_small_sample_counts_less(self):
        m = {"form30_a": 0.60, "form30_b": 0.50}
        d_small = factor(predict_match(dict(m, n30_a=2, n30_b=2)), "form_30d")["delta_logodds"]
        d_big = factor(predict_match(dict(m, n30_a=40, n30_b=40)), "form_30d")["delta_logodds"]
        self.assertGreater(d_small, 0)
        self.assertLess(d_small, 0.5 * d_big)

    @staticmethod
    def _extreme_veto_match():
        # 3-map pool: A picks X, B picks Y, decider Z; A dominant on X and Z.
        return {"map_pool": ["X", "Y", "Z"],
                "maps_a": {"X": [1.0, 1000], "Y": [0.0, 1000], "Z": [1.0, 1000]},
                "maps_b": {"X": [0.0, 1000], "Y": [1.0, 1000], "Z": [0.0, 1000]}}

    def test_veto_not_inflated(self):
        # The veto factor is a down-weighted series log-odds: never larger than
        # the log-odds of the veto-only series probability itself.
        mild = {"maps_a": {"Dust2": [0.7, 15], "Mirage": [0.4, 15], "Nuke": [0.55, 10]},
                "maps_b": {"Dust2": [0.45, 12], "Mirage": [0.6, 12], "Ancient": [0.5, 9]}}
        for m in (self._extreme_veto_match(), mild):
            r = predict_match(m)
            d = factor(r, "map_veto")["delta_logodds"]
            L = predictor.logit(r["veto_only_series_p_a"])
            self.assertNotEqual(d, 0.0)
            self.assertLessEqual(abs(d), abs(L) + 1e-3)

    def test_veto_capped(self):
        r = predict_match(self._extreme_veto_match())
        self.assertGreater(predictor.logit(r["veto_only_series_p_a"]), CONFIG["signal_cap"])
        self.assertLessEqual(factor(r, "map_veto")["delta_logodds"],
                             CONFIG["signal_cap"] * CONFIG["w_veto"] + 1e-3)

    def test_no_data_factors_are_not_conflict(self):
        # Only a rating edge; every other factor has no data -> zero conflict.
        r = predict_match({"rating_a": 1.04, "rating_b": 1.0})
        self.assertEqual(r["conflict"], 0.0)
        self.assertFalse(any("FACTOR SPLIT" in w for w in r["warnings"]))
        # one genuinely opposing factor -> conflict > 0
        r = predict_match({"rating_a": 1.04, "rating_b": 1.0, "roster_a": {"standin": True}})
        self.assertGreater(r["conflict"], 0.0)

    def test_h2h_relative_to_strength(self):
        # A heavy favourite going 2-1 in H2H is doing WORSE than its rating
        # implies: H2H must not add to the favourite.
        r = predict_match({"rating_a": 1.10, "rating_b": 1.00, "h2h": {"a_wins": 2, "b_wins": 1}})
        self.assertLess(factor(r, "head_to_head")["delta_logodds"], 0.0)
        # an underdog beating the favourite in H2H is credited
        r = predict_match({"rating_a": 1.00, "rating_b": 1.10, "h2h": {"a_wins": 2, "b_wins": 1}})
        self.assertGreater(factor(r, "head_to_head")["delta_logodds"], 0.0)

    def test_rating_measurement_noise_counts(self):
        # Even a zero rating gap is measured with noise: the base factor must
        # report non-zero uncertainty when ratings are given.
        r = predict_match({"rating_a": 1.0, "rating_b": 1.0})
        self.assertGreater(factor(r, "base_strength")["sd_logodds"], 0.0)

    def test_scoreline_maps_temperature_zero_all_even(self):
        old = CONFIG["temperature"]
        try:
            CONFIG["temperature"] = 0.0
            r = predict_match(dict(self._extreme_veto_match(), rating_a=1.05))
            for p in r["map_probs"]:
                self.assertAlmostEqual(p, 0.5, places=3)
            for v in r["series_probs_exact"].values():
                self.assertAlmostEqual(v, 0.25, places=6)
        finally:
            CONFIG["temperature"] = old

    def test_scoreline_map_spread_is_shrunk(self):
        # Final per-map probabilities keep the veto's map structure but never
        # amplify it, and shrink it when the model is uncertain.
        m = {"map_pool": ["X", "Y", "Z"],
             "maps_a": {"X": [0.7, 20], "Y": [0.4, 20], "Z": [0.5, 20]},
             "maps_b": {"X": [0.5, 20], "Y": [0.5, 20], "Z": [0.5, 20]},
             "volatility_a": 0.9, "volatility_b": 0.9}
        r = predict_match(m)
        raw = [predictor.logit(p) for p in r["veto_only_map_probs"]]
        fin = [predictor.logit(p) for p in r["map_probs"]]
        spread_raw = max(raw) - min(raw)
        spread_fin = max(fin) - min(fin)
        self.assertGreater(spread_raw, 0.3)
        self.assertGreater(spread_fin, 0.0)
        self.assertLess(spread_fin, 0.9 * spread_raw)
        # more model uncertainty (volatility) -> less map-specific spread
        r_calm = predict_match(dict(m, volatility_a=0.0, volatility_b=0.0))
        fin_calm = [predictor.logit(p) for p in r_calm["map_probs"]]
        self.assertLess(spread_fin, (max(fin_calm) - min(fin_calm)) - 0.02)


class TestCLI(unittest.TestCase):
    def test_backtest_runs_and_formats(self):
        for r in predictor.day4_backtest():
            txt = predictor.format_result(r)
            self.assertIn("uncertainty band", txt)
            txt.encode("ascii")  # report text is ASCII-safe for Windows consoles


if __name__ == "__main__":
    unittest.main()
