#!/usr/bin/env python3
"""
ORIGINAL (v1) predictor.py prediction core, preserved for old-vs-new
comparison. The original file was never committed to git; it was restored
from the pre-fix audit read on 2026-10-06. CONFIG, simulate_veto and
predict_match are verbatim; the CLI, format_result and the built-in Day-4
backtest were omitted. Do not modify — this is the baseline.

CS2 Match Insights Engine — predictor.py
=========================================
A probabilistic match-analysis engine for Counter-Strike 2 (BO3 series).

WHAT THIS IS: a heuristic log-odds blend. Every factor (base skill, recent
form, head-to-head, map veto, roster, stakes) contributes a log-odds
adjustment; the adjustments are summed and passed through a sigmoid to get
a win probability. It is NOT a trained ML model. The weights below are
reasoned defaults, NOT statistically fitted values.

WHAT THIS IS NOT: a gambling tool. This is a fan analytics engine — built
for the deepest possible insight into how a match is likely to play out.
The market-edge figure is market-efficiency analysis (is the market price
consistent with the model's read?), not betting advice.

DESIGN PRINCIPLES (from Jack, 2026-10-06):
  1. No factor unilaterally decides. All factors feed one algorithm.
  2. Larger samples outrank smaller ones: last-30d > last-5 > single match.
     Single matches never enter the model directly.
  3. Head-to-head gets small weight vs larger-sample current form.
  4. Every factor reports its own uncertainty. Final output is a probability
     WITH a confidence interval. "An algorithm that knows and warns."
  5. Genuine upsets are irreducible. Being wrong with a wide interval is
     correct behavior; being wrong with false confidence is the failure mode.

USAGE:
    python3 predictor.py match.json
    python3 predictor.py --test      # runs the built-in EPL S24 Day 4 backtest

Input JSON schema documented in predictor_README.md.
"""

import json
import math
import sys

# ============================================================================
# CONFIG — ALL TUNABLE WEIGHTS LIVE HERE. Change these, not the code below.
# Each weight multiplies a standardized signal (roughly [-2, +2]) into
# log-odds. No single factor can dominate: the largest single-factor
# contribution is about 2 * weight log-odds, and all factors always vote.
# ============================================================================
CONFIG = {
    # --- factor weights (log-odds multipliers) ---
    "w_base": 0.8,
    # Justification: long-run skill (HLTV rating 2.1 / VRS) is the best single
    # predictor of true ability, but it is also the most market-priced
    # information, so it anchors rather than dominates.

    "w_form30": 1.2,
    # Justification: last-30-day form is the largest-sample CURRENT signal.
    # Per 2026-10-06 correction: larger samples outrank smaller ones, so
    # last-30d carries more weight than last-5 and single matches are excluded.

    "w_form5": 0.25,
    # Justification: last-5 captures heaters and slumps, but 5 series is a
    # tiny sample. Tightly capped so one good/bad week can't swing the model.

    "w_h2h": 0.3,
    # Justification: head-to-head is matchup-specific but built on tiny
    # samples. Small relative to larger-sample current form (2026-10-06
    # FURIA lesson: 9-3 historical H2H must not outrank 47%-vs-63% form).

    "w_veto": 1.0,
    # Justification: a BO3 is won map-by-map, and 90-day map win rates are
    # large samples. Structural map edges (e.g. a 2-4 Dust2 over 90 days)
    # are real signal, not noise.

    # --- signal scaling (maps raw inputs to standardized ~[-2, +2] signals) ---
    "rating_scale": 0.05,
    # A 0.05 HLTV-rating gap ~= one standardized unit of skill difference.
    "form30_scale": 0.15,
    # A 15pp last-30d win-rate gap ~= one standardized unit.
    "form5_scale": 0.25,
    # A 25pp last-5 win-rate gap ~= one unit (wider: small sample, more noise).
    "veto_logit_scale": 0.7,
    # Normalizes the veto-implied log-odds so a 60/40 map edge ~= ~0.6 units.
    "base_signal_cap": 2.0,
    # Caps the base-skill signal so a huge rating gap can't decide alone.

    # --- veto simulation ---
    "map_scale": 6.0,
    # Per-map win prob = sigmoid(map_scale * shrunk win-rate diff).
    # 6.0 means a 20pp map edge ~= 77% map win prob before shrinkage.
    "map_shrink_n": 12.0,
    # Win-rate diffs shrink toward 0 when combined map samples are small:
    # shrink = min(1, (n_a + n_b) / map_shrink_n).

    # --- roster penalties: MULTIPLICATIVE on the odds. Reasoned defaults,
    # NOT fitted values. Documented, tunable, and deliberately round. ---
    "standin_penalty": 0.92,
    # -8%: a stand-in disrupts roles, defaults, and team chemistry.
    "no_igl_penalty": 0.88,
    # -12%: a missing IGL destroys mid-round calling and late-round structure.
    # Both stack multiplicatively (0.92 * 0.88 = 0.81).

    # --- stakes / motivation ---
    "stakes_mult": 1.03,
    # +-3% on the odds when one side has clearly more on the line
    # (elimination / qualification) than the other. Motivation moves margins,
    # not outcomes.

    # --- confidence interval ---
    "ci_base": 0.05,
    "ci_conf_weight": 0.22,   # wider when mean factor confidence is low
    "ci_disagree_weight": 0.55,  # wider when factors disagree with each other
    "ci_vol_weight": 0.18,    # wider when either team is volatile
    "ci_max": 0.32,           # never wider than +-32pp (else the model says nothing)
    "vol_conf_damp": 0.4,     # high volatility damps form-factor confidence
}

MAPS = ["Dust2", "Mirage", "Inferno", "Nuke", "Ancient", "Anubis", "Cache"]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    p = min(0.99, max(0.01, p))
    return math.log(p / (1.0 - p))


# ============================================================================
# VETO SIMULATION — standard BO3 veto: ban, ban, pick, pick, ban, ban, decider.
# Each side bans the map where the opponent's edge is largest, and picks the
# map where its own edge is largest. Permabans are honored as first bans.
# Per-map win probability comes from 90-day win-rate differentials with
# sample-size shrinkage; series probability follows from the three maps.
# ============================================================================
def simulate_veto(m):
    maps_a = m.get("maps_a", {})
    maps_b = m.get("maps_b", {})
    remaining = list(MAPS)
    log = []
    forced = {"A": m.get("permaban_a"), "B": m.get("permaban_b")}
    forced_used = {"A": False, "B": False}

    def wr(team_maps, mp):
        if mp in team_maps:
            return team_maps[mp][0], team_maps[mp][1]
        return 0.5, 4

    def do_ban(team):
        if forced[team] and forced[team] in remaining and not forced_used[team]:
            remaining.remove(forced[team])
            forced_used[team] = True
            log.append(f"{team} bans {forced[team]} (permaban)")
            return
        best, bestv = None, -1e9
        for mp in remaining:
            wa, _ = wr(maps_a, mp)
            wb, _ = wr(maps_b, mp)
            edge = (wb - wa) if team == "A" else (wa - wb)
            if edge > bestv:
                bestv, best = edge, mp
        assert best is not None
        remaining.remove(best)
        log.append(f"{team} bans {best}")

    def do_pick(team):
        best, bestv = None, -1e9
        for mp in remaining:
            wa, _ = wr(maps_a, mp)
            wb, _ = wr(maps_b, mp)
            edge = (wa - wb) if team == "A" else (wb - wa)
            if edge > bestv:
                bestv, best = edge, mp
        assert best is not None
        remaining.remove(best)
        log.append(f"{team} picks {best}")
        return best

    picks = []
    for act in ["A_ban", "B_ban", "A_pick", "B_pick", "A_ban", "B_ban"]:
        if len(remaining) <= 1:
            break
        team, kind = act.split("_")
        if kind == "ban":
            do_ban(team)
        else:
            picks.append(do_pick(team))
    decider = remaining[0]
    log.append(f"decider: {decider}")

    # per-map win probability for A, with sample-size shrinkage
    p_maps = []
    shrinks = []
    for mp in picks + [decider]:
        wa, na = wr(maps_a, mp)
        wb, nb = wr(maps_b, mp)
        shrink = min(1.0, (na + nb) / CONFIG["map_shrink_n"])
        shrinks.append(shrink)
        p = sigmoid(CONFIG["map_scale"] * (wa - wb) * shrink)
        p_maps.append(p)
    p1, p2, p3 = p_maps[0], p_maps[1], p_maps[2]

    # series probabilities (map1 = A's pick, map2 = B's pick, map3 = decider)
    p_2_0 = p1 * p2
    p_2_1 = p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
    p_series_a = p_2_0 + p_2_1
    p_1_2 = (1 - p1) * p2 * (1 - p3) + p1 * (1 - p2) * (1 - p3)
    p_0_2 = (1 - p1) * (1 - p2)

    mean_shrink = sum(shrinks) / len(shrinks)
    confidence = 0.35 + 0.55 * mean_shrink  # 0.35..0.90
    note = (f"veto -> {picks[0]} (A {p1:.0%}), {picks[1]} (A {p2:.0%}), "
            f"decider {decider} (A {p3:.0%}); series P(A)={p_series_a:.1%}")

    return {
        "picks": picks,
        "decider": decider,
        "p_map": p_maps,
        "p_series_a": p_series_a,
        "p_2_0": p_2_0, "p_2_1": p_2_1, "p_1_2": p_1_2, "p_0_2": p_0_2,
        "confidence": confidence,
        "note": note,
        "veto_log": log,
    }


# ============================================================================
# MAIN PREDICTION
# ============================================================================
def predict_match(m):
    a_name = m.get("team_a", "A")
    b_name = m.get("team_b", "B")
    vol_a = m.get("volatility_a", 0.3)
    vol_b = m.get("volatility_b", 0.3)
    vol_max = max(vol_a, vol_b)
    damp = 1.0 - CONFIG["vol_conf_damp"] * vol_max  # volatility damps confidence

    factors = []  # (name, delta_logodds, confidence, note)

    # 1. base strength
    rd = m.get("rating_a", 1.0) - m.get("rating_b", 1.0)
    sig = max(-CONFIG["base_signal_cap"],
              min(CONFIG["base_signal_cap"], rd / CONFIG["rating_scale"]))
    factors.append(("base_strength", CONFIG["w_base"] * sig, 0.85,
                    f"rating {m.get('rating_a', 1.0):.2f} vs {m.get('rating_b', 1.0):.2f}"))

    # 2. form, last 30 days (larger sample -> more weight)
    f30 = m.get("form30_a", 0.5) - m.get("form30_b", 0.5)
    d = CONFIG["w_form30"] * (f30 / CONFIG["form30_scale"])
    n30 = min(m.get("n30_a", 8), m.get("n30_b", 8))
    c = min(0.9, n30 / 15.0) * damp
    factors.append(("form_30d", d, c,
                    f"{m.get('form30_a', 0.5):.0%} vs {m.get('form30_b', 0.5):.0%} last 30d"
                    f" (~{n30} series/side)"))

    # 3. form, last 5 (small sample -> capped)
    f5 = m.get("form5_a", 0.5) - m.get("form5_b", 0.5)
    d = CONFIG["w_form5"] * (f5 / CONFIG["form5_scale"])
    c = 0.5 * damp
    factors.append(("form_last5", d, c,
                    f"{m.get('form5_a', 0.5):.0%} vs {m.get('form5_b', 0.5):.0%} last 5"))

    # 4. head-to-head (tiny samples -> shrinkage + small weight)
    h = m.get("h2h", {"a_wins": 0, "b_wins": 0, "meetings": 0})
    n = h.get("meetings", 0)
    if n == 0:
        factors.append(("head_to_head", 0.0, 0.2, "no recent meetings -> neutral"))
    else:
        wr_h = h["a_wins"] / n
        shrink = n / (n + 4.0)
        d = CONFIG["w_h2h"] * (wr_h - 0.5) * 2 * shrink
        factors.append(("head_to_head", d, shrink,
                        f"{h['a_wins']}-{h['b_wins']} last {n} (shrunk x{shrink:.2f})"))

    # 5. map veto
    veto = simulate_veto(m)
    p_v = min(0.85, max(0.15, veto["p_series_a"]))
    d = CONFIG["w_veto"] * (logit(p_v) / CONFIG["veto_logit_scale"])
    factors.append(("map_veto", d, veto["confidence"], veto["note"]))

    # 6. roster — applied as a DIRECT probability multiplier AFTER the log-odds
    # blend (spec: "-8% win prob", "-12% win prob", multiplicative, stacking).
    # A log-odds formulation would dilute the penalty at the extremes; the
    # direct multiplier keeps the stated meaning: P(win) *= 0.92 / 0.88.
    pen_a, pen_b = 1.0, 1.0
    rnotes = []
    ra, rb = m.get("roster_a", {}), m.get("roster_b", {})
    if ra.get("standin"):
        pen_a *= CONFIG["standin_penalty"]
        rnotes.append(f"{a_name} stand-in x{CONFIG['standin_penalty']}")
    if ra.get("missing_igl"):
        pen_a *= CONFIG["no_igl_penalty"]
        rnotes.append(f"{a_name} missing IGL x{CONFIG['no_igl_penalty']}")
    if rb.get("standin"):
        pen_b *= CONFIG["standin_penalty"]
        rnotes.append(f"{b_name} stand-in x{CONFIG['standin_penalty']}")
    if rb.get("missing_igl"):
        pen_b *= CONFIG["no_igl_penalty"]
        rnotes.append(f"{b_name} missing IGL x{CONFIG['no_igl_penalty']}")
    roster_note = "; ".join(rnotes) if rnotes else "both full strength"
    # log-odds equivalent, used for the confidence-interval math and breakdown
    d_roster = math.log(pen_a / pen_b) if (pen_a != 1.0 or pen_b != 1.0) else 0.0

    # 7. stakes / motivation (small)
    sa, sb = m.get("stakes_a", "none"), m.get("stakes_b", "none")
    if sa != "none" and sb == "none":
        d_stakes, snote = math.log(CONFIG["stakes_mult"]), f"{a_name} more on the line ({sa})"
    elif sb != "none" and sa == "none":
        d_stakes, snote = -math.log(CONFIG["stakes_mult"]), f"{b_name} more on the line ({sb})"
    else:
        d_stakes, snote = 0.0, "even stakes"
    factors.append(("stakes", d_stakes, 0.4, snote))

    # --- combine: every log-odds factor contributes; none decides alone ---
    total = sum(d for _, d, _, _ in factors)
    p_raw = sigmoid(total)

    # --- roster penalty: direct probability multiplier (see factor 6) ---
    p_a_pre, p_b_pre = p_raw * pen_a, (1.0 - p_raw) * pen_b
    if pen_a != 1.0 and pen_b != 1.0:
        # both sides penalized (rare): renormalize so probabilities sum to 1
        s = p_a_pre + p_b_pre
        p_a, p_b = p_a_pre / s, p_b_pre / s
    elif pen_a != 1.0:
        p_a, p_b = p_a_pre, 1.0 - p_a_pre
    elif pen_b != 1.0:
        p_b, p_a = p_b_pre, 1.0 - p_b_pre
    else:
        p_a, p_b = p_raw, 1.0 - p_raw

    # --- per-factor implied probabilities (for disagreement measurement) ---
    # roster participates here via its log-odds equivalent
    ci_factors = factors + [("roster", d_roster, 0.95, roster_note)]
    p_is = [sigmoid(d) for _, d, _, _ in ci_factors]
    confs = [c for _, _, c, _ in ci_factors]
    wsum = sum(abs(d) for _, d, _, _ in ci_factors)
    if wsum < 1e-9:
        mean_conf = sum(confs) / len(confs)
    else:
        mean_conf = sum(c * abs(d) for (_, d, c, _) in ci_factors) / wsum
    csum = sum(confs)
    mean_p = sum(p * c for p, c in zip(p_is, confs)) / csum
    var = sum(c * (p - mean_p) ** 2 for p, c in zip(p_is, confs)) / csum
    disagree = math.sqrt(var)

    # --- confidence interval: wider when confidence is low, factors disagree,
    #     or either team is volatile. Being wrong inside a wide interval is
    #     correct behavior; false confidence is the failure mode. ---
    hw = (CONFIG["ci_base"]
          + CONFIG["ci_conf_weight"] * (1 - mean_conf)
          + CONFIG["ci_disagree_weight"] * disagree
          + CONFIG["ci_vol_weight"] * vol_max)
    hw = max(0.05, min(CONFIG["ci_max"], hw))
    ci_lo, ci_hi = max(0.01, p_a - hw), min(0.99, p_a + hw)

    # --- scoreline: 2-0 needs real dominance, else 2-1 for the favored side ---
    fav_is_a = p_a >= 0.5
    fav_p = p_a if fav_is_a else p_b
    fav_name = a_name if fav_is_a else b_name
    dog_name = b_name if fav_is_a else a_name
    score = "2-0" if fav_p >= 0.65 else "2-1"
    scoreline = f"{fav_name} {score}"

    # --- marginal contribution of each factor (pp impact of removing it) ---
    # For log-odds factors: P(total) - P(total - delta). For roster: the
    # direct-multiplier impact vs the pre-roster probability.
    breakdown = []
    for (name, d, c, note) in factors:
        p_without = sigmoid(total - d)
        contrib_pp = (p_raw - p_without) * 100
        breakdown.append({
            "factor": name,
            "delta_logodds": round(d, 3),
            "implied_win_pct_alone": round(sigmoid(d) * 100, 1),
            "marginal_pp": round(contrib_pp, 1),
            "confidence": round(c, 2),
            "note": note,
        })
    breakdown.append({
        "factor": "roster",
        "delta_logodds": round(d_roster, 3),
        "implied_win_pct_alone": round(sigmoid(d_roster) * 100, 1),
        "marginal_pp": round((p_a - p_raw) * 100, 1),
        "confidence": 0.95,
        "note": roster_note + " (direct probability multiplier)",
    })

    # --- market-efficiency edge (NOT betting advice) ---
    edge = None
    edge_note = None
    mp = m.get("market_price_a")
    if mp is not None:
        edge = round((p_a - mp) * 100, 1)  # in percentage points
        if edge > 0:
            edge_note = (f"model prices {a_name} {edge:.1f}pp above the market "
                         f"({p_a:.0%} vs {mp:.0%}) — market may be underpricing {a_name}.")
        elif edge < 0:
            edge_note = (f"model prices {a_name} {abs(edge):.1f}pp below the market "
                         f"({p_a:.0%} vs {mp:.0%}) — market may be overpricing {a_name}.")
        else:
            edge_note = "model agrees with the market."
        edge_note += " Market-efficiency read only, not a recommendation."

    # --- warnings: the 'knows and warns' part ---
    warnings = []
    if hw > 0.20:
        warnings.append(
            f"WIDE INTERVAL (+-{hw:.0%}): factors disagree or confidence is low. "
            f"Treat the point estimate with caution.")
    if vol_max > 0.6:
        warnings.append(
            "HIGH VOLATILITY: recent form contradicts the base rating — "
            "upset risk is elevated regardless of the point estimate.")
    # name the biggest disagreement (only among factors with real signal)
    if disagree > 0.08:
        sig_factors = [f for f in ci_factors if abs(f[1]) >= 0.05]
        if len(sig_factors) >= 2:
            most_bull = max(sig_factors, key=lambda f: sigmoid(f[1]))
            most_bear = min(sig_factors, key=lambda f: sigmoid(f[1]))
            warnings.append(
                f"FACTOR SPLIT: '{most_bull[0]}' implies {sigmoid(most_bull[1]):.0%} for {a_name}, "
                f"'{most_bear[0]}' implies {sigmoid(most_bear[1]):.0%}.")
    reliability = "LOW" if hw > 0.20 else ("MEDIUM" if hw > 0.12 else "HIGH")

    return {
        "match": f"{a_name} vs {b_name}",
        "p_a": round(p_a, 3),
        "p_b": round(p_b, 3),
        "pick": fav_name,
        "scoreline": scoreline,
        "confidence_interval": [round(ci_lo, 3), round(ci_hi, 3)],
        "interval_width_pp": round(hw * 200, 1),
        "reliability": reliability,
        "factor_breakdown": breakdown,
        "veto_log": veto["veto_log"],
        "series_probs": {
            "p_2_0": round(veto["p_2_0"], 3),
            "p_2_1": round(veto["p_2_1"], 3),
            "p_1_2": round(veto["p_1_2"], 3),
            "p_0_2": round(veto["p_0_2"], 3),
        },
        "market_edge_pp": edge,
        "market_edge_note": edge_note,
        "warnings": warnings,
        "volatility": {"a": vol_a, "b": vol_b},
    }
