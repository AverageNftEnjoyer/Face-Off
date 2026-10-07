#!/usr/bin/env python3
"""
CS2 Match Insights Engine -- predictor.py
=========================================
A probabilistic match-analysis engine for Counter-Strike 2 (BO3 series).

WHAT THIS IS: a heuristic log-odds blend. Every factor (base skill, recent
form, head-to-head, map veto, roster, stakes) contributes a log-odds
adjustment; the adjustments are summed, shrunk toward 0 by the model's own
uncertainty, scaled by a temperature, and passed through a sigmoid to get a
win probability. It is NOT a trained ML model. The weights below are
reasoned defaults, NOT statistically fitted values. The only parameter meant
to be fitted on real results is CONFIG["temperature"] (see backtest.py).

WHAT THIS IS NOT: a gambling tool. This is a fan analytics engine -- built
for the deepest possible insight into how a match is likely to play out.
The market-edge figure is market-efficiency analysis (is the market price
consistent with the model's read?), not betting advice.

DESIGN PRINCIPLES (from Jack, 2026-10-06):
  1. No factor unilaterally decides. All factors feed one algorithm.
  2. Larger samples outrank smaller ones: last-30d > last-5 > single match.
     Single matches never enter the model directly.
  3. Head-to-head gets small weight vs larger-sample current form.
  4. Every factor reports its own uncertainty. Final output is a probability
     WITH an uncertainty band. "An algorithm that knows and warns."
  5. Genuine upsets are irreducible. Being wrong with a wide band is
     correct behavior; being wrong with false confidence is the failure mode.

STRENGTH IS COUNTED ONCE: the base rating is the only factor that says "who
is better". Every other factor is a RESIDUAL against what the rating already
implies: 30-day form vs the rating-implied form gap, last-5 vs 30-day form,
head-to-head vs the rating-implied result, and map edges relative to each
team's own average across the pool (a team that is uniformly better on every
map gets ~0 veto signal, because that edge is already in the rating).

USAGE:
    python predictor.py match.json
    python predictor.py --test      # runs the built-in EPL S24 Day 4 backtest

INPUT JSON SCHEMA (every key optional; defaults in brackets):
    team_a, team_b            names                               ["A", "B"]
    rating_a, rating_b        HLTV-2.1-like team rating, ~1.0     [1.0]
    form30_a, form30_b        series win rate, last 30 days, 0..1 [absent = no data]
    n30_a, n30_b              series played in that window        [8 if form30 given]
    opp_rating30_a/_b         avg opponent rating in that window  [absent = assume equal]
    form5_a, form5_b          series win rate, last 5 series      [absent = no data]
    h2h                       {"a_wins", "b_wins", "meetings"}    [no meetings]
                              (sample = a_wins + b_wins; "meetings" is ignored)
    maps_a, maps_b            {map: [win_rate_90d, maps_played]}  [map absent = no data]
    map_pool                  list of map names                   [CONFIG["map_pool"]]
    permaban_a, permaban_b    map always banned first             [None]
    roster_a, roster_b        {"standin": bool, "missing_igl": bool}  [{}]
    stakes_a, stakes_b        "none" | "qualification" | "elimination" | "title"  ["none"]
    volatility_a/_b           0..1, how much recent results contradict the rating [0.3]
    market_price_a            market-implied P(A), 0..1, for the efficiency read  [None]
"""

import json
import math
import sys

# ============================================================================
# CONFIG -- ALL TUNABLE WEIGHTS LIVE HERE. Change these, not the code below.
# Each weight multiplies a standardized signal that is HARD-CAPPED to
# [-signal_cap, +signal_cap] (2.0), so the largest single-factor contribution
# is exactly signal_cap * weight log-odds (roster and stakes have their own
# documented bounds). No factor can decide alone; every factor with data votes.
# ============================================================================
CONFIG = {
    # --- calibration ---
    "temperature": 1.0,
    # Final p = sigmoid(temperature * total_logodds). The fitted weights below
    # already carry the overall scale (fit_weights.py fits w_i with the
    # temperature held at 1), so this stays 1.0. If weights are changed by
    # hand again, re-fit it with backtest.py on the TRAIN split.

    # --- factor weights (log-odds multipliers) ---
    # FITTED by fit_weights.py (2026-10-06): penalised logistic regression of
    # series outcomes on each factor's log-odds contribution, multipliers
    # constrained >= 0, L2 strength chosen on the last 20% of TRAIN (lambda 0
    # won). TRAIN = 917 real BO3 series 2023-10-22 to 2025-10-08 (Liquipedia);
    # the 611 later series were held out. Held-out result vs the previous
    # reasoned weights: Brier 0.2134 -> 0.2109, log loss 0.6161 -> 0.6104,
    # ECE 0.0275 -> 0.0247, accuracy unchanged (0.661); both paired 95% CIs
    # include 0, so the gain is small and not yet significant.
    # Caveat: the >= 0 constraint was added after seeing an unconstrained fit
    # (which flipped w_veto negative) evaluated on the held-out split.
    "w_base": 0.7955,
    # Elo-based strength: the only factor that carries clear predictive
    # weight; fitted value is essentially the reasoned 0.8.

    "w_form30": 0.1049,
    # 30-day form residual (vs the rating-implied gap). Fitted far below the
    # reasoned 0.7: series Elo already absorbs recent results, so form beyond
    # it adds little (single-factor multiplier ~0.17 on both train and test).

    "w_form5": 0.0436,
    # Last-5 swing vs 30-day form: nearly switched off by the fit.

    "w_h2h": 0.177,
    # Head-to-head residual vs strength: small positive weight; noisy
    # (single-factor multiplier 0.6 on train, 3.0 on test, few meetings).

    "w_veto": 0.0,
    # Simulated-veto series edge: OFF. Its best-fit multiplier on top of base
    # strength was negative on both train and test, with plain map win rates
    # (-1.07 / -0.98) and with Elo-residual map rates (-0.77 / -0.38). Until a
    # map signal shows positive held-out value it does not move p_a. The veto
    # simulation still runs. Per-map win chances keep that map-to-map shape
    # (map_shape, below) and are shifted together so the series probability
    # stays equal to p_a.

    # --- signal scaling and caps ---
    "signal_cap": 2.0,
    # Every standardized signal (base, form30, form5, h2h, veto) is clamped to
    # [-2, +2] before weighting. This is what makes the header claim true.
    "rating_scale": 0.05,
    # A 0.05 HLTV-rating gap ~= one standardized unit of skill difference.
    "form30_scale": 0.15,
    # A 15pp deviation of the 30-day form gap from expectation ~= one unit.
    "form5_scale": 0.25,
    # A 25pp last-5 vs 30-day swing gap ~= one unit (wider: small sample).
    "form_rating_beta": 2.0,
    # Expected 30-day series-win-rate gap per unit of rating gap: a 0.05
    # rating gap ~= 10pp expected form gap. HEURISTIC; should be estimated by
    # regressing the observed form gap on the rating gap across many pairs.
    "form_shrink_k": 10.0,
    # Sample-size shrinkage for form signals: shrink(n) = n / (n + k).
    # n=12 -> 0.55, n=5 -> 0.33, n=30 -> 0.75. k=10 corresponds to a prior sd
    # of ~16pp on a team's true form deviation (k ~= 0.25 / tau^2).
    "h2h_shrink_k": 4.0,
    # H2H shrinkage: n/(n+4). 2 meetings -> 0.33, 10 -> 0.71.
    "h2h_vs_strength": True,
    # True: H2H is a residual vs the rating-implied result (a favourite beating
    # an underdog in H2H is not new information). False: residual vs 50%.

    # --- veto simulation ---
    "map_pool": ["Dust2", "Mirage", "Inferno", "Nuke", "Ancient", "Anubis", "Cache"],
    # VERIFY against the current CS2 Active Duty pool before use; the Day 4
    # backtest data below uses this 7-map pool. A per-match "map_pool" input
    # overrides it. Pools of 3..N maps are supported.
    "map_scale": 6.0,
    # Per-map logit for A = map_scale * (relative edge A - relative edge B),
    # where a relative edge is the team's shrunk map rate minus its own
    # pool-wide average. A 10pp relative edge each way (20pp) ~= 1.2 logit.
    "map_shape": 0.75,
    # Share of that veto gap kept in the three map win chances. Separate from
    # w_veto, which only decides whether the veto moves the series winner.
    # _scoreline_maps then adds one constant to all three logits so they still
    # imply p_a. 0 would print the same win chance on every map.
    "map_shrink_k": 10.0,
    # Per-team map-rate shrinkage toward the team's own pool average:
    # n/(n+10). A missing map is n=0 (no data), never a fabricated sample.

    # --- roster: penalties are ODDS multipliers, entered as log(pen_a) -
    # log(pen_b) into the log-odds sum. Reasoned defaults, NOT fitted. ---
    "standin_penalty": 0.92,
    # -8% on the odds: a stand-in disrupts roles, defaults, and chemistry.
    "no_igl_penalty": 0.88,
    # -12% on the odds: missing IGL, no stand-in flagged (role reshuffle).
    "standin_igl_penalty": 0.84,
    # Both flags = usually ONE absent player (the IGL) replaced by a stand-in.
    # Not 0.92*0.88=0.81 (that double-penalizes one absence); the IGL loss is
    # treated as incremental on top of the stand-in disruption.

    # --- stakes / motivation ---
    "stakes_mult": 1.03,
    # Odds multiplier per stakes LEVEL of difference (none=0, qualification=1,
    # elimination=2, title=2; unknown non-"none" strings=1), level diff capped
    # at +-2. 1.03 ~= 0.7pp near 50% -- unfitted and near the noise floor;
    # kept only as a tie-breaker-sized nudge.

    # --- uncertainty propagation (all in log-odds units; heuristic) ---
    "tau_base_missing": 0.6,
    # sd of the base factor when either rating is missing (strength unknown).
    "rating_noise_sd": 0.01,
    # Measurement noise of each team rating; propagated through w_base.
    "tau_form30": 0.35,
    "tau_form5": 0.20,
    "tau_h2h": 0.20,
    # Prior sd of each factor's true log-odds effect. Posterior variance =
    # tau^2 * k/(n+k): shrinks with sample size; no data -> full tau^2.
    "tau_map": 0.5,
    # Prior sd of a team's true per-map logit deviation; per-map posterior
    # variance tau_map^2 * k/(n+k) per team, propagated to the series logit
    # (d series-logit / d map-logit ~= 0.5 per map near 50%).
    "veto_path_cv": 0.3,
    "veto_path_floor": 0.05,
    # Veto-path uncertainty (the simulated bans/picks may not happen):
    # sd += 30% of |veto delta|, plus a 0.05 floor.
    "weight_cv": 0.25,
    # Weight uncertainty: every unfitted weight is +-25% (1 sd), so each
    # factor adds (0.25 * delta)^2 of variance.
    "roster_cv": 0.5,
    "stakes_cv": 1.0,
    # Roster/stakes penalties are guesses: +-50% and +-100% of their effect.
    "vol_sd": 1.0,
    # Volatility (0..1) -> latent log-odds sd: vol_sd * rms(vol_a, vol_b).
    # Total sd s combines this with every factor variance; the point estimate
    # is shrunk as total / sqrt(1 + pi*s^2/8) (probit approximation to the
    # expected sigmoid under that uncertainty).

    # --- uncertainty band and reliability ---
    "band_z": 1.2816,
    # Band = sigmoid(logit(p) +- z * temperature * s); z=1.28 is a nominal 80%
    # band IF s were calibrated. It is NOT yet coverage-validated.
    "rel_high_width_pp": 29.3,
    "rel_med_width_pp": 33.6,
    # Reliability tier from the band WIDTH in probability points (interval_width_pp):
    # HIGH if width <= 29.3, MEDIUM if <= 33.6, else LOW. Cut-offs are the
    # 33rd/67th percentiles of width on the TRAIN split (917 series, 2023-10-22
    # to 2025-10-08) under the fitted weights. Brier by tier: train 0.196 /
    # 0.232 / 0.229 (MEDIUM and LOW about level), held-out test 0.188 / 0.216 /
    # 0.240. Width is largely a proxy for how close p is to 50%, so the tiers
    # mostly say "how clear-cut the read is", not "extra error beyond p".
    # Tiers were NOT based on logit_sd: on train, Brier FELL as logit_sd rose.

    # --- factor-split warning ---
    "split_min_conflict": 0.25,
    "split_min_opposing": 0.15,
    # FACTOR SPLIT only if factors WITH DATA that oppose the overall lean carry
    # >= 25% of the total |log-odds| mass AND >= 0.15 log-odds in absolute terms.
}

MAPS = CONFIG["map_pool"]  # backward-compat alias (same list object)

STAKES_LEVELS = {"none": 0, "": 0, "qualification": 1, "elimination": 2, "title": 2}


# ============================================================================
# NUMERIC HELPERS (overflow-safe)
# ============================================================================
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p, eps=1e-12):
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _cap(x):
    c = CONFIG["signal_cap"]
    return _clamp(x, -c, c)


def _num(v, default, lo=None, hi=None):
    """Parse a number defensively: bad / missing / non-finite -> default."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


MAX_COUNT = 1e6  # cap on any sample count read from input (1e308 would overflow sums)
RATING_BOUND = 100.0  # ratings/opp ratings outside +-100 are junk; clamp to keep arithmetic finite


def _count(v, default=0.0):
    """Sample counts: finite, >= 0, <= MAX_COUNT."""
    return _num(v, default, 0.0, MAX_COUNT)


_TRUE = {"true", "t", "yes", "y", "1", "on"}


def _flag(v):
    """Strict boolean parse: {"standin": "false"} must NOT count as a stand-in.
    True only for True, a non-zero finite number, or a string in _TRUE."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return math.isfinite(v) and v != 0
    if isinstance(v, str):
        return v.strip().lower() in _TRUE
    return False


def _has(m, key):
    return m.get(key) is not None and _num(m.get(key), None) is not None


def _shrink(n, k):
    return n / (n + k) if n > 0 else 0.0


def series_probs(p1, p2, p3):
    """BO3 scorelines from per-map P(A) (map1=A pick, map2=B pick, map3=decider).
    Returns (p_2_0, p_2_1, p_1_2, p_0_2); they sum to 1.
    Map outcomes are independent given those three probabilities."""
    p_2_0 = p1 * p2
    p_2_1 = (p1 * (1 - p2) + (1 - p1) * p2) * p3
    p_1_2 = (p1 * (1 - p2) + (1 - p1) * p2) * (1 - p3)
    p_0_2 = (1 - p1) * (1 - p2)
    return p_2_0, p_2_1, p_1_2, p_0_2


BO5_KEYS = ["p_3_0", "p_3_1", "p_3_2", "p_2_3", "p_1_3", "p_0_3"]


def series_scorelines(p_maps):
    """Every final score of a best-of-N series, N = len(p_maps) (odd), from
    per-map P(A); map i uses p_maps[i] and play stops once a side has
    (N + 1) // 2 maps. Returns A's wins by margin then B's, from A's view:
    BO3 [2-0, 2-1, 1-2, 0-2] (same values as series_probs), BO5 [3-0, 3-1,
    3-2, 2-3, 1-3, 0-3]. Sums to 1; map outcomes are independent."""
    n = len(p_maps)
    if n < 1 or n % 2 == 0:
        raise ValueError(f"best-of needs an odd number of maps, got {n}")
    need = (n + 1) // 2
    a_end, b_end = [0.0] * need, [0.0] * need   # indexed by the loser's map count
    state = {(0, 0): 1.0}
    for p in p_maps:
        nxt = {}
        for (i, j), pr_ in state.items():
            for ii, jj, q in ((i + 1, j, p), (i, j + 1, 1.0 - p)):
                if ii == need:
                    a_end[jj] += pr_ * q
                elif jj == need:
                    b_end[ii] += pr_ * q
                else:
                    nxt[(ii, jj)] = nxt.get((ii, jj), 0.0) + pr_ * q
        state = nxt
    return a_end + b_end[::-1]


def flat_map_prob(p_series, n_maps):
    """The one per-map P(A) that, played on every map of a best-of-n_maps,
    gives series P(A) = p_series (bisection, full precision)."""
    need = (n_maps + 1) // 2
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if sum(series_scorelines([mid] * n_maps)[:need]) < p_series:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def display_percent(probs, places=2):
    """Largest-remainder percentages for one complete probability set.

    `probs` are shares of a single whole (a scoreline, or a team and its
    opponent). Returns one float per input, each with exactly `places`
    decimal places, summing to exactly 100. Ties in the remainder go to the
    earlier input. The viewer copies this rule; keep the two in step.
    """
    n = len(probs)
    if n == 0:
        return []
    scale = 10 ** places
    target = 100 * scale
    weights = []
    for p in probs:
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0:
            weights.append(0.0)
        else:
            weights.append(float(p))
    total = sum(weights)
    if total <= 0:
        base = [0] * n
        base[0] = target
        return [b / scale for b in base]
    raw = [w / total * target for w in weights]
    base = [int(math.floor(x + 1e-9)) for x in raw]
    leftover = target - sum(base)
    if leftover > 0:
        order = sorted(range(n), key=lambda i: (raw[i] - math.floor(raw[i] + 1e-9), -i), reverse=True)
        for k in range(leftover):
            base[order[k]] += 1
    elif leftover < 0:
        order = sorted(range(n), key=lambda i: (raw[i] - math.floor(raw[i] + 1e-9), i))
        k = 0
        while leftover < 0 and k < n:
            i = order[k]
            if base[i] > 0:
                base[i] -= 1
                leftover += 1
            k += 1
    return [b / scale for b in base]


def _series_a(logits, c=0.0):
    p = [sigmoid(L + c) for L in logits]
    s = series_probs(*p)
    return s[0] + s[1]


# ============================================================================
# VETO SIMULATION -- standard BO3 veto: ban, ban, pick, pick, ban, ban, decider
# (generalised: bans are skipped when the pool is too small, and extra
# alternating bans are added when it is larger than 7). Each side bans the
# map where the opponent's RELATIVE edge is largest and picks the map where
# its own relative edge is largest. Permabans are honored as first bans.
# Relative edge = team's shrunk map rate minus its own pool-wide average, so
# the veto carries map-specific structure only, never overall strength.
# ============================================================================
def _team_maps(raw, pool):
    raw = raw if isinstance(raw, dict) else {}
    k = CONFIG["map_shrink_k"]
    rates = {}
    for mp in pool:
        e = raw.get(mp)
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            wr = _num(e[0], 0.5, 0.0, 1.0)
            n = _count(e[1])
        else:
            wr, n = 0.5, 0.0  # no data: n=0, contributes nothing
        rates[mp] = (wr, n)
    tot_n = sum(n for _, n in rates.values())
    overall = (sum(wr * n for wr, n in rates.values()) / tot_n) if tot_n > 0 else 0.5
    rel, shr = {}, {}
    for mp, (wr, n) in rates.items():
        s = _shrink(n, k)
        shr[mp] = s
        rel[mp] = s * (wr - overall)
    return rel, shr, overall


def _pool(m):
    pool = m.get("map_pool")
    if (not isinstance(pool, (list, tuple))
            or not any(isinstance(mp, str) for mp in pool)):
        # junk types (7, "abc", {}) or lists with no map names ([1, 2, 3]) fall
        # back to the default; a real but too-short list of names still raises
        pool = CONFIG["map_pool"]
    out = []
    for mp in pool:
        if isinstance(mp, str) and mp not in out:
            out.append(mp)
    if len(out) < 3:
        raise ValueError(f"map pool needs at least 3 distinct maps for a BO3 veto, got {len(out)}: {out}")
    return out


def simulate_veto(m):
    pool = _pool(m)
    rel_a, shr_a, ov_a = _team_maps(m.get("maps_a"), pool)
    rel_b, shr_b, ov_b = _team_maps(m.get("maps_b"), pool)
    edge = {mp: rel_a[mp] - rel_b[mp] for mp in pool}  # A's relative edge

    remaining = list(pool)
    log = []
    forced = {"A": m.get("permaban_a"), "B": m.get("permaban_b")}
    forced_used = {"A": False, "B": False}
    picks = []

    def do_ban(team):
        f = forced[team]
        if f and not forced_used[team] and f in remaining:
            remaining.remove(f)
            forced_used[team] = True
            log.append(f"{team} bans {f} (permaban)")
            return
        forced_used[team] = True  # permaban only applies to the first ban
        # A bans the map where B's edge is largest (min A edge); B the reverse.
        sign = 1.0 if team == "A" else -1.0
        best = min(remaining, key=lambda mp: sign * edge[mp])
        remaining.remove(best)
        log.append(f"{team} bans {best}")

    def do_pick(team):
        sign = 1.0 if team == "A" else -1.0
        best = max(remaining, key=lambda mp: sign * edge[mp])
        remaining.remove(best)
        log.append(f"{team} picks {best}")
        picks.append(best)

    for act in ["A_ban", "B_ban", "A_pick", "B_pick", "A_ban", "B_ban"]:
        team, kind = act.split("_")
        if kind == "pick":
            do_pick(team)
        elif len(remaining) - (2 - len(picks)) > 1:  # keep room for picks + decider
            do_ban(team)
    turn = "A"
    while len(remaining) > 1:  # pools larger than 7: keep alternating bans
        do_ban(turn)
        turn = "B" if turn == "A" else "A"
    decider = remaining[0]
    log.append(f"decider: {decider}")

    veto_maps = picks + [decider]
    map_logits = [CONFIG["map_scale"] * edge[mp] for mp in veto_maps]
    p_maps = [sigmoid(L) for L in map_logits]
    p_series_a = _series_a(map_logits)

    # posterior variance of each per-map logit (both teams' shrinkage)
    tau2 = CONFIG["tau_map"] ** 2
    map_vars = []
    for mp in veto_maps:
        na_term = 1.0 - shr_a[mp]  # = k/(n+k), 1 when no data
        nb_term = 1.0 - shr_b[mp]
        map_vars.append(tau2 * (na_term + nb_term))
    mean_shrink = sum((shr_a[mp] + shr_b[mp]) / 2 for mp in veto_maps) / 3.0

    note = (f"veto-only (pool-relative) -> {picks[0]} (A {p_maps[0]:.2%}), "
            f"{picks[1]} (A {p_maps[1]:.2%}), decider {decider} (A {p_maps[2]:.2%}); "
            f"veto-only series P(A)={p_series_a:.1%}")
    s20, s21, s12, s02 = series_probs(*p_maps)
    return {
        "picks": picks,
        "decider": decider,
        "maps": veto_maps,
        "map_logits": map_logits,
        "map_vars": map_vars,
        "p_map": p_maps,
        "p_series_a": p_series_a,
        "p_2_0": s20, "p_2_1": s21, "p_1_2": s12, "p_0_2": s02,
        "confidence": mean_shrink,
        "overall_rate": {"a": ov_a, "b": ov_b},
        "note": note,
        "veto_log": log,
    }


# ============================================================================
# FACTORS -- each returns (name, delta_logodds, variance, reliability, note)
# ============================================================================
def _compute(m):
    if not isinstance(m, dict):
        m = {}
    a_name = str(m.get("team_a", "A"))
    b_name = str(m.get("team_b", "B"))
    vol_a = _num(m.get("volatility_a", 0.3), 0.3, 0.0, 1.0)
    vol_b = _num(m.get("volatility_b", 0.3), 0.3, 0.0, 1.0)
    wcv2 = CONFIG["weight_cv"] ** 2
    factors = []

    # 1. base strength -- the ONLY factor that encodes "who is better"
    ra = _num(m.get("rating_a", 1.0), 1.0, -RATING_BOUND, RATING_BOUND)
    rb = _num(m.get("rating_b", 1.0), 1.0, -RATING_BOUND, RATING_BOUND)
    rd = ra - rb
    sig = _cap(rd / CONFIG["rating_scale"])
    d_base = CONFIG["w_base"] * sig
    if _has(m, "rating_a") and _has(m, "rating_b"):
        noise = CONFIG["w_base"] * CONFIG["rating_noise_sd"] * math.sqrt(2) / CONFIG["rating_scale"]
        var = noise ** 2 + wcv2 * d_base ** 2
        rel_base = 1.0
    else:
        var = CONFIG["tau_base_missing"] ** 2 + wcv2 * d_base ** 2
        rel_base = 0.0
    factors.append(["base_strength", d_base, var, rel_base, f"rating {ra:.2f} vs {rb:.2f}"])

    # Rating-implied form gap. The rating gap is clamped FIRST, exactly where
    # the base signal saturates, so the rating-dependent part of "expected
    # form" moves only where the base factor moves (keeps p monotone in the
    # rating gap; the base slope always dominates the form30 slope it offsets).
    # Optional opponent-strength adjustment: a team that faced tougher
    # opposition is expected to have a lower win rate. It is added AFTER the
    # clamp as a rating-independent offset (itself clamped to the same range),
    # so it can never move the saturation point. Without opp_rating30_* we
    # cannot adjust (documented limitation).
    cap_rd = CONFIG["signal_cap"] * CONFIG["rating_scale"]
    opp_note = ""
    eff_rd = _clamp(rd, -cap_rd, cap_rd)
    if _has(m, "opp_rating30_a") and _has(m, "opp_rating30_b"):
        oa = _num(m["opp_rating30_a"], 1.0, -RATING_BOUND, RATING_BOUND)
        ob = _num(m["opp_rating30_b"], 1.0, -RATING_BOUND, RATING_BOUND)
        eff_rd -= _clamp(oa - ob, -cap_rd, cap_rd)
        opp_note = f", opp-adj (opp {oa:.2f} vs {ob:.2f})"
    exp_gap = CONFIG["form_rating_beta"] * eff_rd

    # 2. form, last 30 days -- deviation from the rating-implied gap
    fk = CONFIG["form_shrink_k"]
    has30 = _has(m, "form30_a") and _has(m, "form30_b")
    f30a = _num(m.get("form30_a", 0.5), 0.5, 0.0, 1.0)
    f30b = _num(m.get("form30_b", 0.5), 0.5, 0.0, 1.0)
    if has30:
        n30a = _count(m.get("n30_a", 8), 8.0)
        n30b = _count(m.get("n30_b", 8), 8.0)
        n_eff = 2.0 * n30a * n30b / (n30a + n30b) if n30a > 0 and n30b > 0 else 0.0
        sh = _shrink(n_eff, fk)
        dev = (f30a - f30b) - exp_gap
        z30 = _cap(sh * dev / CONFIG["form30_scale"])
        d = CONFIG["w_form30"] * z30
        # The form gap the model actually CREDITED: the rating expectation plus
        # the SHRUNK, CAPPED deviation (the model's posterior estimate of the
        # current form gap). form5 is measured against this. Because form5's
        # dependence on form30 is then scaled by the same shrink sh as form30's
        # own credit, p stays monotone in form30 for ANY n (incl. fractional
        # n near 0) and a capped form30 cannot leak back in through form5.
        credited_gap = exp_gap + z30 * CONFIG["form30_scale"]
        note = (f"{f30a:.0%} vs {f30b:.0%} last 30d (~{n_eff:.0f} series eff.); "
                f"rating expects {exp_gap:+.0%} gap{opp_note}, deviation {dev:+.0%} x shrink {sh:.2f}")
    else:
        n_eff, sh, d = 0.0, 0.0, 0.0
        credited_gap = exp_gap
        note = "no 30-day form data -> neutral"
    var = CONFIG["tau_form30"] ** 2 * (1.0 - sh) + wcv2 * d ** 2
    factors.append(["form_30d", d, var, sh, note])

    # 3. form, last 5 -- last-5 gap minus the CREDITED 30d gap (see above):
    #    with full trust in form30 this is (form5_a - form30_a) - (form5_b -
    #    form30_b); with less trust it is measured against the shrunk
    #    estimate; with no 30d data, against the rating-implied gap.
    if _has(m, "form5_a") and _has(m, "form5_b"):
        f5a = _num(m["form5_a"], 0.5, 0.0, 1.0)
        f5b = _num(m["form5_b"], 0.5, 0.0, 1.0)
        swing = (f5a - f5b) - credited_gap
        ref = "vs 30d form" if (has30 and sh > 0) else "vs rating-implied gap (no 30d form)"
        sh5 = _shrink(5.0, fk)
        d = CONFIG["w_form5"] * _cap(sh5 * swing / CONFIG["form5_scale"])
        note = f"{f5a:.0%} vs {f5b:.0%} last 5; swing {swing:+.0%} {ref} x shrink {sh5:.2f}"
    else:
        sh5, d = 0.0, 0.0
        note = "no last-5 data -> neutral"
    var = CONFIG["tau_form5"] ** 2 * (1.0 - sh5) + wcv2 * d ** 2
    factors.append(["form_last5", d, var, sh5, note])

    # 4. head-to-head -- tiny samples, shrunk, residual vs rating expectation
    h = m.get("h2h") if isinstance(m.get("h2h"), dict) else {}
    aw = _count(h.get("a_wins", 0))
    bw = _count(h.get("b_wins", 0))
    n = aw + bw
    if n > 0:
        wr_h = aw / n
        exp_h = sigmoid(d_base) if CONFIG["h2h_vs_strength"] else 0.5
        shh = _shrink(n, CONFIG["h2h_shrink_k"])
        d = CONFIG["w_h2h"] * _cap(2.0 * (wr_h - exp_h) * shh)
        note = (f"{aw:.0f}-{bw:.0f} in {n:.0f} meetings vs {exp_h:.0%} expected "
                f"(shrunk x{shh:.2f})")
    else:
        shh, d = 0.0, 0.0
        note = "no recent meetings -> neutral"
    var = CONFIG["tau_h2h"] ** 2 * (1.0 - shh) + wcv2 * d ** 2
    factors.append(["head_to_head", d, var, shh, note])

    # 5. map veto -- pool-relative, series log-odds, capped, weight <= 1
    veto = simulate_veto(m)
    d = CONFIG["w_veto"] * _cap(logit(veto["p_series_a"]))
    var = (CONFIG["w_veto"] ** 2 * 0.25 * sum(veto["map_vars"])
           + (CONFIG["veto_path_cv"] * abs(d) + CONFIG["veto_path_floor"]) ** 2
           + wcv2 * d ** 2)
    factors.append(["map_veto", d, var, veto["confidence"], veto["note"]])

    # 6. stakes / motivation -- level difference, tiny
    def lvl(s):
        s = str(s if s is not None else "none").strip().lower()
        return STAKES_LEVELS.get(s, 1)
    sa, sb = m.get("stakes_a", "none"), m.get("stakes_b", "none")
    ldiff = _clamp(lvl(sa) - lvl(sb), -2, 2)
    d = math.log(CONFIG["stakes_mult"]) * ldiff
    if ldiff > 0:
        note = f"{a_name} more on the line ({sa} vs {sb})"
    elif ldiff < 0:
        note = f"{b_name} more on the line ({sb} vs {sa})"
    else:
        note = "even stakes"
    factors.append(["stakes", d, (CONFIG["stakes_cv"] * d) ** 2, 1.0 if ldiff else 0.0, note])

    # 7. roster -- odds multipliers, entered in log-odds like every other factor
    def pen(r, name, notes):
        r = r if isinstance(r, dict) else {}
        si, igl = _flag(r.get("standin")), _flag(r.get("missing_igl"))
        if si and igl:
            notes.append(f"{name} stand-in for missing IGL x{CONFIG['standin_igl_penalty']}")
            return CONFIG["standin_igl_penalty"]
        if si:
            notes.append(f"{name} stand-in x{CONFIG['standin_penalty']}")
            return CONFIG["standin_penalty"]
        if igl:
            notes.append(f"{name} missing IGL x{CONFIG['no_igl_penalty']}")
            return CONFIG["no_igl_penalty"]
        return 1.0
    rnotes = []
    pen_a = pen(m.get("roster_a"), a_name, rnotes)
    pen_b = pen(m.get("roster_b"), b_name, rnotes)
    d = math.log(pen_a) - math.log(pen_b)
    note = ("; ".join(rnotes) if rnotes else "both full strength") + " (odds multiplier)"
    factors.append(["roster", d, (CONFIG["roster_cv"] * d) ** 2,
                    1.0 if rnotes else 0.0, note])

    # --- combine ---
    raw = sum(f[1] for f in factors)
    s_vol = CONFIG["vol_sd"] * math.sqrt((vol_a ** 2 + vol_b ** 2) / 2.0)
    s2 = s_vol ** 2 + sum(f[2] for f in factors)
    s = math.sqrt(s2)
    shrink_k = math.sqrt(1.0 + math.pi * s2 / 8.0)
    total = raw / shrink_k
    T = _num(CONFIG["temperature"], 1.0, 0.0)
    return {
        "m": m, "a_name": a_name, "b_name": b_name, "vol_a": vol_a, "vol_b": vol_b,
        "factors": factors, "raw": raw, "s": s, "s_vol": s_vol,
        "shrink_k": shrink_k, "total": total, "T": T, "veto": veto,
    }


def total_logodds(m):
    """Pre-temperature log-odds (factor sum after uncertainty shrinkage).
    Final p_a = sigmoid(CONFIG['temperature'] * total_logodds(m)); this is the
    quantity backtest.py fits the temperature on."""
    return _compute(m)["total"]


def _scoreline_maps(veto, p_target, scale):
    """Shift the veto map logits by one constant c (bisection) so the implied
    series P(A) equals the final p_a. The map SHAPE comes from the veto; the
    level comes from the full blend, so series_probs is consistent with p_a."""
    base = [scale * L for L in veto["map_logits"]]
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _series_a(base, mid) < p_target:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2.0
    return [sigmoid(L + c) for L in base], c


# ============================================================================
# MAIN PREDICTION
# ============================================================================
def predict_match(m):
    C = _compute(m)
    m = C["m"]
    a_name, b_name = C["a_name"], C["b_name"]
    factors, raw, s, k, T = C["factors"], C["raw"], C["s"], C["shrink_k"], C["T"]
    vol_a, vol_b = C["vol_a"], C["vol_b"]
    vol_max = max(vol_a, vol_b)
    veto = C["veto"]

    p_a = sigmoid(T * raw / k)
    p_b = 1.0 - p_a

    # --- uncertainty band: symmetric in logit space, asymmetric in probability ---
    lp = T * raw / k
    half = CONFIG["band_z"] * T * s
    ci_lo, ci_hi = sigmoid(lp - half), sigmoid(lp + half)
    width_pp = round((ci_hi - ci_lo) * 100, 1)  # same value as the output key
    reliability = ("HIGH" if width_pp <= CONFIG["rel_high_width_pp"]
                   else "MEDIUM" if width_pp <= CONFIG["rel_med_width_pp"] else "LOW")

    # --- scoreline: veto's map shape, level-shifted so series P(A) equals p_a ---
    # map_shape, not w_veto: the veto weight is 0 and must not move the series
    # winner, but the three maps still have different win chances.
    p_maps, c_shift = _scoreline_maps(veto, p_a, T * CONFIG["map_shape"] / k)
    s20, s21, s12, s02 = series_probs(*p_maps)
    fav_is_a = p_a >= 0.5
    fav_name = a_name if fav_is_a else b_name
    opts = [(s20, a_name, "2-0", 0), (s21, a_name, "2-1", 1),
            (s12, b_name, "2-1", 2), (s02, b_name, "2-0", 3)]
    best = max(opts, key=lambda o: o[0])
    scoreline = f"{best[1]} {best[2]}"

    # --- conflict among factors WITH data (zero-delta factors don't vote) ---
    live = [f for f in factors if abs(f[1]) > 1e-9]
    mass = sum(abs(f[1]) for f in live)
    lean = 1.0 if raw >= 0 else -1.0
    opposing = [f for f in live if f[1] * lean < 0]
    opp_mass = sum(abs(f[1]) for f in opposing)
    conflict = opp_mass / mass if mass > 1e-9 else 0.0

    # --- marginal impact on the FINAL p: same shrinkage k and temperature ---
    breakdown = []
    for (name, d, var, rel, note) in factors:
        p_without = sigmoid(T * (raw - d) / k)
        breakdown.append({
            "factor": name,
            "delta_logodds": round(d, 3),
            # sigmoid(delta): what this increment alone would do from 50/50.
            # NOT a standalone forecast (factors are residuals of each other).
            "implied_win_pct_alone": round(sigmoid(d) * 100, 1),
            "marginal_pp": round((p_a - p_without) * 100, 1),
            "confidence": round(rel, 2),   # data reliability (sample shrinkage)
            "sd_logodds": round(math.sqrt(var), 3),
            "note": note,
        })

    # --- market-efficiency edge (NOT betting advice) ---
    edge = None
    edge_note = None
    # validated: non-numeric / NaN / inf / outside [0, 1] -> ignored (an
    # out-of-range price is a data error, not a 0% or 100% market read)
    mp = _num(m.get("market_price_a"), None)
    if mp is not None and not 0.0 <= mp <= 1.0:
        mp = None
    if mp is not None:
        edge = round((p_a - mp) * 100, 1)  # in percentage points
        if edge > 0:
            edge_note = (f"model prices {a_name} {edge:.1f}pp above the market "
                         f"({p_a:.2%} vs {mp:.2%}) - market may be underpricing {a_name}.")
        elif edge < 0:
            edge_note = (f"model prices {a_name} {abs(edge):.1f}pp below the market "
                         f"({p_a:.2%} vs {mp:.2%}) - market may be overpricing {a_name}.")
        else:
            edge_note = "model agrees with the market."
        edge_note += " Market-efficiency read only, not a recommendation."

    # --- warnings: the 'knows and warns' part ---
    warnings = []
    if reliability == "LOW":
        warnings.append(
            f"WIDE BAND ({ci_lo:.2%}-{ci_hi:.2%}, logit sd {s:.2f}): close read "
            f"and/or small samples, missing data or volatility. Treat the point estimate with caution.")
    if vol_max > 0.6:
        warnings.append(
            "HIGH VOLATILITY: recent form contradicts the base rating - "
            "upset risk is elevated regardless of the point estimate.")
    if (conflict >= CONFIG["split_min_conflict"] and opp_mass >= CONFIG["split_min_opposing"]
            and opposing):
        top_for = max((f for f in live if f[1] * lean > 0), key=lambda f: abs(f[1]), default=None)
        top_against = max(opposing, key=lambda f: abs(f[1]))
        lean_name = a_name if lean > 0 else b_name
        msg = (f"FACTOR SPLIT ({conflict:.0%} of signal opposes the lean to {lean_name}): "
               f"'{top_against[0]}' {top_against[1]:+.2f} log-odds")
        if top_for is not None:
            msg += f" vs '{top_for[0]}' {top_for[1]:+.2f}"
        warnings.append(msg + ".")

    # Display grid: 2 decimal places on the percentage, largest remainder so a
    # pair or a scoreline always adds to 100.00. Exact values stay unrounded.
    pa_shown, pb_shown = display_percent([p_a, p_b])
    score_shown = display_percent([s20, s21, s12, s02])

    # --- BO5 (grand finals): one flat per-map chance on all five maps, set so
    # the BO5 series P(A) equals p_a. On 44 real BO5s no map-specific or
    # length-based alternative scored measurably better (scripts/bo5_check.py).
    q5 = flat_map_prob(p_a, 5)
    s5 = series_scorelines([q5] * 5)
    s5_shown = display_percent(s5)
    return {
        "match": f"{a_name} vs {b_name}",
        "p_a": pa_shown / 100.0,
        "p_b": pb_shown / 100.0,
        "pick": fav_name,
        "scoreline": scoreline,
        "scoreline_prob": score_shown[best[3]] / 100.0,
        "confidence_interval": [round(ci_lo, 3), round(ci_hi, 3)],  # uncertainty band
        "band_kind": "heuristic uncertainty band (not yet coverage-validated)",
        "interval_width_pp": width_pp,
        "reliability": reliability,
        "logit_sd": round(s, 3),
        "raw_logodds": round(raw, 4),
        "total_logodds": raw / k,  # unrounded: backtest.py fits temperature on it
        "uncertainty_shrink": round(1.0 / k, 4),
        "temperature": T,
        "conflict": round(conflict, 3),
        "factor_breakdown": breakdown,
        "veto_log": veto["veto_log"],
        "veto_maps": veto["maps"],
        "map_probs": [round(p, 3) for p in p_maps],  # final-p-consistent, per veto_maps
        "map_probs_exact": list(p_maps),
        # veto-only view (pool-relative map edges, before blending/shrinkage):
        "veto_only_series_p_a": veto["p_series_a"],
        "veto_only_map_probs": list(veto["p_map"]),
        "map_logit_shift": round(c_shift, 4),
        "series_probs": {
            "p_2_0": score_shown[0] / 100.0,
            "p_2_1": score_shown[1] / 100.0,
            "p_1_2": score_shown[2] / 100.0,
            "p_0_2": score_shown[3] / 100.0,
        },
        "series_probs_exact": {"p_2_0": s20, "p_2_1": s21, "p_1_2": s12, "p_0_2": s02},
        # BO5 scorelines from A's view (3-0 .. 0-3); the BO3 fields above are unchanged
        "series_probs_bo5": {k: v / 100.0 for k, v in zip(BO5_KEYS, s5_shown)},
        "series_probs_bo5_exact": dict(zip(BO5_KEYS, s5)),
        "map_prob_bo5_exact": q5,
        "p_a_exact": p_a,
        "market_edge_pp": edge,
        "market_edge_note": edge_note,
        "warnings": warnings,
        "volatility": {"a": vol_a, "b": vol_b},
    }


def format_result(r):
    """Human-readable one-match report for the briefing write-up (ASCII only)."""
    lines = []
    a, b = r["match"].split(" vs ", 1)
    lines.append(f"### {r['match']}")
    pa_pct, pb_pct = display_percent([r["p_a_exact"], 1.0 - r["p_a_exact"]])
    lines.append(f"Model: {a} {pa_pct:.2f}% / {b} {pb_pct:.2f}% "
                 f"(uncertainty band {r['confidence_interval'][0] * 100:.2f}%-"
                 f"{r['confidence_interval'][1] * 100:.2f}%, "
                 f"heuristic, not coverage-validated)")
    sp_exact = r["series_probs_exact"]
    shown = display_percent([sp_exact["p_2_0"], sp_exact["p_2_1"],
                             sp_exact["p_1_2"], sp_exact["p_0_2"]])
    top = max(range(4), key=lambda i: (sp_exact[["p_2_0", "p_2_1", "p_1_2", "p_0_2"][i]], -i))
    lines.append(f"Pick: {r['pick']}  |  most likely scoreline: {r['scoreline']} "
                 f"({shown[top]:.2f}%)  |  reliability: {r['reliability']}")
    lines.append(f"Series: {a} 2-0 {shown[0]:.2f}%, 2-1 {shown[1]:.2f}% | "
                 f"{b} 2-1 {shown[2]:.2f}%, 2-0 {shown[3]:.2f}%")
    lines.append("Factor breakdown (marginal pp impact on A's final win prob):")
    for f in r["factor_breakdown"]:
        arrow = "+" if f["marginal_pp"] >= 0 else ""
        lines.append(f"  - {f['factor']}: {arrow}{f['marginal_pp']}pp "
                     f"(delta {f['delta_logodds']:+.2f} +- {f['sd_logodds']:.2f}, "
                     f"data conf {f['confidence']}) - {f['note']}")
    if r["market_edge_note"]:
        lines.append(f"Market efficiency: {r['market_edge_note']}")
    for w in r["warnings"]:
        lines.append(f"  ! {w}")
    return "\n".join(lines)


# ============================================================================
# BACKTEST -- EPL S24 Day 4 (2026-10-06), morning matches.
# Inputs are APPROXIMATE, built from the real facts available pre-match.
# Ratings are approximate HLTV-style 2.1 ratings; map win rates are 90-day
# approximations grounded in the numbers stated in the Day 4 previews
# (flagged REAL below); the rest are labeled estimates.
# ============================================================================
def day4_backtest():
    matches = [
        {
            "team_a": "G2", "team_b": "PARIVISION",
            "rating_a": 1.03, "rating_b": 1.01,  # estimate: slumping G2 vs competitive PARI
            "form30_a": 0.42, "form30_b": 0.45, "n30_a": 12, "n30_b": 11,  # estimate: both sub-50%, G2 broke 3-series slide vs 0-3 ShindeN
            "form5_a": 0.40, "form5_b": 0.20,  # REAL: 2-3 vs 1-4
            "h2h": {"a_wins": 1, "b_wins": 1, "meetings": 2},  # REAL: 1-1 in 2026
            "maps_a": {"Dust2": [0.33, 6], "Mirage": [0.55, 10], "Inferno": [0.70, 8],
                       "Nuke": [0.50, 8], "Ancient": [0.14, 7], "Anubis": [0.55, 8], "Cache": [0.45, 6]},
            # REAL: Dust2 2-4/90d, Ancient 1-6/90d; rest estimated
            "maps_b": {"Dust2": [0.55, 9], "Mirage": [0.50, 10], "Inferno": [0.50, 6],
                       "Nuke": [0.45, 8], "Ancient": [0.80, 5], "Anubis": [0.50, 8], "Cache": [0.45, 6]},
            # REAL: Ancient 4-1 (80%); Dust2 comfort (46% pick rate); rest estimated
            "permaban_a": None, "permaban_b": "Inferno",  # REAL: PARI bans Inferno
            "roster_a": {}, "roster_b": {},
            "stakes_a": "elimination", "stakes_b": "elimination",
            "volatility_a": 0.7, "volatility_b": 0.4,  # G2: slide + dead-cat bounce vs worst team
            "market_price_a": 0.65,  # estimate: market had G2 ~65%
            "_actual": "PARIVISION won",
        },
        {
            "team_a": "Spirit", "team_b": "1WIN",
            "rating_a": 1.11, "rating_b": 1.00,  # estimate: world #1 vs 33rd VRS
            "form30_a": 0.70, "form30_b": 0.55, "n30_a": 13, "n30_b": 12,  # estimate: Porto champs + MOUZ loss vs G2+Legacy wins
            "form5_a": 0.80, "form5_b": 0.40,  # REAL: 4-1 vs 2-3
            "h2h": {"a_wins": 0, "b_wins": 0, "meetings": 0},  # no relevant meetings
            "maps_a": {"Dust2": [0.45, 10], "Mirage": [0.62, 10], "Inferno": [0.40, 6],
                       "Nuke": [0.65, 10], "Ancient": [0.60, 8], "Anubis": [0.87, 8], "Cache": [0.40, 5]},
            # REAL: Anubis 87.5% S-map, Cache 40% worst; Inferno permaban
            "maps_b": {"Dust2": [0.55, 8], "Mirage": [0.58, 8], "Inferno": [0.55, 6],
                       "Nuke": [0.55, 8], "Ancient": [0.52, 8], "Anubis": [0.50, 6], "Cache": [0.50, 6]},
            # estimate: solid across the pool, no huge edges
            "permaban_a": "Inferno", "permaban_b": None,  # REAL: Spirit permaban Inferno
            "roster_a": {}, "roster_b": {},
            "stakes_a": "qualification", "stakes_b": "qualification",
            "volatility_a": 0.8, "volatility_b": 0.5,
            # Spirit: swept by MOUZ, donk's worst-ever map - form contradicts rating.
            # 1WIN: genuine heater (beat G2, blanked Legacy).
            "market_price_a": 0.78,  # estimate
            "_actual": "1WIN won (genuine upset - model must still pick Spirit)",
        },
        {
            "team_a": "FURIA", "team_b": "Aurora",
            "rating_a": 1.06, "rating_b": 1.02,  # estimate: Major finalists vs hot tier-2
            "form30_a": 0.47, "form30_b": 0.63, "n30_a": 15, "n30_b": 8,  # REAL: 47% (7-8) vs 63% (5-3)
            "form5_a": 0.60, "form5_b": 0.60,  # REAL: 3-2 vs 3-2
            "h2h": {"a_wins": 2, "b_wins": 0, "meetings": 2},  # REAL: won last two
            "maps_a": {"Dust2": [0.50, 10], "Mirage": [0.55, 10], "Inferno": [0.55, 8],
                       "Nuke": [0.69, 16], "Ancient": [0.50, 8], "Anubis": [0.45, 6], "Cache": [0.50, 6]},
            # REAL: Nuke 11-5 (69%); Anubis is their ban
            "maps_b": {"Dust2": [0.50, 8], "Mirage": [0.55, 8], "Inferno": [0.50, 8],
                       "Nuke": [0.55, 8], "Ancient": [0.55, 8], "Anubis": [0.60, 10], "Cache": [0.50, 6]},
            # REAL: Anubis most-picked; rest estimated
            "permaban_a": "Anubis", "permaban_b": None,  # REAL: FURIA bans Anubis
            "roster_a": {}, "roster_b": {},
            "stakes_a": "qualification", "stakes_b": "qualification",
            "volatility_a": 0.5, "volatility_b": 0.3,
            "market_price_a": 0.60,  # estimate
            "_actual": "Aurora won",
        },
        {
            "team_a": "Legacy", "team_b": "M80",
            "rating_a": 1.02, "rating_b": 1.00,  # estimate; Legacy slightly better on paper WITH arT
            "form30_a": 0.55, "form30_b": 0.40, "n30_a": 11, "n30_b": 10,  # estimate
            "form5_a": 0.40, "form5_b": 0.40,  # estimate: both 2-3ish
            "h2h": {"a_wins": 2, "b_wins": 0, "meetings": 2},  # REAL: Legacy 2-0 in 2026 (arT-era)
            "maps_a": {"Dust2": [0.67, 9], "Mirage": [0.55, 8], "Inferno": [0.55, 8],
                       "Nuke": [0.50, 8], "Ancient": [0.55, 8], "Anubis": [0.50, 8], "Cache": [0.50, 6]},
            # REAL: Dust2 6-3 (67%)
            "maps_b": {"Dust2": [0.50, 8], "Mirage": [0.67, 6], "Inferno": [0.50, 8],
                       "Nuke": [0.50, 8], "Ancient": [0.50, 8], "Anubis": [0.50, 8], "Cache": [0.45, 6]},
            # REAL: Mirage 4-2 (67%) A-map
            "permaban_a": None, "permaban_b": None,
            "roster_a": {"standin": True, "missing_igl": True},  # REAL: arT out (meningitis), bobz in
            "roster_b": {},
            "stakes_a": "elimination", "stakes_b": "elimination",
            "volatility_a": 0.7, "volatility_b": 0.4,
            "market_price_a": 0.55,  # estimate
            "_actual": "demonstrates roster penalty (arT out); not scored as backtest",
        },
    ]
    results = []
    for m in matches:
        actual = m.pop("_actual")
        r = predict_match(m)
        r["_actual"] = actual
        results.append(r)
        m["_actual"] = actual
    return results


def main():
    # Windows consoles default to a legacy code page; force UTF-8 so any
    # non-ASCII team names print correctly (report text itself is ASCII).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        for r in day4_backtest():
            print(format_result(r))
            print(f"  actual: {r['_actual']}")
            print()
        return
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            m = json.load(f)
        r = predict_match(m)
        print(json.dumps(r, indent=2))
        print()
        print(format_result(r))
        return
    print(__doc__)
    print("usage: python predictor.py match.json | python predictor.py --test")


if __name__ == "__main__":
    main()
