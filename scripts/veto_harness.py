#!/usr/bin/env python3
"""
veto_harness.py -- validation harness for the veto subsystem v2 (VetoHarness).
==============================================================================
Fan analytics only. Scores veto models against the order maps were actually
played in. Point-in-time rows from backtest.build_dataset; BO3 rows whose
played maps are all in that date's pool (others counted as pool_mismatch).

Split: the backtest's own date boundary, train <= 2025-11-05, test >=
2025-11-07; inner validation fold = the last 25% of training dates. Every
look at the test window increments test_evaluations in
data/veto_fit_report.json. Bootstrap: paired, resampling whole match days,
2,000 resamples, seed 20261009, 95% percentile interval.

Metrics (design "Validation harness"):
  pick_set_hit       |modal pick set & {map 1, map 2}| / 2 (modal outcome
                     marginalised over the starter)
  loglik             mean log P(observed map order)
  decider_hit        modal decider == map 3 on 2-1 series
  coverage80         share of series whose observed (map 1, map 2) is in the
                     80% highest-probability set of ordered pick pairs (the
                     observable part of the outcome for every BO3)
  played_cov01       share of series where every played map has P(played) >= 0.01
  ece_played         ECE (10 bins) of P(map played | number of maps played)
  unplayed_pick      share of modal picks on maps the picker has not played in 90 days
  zero_prob          share of series the model gave probability exactly 0 (gate: must be 0)
  masked_but_picked  share of series where an observed pick was in the picker's
                     exclusion mask under at least one starter hypothesis
  (ece_played is conditional on the realised series length: a decider counts
  as played only on 2-1 series)
  scoreline_logloss  held-out log loss of the actual BO3 score (frozen four fields)

USAGE (repo root):
  python scripts/veto_harness.py --phase p0          baselines + sanity check
  python scripts/veto_harness.py --phase p1 [...]    gate for a fitted phase
Stdlib only, deterministic (the bootstrap is seeded).
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402
import veto  # noqa: E402

TRAIN_END = "2025-11-05"
TEST_START = "2025-11-07"
INNER_FRAC = 0.25
SEED = 20261009
N_BOOT = 2000
FIT_REPORT = os.path.join(ROOT, "data", "veto_fit_report.json")
SNAPSHOT_KEYS = ("map_scale", "map_shrink_k", "veto_comfort", "veto_comfort_k", "map_shape", "temperature")


# ============================================================================
# data
# ============================================================================
def sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def data_hashes():
    return {"matches.json": sha256(os.path.join(ROOT, "data", "matches.json")),
            "lineups.json": sha256(os.path.join(ROOT, "data", "lineups.json"))}


def load_rows(best_of=3, history_factory=None):
    """(rows, counts). Each row: {"date", "input", "played", "match"}."""
    ms = bt.load_matches()
    built = build_dataset(ms, history_factory) if history_factory else bt.build_dataset(ms)
    rows, cnt = [], defaultdict(int)
    need = 1 if best_of == 1 else 2
    for r in built:
        m = r["match"]
        if m.get("best_of") != best_of or len(m.get("maps") or []) < need:
            continue
        pool = pr._pool(r["input"])
        played = [x["map"] for x in m["maps"]]
        if best_of == 1:
            played = played[:1]
        if any(p not in pool for p in played) or len(set(played)) != len(played):
            cnt["pool_mismatch"] += 1
            continue
        if len(pool) < best_of:
            cnt["pool_too_small"] += 1
            continue
        rows.append({"date": m["date"], "input": r["input"], "played": played, "match": m})
        cnt["kept"] += 1
    return rows, dict(cnt)


def build_dataset(matches, history_factory):
    """bt.build_dataset with a custom History (e.g. with a HabitAccumulator)."""
    ms = sorted(matches, key=lambda m: (m["date"], m.get("id", 0)))
    h = history_factory()
    out, i = [], 0
    while i < len(ms):
        j = i
        while j < len(ms) and ms[j]["date"] == ms[i]["date"]:
            j += 1
        for m in ms[i:j]:
            inp, meta = h.features(m)
            out.append({"match": m, "input": inp, "meta": meta})
        for m in ms[i:j]:
            h.add(m)
        i = j
    return out


def split(rows):
    train = [r for r in rows if r["date"] <= TRAIN_END]
    test = [r for r in rows if r["date"] >= TEST_START]
    days = sorted({r["date"] for r in train})
    cut = days[int(len(days) * (1 - INNER_FRAC))]
    inner_fit = [r for r in train if r["date"] < cut]
    inner_val = [r for r in train if r["date"] >= cut]
    return train, test, inner_fit, inner_val


# ============================================================================
# reference: the comfort sim (deterministic), epsilon-smoothed for likelihood
# ============================================================================
def swap_input(inp):
    out = dict(inp)
    for a, b in (("maps_a", "maps_b"), ("maps_raw_a", "maps_raw_b"), ("permaban_a", "permaban_b"),
                 ("veto_ev_a", "veto_ev_b"), ("team_a", "team_b")):
        out[a], out[b] = inp.get(b), inp.get(a)
    return out


def sim_orders(inp, best_of=3):
    """Played order the comfort sim implies under each starter."""
    va = pr.simulate_veto(inp, best_of)
    vb = pr.simulate_veto(swap_input(inp), best_of)
    return {"a": tuple(va["maps"]), "b": tuple(vb["maps"]), "picks_a": va["picks"], "log_a": va["veto_log"],
            "picks_b": [vb["maps"][1], vb["maps"][0]] if best_of == 3 else vb["picks"]}


def uniform_params(best_of=3):
    w = dict(pr.veto_params(best_of))
    for k in list(w):
        if k.startswith(("alpha_", "gamma_", "rho_", "eta_")):
            w[k] = 0.0
    w["lookahead"] = False
    return w


def ref_prepare(rows, best_of=3):
    """Per row: sim orders and the uniform-legal-veto probability of the
    observed order (no exclusion mask in the uniform baseline)."""
    wu = uniform_params(best_of)
    saved = pr.CONFIG["veto_excl_alpha"]
    pr.CONFIG["veto_excl_alpha"] = 0.0
    try:
        out = []
        for r in rows:
            ctx = pr.veto_context(r["input"], best_of)
            ll_u = veto.loglik(ctx, wu, r["played"])[0]
            out.append({"sim": sim_orders(r["input"], best_of), "u": math.exp(ll_u), "ll_u": ll_u})
        return out
    finally:
        pr.CONFIG["veto_excl_alpha"] = saved


def _match(order, played):
    return tuple(order[:len(played)]) == tuple(played)


def ref_loglik_rows(rows, prep, eps, pi):
    out = []
    for r, p in zip(rows, prep):
        pa = (1 - eps) * _match(p["sim"]["a"], r["played"]) + eps * p["u"]
        pb = (1 - eps) * _match(p["sim"]["b"], r["played"]) + eps * p["u"]
        out.append(math.log(pi * pa + (1 - pi) * pb))
    return out


def golden(f, lo, hi, tol=1e-7, iters=200):
    """Maximise a unimodal f on [lo, hi]."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if b - a < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


def fit_ref(rows, prep):
    """epsilon and pi of the smoothed comfort sim, on training rows only
    (coordinate golden-section; the log likelihood is concave in each)."""
    flags = [(float(_match(p["sim"]["a"], r["played"])), float(_match(p["sim"]["b"], r["played"])), p["u"])
             for r, p in zip(rows, prep)]

    def ll(e, pi):
        return sum(math.log(pi * ((1 - e) * fa + e * u) + (1 - pi) * ((1 - e) * fb + e * u))
                   for fa, fb, u in flags) / len(flags)

    e, pi = 0.5, 0.5
    v = ll(e, pi)
    for _ in range(50):
        e, _ = golden(lambda x: ll(x, pi), 1e-6, 1 - 1e-6)
        pi, v2 = golden(lambda x: ll(e, x), 0.0, 1.0)
        if abs(v2 - v) < 1e-12:
            v = v2
            break
        v = v2
    return {"eps": e, "pi": pi, "train_loglik": v}


# ============================================================================
# model metrics per row
# ============================================================================
def row_metrics(r, dist, internals, ll, best_of=3):
    played = r["played"]
    ranked = internals["outcomes"]
    pool = list(dist["marginals"])
    out = {"ll": ll, "zero_prob": float(ll == float("-inf"))}
    if best_of > 1:
        # observed picks that a team's exclusion mask forbade, under either starter
        E = {t: set(dist["masked"][t]) for t in "ab"}
        m1, m2 = played[0], played[1]
        under_a = m1 in E["a"] or m2 in E["b"]
        under_b = m1 in E["b"] or m2 in E["a"]
        out["masked_but_picked"] = float(under_a or under_b)
    # modal outcome marginalised over pickers
    agg = defaultdict(float)
    for (maps, _), p in ranked:
        agg[maps] += p
    modal = max(sorted(agg), key=lambda k: agg[k])
    ctx_pool = sorted(pool)
    mnames = [ctx_pool[i] for i in modal]
    if best_of == 1:
        out["hit"] = float(mnames[0] == played[0])
    else:
        out["hit"] = len(set(mnames[:2]) & set(played[:2])) / 2.0
        if len(played) >= 3:
            dec = max(sorted(pool), key=lambda mp: dist["marginals"][mp]["decider"])
            out["dec_hit"] = float(dec == played[2])
    # coverage: ordered pick pairs (or the played map for BO1)
    if best_of == 1:
        pairs = defaultdict(float)
        for (maps, _), p in ranked:
            pairs[(maps[-1],)] += p
        obs = (ctx_pool.index(played[0]),)
    else:
        pairs = defaultdict(float)
        for (maps, _), p in ranked:
            pairs[maps[:2]] += p
        obs = tuple(ctx_pool.index(mp) for mp in played[:2])
    acc, inset = 0.0, False
    for k, p in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
        if acc >= 0.8:
            break
        acc += p
        if k == obs:
            inset = True
    out["cov80"] = float(inset)
    out["cov80_mass"] = acc
    # P(played | number of maps played)
    pp = {}
    for mp in pool:
        mg = dist["marginals"][mp]
        if best_of == 1:
            pp[mp] = mg["decider"]
        else:
            pp[mp] = mg["picked_a"] + mg["picked_b"] + (mg["decider"] if len(played) >= 3 else 0.0)
    out["pp"] = [(pp[mp], 1.0 if mp in played else 0.0) for mp in sorted(pool)]
    out["cov01"] = float(all(pp[mp] >= 0.01 for mp in played))
    # unplayed modal picks (attributed-pick accuracy is dropped: with the
    # starter latent it is structurally 0 or 1 by the sign of pi - 0.5)
    best_pk = None
    for (maps, pk), p in ranked:
        if maps == modal:
            best_pk = pk
            break
    up = n = 0
    for j in range(best_of - 1):
        mp = ctx_pool[modal[j]]
        side = best_pk[j]
        raw = r["input"].get("maps_a" if side == "a" else "maps_b") or {}
        e = raw.get(mp)
        n += 1
        up += not (e and e[1])
    if n:
        out["unplayed"] = up / n
    return out


def model_rows(rows, w, best_of=3, ctx_fn=None):
    res = []
    for r in rows:
        ctx = (ctx_fn or pr.veto_context)(r["input"], best_of)
        dist, internals = veto.distribution(ctx, w)
        ctx2 = (ctx_fn or pr.veto_context)(r["input"], best_of)
        ll = veto.loglik(ctx2, w, r["played"])[0]
        res.append(row_metrics(r, dist, internals, ll, best_of))
    return res


def ref_rows(rows, prep, refp, best_of=3):
    lls = ref_loglik_rows(rows, prep, refp["eps"], refp["pi"])
    out = []
    for r, p, ll in zip(rows, prep, lls):
        played = r["played"]
        sa = p["sim"]["a"]
        d = {"ll": ll}
        if best_of == 1:
            so = p["sim"]["a" if refp["pi"] >= 0.5 else "b"]
            d["hit"] = float(so[-1] == played[0])
        else:
            # modal outcome of the pi-mixture of the two starter sims
            s_star = "a" if refp["pi"] >= 0.5 else "b"
            so = p["sim"][s_star]
            d["hit"] = len(set(so[:2]) & set(played[:2])) / 2.0
            d["hit_starter_a"] = len(set(p["sim"]["picks_a"]) & set(played[:2])) / 2.0
            if len(played) >= 3:
                d["dec_hit"] = float(so[-1] == played[2])
            up = n = 0
            for line in p["sim"]["log_a"]:
                if " picks " in line:
                    side, mp = line.split(" picks ")
                    raw = r["input"].get("maps_a" if side == "A" else "maps_b") or {}
                    e = raw.get(mp)
                    n += 1
                    up += not (e and e[1])
            d["unplayed"] = up / max(1, n)
        out.append(d)
    return out


def ece(pairs, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for p, y in pairs:
        bins[min(n_bins - 1, int(p * n_bins))].append((p, y))
    n = len(pairs)
    return sum(len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(y for _, y in b) / len(b))
               for b in bins if b)


def summarize(res):
    out = {"n": len(res)}
    for k in ("hit", "ll", "dec_hit", "cov80", "cov80_mass", "cov01", "unplayed", "hit_starter_a",
              "zero_prob", "masked_but_picked"):
        vals = [x[k] for x in res if k in x]
        if vals:
            out[k] = sum(vals) / len(vals)
            out[k + "_n"] = len(vals)
    pp = [t for x in res if "pp" in x for t in x["pp"]]
    if pp:
        out["ece_played"] = ece(pp)
    return out


# ============================================================================
# bootstrap (match-day clusters)
# ============================================================================
def cluster_boot(dates, diffs, n_boot=N_BOOT, seed=SEED):
    """95% percentile CI of mean(diffs), resampling whole match days."""
    by = defaultdict(list)
    for d, x in zip(dates, diffs):
        by[d].append(x)
    days = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = n = 0
        for _ in range(len(days)):
            xs = by[days[rng.randrange(len(days))]]
            s += sum(xs)
            n += len(xs)
        vals.append(s / n)
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]]


def paired(rows, new, ref, key):
    idx = [i for i in range(len(rows)) if key in new[i] and key in ref[i]]
    diffs = [new[i][key] - ref[i][key] for i in idx]
    dates = [rows[i]["date"] for i in idx]
    if not diffs:
        return None
    return {"diff": sum(diffs) / len(diffs), "ci95": cluster_boot(dates, diffs), "n": len(diffs)}


# ============================================================================
# scoreline log loss and I6 (predict_match in both modes)
# ============================================================================
SC_KEY = {"2-0": "p_2_0", "2-1": "p_2_1", "1-2": "p_1_2", "0-2": "p_0_2"}


def scoreline_rows(rows, cfg_new, cfg_ref):
    """Per-row log loss of the actual scoreline under each config, and I6."""
    out_new, out_ref, i6 = [], [], True
    saved = copy.deepcopy(pr.CONFIG)
    try:
        for r in rows:
            sc = bt.actual_scoreline(r["match"])
            vals = []
            pas = []
            for cfg in (cfg_new, cfg_ref):
                pr.CONFIG.clear()
                pr.CONFIG.update(cfg)
                res = pr.predict_match(dict(r["input"]))
                pas.append(res["p_a_exact"])
                vals.append(-math.log(max(1e-15, res["series_probs_exact"][SC_KEY[sc]])) if sc else None)
            i6 &= pas[0] == pas[1]
            out_new.append({"sll": vals[0]} if vals[0] is not None else {})
            out_ref.append({"sll": vals[1]} if vals[1] is not None else {})
    finally:
        pr.CONFIG.clear()
        pr.CONFIG.update(saved)
    return out_new, out_ref, i6


# ============================================================================
# fit report / counter
# ============================================================================
def read_report():
    if os.path.exists(FIT_REPORT):
        with open(FIT_REPORT, encoding="utf-8") as f:
            return json.load(f)
    return {"version": "veto-v2", "test_evaluations": 0, "phases": {}}


def write_report(rep):
    with open(FIT_REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, sort_keys=True)


def bump_test_counter(reason):
    rep = read_report()
    rep["test_evaluations"] = rep.get("test_evaluations", 0) + 1
    rep.setdefault("test_evaluation_log", []).append(reason)
    write_report(rep)
    return rep["test_evaluations"]


def snapshot():
    return {k: pr.CONFIG[k] for k in SNAPSHOT_KEYS}


# ============================================================================
# phase 0: baselines + sanity check vs scripts/veto_check.py
# ============================================================================
def sanity_veto_check():
    """Reproduce scripts/veto_check.py (old 60/40 split, comfort 0.8)."""
    import veto_check as vc
    data = vc.rows()
    k = int(len(data) * 0.6)
    old = vc.score(data[k:], pr.CONFIG["veto_comfort"])
    # independent re-implementation over the same rows
    hit = dec = dn = 0.0
    for inp, played in data[k:]:
        v = pr.simulate_veto(inp, 3)
        hit += len(set(v["picks"]) & set(played[:2])) / 2
        if len(played) >= 3:
            dn += 1
            dec += v["decider"] == played[2]
    mine = (hit / len(data[k:]), dec / dn)
    return {"veto_check": list(old[:2]), "harness": list(mine),
            "max_abs_diff": max(abs(old[0] - mine[0]), abs(old[1] - mine[1]))}


def phase0(out_path):
    rows, cnt = load_rows(3)
    train, test, _, _ = split(rows)
    prep_tr = ref_prepare(train)
    refp = fit_ref(train, prep_tr)
    prep_te = ref_prepare(test)
    n_eval = bump_test_counter("p0 baselines on test")
    ref_tr = summarize(ref_rows(train, prep_tr, refp))
    ref_te = summarize(ref_rows(test, prep_te, refp))
    uni_te = sum(p["ll_u"] for p in prep_te) / len(prep_te)
    uni_tr = sum(p["ll_u"] for p in prep_tr) / len(prep_tr)
    # comfort off (the live sim before the comfort change), pick-set only
    saved = pr.CONFIG["veto_comfort"]
    pr.CONFIG["veto_comfort"] = 0.0
    try:
        off_te = summarize(ref_rows(test, ref_prepare(test), refp))
    finally:
        pr.CONFIG["veto_comfort"] = saved
    rep = {"phase": "p0", "data_hashes": data_hashes(), "rows": cnt,
           "split": {"train_end": TRAIN_END, "test_start": TEST_START, "train": len(train), "test": len(test)},
           "reference_comfort_sim": {"eps_pi_fit_on_train": refp, "train": ref_tr, "test": ref_te},
           "uniform_legal_veto": {"train_loglik": uni_tr, "test_loglik": uni_te},
           "comfort_off_test": {"hit": off_te["hit"], "dec_hit": off_te.get("dec_hit")},
           "chance": {"pick_set_hit": 2 / 7, "decider_hit": 1 / 7},
           "sanity_veto_check": sanity_veto_check(),
           "test_evaluations": n_eval}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


# ============================================================================
# phase gate
# ============================================================================
def ablation_params(c, pi, best_of=3):
    """Same quantal family as the candidate with baseline-equivalent weights
    (the comfort sim's edge + veto_comfort x comfort, i.e. alpha = c /
    map_scale, gamma = c x veto_comfort, no rho, myopic): only the overall
    sharpness c (a temperature) and pi are free."""
    a, g = c / pr.CONFIG["map_scale"], c * pr.CONFIG["veto_comfort"]
    w = {"version": "ablation", "lookahead": False, "pi": pi, "temp": 1.0,
         "alpha_ban1": a, "gamma_ban1": g, "rho_ban1": 0.0, "alpha_ban2": a, "gamma_ban2": g, "rho_ban2": 0.0,
         "alpha_pick": a if best_of > 1 else 0.0, "gamma_pick": g if best_of > 1 else 0.0}
    return w


def fit_ablation(train, best_of=3, ctx_fn=None):
    """c and pi of the ablation by coordinate golden-section on train."""
    ctxs = [(ctx_fn or pr.veto_context)(r["input"], best_of) for r in train]

    def ll(c, pi):
        w = ablation_params(c, pi, best_of)
        return sum(veto.loglik(x, w, r["played"])[0] for x, r in zip(ctxs, train)) / len(train)
    c, pi = 1.0, 0.5
    v = ll(c, pi)
    for _ in range(20):
        c, _ = golden(lambda x: ll(x, pi), 0.0, 20.0, tol=1e-4)
        pi, v2 = golden(lambda x: ll(c, x), 0.05, 0.95, tol=1e-4)
        if abs(v2 - v) < 1e-7:
            v = v2
            break
        v = v2
    return {"c": c, "pi": pi, "train_loglik": v, "params": ablation_params(c, pi, best_of)}


def gate(phase, w, cfg_overrides, out_path, rows=None, best_of=3, ctx_fn=None, ref_model=None,
         leakage_fn=None, note=None):
    """Evaluate candidate params `w` (with CONFIG overrides, e.g. the exclusion
    alpha or habit flags) on the test window against TWO references: the
    comfort sim (eps-smoothed for likelihood; or `ref_model`, a shipped
    distribution) and the ablation (same quantal family, baseline-equivalent
    weights, only sharpness and pi fitted on train, same mask). Every paired
    check must pass against both. `leakage_fn()` -> (ok, detail) adds the
    leakage check to the gate (habit / roster phases)."""
    if rows is None:
        rows, cnt = load_rows(best_of)
    else:
        cnt = {"kept": len(rows)}
    train, test, _, _ = split(rows)
    saved = copy.deepcopy(pr.CONFIG)

    def run(params, over, rr):
        try:
            pr.CONFIG.update(over)
            out = model_rows(rr, params, best_of, ctx_fn)
            cfg = copy.deepcopy(pr.CONFIG)
        finally:
            pr.CONFIG.clear()
            pr.CONFIG.update(saved)
        return out, cfg

    # ablation fitted on train (same CONFIG overrides, i.e. the same mask)
    try:
        pr.CONFIG.update(cfg_overrides)
        abl = fit_ablation(train, best_of, ctx_fn)
    finally:
        pr.CONFIG.clear()
        pr.CONFIG.update(saved)
    n_eval = bump_test_counter(f"{phase} gate on test" + (f" ({note})" if note else ""))
    new, cfg_new = run(w, cfg_overrides, test)
    new_tr, _ = run(w, cfg_overrides, train)
    abl_rows, cfg_abl = run(abl["params"], cfg_overrides, test)
    cfg_abl["veto_mode"] = "dist"
    cfg_abl["veto_params_bo3"] = abl["params"]
    if ref_model is None:
        prep_tr = ref_prepare(train, best_of)
        refp = fit_ref(train, prep_tr)
        prep_te = ref_prepare(test, best_of)
        ref = ref_rows(test, prep_te, refp, best_of)
        cfg_ref = copy.deepcopy(saved)
        cfg_ref["veto_mode"] = "point"
        ref_desc = {"model": "comfort sim (point), eps-smoothed", "eps_pi_fit_on_train": refp}
    else:
        ref, cfg_ref = run(ref_model["params"], ref_model["overrides"], test)
        cfg_ref["veto_mode"] = "dist"
        cfg_ref["veto_params_bo3"] = ref_model["params"]
        ref_desc = {"model": "shipped distribution", "params": ref_model["params"],
                    "overrides": ref_model["overrides"]}
    S_new, S_ref, S_abl = summarize(new), summarize(ref), summarize(abl_rows)
    rep = {"phase": phase, "data_hashes": data_hashes(), "rows": cnt,
           "split": {"train_end": TRAIN_END, "test_start": TEST_START, "train": len(train), "test": len(test)},
           "params": w, "config_overrides": cfg_overrides,
           "reference": dict(ref_desc, test=S_ref),
           "ablation": {"c": abl["c"], "pi": abl["pi"], "train_loglik": abl["train_loglik"], "test": S_abl},
           "candidate": {"test": S_new, "train": summarize(new_tr)},
           "metric_notes": {"ece_played": "conditional on the realised series length (decider counted only on "
                                          "2-1 series)",
                            "coverage80": "80% highest-probability set of ordered (map 1, map 2) pairs",
                            "reference_hit": "comfort sim of the more likely starter under its fitted pi"},
           "test_evaluations": n_eval}
    checks = {}
    for tag, rr in (("", ref), ("_vs_ablation", abl_rows)):
        rep["pick_set_hit_diff" + tag] = paired(test, new, rr, "hit")
        rep["loglik_diff" + tag] = paired(test, new, rr, "ll")
        rep["decider_hit_diff" + tag] = paired(test, new, rr, "dec_hit")
        checks["pick_set_hit" + tag] = rep["pick_set_hit_diff" + tag]["ci95"][0] > 0
        ld = rep["loglik_diff" + tag]
        checks["loglik" + tag] = ld["ci95"][0] > 0 and math.isfinite(ld["diff"])
    checks["coverage80"] = 0.75 <= S_new.get("cov80", 0) <= 0.85
    checks["ece_played"] = S_new.get("ece_played", 1) <= 0.03
    checks["zero_prob"] = S_new.get("zero_prob", 1) == 0
    if best_of == 3:
        cfg_new["veto_mode"] = "dist"
        cfg_new["veto_params_bo3"] = w
        sn, sr, i6 = scoreline_rows(test, cfg_new, cfg_ref)
        sa, _, i6b = scoreline_rows(test, cfg_abl, cfg_ref)
        d = paired(test, sn, sr, "sll")
        da = paired(test, sn, sa, "sll")
        rep["scoreline_logloss"] = {"new": summarize_key(sn, "sll"), "ref": summarize_key(sr, "sll"),
                                    "ablation": summarize_key(sa, "sll"), "diff": d, "diff_vs_ablation": da}
        checks["scoreline_logloss"] = d["ci95"][1] < 0.002
        checks["scoreline_logloss_vs_ablation"] = da["ci95"][1] < 0.002
        checks["I6_p_a_identical"] = i6 and i6b
    if leakage_fn is not None:
        ok, detail = leakage_fn()
        rep["leakage_check"] = detail
        checks["leakage"] = ok
    rep["checks"] = checks
    rep["passed"] = all(checks.values())
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


def summarize_key(res, key):
    vals = [x[key] for x in res if key in x]
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="p0")
    a = ap.parse_args(argv)
    if a.phase == "p0":
        rep = phase0(os.path.join(ROOT, "data", "veto_phase0_report.json"))
        print(json.dumps(rep, indent=1))
        return 0
    print("phase gates are run from scripts/veto_fit.py --gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
