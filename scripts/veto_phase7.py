#!/usr/bin/env python3
"""
veto_phase7.py -- phase 7 of the veto model: three evidence changes, each
fitted on TRAINING data only and gated once on the test window.

  decay    recency-decayed habit / exclusion evidence (CONFIG veto_decay_halflife)
  soft     soft avoid score instead of the binary exclusion, with pick habits
           shrunk 4x harder than ban habits (veto_soft_avoid, veto_habit_k_ban)
  roster   habit evidence discounted after lineup changes (veto_roster_discount)
  combined the three together, each at its own inner-fold choice

Every candidate sits on the phase-3 model (lookahead policy with the phase-2
weights, exclusion alpha 0.01, habits on, veto_habit_k 4) and refits only the
weights its change touches (eta_* habit weights; plus zeta_* avoid weights for
soft) with the rest held at the phase-3 fit. The one hyperparameter per change
is chosen on the inner fold (fit on the first 75% of training dates, score on
the last 25%); then the free weights are refitted on the whole training
window. Nothing is chosen on dates >= 2025-11-07.

The labels of past vetoes that feed the habit counts (veto_posterior) always
use the plain evidence (flat 90-day exclusion, no phase-7 option), so the
counts are comparable across candidates and cacheable.

Gate (same as phases 1-6, against the shipped config = the comfort-sim point
veto, eps-smoothed for likelihood): pick-set hit and veto log-likelihood paired
95% CI lower bound > 0; scoreline log loss CI upper bound of (new - ref) <
+0.002; 80%-set coverage in [0.75, 0.85]; map ECE <= 0.03; zero_prob == 0;
p_a bit-identical on every held-out row. Also reported (not gating): the same
paired differences against the phase-3 model.

USAGE (repo root):
  python scripts/veto_phase7.py cache                       posterior cache
  python scripts/veto_phase7.py fit  --change decay --stage grid --grid 60   (per grid value)
  python scripts/veto_phase7.py fit  --change decay --stage final            (best value, refit on train)
  python scripts/veto_phase7.py gate --change decay         one test look
  python scripts/veto_phase7.py report                      data/veto_phase7_report.json
"""
import argparse
import contextlib
import copy
import json
import math
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backtest as bt  # noqa: E402
import lineup_features as LF  # noqa: E402
import predictor as pr  # noqa: E402
import veto  # noqa: E402
import veto_fit as F  # noqa: E402
import veto_harness as H  # noqa: E402

REPORT = os.path.join(ROOT, "data", "veto_phase7_report.json")
BASE_OVER = {"veto_excl_alpha": 0.01, "veto_use_habits": True, "veto_habit_k": 4.0}
GRIDS = {"decay": ("halflife", [45, 60, 90, 120]),
         "soft": ("habit_k_ban", [1.0, 2.0, 4.0, 8.0]),
         "roster": ("floor", [0.5])}
FREE = {"decay": ["eta_ban1", "eta_ban2", "eta_pick"],
        "soft": ["eta_ban1", "eta_ban2", "eta_pick", "zeta_ban1", "zeta_ban2", "zeta_pick"],
        "roster": ["eta_ban1", "eta_ban2", "eta_pick"]}
FREE["combined"] = FREE["soft"]
W_BOUNDS = (-1.0, 3.0)
W_TOL = 0.1
SWEEPS = 2
SWEEP_TOL = 1e-3


# ---------------------------------------------------------------- plumbing
@contextlib.contextmanager
def config(over):
    saved = copy.deepcopy(pr.CONFIG)
    pr.CONFIG.update(over)
    try:
        yield
    finally:
        pr.CONFIG.clear()
        pr.CONFIG.update(saved)


def overrides(change, hp):
    """CONFIG overrides of a candidate. hp: {"halflife", "habit_k_ban", "floor"}."""
    o = dict(BASE_OVER)
    if change in ("decay", "combined"):
        o["veto_decay_halflife"] = hp["halflife"]
    if change in ("soft", "combined"):
        o["veto_soft_avoid"] = True
        o["veto_habit_k_ban"] = hp["habit_k_ban"]
    if change in ("roster", "combined"):
        o["veto_roster_discount"] = True
        o["veto_roster_floor"] = hp["floor"]
    return o


def tmp_dir():
    d = os.environ.get("VETO7_DIR") or os.path.join(ROOT, "data", "veto_phase7_work")
    os.makedirs(d, exist_ok=True)
    return d


def cache_path():
    return os.path.join(tmp_dir(), "posterior_cache.pkl")


class CachedPost:
    """veto_posterior with a cache keyed by the plain (baseline) parts of the
    input, which every candidate shares."""

    def __init__(self, w_post, cache):
        self.w, self.cache = w_post, cache

    def __call__(self, inp, played):
        key = json.dumps([inp["team_a"], inp["team_b"], inp["rating_a"], inp["rating_b"], inp["maps_a"],
                          inp["maps_b"], inp["map_pool"], inp["veto_ev_a"]["maps"], inp["veto_ev_b"]["maps"],
                          inp.get("permaban_a"), inp.get("permaban_b"), list(played)], sort_keys=True)
        if key not in self.cache:
            self.cache[key] = pr.veto_posterior(inp, played, self.w)
        return self.cache[key]


def factory(w_post, cache):
    post = CachedPost(w_post, cache)
    lineups = {}

    def make(leaky=False):
        h = bt.History()
        if pr.CONFIG.get("veto_roster_discount"):
            if not lineups:
                lineups.update(LF.load())
            h.lineups = lineups
        h.habits = bt.HabitAccumulator(post, leaky=leaky)
        return h
    return make


def load_cache():
    p = cache_path()
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return {}


def build_rows(over, w_post, cache):
    with config(over):
        rows, cnt = H.load_rows(3, history_factory=factory(w_post, cache))
    return rows, cnt


def contexts(rows, over):
    with config(over):
        return [pr.veto_context(r["input"], 3) for r in rows]


# ---------------------------------------------------------------- fitting
def coord_descent(obj, w0, names, log=print, sweeps=SWEEPS):
    w = dict(w0)
    cur = obj(w)
    for sweep in range(sweeps):
        start = cur
        for k in names:
            def f(x, k=k):
                return obj(dict(w, **{k: x}))
            x, v = F.golden_max(f, W_BOUNDS[0], W_BOUNDS[1], tol=W_TOL)
            if v > cur:
                w[k], cur = x, v
        log(f"    sweep {sweep + 1}: mean loglik {cur:.6f} ({obj.evals} evals)")
        if cur - start < SWEEP_TOL:
            break
    return w, cur


def fit_free(rows, over, w0, names, procs, log=print, sweeps=SWEEPS):
    obj = F.Objective(contexts(rows, over), [r["played"] for r in rows], procs)
    try:
        return coord_descent(obj, w0, names, log, sweeps)
    finally:
        obj.close()


def score(rows, over, w):
    obj = F.Objective(contexts(rows, over), [r["played"] for r in rows], 1)
    return obj(w)


def base_params(rep):
    w = dict(rep["phases"]["p3"]["params"])
    for k in ("zeta_ban1", "zeta_ban2", "zeta_pick"):
        w.setdefault(k, 0.0)
    return w


def cmd_cache(a):
    rep = H.read_report()
    cache = {}
    t0 = time.time()
    build_rows(BASE_OVER, rep["phases"]["p2"]["params"], cache)
    with open(cache_path(), "wb") as f:
        pickle.dump(cache, f)
    print(f"cache: {len(cache)} posteriors ({time.time() - t0:.0f}s)")


def chosen_hp(change):
    """Hyperparameters of the candidate: the single-change fits' inner-fold choices."""
    hp = {}
    for ch, (key, _) in GRIDS.items():
        if change in (ch, "combined"):
            with open(os.path.join(tmp_dir(), f"fit_{ch}.json"), encoding="utf-8") as f:
                hp[key] = json.load(f)["hp"][key]
    return hp


def _setup(rep):
    return rep["phases"]["p2"]["params"], base_params(rep), load_cache()


def _check_inner(*parts):
    for rr in parts:
        for r in rr:
            assert r["date"] <= H.TRAIN_END < H.TEST_START, "inner-fold row past the training boundary"


def _dump(name, obj):
    with open(os.path.join(tmp_dir(), name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def cmd_fit(a):
    """stages: base (phase-3 reference refit on the inner fold), grid (one
    hyperparameter value on the inner fold; --grid v), final (pick the best
    grid value by inner-fold validation log-likelihood, refit the free weights
    on the whole training window)."""
    rep = H.read_report()
    w_post, w3, cache = _setup(rep)
    change, t0 = a.change, time.time()
    log = lambda s: print(s, flush=True)
    names = FREE[change]
    if a.stage == "base":
        rows0, _ = build_rows(BASE_OVER, w_post, cache)
        train0, _, fit0, val0 = H.split(rows0)
        _check_inner(train0, fit0, val0)
        wb, vb = fit_free(fit0, BASE_OVER, w3, FREE["decay"], a.procs, log, sweeps=1)
        out = {"inner_fit": vb, "inner_val": score(val0, BASE_OVER, wb), "params": wb, "n_fit": len(fit0),
               "n_val": len(val0)}
        _dump("fit_base.json", out)
        log(json.dumps(out))
        return
    key, grid = GRIDS[change] if change != "combined" else (None, None)
    if a.stage == "grid":
        hp = {key: type(grid[0])(a.grid)}
        over = overrides(change, hp)
        rows, _ = build_rows(over, w_post, cache)
        train, _, fit, val = H.split(rows)
        _check_inner(train, fit, val)
        w, v = fit_free(fit, over, w3, names, a.procs, log, sweeps=1)
        out = {"hp": hp, "inner_fit": v, "inner_val": score(val, over, w), "params": w,
               "seconds": round(time.time() - t0, 1)}
        _dump(f"grid_{change}_{a.grid}.json", out)
        log(json.dumps({k: out[k] for k in ("hp", "inner_fit", "inner_val", "seconds")}))
        return
    inner = []
    if change == "combined":
        hp = chosen_hp("combined")
    else:
        for g in grid:
            p = os.path.join(tmp_dir(), f"grid_{change}_{g}.json")
            with open(p, encoding="utf-8") as f:
                inner.append(json.load(f))
        hp = max(inner, key=lambda x: x["inner_val"])["hp"]
    over = overrides(change, hp)
    rows, cnt = build_rows(over, w_post, cache)
    train, _, _, _ = H.split(rows)
    assert all(r["date"] <= H.TRAIN_END for r in train)
    w, v = fit_free(train, over, w3, names, a.procs, log)
    w["version"] = f"veto-v2.p7.{change}"
    out = {"change": change, "free": names, "hp": hp, "overrides": over, "params": w, "train_loglik": v,
           "n_train": len(train), "rows": cnt, "seconds": round(time.time() - t0, 1),
           "inner_grid": [{k: x[k] for k in ("hp", "inner_fit", "inner_val")} for x in inner]}
    bp = os.path.join(tmp_dir(), "fit_base.json")
    if os.path.exists(bp):
        with open(bp, encoding="utf-8") as f:
            out["base_inner"] = {k: v2 for k, v2 in json.load(f).items() if k != "params"}
    _dump(f"fit_{change}.json", out)
    log(json.dumps({k: out[k] for k in ("hp", "train_loglik", "seconds")}))
    log(json.dumps(w))


# ---------------------------------------------------------------- gate
def cmd_gate(a):
    rep = H.read_report()
    w_post = rep["phases"]["p2"]["params"]
    cache = load_cache()
    change = a.change
    with open(os.path.join(tmp_dir(), f"fit_{change}.json"), encoding="utf-8") as f:
        fit = json.load(f)
    over, w = fit["overrides"], fit["params"]
    rows, cnt = build_rows(over, w_post, cache)
    train, test, _, _ = H.split(rows)
    saved = copy.deepcopy(pr.CONFIG)
    n_eval = H.bump_test_counter(f"p7 {change} gate on test" + (f" ({a.note})" if a.note else ""))

    def run(params, ov, rr):
        try:
            pr.CONFIG.update(ov)
            out = H.model_rows(rr, params, 3)
            cfg = copy.deepcopy(pr.CONFIG)
        finally:
            pr.CONFIG.clear()
            pr.CONFIG.update(saved)
        return out, cfg

    new, cfg_new = run(w, over, test)
    new_tr, _ = run(w, over, train)
    # reference 1: the shipped config (comfort-sim point veto), as in phases 1-6
    prep_tr = H.ref_prepare(train)
    refp = H.fit_ref(train, prep_tr)
    ref = H.ref_rows(test, H.ref_prepare(test), refp)
    cfg_ref = copy.deepcopy(saved)
    cfg_ref["veto_mode"] = "point"
    # reference 2 (not gating): the phase-3 model on its own evidence
    base_rows, _ = build_rows(BASE_OVER, w_post, cache)
    _, base_test, _, _ = H.split(base_rows)
    p3, _ = run(base_params(rep), BASE_OVER, base_test)
    S_new, S_ref, S_p3 = H.summarize(new), H.summarize(ref), H.summarize(p3)
    res = {"change": change, "hp": fit["hp"], "overrides": over, "params": w,
           "rows": cnt, "split": {"train": len(train), "test": len(test)},
           "reference_shipped": dict(test=S_ref, eps_pi_fit_on_train=refp),
           "reference_phase3": {"test": S_p3},
           "candidate": {"test": S_new, "train": H.summarize(new_tr)},
           "test_evaluations": n_eval}
    res["pick_set_hit_diff"] = H.paired(test, new, ref, "hit")
    res["loglik_diff"] = H.paired(test, new, ref, "ll")
    res["decider_hit_diff"] = H.paired(test, new, ref, "dec_hit")
    res["pick_set_hit_diff_vs_phase3"] = H.paired(test, new, p3, "hit")
    res["loglik_diff_vs_phase3"] = H.paired(test, new, p3, "ll")
    checks = {"pick_set_hit": res["pick_set_hit_diff"]["ci95"][0] > 0,
              "loglik": res["loglik_diff"]["ci95"][0] > 0 and math.isfinite(res["loglik_diff"]["diff"]),
              "coverage80": 0.75 <= S_new.get("cov80", 0) <= 0.85,
              "ece_played": S_new.get("ece_played", 1) <= 0.03,
              "zero_prob": S_new.get("zero_prob", 1) == 0}
    cfg_new["veto_mode"] = "dist"
    cfg_new["veto_params_bo3"] = w
    sn, sr, i6 = H.scoreline_rows(test, cfg_new, cfg_ref)
    d = H.paired(test, sn, sr, "sll")
    res["scoreline_logloss"] = {"new": H.summarize_key(sn, "sll"), "ref": H.summarize_key(sr, "sll"), "diff": d}
    checks["scoreline_logloss"] = d["ci95"][1] < 0.002
    checks["I6_p_a_identical"] = bool(i6)
    res["checks"] = checks
    res["passed"] = all(checks.values())
    with open(os.path.join(tmp_dir(), f"gate_{change}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"checks": checks, "passed": res["passed"], "hit": res["pick_set_hit_diff"],
                      "ll": res["loglik_diff"], "scoreline": d, "cov80": S_new.get("cov80"),
                      "ece": S_new.get("ece_played")}, indent=1))


def cmd_leak(a):
    """Leakage check (veto test 6 with habits on) for a candidate's CONFIG."""
    rep = H.read_report()
    with open(os.path.join(tmp_dir(), f"fit_{a.change}.json"), encoding="utf-8") as f:
        fit = json.load(f)
    import leakage_check as LC
    cache = load_cache()
    with config(dict(fit["overrides"], veto_params_bo3=fit["params"])):
        ms = bt.load_matches()
        days = LC.sample_days(sorted({m["date"] for m in ms})[30:], a.days)
        probs, excl, status = LC.check_veto(ms, days, factory(rep["phases"]["p2"]["params"], cache))
    out = {"change": a.change, "differing_distributions": len(probs), "exclusion_canary_days": excl,
           "habit_canary": status, "days": len(days), "ok": not probs and excl > 0 and status == "caught"}
    with open(os.path.join(tmp_dir(), f"leak_{a.change}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


def cmd_report(a):
    out = {"phase": "p7", "split": {"train_end": H.TRAIN_END, "test_start": H.TEST_START},
           "data_hashes": H.data_hashes(), "candidates": {}}
    for ch in ("decay", "soft", "roster", "combined"):
        p = os.path.join(tmp_dir(), f"gate_{ch}.json")
        q = os.path.join(tmp_dir(), f"fit_{ch}.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            g = json.load(f)
        with open(q, encoding="utf-8") as f:
            ft = json.load(f)
        g["fit"] = {k: ft.get(k) for k in ("inner_grid", "base_inner", "free", "train_loglik", "seconds", "n_train")}
        lp = os.path.join(tmp_dir(), f"leak_{ch}.json")
        if os.path.exists(lp):
            with open(lp, encoding="utf-8") as f:
                g["leakage_check"] = json.load(f)
        out["candidates"][ch] = g
    out["enabled"] = [c for c, g in out["candidates"].items() if g["passed"]]
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", REPORT, "enabled:", out["enabled"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "fit", "gate", "leak", "report"])
    ap.add_argument("--change", choices=["decay", "soft", "roster", "combined"])
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--stage", choices=["base", "grid", "final"], default="final")
    ap.add_argument("--grid", default=None, help="hyperparameter value (stage grid)")
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--note", default=None)
    a = ap.parse_args(argv)
    {"cache": cmd_cache, "fit": cmd_fit, "gate": cmd_gate, "leak": cmd_leak, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
