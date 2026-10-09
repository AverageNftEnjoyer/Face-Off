#!/usr/bin/env python3
"""
veto_fit.py -- fit the veto policy weights by maximum likelihood (VetoFit).
==========================================================================
Fan analytics only. Objective: mean log P(observed map order) over TRAINING
BO3 series (dated <= 2025-11-05, asserted), summing over every consistent
veto and both starters with prior pi. Stdlib coordinate descent with a
golden-section line search per weight (bounds [-10, 10], pi in [0.05,
0.95]), fixed parameter order, stop when a sweep gains < 1e-6, at most 50
sweeps. A line-search result is kept only if it improves the objective.
Series are evaluated in parallel; results are summed in series-index order,
so the fit is bit-reproducible whatever the number of processes.

Hyperparameter (exclusion alpha): grid on the inner fold (fit on the first
75% of training dates, score on the last 25%), then a refit on the whole
training window with the chosen value.

USAGE (repo root):
  python scripts/veto_fit.py --phase p1            fit, write data/veto_fit_report.json
  python scripts/veto_fit.py --phase p1 --gate     then score the gate (one test look)
"""
import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import predictor as pr  # noqa: E402
import veto  # noqa: E402
import veto_harness as H  # noqa: E402

EXCL_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
BOUNDS = (-10.0, 10.0)
PI_BOUNDS = (0.05, 0.95)
TOL = 1e-6
MAX_SWEEPS = 50
LINE_TOL = 1e-4
# Mechanical exclusion-alpha rule (screened on TRAINING rows only, before any
# fit): an alpha is rejected if its masks give any training series
# probability 0 (an observed pick masked for its picker under both starter
# hypotheses), or if more than MASK_CAP of training series have an observed
# pick masked under at least one hypothesis. Added after the phase-1 test
# look showed a probability-0 series at the inner-fold choice 0.05 (logged).
MASK_CAP = 0.01

PHASE_PARAMS = {
    "p1": ["alpha_ban1", "gamma_ban1", "rho_ban1", "alpha_ban2", "gamma_ban2", "rho_ban2",
           "alpha_pick", "gamma_pick", "pi"],
    "p2": ["alpha_ban1", "gamma_ban1", "alpha_ban2", "gamma_ban2", "alpha_pick", "gamma_pick", "pi"],
    "p3": ["alpha_ban1", "gamma_ban1", "alpha_ban2", "gamma_ban2", "alpha_pick", "gamma_pick",
           "eta_ban1", "eta_ban2", "eta_pick", "pi"],
}
PHASE_PARAMS["p4"] = PHASE_PARAMS["p3"]
PHASE_PARAMS["p6"] = ["alpha_ban1", "gamma_ban1", "rho_ban1", "alpha_ban2", "gamma_ban2", "rho_ban2", "pi"]

# ---------------------------------------------------------------- workers
_CTX = None
_OBS = None


def _init(ctxs, obs):
    global _CTX, _OBS
    _CTX, _OBS = ctxs, obs


def _chunk(args):
    w, lo, hi = args
    return [veto.loglik(_CTX[i], w, _OBS[i])[0] for i in range(lo, hi)]


class Objective:
    def __init__(self, ctxs, obs, procs):
        self.n = len(ctxs)
        self.procs = procs
        self.pool = mp.Pool(procs, initializer=_init, initargs=(ctxs, obs)) if procs > 1 else None
        if self.pool is None:
            _init(ctxs, obs)
        k = max(1, (self.n + procs * 4 - 1) // (procs * 4))
        self.chunks = [(i, min(self.n, i + k)) for i in range(0, self.n, k)]
        self.evals = 0

    def per_row(self, w):
        self.evals += 1
        tasks = [(w, lo, hi) for lo, hi in self.chunks]
        parts = self.pool.map(_chunk, tasks) if self.pool else [_chunk(t) for t in tasks]
        return [x for p in parts for x in p]

    def __call__(self, w):
        rows = self.per_row(w)
        s = 0.0
        for x in rows:            # fixed series-index order
            s += x
        return s / self.n

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


def golden_max(f, lo, hi, tol=LINE_TOL):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
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


def coord_descent(obj, w0, names, log=print):
    w = dict(w0)
    cur = obj(w)
    hist = [cur]
    for sweep in range(MAX_SWEEPS):
        start = cur
        for k in names:
            lo, hi = PI_BOUNDS if k == "pi" else BOUNDS

            def f(x, k=k):
                return obj(dict(w, **{k: x}))
            x, v = golden_max(f, lo, hi)
            if v > cur:
                w[k], cur = x, v
        hist.append(cur)
        log(f"  sweep {sweep + 1}: mean loglik {cur:.8f} ({obj.evals} evals)")
        if cur - start < TOL:
            return w, cur, sweep + 1, True, hist
    return w, cur, MAX_SWEEPS, False, hist


def contexts(rows, value, best_of=3, ctx_fn=None, key="veto_excl_alpha"):
    saved = pr.CONFIG[key]
    pr.CONFIG[key] = value
    try:
        return [(ctx_fn or pr.veto_context)(r["input"], best_of) for r in rows]
    finally:
        pr.CONFIG[key] = saved


def history_factory(w_post, roster=False):
    """History with a HabitAccumulator whose posteriors use `w_post` (phase 3),
    plus lineups for the roster-core level (phase 4)."""
    import backtest as bt
    import lineup_features as LF
    lineups = LF.load() if roster else None

    def make(leaky=False):
        h = bt.History()
        h.habits = bt.HabitAccumulator(lambda inp, pl: pr.veto_posterior(inp, pl, w_post), leaky=leaky,
                                       core_fn=h.core_five if roster else None)
        if roster:
            h.lineups = lineups
        return h
    return make


def start_params(phase, best_of=3, warm=None):
    # always the baseline-equivalent point (comfort sim: edge + veto_comfort x
    # comfort, map logits = map_scale x edge), never the CONFIG params, which
    # may hold a later phase's fit
    a, g = 1.0 / pr.CONFIG["map_scale"], pr.CONFIG["veto_comfort"]
    w = {"lookahead": False, "pi": 0.5, "temp": 1.0,
         "alpha_ban1": a, "gamma_ban1": g, "rho_ban1": 0.0, "alpha_ban2": a, "gamma_ban2": g, "rho_ban2": 0.0,
         "alpha_pick": a if best_of > 1 else 0.0, "gamma_pick": g if best_of > 1 else 0.0}
    w["version"] = f"veto-v2.{phase}"
    if phase in ("p2", "p3", "p4"):
        w["lookahead"] = True
        for k in ("rho_ban1", "rho_ban2"):
            w.pop(k, None)               # rho removed from phase 2 on (no double counting)
        for k in ("alpha_ban1", "alpha_ban2", "alpha_pick"):
            w[k] = 1.0
    if phase in ("p3", "p4"):
        for k in ("eta_ban1", "eta_ban2", "eta_pick"):
            w[k] = 0.0
    if warm:
        for k, v in warm.items():
            if (k in w or k.startswith("eta_")) and k != "version":
                w[k] = v
    return w


def mask_stats(rows, alpha, best_of=3, ctx_fn=None):
    """(zero-probability series, share with a masked observed pick) on rows."""
    if best_of == 1:
        return 0, 0.0
    zero = anyk = 0
    for c, r in zip(contexts(rows, alpha, best_of, ctx_fn), rows):
        E = {t: set(c.masked[t]) for t in "ab"}
        m1, m2 = r["played"][0], r["played"][1]
        ua = m1 in E["a"] or m2 in E["b"]
        ub = m1 in E["b"] or m2 in E["a"]
        zero += ua and ub
        anyk += ua or ub
    return zero, anyk / len(rows)


def fit_phase(phase, rows, procs, best_of=3, ctx_fn=None, warm=None, excl_grid=None, log=print,
              grid_key="veto_excl_alpha"):
    train, _, inner_fit, inner_val = H.split(rows)
    for r in train + inner_fit + inner_val:
        assert r["date"] <= H.TRAIN_END < H.TEST_START, "training row past the split boundary"
    names = PHASE_PARAMS[phase]
    # phases 3-4: the hyperparameter grid is screened with only the habit
    # weights free (the phase-2 weights fixed), then one joint refit
    screen = [k for k in names if k.startswith("eta_")] if phase in ("p3", "p4") else names
    w0 = start_params(phase, best_of, warm)
    grid = []
    excl_grid = EXCL_GRID if excl_grid is None else excl_grid
    best = None
    for a in excl_grid:
        t0 = time.time()
        if grid_key == "veto_excl_alpha":
            zero, share = mask_stats(train, a, best_of, ctx_fn)
            if zero or share > MASK_CAP:
                grid.append({grid_key: a, "rejected": True, "train_zero_prob_series": zero,
                             "train_masked_but_picked": share})
                log(f"{grid_key} {a}: rejected ({zero} zero-probability series, masked-but-picked {share:.4f})")
                continue
        obj = Objective(contexts(inner_fit, a, best_of, ctx_fn, grid_key), [r["played"] for r in inner_fit], procs)
        try:
            w, v, sweeps, conv, _ = coord_descent(obj, w0, screen, log)
        finally:
            obj.close()
        objv = Objective(contexts(inner_val, a, best_of, ctx_fn, grid_key), [r["played"] for r in inner_val], 1)
        vv = objv(w)
        grid.append({grid_key: a, "rejected": False, "inner_fit_loglik": v, "inner_val_loglik": vv, "sweeps": sweeps,
                     "converged": conv})
        log(f"{grid_key} {a}: inner fit {v:.6f}, inner val {vv:.6f} ({time.time() - t0:.0f}s)")
        if best is None or vv > best[1]:
            best = (a, vv, w)
    a = best[0]
    obj = Objective(contexts(train, a, best_of, ctx_fn, grid_key), [r["played"] for r in train], procs)
    try:
        w, v, sweeps, conv, hist = coord_descent(obj, best[2], names, log)
    finally:
        obj.close()
    return {"params": w, "grid_key": grid_key, "grid_value": a,
            "excl_alpha": a if grid_key == "veto_excl_alpha" else pr.CONFIG["veto_excl_alpha"], "train_loglik": v, "sweeps": sweeps, "converged": conv,
            "history": hist, "inner_grid": grid, "n_train": len(train),
            "n_inner_fit": len(inner_fit), "n_inner_val": len(inner_val)}


def record(phase, res, extra=None):
    rep = H.read_report()
    rep["data_hashes"] = H.data_hashes()
    rep["split"] = {"train_end": H.TRAIN_END, "test_start": H.TEST_START, "inner_frac": H.INNER_FRAC}
    entry = dict(res, config_snapshot=H.snapshot(), version=f"veto-v2.{phase}")
    if extra:
        entry.update(extra)
    rep.setdefault("phases", {})[phase] = entry
    H.write_report(rep)


GRID = {"p1": ("veto_excl_alpha", None), "p2": ("veto_excl_alpha", "fixed"),
        "p3": ("veto_habit_k", [1.0, 2.0, 4.0, 8.0, 16.0]), "p4": ("veto_roster_k", [1.0, 2.0, 4.0, 8.0, 16.0]),
        "p6": ("veto_excl_alpha", None)}


def phase_setup(phase, rep, excl_alpha=None):
    """(rows, counts, CONFIG overrides, warm start, grid) for a phase. Later
    phases build on the stored fit of the phase before them."""
    phases = rep.get("phases", {})
    best_of = 1 if phase == "p6" else 3
    over = {}
    warm = None
    factory = None
    key, grid = GRID[phase]
    if phase in ("p2", "p3", "p4"):
        over["veto_excl_alpha"] = phases["p1"]["excl_alpha"] if excl_alpha is None else excl_alpha
    if phase == "p2":
        warm = {k: v for k, v in phases["p1"]["params"].items() if k in ("gamma_ban1", "gamma_ban2",
                                                                         "gamma_pick", "pi")}
        grid = [over["veto_excl_alpha"]]
    if phase == "p6" and excl_alpha is not None:
        grid = [excl_alpha]
    if phase in ("p3", "p4"):
        w_post = phases["p2"]["params"]
        warm = dict(w_post)
        over["veto_use_habits"] = True
        if phase == "p4":
            over["veto_habit_k"] = phases["p3"]["grid_value"]
            warm = dict(phases["p3"]["params"])
            over["veto_use_roster"] = True
        factory = history_factory(w_post, roster=(phase == "p4"))
    rows, cnt = H.load_rows(best_of, history_factory=factory)
    return rows, cnt, over, warm, key, grid, best_of, factory


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="p1")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--gate-only", action="store_true", help="score the stored fit, no refit")
    ap.add_argument("--ref", default=None, help="phase whose stored fit is the gate reference "
                                                "(default: the comfort sim)")
    ap.add_argument("--note", default=None, help="reason logged with the test evaluation")
    ap.add_argument("--excl-alpha", type=float, default=None,
                    help="fixed exclusion alpha for phases 2-4 / 6 (default: the phase 1 inner-fold choice)")
    a = ap.parse_args(argv)
    rep = H.read_report()
    t0 = time.time()
    rows, cnt, over, warm, key, grid, best_of, _ = phase_setup(a.phase, rep, a.excl_alpha)
    print(f"{a.phase}: {len(rows)} rows ({time.time() - t0:.0f}s to build)", flush=True)
    saved = copy.deepcopy(pr.CONFIG)
    if a.gate_only:
        res = rep["phases"][a.phase]
        assert res["config_snapshot"] == H.snapshot(), "CONFIG differs from the fit report snapshot"
        a.gate = True
    else:
        try:
            pr.CONFIG.update(over)
            res = fit_phase(a.phase, rows, a.procs, best_of, warm=warm, excl_grid=grid, grid_key=key)
        finally:
            pr.CONFIG.clear()
            pr.CONFIG.update(saved)
        res["seconds"] = round(time.time() - t0, 1)
        res["rows"] = cnt
        res["config_overrides"] = over
        record(a.phase, res)
    print(json.dumps({k: res[k] for k in ("params", "grid_key", "grid_value", "train_loglik", "sweeps",
                                          "converged")}, indent=1), flush=True)
    if a.gate:
        out = os.path.join(ROOT, "data", f"veto_phase{a.phase[1:]}_report.json")
        over_g = dict(res.get("config_overrides", {}), **{res["grid_key"]: res["grid_value"]})
        ref = None
        if a.ref:
            rr = rep["phases"][a.ref]
            ref = {"params": rr["params"],
                   "overrides": dict(rr.get("config_overrides", {}), **{rr["grid_key"]: rr["grid_value"]})}
        leak = None
        if a.phase in ("p3", "p4"):
            def leak():
                import leakage_check as LC
                import backtest as bt
                saved_c = copy.deepcopy(pr.CONFIG)
                try:
                    pr.CONFIG.update(over_g)
                    pr.CONFIG["veto_params_bo3"] = res["params"]
                    ms = bt.load_matches()
                    days = LC.sample_days(sorted({m["date"] for m in ms})[30:], 40)
                    fac = history_factory(rep["phases"]["p2"]["params"], roster=(a.phase == "p4"))
                    probs, excl, status = LC.check_veto(ms, days, fac)
                finally:
                    pr.CONFIG.clear()
                    pr.CONFIG.update(saved_c)
                ok = not probs and excl > 0 and status == "caught"
                return ok, {"differing_distributions": len(probs), "exclusion_canary_days": excl,
                            "habit_canary": status, "days_with_habits": 4}
        g = H.gate(a.phase, res["params"], over_g, out, rows, best_of, ref_model=ref, leakage_fn=leak,
                   note=a.note)
        print(json.dumps({k: g[k] for k in ("checks", "passed", "pick_set_hit_diff", "loglik_diff")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
