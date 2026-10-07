#!/usr/bin/env python3
"""
walkforward.py -- walk-forward fit and evaluation of the factor weights.

MODEL. Each engine factor i is linear in its weight: d_i = w_i * u_i, where
u_i is the capped, shrunk, residualised signal. With the engine's
uncertainty shrink k (which itself depends on the weights):

    P(team_a wins) = sigmoid( offset/k + sum_i w_i * u_i / k )

offset = roster + stakes log-odds (fixed priors, not fitted). Temperature is
held at 1; the weights carry the scale.

OBJECTIVE (per training window, in WEIGHT space):

    minimise  sum_n logloss(y_n, p_n)  +  (lam / 2) * sum_i (w_i - w0_i)^2
    subject to w_i >= 0

w0 = the REASONED pre-data weights (git 73efb78), never weights fitted on
any data, so no fold's prior encodes the future. w_i >= 0 is a sign prior
(better form / map edge can't lower a team's chances); a factor whose best
fit is negative is switched off. Because k depends on w, the fit is a fixed
point: fit w with k frozen, recompute k, refit, until max |dw| < 1e-4. The
penalty is always measured against w0 (fixes the drifting anchor in
fit_weights.py, which penalises toward the previous round's weights).

WALK-FORWARD (expanding window, never shuffled):
  * eval rows = BO3 series where both teams have >= 5 prior series, by date;
  * the first test block starts at the date of eval row MIN_TRAIN; blocks are
    BLOCK_DAYS long;
  * for each block: train = every eval row dated before the block start;
    lam is picked from LAMBDAS on the last 20% of train (by date, after
    fitting on the first 80%); the weights are refit on all of train with
    that lam and score the block. Asserted: max train date < block start.
  * the out-of-fold (OOF) predictions of all blocks form the evaluation set.

MODELS SCORED ON THE SAME OOF ROWS
  elo_only    point-in-time Elo expectation, no fitting.
  elo_platt   sigmoid(a * logit(p_elo)), a fitted walk-forward (1 parameter).
  shipped     predictor.py CONFIG as it is today. Its weights were fitted on
              data up to 2025-10-08, so before that date it is IN-SAMPLE; it
              is compared only on the CLEAN window after SHIPPED_FIT_END.
  wf_fit      the walk-forward fit above.

ACCEPTANCE TEST (fixed before running; all must hold to replace CONFIG):
  A1  wf_fit OOF log loss <= elo_platt OOF log loss (the engine must earn
      its six extra factors against a 1-parameter recalibrated Elo).
  A2  on the clean window, wf_fit log loss <= shipped log loss.
  A3  calibration, favourite view (p_fav = max(p, 1-p)): every bucket with
      n >= 30 is within 10pp AND its Wilson 95% interval for the actual rate
      contains the mean stated probability.
  A4  ECE10 (favourite view) of wf_fit <= shipped + 0.005 on the clean window.
  An improvement is only CLAIMED when the paired-bootstrap 95% CI of the log
  loss difference excludes 0; otherwise the report says "not significant".

USAGE (from D:/Face-Off):
    python walkforward.py                 # writes data/walkforward_report.json
    python walkforward.py --block-days 91 --min-train 400
Stdlib only, deterministic (bootstrap seeded).
"""
import argparse
import copy
import json
import math
import os
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backtest as bt  # noqa: E402
import fit_weights as fw  # noqa: E402
import predictor as pr  # noqa: E402

FACTORS = ["base_strength", "form_30d", "form_last5", "head_to_head", "map_veto"]
WKEY = {"base_strength": "w_base", "form_30d": "w_form30", "form_last5": "w_form5",
        "head_to_head": "w_h2h", "map_veto": "w_veto"}
# reasoned, pre-data weights (predictor.py at commit 73efb78)
W0 = {"w_base": 0.8, "w_form30": 0.7, "w_form5": 0.25, "w_h2h": 0.3, "w_veto": 0.75}
LAMBDAS = [0.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
W_FLOOR = 1e-6          # engine weight used while fitting so u_i = d_i / w_i stays recoverable
MAX_ROUNDS = 10
SHIPPED_FIT_END = "2025-10-08"   # last training date of the weights in predictor.py
CFG_SHIPPED = copy.deepcopy(pr.CONFIG)
FAV_BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)]


def label(r):
    return 1 if r["match"]["winner"] == r["match"]["team_a"] else 0


def with_config(cfg, fn):
    """Run fn() with predictor.CONFIG temporarily replaced by cfg."""
    saved = copy.deepcopy(pr.CONFIG)
    pr.CONFIG.clear()
    pr.CONFIG.update(cfg)
    try:
        return fn()
    finally:
        pr.CONFIG.clear()
        pr.CONFIG.update(saved)


def engine_cfg(w):
    cfg = copy.deepcopy(CFG_SHIPPED)
    for key, v in w.items():
        cfg[key] = max(v, W_FLOOR)
    cfg["temperature"] = 1.0
    return cfg


def final_cfg(w):
    cfg = copy.deepcopy(CFG_SHIPPED)
    cfg.update({k: v for k, v in w.items()})
    cfg["temperature"] = 1.0
    return cfg


def design(rows, w):
    """X[n][i] = u_i / k and offset = (non-fitted factors) / k under weights w."""
    def run():
        X, off = [], []
        for r in rows:
            c = pr._compute(dict(r["input"]))
            d = {f[0]: f[1] for f in c["factors"]}
            k = c["shrink_k"]
            X.append([d[f] / pr.CONFIG[WKEY[f]] / k for f in FACTORS])
            off.append(sum(v for n, v in d.items() if n not in WKEY) / k)
        return X, off
    return with_config(engine_cfg(w), run)


def fit_nonneg(X, off, y, lam, prior):
    """Penalised logistic fit toward `prior`, every weight >= 0 (active set)."""
    active = list(range(len(prior)))
    while active:
        sub = fw.fit_logistic([[x[i] for i in active] for x in X], off, y, lam,
                              prior=[prior[i] for i in active])
        c = [0.0] * len(prior)
        for i, v in zip(active, sub):
            c[i] = v
        neg = [i for i in active if c[i] < 0]
        if not neg:
            return c
        active.remove(min(neg, key=lambda i: c[i]))
    return [0.0] * len(prior)


def fit_weights(rows, lam):
    y = [label(r) for r in rows]
    prior = [W0[WKEY[f]] for f in FACTORS]
    w = dict(W0)
    for _ in range(MAX_ROUNDS):
        X, off = design(rows, w)
        c = fit_nonneg(X, off, y, lam, prior)
        new = {WKEY[f]: v for f, v in zip(FACTORS, c)}
        delta = max(abs(new[k] - w[k]) for k in new)
        w = new
        if delta < 1e-4:
            break
    return w


def predict(rows, w):
    return with_config(final_cfg(w), lambda: [pr.predict_match(dict(r["input"]))["p_a_exact"] for r in rows])


def fit_platt(rows):
    xs = [bt.logit(r["meta"]["p_elo"]) for r in rows]
    return bt.fit_temperature(xs, [label(r) for r in rows])


# ------------------------------------------------------------------ metrics
def fav_view(ps, ys):
    return [(max(p, 1 - p), (y if p >= 0.5 else 1 - y)) for p, y in zip(ps, ys)]


def fav_buckets(ps, ys):
    rows = []
    fv = fav_view(ps, ys)
    for lo, hi in FAV_BUCKETS:
        b = [(p, y) for p, y in fv if lo <= p < hi]
        n = len(b)
        if not n:
            rows.append({"bucket": f"{lo:.0%}-{min(hi, 1):.0%}", "n": 0})
            continue
        mp = sum(p for p, _ in b) / n
        k = sum(y for _, y in b)
        lo95, hi95 = bt.wilson(k, n)
        gap = k / n - mp
        rows.append({"bucket": f"{lo:.0%}-{min(hi, 1):.0%}", "n": n, "stated": mp, "actual": k / n,
                     "gap_pp": 100 * gap, "wilson95": [lo95, hi95],
                     "pass": (n < 30) or (abs(gap) <= 0.10 and lo95 <= mp <= hi95),
                     "scored": n >= 30})
    return rows


def fav_ece(ps, ys):
    n = len(ps)
    return sum(b["n"] / n * abs(b["actual"] - b["stated"]) for b in fav_buckets(ps, ys) if b["n"])


def metrics(ps, ys):
    n = len(ys)
    acc = bt.accuracy(ps, ys)
    return {"n": n, "accuracy": acc, "accuracy_wilson95": list(bt.wilson(round(acc * n), n)),
            "brier": bt.brier(ps, ys), "log_loss": bt.log_loss(ps, ys),
            "ece10_fav": fav_ece(ps, ys), "fav_buckets": fav_buckets(ps, ys)}


def paired(ps1, ps2, ys):
    """log loss and Brier of ps1 minus ps2, with paired-bootstrap 95% CIs."""
    out = {}
    for name, fn in (("log_loss", bt.log_loss), ("brier", bt.brier)):
        lo, hi = bt.paired_bootstrap_diff(fn, ps1, ps2, ys)
        d = fn(ps1, ys) - fn(ps2, ys)
        out[name] = {"diff": d, "ci95": [lo, hi],
                     "verdict": "better (significant)" if hi < 0 else
                                "worse (significant)" if lo > 0 else "not significant"}
    return out


# ------------------------------------------------------------------ walk-forward
def blocks(ev, min_train, block_days):
    start = bt._d(ev[min_train]["match"]["date"])
    last = bt._d(ev[-1]["match"]["date"])
    out = []
    while start <= last:
        end = start + timedelta(days=block_days)
        tr = [r for r in ev if bt._d(r["match"]["date"]) < start]
        te = [r for r in ev if start <= bt._d(r["match"]["date"]) < end]
        if te:
            assert max(r["match"]["date"] for r in tr) < min(r["match"]["date"] for r in te)
            out.append((start, end, tr, te))
        start = end
    return out


def run(min_train, block_days, verbose=True):
    rows = bt.build_dataset(bt.load_matches())
    ev, _ = bt.select_eval(rows, bt.MIN_HISTORY)
    folds = blocks(ev, min_train, block_days)
    oof = {"elo_only": [], "elo_platt": [], "shipped": [], "wf_fit": []}
    oof_rows, fold_log = [], []
    for start, end, tr, te in folds:
        fit_part, val_part = bt.chrono_split(tr, 0.8)
        yv = [label(r) for r in val_part]
        scores = []
        for lam in LAMBDAS:
            ll = bt.log_loss(predict(val_part, fit_weights(fit_part, lam)), yv)
            scores.append((ll, lam))
        best_lam = min(scores)[1]
        w = fit_weights(tr, best_lam)
        a = fit_platt(tr)
        yt = [label(r) for r in te]
        p_wf = predict(te, w)
        p_ship = with_config(CFG_SHIPPED, lambda: [pr.predict_match(dict(r["input"]))["p_a_exact"] for r in te])
        oof["wf_fit"] += p_wf
        oof["shipped"] += p_ship
        oof["elo_only"] += [r["meta"]["p_elo"] for r in te]
        oof["elo_platt"] += [bt.sigmoid(a * bt.logit(r["meta"]["p_elo"])) for r in te]
        oof_rows += te
        entry = {"block": [start.isoformat(), (end - timedelta(days=1)).isoformat()],
                 "n_train": len(tr), "n_test": len(te), "lambda": best_lam,
                 "weights": {k: round(v, 4) for k, v in w.items()}, "elo_platt_a": round(a, 4),
                 "test_log_loss": {"wf_fit": bt.log_loss(p_wf, yt), "shipped": bt.log_loss(p_ship, yt)}}
        fold_log.append(entry)
        if verbose:
            print(f"{entry['block'][0]}..{entry['block'][1]}  train {len(tr):4d} test {len(te):3d}  "
                  f"lam {best_lam:>6}  " + " ".join(f"{k[2:]}={v:.3f}" for k, v in entry["weights"].items())
                  + f"  LL wf {entry['test_log_loss']['wf_fit']:.4f} ship {entry['test_log_loss']['shipped']:.4f}")
    y = [label(r) for r in oof_rows]
    clean = [i for i, r in enumerate(oof_rows) if r["match"]["date"] > SHIPPED_FIT_END]
    yc = [y[i] for i in clean]
    sub = lambda ps: [ps[i] for i in clean]

    rep = {"oof_range": [oof_rows[0]["match"]["date"], oof_rows[-1]["match"]["date"]],
           "clean_window_start_after": SHIPPED_FIT_END, "n_oof": len(y), "n_clean": len(clean),
           "folds": fold_log, "prior_w0": W0, "lambdas": LAMBDAS,
           "oof": {m: metrics(ps, y) for m, ps in oof.items() if m != "shipped"},
           "clean": {m: metrics(sub(ps), yc) for m, ps in oof.items()},
           "paired_oof": {"wf_fit - elo_platt": paired(oof["wf_fit"], oof["elo_platt"], y),
                          "wf_fit - elo_only": paired(oof["wf_fit"], oof["elo_only"], y)},
           "paired_clean": {"wf_fit - shipped": paired(sub(oof["wf_fit"]), sub(oof["shipped"]), yc),
                            "shipped - elo_platt": paired(sub(oof["shipped"]), sub(oof["elo_platt"]), yc)}}
    o, c = rep["oof"], rep["clean"]
    rep["acceptance"] = {
        "A1_wf_le_elo_platt_oof": o["wf_fit"]["log_loss"] <= o["elo_platt"]["log_loss"],
        "A2_wf_le_shipped_clean": c["wf_fit"]["log_loss"] <= c["shipped"]["log_loss"],
        "A3_buckets_pass_oof": all(b.get("pass", True) for b in o["wf_fit"]["fav_buckets"]),
        "A4_ece_not_worse_clean": c["wf_fit"]["ece10_fav"] <= c["shipped"]["ece10_fav"] + 0.005,
    }
    rep["acceptance"]["ALL"] = all(rep["acceptance"].values())
    # proposed weights = the final fold's fit (trained on everything before the last block)
    # plus a fit on ALL eval rows for the live engine
    lam_all = fold_log[-1]["lambda"]
    rep["proposed_live_weights"] = {"lambda": lam_all,
                                    "weights": {k: round(v, 4) for k, v in fit_weights(ev, lam_all).items()}}
    return rep


def render(rep):
    L = [f"OOF {rep['oof_range'][0]}..{rep['oof_range'][1]}: n={rep['n_oof']} "
         f"(clean window after {rep['clean_window_start_after']}: n={rep['n_clean']})", ""]
    for view in ("oof", "clean"):
        L.append(f"[{view}]  model        acc   [Wilson95]       Brier   logloss  ECE(fav)")
        for m, b in rep[view].items():
            L.append(f"  {m:12s} {b['accuracy']:.3f} [{b['accuracy_wilson95'][0]:.3f},{b['accuracy_wilson95'][1]:.3f}]  "
                     f"{b['brier']:.4f}  {b['log_loss']:.4f}  {b['ece10_fav']:.4f}")
        L.append("")
    for key in ("paired_oof", "paired_clean"):
        for name, d in rep[key].items():
            ll = d["log_loss"]
            L.append(f"{key:13s} {name:22s} logloss {ll['diff']:+.4f} [{ll['ci95'][0]:+.4f},{ll['ci95'][1]:+.4f}] {ll['verdict']}")
    L += ["", "Favourite-view calibration (OOF), stated -> actual [Wilson95], n:"]
    for m in ("wf_fit", "elo_platt"):
        L.append(f"  {m}")
        for b in rep["oof"][m]["fav_buckets"]:
            if b["n"]:
                L.append(f"    {b['bucket']:9s} {b['stated']:.3f} -> {b['actual']:.3f} "
                         f"[{b['wilson95'][0]:.3f},{b['wilson95'][1]:.3f}]  n={b['n']:4d}  gap {b['gap_pp']:+5.1f}pp  "
                         f"{'PASS' if b['pass'] else 'FAIL'}{'' if b['scored'] else ' (n<30, not scored)'}")
    L += ["", "Acceptance: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in rep["acceptance"].items()),
          f"Proposed live weights (all eval rows, lambda {rep['proposed_live_weights']['lambda']}): "
          f"{rep['proposed_live_weights']['weights']}"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="walk-forward weight fit")
    ap.add_argument("--min-train", type=int, default=400)
    ap.add_argument("--block-days", type=int, default=91)
    a = ap.parse_args(argv)
    rep = run(a.min_train, a.block_days)
    print()
    print(render(rep))
    with open(os.path.join(HERE, "data", "walkforward_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print("wrote data/walkforward_report.json")


if __name__ == "__main__":
    main()
