#!/usr/bin/env python3
"""
fit_weights.py -- fit the engine's factor weights on the TRAIN split.

Model. For each series the engine produces per-factor log-odds contributions
d_i (already capped, sample-shrunk and residualised) and an uncertainty
shrink k. We fit one multiplier c_i per factor:

    P(team_a wins) = sigmoid( sum_i c_i * d_i / k )

by penalised maximum likelihood (log loss), with an L2 penalty pulling each
c_i toward 1, i.e. toward the current reasoned weights:

    minimise  sum_n logloss(y_n, p_n)  +  (lam / 2) * sum_i (c_i - 1)^2

There is no intercept: the engine is antisymmetric in (team_a, team_b), so a
fitted intercept would only learn which side the data source lists first.

Because k depends on the weights (weight-uncertainty and rating-noise
terms), the fit is repeated: apply w_i <- c_i * w_i with temperature 1,
recompute d_i and k, refit, until every c_i is within 1% of 1. The fitted
scale absorbs the temperature, so temperature is set to 1.0 afterwards.

Factors: base_strength, form_30d, form_last5, head_to_head, map_veto,
roster. Stakes is NOT fitted: every series in the data has stakes "none", so
it carries no information (its weight is left as is).

Roster: c_roster scales the log of all three penalties together
(pen -> pen ** c_roster), so the ratios between stand-in, missing-IGL and
both stay as reasoned; there are too few events (5.8% of sides, 42 IGL
cases) to fit three separate values.

Splits (chronological, by match day, same as backtest.py):
  * test  = last 40% -- never touched here except in --evaluate
  * train = first 60%, itself split 80/20 into fit / validation to pick lam
    from LAMBDAS by validation log loss; then refit on all of train.

USAGE (from D:/Face-Off):
    python fit_weights.py              # fit, print proposed CONFIG, write data/fit_report.json
    python fit_weights.py --evaluate   # also score current vs fitted on the TEST split
"""

import argparse
import copy
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402

FACTORS = ["base_strength", "form_30d", "form_last5", "head_to_head", "map_veto"]
# Roster is NOT fitted: with 95 flagged series in train (49 in test) its
# multiplier is unidentifiable (single-factor fits: -1.4 on train, +9.3 on
# test). The reasoned penalties stay as fixed priors and enter as an offset;
# see bootstrap_roster() for the evidence report.
WEIGHT_KEY = {"base_strength": "w_base", "form_30d": "w_form30", "form_last5": "w_form5",
              "head_to_head": "w_h2h", "map_veto": "w_veto"}
ROSTER_KEYS = ("standin_penalty", "no_igl_penalty", "standin_igl_penalty")
LAMBDAS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
MAX_ROUNDS = 8


# ------------------------------------------------------------------ data
def label(row):
    m = row["match"]
    return 1 if m["winner"] == m["team_a"] else 0


def design(rows):
    """X[n][i] = d_i / k for the current CONFIG; also returns the rest of the
    log-odds (unfitted factors) as an offset."""
    X, off = [], []
    for r in rows:
        c = pr._compute(dict(r["input"]))
        d = {f[0]: f[1] for f in c["factors"]}
        k = c["shrink_k"]
        X.append([d.get(f, 0.0) / k for f in FACTORS])
        off.append(sum(v for name, v in d.items() if name not in FACTORS) / k)
    return X, off


# ------------------------------------------------------------------ fitting
def solve(A, b):
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            raise ValueError("singular system")
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for cc in range(col, n + 1):
                M[r][cc] -= f * M[col][cc]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][cc] * x[cc] for cc in range(r + 1, n))) / M[r][r]
    return x


def fit_nonneg(X, off, y, lam):
    """fit_logistic with every multiplier constrained to >= 0 (active set): a
    factor whose best fit is negative is anti-predictive on this data and is
    switched off (multiplier 0) rather than allowed to flip sign."""
    p_n = len(X[0])
    active = list(range(p_n))
    while True:
        c_act = fit_logistic([[x[i] for i in active] for x in X], off, y, lam)
        c = [0.0] * p_n
        for i, v in zip(active, c_act):
            c[i] = v
        neg = [i for i in active if c[i] < 0]
        if not neg:
            return c
        active.remove(min(neg, key=lambda i: c[i]))
        if not active:
            return c


def fit_logistic(X, off, y, lam, prior=None, iters=50):
    """Newton-Raphson for penalised logistic regression without intercept."""
    p_n = len(X[0])
    prior = prior or [1.0] * p_n
    c = prior[:]
    for _ in range(iters):
        g = [lam * (c[i] - prior[i]) for i in range(p_n)]
        H = [[lam if i == j else 0.0 for j in range(p_n)] for i in range(p_n)]
        for x, o, t in zip(X, off, y):
            z = o + sum(ci * xi for ci, xi in zip(c, x))
            p = bt.sigmoid(z)
            w = p * (1 - p)
            for i in range(p_n):
                g[i] += (p - t) * x[i]
                if x[i] == 0.0:
                    continue
                for j in range(p_n):
                    H[i][j] += w * x[i] * x[j]
        # columns with no signal at all (e.g. no stand-ins in a split) keep their prior
        for i in range(p_n):
            if all(x[i] == 0.0 for x in X) and lam == 0.0:
                H[i][i] = 1.0
                g[i] = 0.0
        step = solve(H, g)
        c = [ci - si for ci, si in zip(c, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    return c


def logloss(X, off, y, c):
    tot = 0.0
    for x, o, t in zip(X, off, y):
        p = min(1 - 1e-12, max(1e-12, bt.sigmoid(o + sum(ci * xi for ci, xi in zip(c, x)))))
        tot -= t * math.log(p) + (1 - t) * math.log(1 - p)
    return tot / len(y)


# ------------------------------------------------------------------ config application
def apply(cfg, c):
    """Return a CONFIG with w_i *= c_i, roster penalties ** c_roster, temperature 1."""
    new = copy.deepcopy(cfg)
    for name, ci in zip(FACTORS, c):
        if name in WEIGHT_KEY:
            new[WEIGHT_KEY[name]] = cfg[WEIGHT_KEY[name]] * ci
        elif name == "roster":
            for key in ROSTER_KEYS:
                new[key] = cfg[key] ** ci
    new["temperature"] = 1.0
    return new


def fit_config(train, lam, cfg0, verbose=False):
    """Fixed-point fit: refit multipliers on the current engine until they are ~1."""
    cfg = copy.deepcopy(cfg0)
    cfg["temperature"] = 1.0
    y = [label(r) for r in train]
    for rnd in range(MAX_ROUNDS):
        pr.CONFIG.clear()
        pr.CONFIG.update(cfg)
        X, off = design(train)
        c = fit_nonneg(X, off, y, lam)
        if verbose:
            print(f"  round {rnd + 1}: multipliers " + ", ".join(f"{n} {v:.3f}" for n, v in zip(FACTORS, c)))
        cfg = apply(cfg, c)
        if max(abs(v - 1) for v in c if v != 0.0) < 0.01 if any(c) else True:
            break
    pr.CONFIG.clear()
    pr.CONFIG.update(cfg0)
    return cfg


def bootstrap_roster(train, cfg, lam, n_boot=200, seed=7):
    """Bootstrap the roster multiplier with every other weight fixed at the fit."""
    pr.CONFIG.clear()
    pr.CONFIG.update(cfg)
    X, off = design(train)
    y = [label(r) for r in train]
    pr.CONFIG.clear()
    pr.CONFIG.update(CFG0)
    # roster contribution per series (in the offset), re-derived for the report
    pr.CONFIG.clear(); pr.CONFIG.update(cfg)
    rcol = []
    for r in train:
        c = pr._compute(dict(r["input"]))
        rcol.append(next(f[1] for f in c["factors"] if f[0] == "roster") / c["shrink_k"])
    pr.CONFIG.clear(); pr.CONFIG.update(CFG0)
    X = [x + [rc] for x, rc in zip(X, rcol)]
    off = [o - rc for o, rc in zip(off, rcol)]
    ri = len(X[0]) - 1
    idx = [n for n, x in enumerate(X) if x[ri] != 0.0]
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [rng.choice(range(len(X))) for _ in range(len(X))]
        Xs = [[X[n][ri]] for n in samp]
        offs = [off[n] + sum(X[n][i] for i in range(ri)) for n in samp]
        ys = [y[n] for n in samp]
        if not any(x[0] for x in Xs):
            continue
        vals.append(fit_logistic(Xs, offs, ys, lam / len(FACTORS))[0])
    vals.sort()
    if not vals:
        return None
    return {"events_in_train": len(idx), "median": vals[len(vals) // 2],
            "ci95": [vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]]}


# ------------------------------------------------------------------ evaluation
def predict_all(rows, cfg):
    pr.CONFIG.clear()
    pr.CONFIG.update(cfg)
    ps = [pr.predict_match(dict(r["input"]))["p_a_exact"] for r in rows]
    pr.CONFIG.clear()
    pr.CONFIG.update(CFG0)
    return ps


def evaluate(test, cfgs, seed=12345):
    y = [label(r) for r in test]
    out = {}
    preds = {name: predict_all(test, cfg) for name, cfg in cfgs.items()}
    for name, ps in preds.items():
        out[name] = {
            "n": len(y), "brier": bt.brier(ps, y), "log_loss": bt.log_loss(ps, y),
            "accuracy": bt.accuracy(ps, y), "ece10": bt.ece(ps, y),
            "reliability_table": bt.reliability_table(ps, y),
        }
    names = list(preds)
    if len(names) >= 2:
        a, b = names[0], names[1]
        out["paired_" + b + "_minus_" + a] = {
            fn_name: {"diff": fn(preds[b], y) - fn(preds[a], y),
                      "ci95": list(bt.paired_bootstrap_diff(fn, preds[b], preds[a], y, seed=seed))}
            for fn_name, fn in (("brier", bt.brier), ("log_loss", bt.log_loss))
        }
    return out


# ------------------------------------------------------------------ main
CFG0 = copy.deepcopy(pr.CONFIG)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evaluate", action="store_true", help="also score current vs fitted on the TEST split")
    args = ap.parse_args(argv)

    rows = bt.build_dataset(bt.load_matches())
    ev, _ = bt.select_eval(rows, bt.MIN_HISTORY)
    train, test = bt.chrono_split(ev, 0.6)
    fit_part, val_part = bt.chrono_split(train, 0.8)
    print(f"train {len(train)} (fit {len(fit_part)} / validation {len(val_part)}), test {len(test)} held out")

    # 1) choose lambda on the validation slice of TRAIN
    yv = [label(r) for r in val_part]
    scores = []
    for lam in LAMBDAS:
        cfg = fit_config(fit_part, lam, CFG0)
        pr.CONFIG.clear(); pr.CONFIG.update(cfg)
        Xv, offv = design(val_part)
        pr.CONFIG.clear(); pr.CONFIG.update(CFG0)
        ll = logloss(Xv, offv, yv, [1.0] * len(FACTORS))
        scores.append((ll, lam))
        print(f"lambda {lam:>6}: validation log loss {ll:.4f}")
    # baseline on the same validation slice: the current engine as shipped
    ps0 = predict_all(val_part, CFG0)
    ll0 = bt.log_loss(ps0, yv)
    print(f"current engine (T={CFG0['temperature']}): validation log loss {ll0:.4f}")
    best_ll, best_lam = min(scores)

    # 2) refit on all of TRAIN with the chosen lambda
    print(f"refit on all train with lambda {best_lam}:")
    fitted = fit_config(train, best_lam, CFG0, verbose=True)
    roster_ci = bootstrap_roster(train, fitted, best_lam)

    keys = [WEIGHT_KEY[f] for f in FACTORS if f in WEIGHT_KEY] + ["temperature"]
    proposal = {k: round(fitted[k], 4) for k in keys}
    print("proposed CONFIG:")
    for k in keys:
        print(f"  {k:22s} {CFG0[k]:>8.4f} -> {fitted[k]:.4f}")
    if roster_ci:
        print(f"roster multiplier bootstrap (train, {roster_ci['events_in_train']} series with a flag): "
              f"median {roster_ci['median']:.2f}, 95% CI {roster_ci['ci95'][0]:.2f} to {roster_ci['ci95'][1]:.2f}")

    report = {"train": len(train), "validation": len(val_part), "test": len(test),
              "lambda_scores": [{"lambda": l, "val_log_loss": s} for s, l in scores],
              "current_val_log_loss": ll0, "chosen_lambda": best_lam,
              "current": {k: CFG0[k] for k in keys}, "proposed": proposal, "roster_bootstrap": roster_ci}

    if args.evaluate:
        res = evaluate(test, {"current": CFG0, "fitted": fitted})
        report["test"] = res
        for name in ("current", "fitted"):
            r = res[name]
            print(f"TEST {name:8s}: Brier {r['brier']:.4f}  log loss {r['log_loss']:.4f}  "
                  f"acc {r['accuracy']:.3f}  ECE {r['ece10']:.4f}")
        d = res["paired_fitted_minus_current"]
        print(f"TEST fitted - current: Brier {d['brier']['diff']:+.4f} [{d['brier']['ci95'][0]:+.4f}, {d['brier']['ci95'][1]:+.4f}]  "
              f"log loss {d['log_loss']['diff']:+.4f} [{d['log_loss']['ci95'][0]:+.4f}, {d['log_loss']['ci95'][1]:+.4f}]")
        print("TEST calibration buckets (predicted -> actual, n):")
        for a, b in zip(res["current"]["reliability_table"], res["fitted"]["reliability_table"]):
            fmt = lambda r: f"{r['mean_pred']:.3f}->{r['actual']:.3f} ({r['n']})" if r["n"] else "-"
            print(f"  {a['bin']:8s} current {fmt(a):24s} fitted {fmt(b)}")

    with open(os.path.join(HERE, "data", "fit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("wrote data/fit_report.json")


if __name__ == "__main__":
    main()
