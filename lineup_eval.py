#!/usr/bin/env python3
"""
lineup_eval.py -- do per-match lineups add held-out signal beyond Elo?

PRE-REGISTERED TEST (written before any lineup result was seen)
  Rows, blocks and folds: identical to walkforward.py (BO3, >= 5 prior series
  per team, expanding window, 91-day blocks from eval row 400).
  Features: lineup_features.py, strict cutoff (lineups dated < D only).
  Models, all logistic without intercept, fitted per fold on rows before the
  block, penalty lam toward the prior chosen on the last 20% of the fold's
  training rows from LAMBDAS:
    elo_only   logit p = L                       (no fitting)
    elo_platt  logit p = a L                     (prior a=1)
    lineup     logit p = a L + b (S_a - S_b) + c L ((1-C_a) + (1-C_b))
               (prior a=1, b=0, c=0)  <- PRIMARY
  L = logit(point-in-time Elo expectation), S = stand-in flag, C = continuity.
  PASS: lineup beats elo_only on OOF log loss AND the paired-bootstrap 95% CI
  of (lineup - elo_only) lies entirely below 0. Otherwise: do not ship.
  Also reported (information only, NOT the test):
    lineup vs elo_platt; and an ORACLE variant that also sees day D's own
    lineups (violates the cutoff rule; an upper bound on what pre-match
    lineup announcements could add).

USAGE (from D:/Face-Off):  python lineup_eval.py
"""
import json
import os
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backtest as bt  # noqa: E402
import fit_weights as fw  # noqa: E402
import lineup_features as LF  # noqa: E402
import walkforward as wf  # noqa: E402

LAMBDAS = [0.0, 1.0, 10.0, 100.0]
MODELS = {"elo_platt": [1.0], "lineup": [1.0, 0.0, 0.0]}


def design(rows, hist, oracle=False):
    X = {"elo_platt": [], "lineup": []}
    for r in rows:
        m = r["match"]
        day = m["date"]
        if oracle:   # include day D itself (cutoff violated on purpose)
            day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        fa = LF.features(hist.get(m["team_a"], []), day)
        fb = LF.features(hist.get(m["team_b"], []), day)
        L = bt.logit(r["meta"]["p_elo"])
        X["elo_platt"].append([L])
        X["lineup"].append([L, fa["standin"] - fb["standin"],
                            L * ((1 - fa["continuity"]) + (1 - fb["continuity"]))])
    return X


def fit(X, y, lam, prior):
    return fw.fit_logistic(X, [0.0] * len(X), y, lam, prior=list(prior))


def predict(X, c):
    return [bt.sigmoid(sum(ci * xi for ci, xi in zip(c, x))) for x in X]


def run(oracle=False):
    matches = bt.load_matches()
    hist = LF.team_history(matches, LF.load())
    rows = bt.build_dataset(matches)
    ev, _ = bt.select_eval(rows, bt.MIN_HISTORY)
    folds = wf.blocks(ev, 400, 91)
    oof = {"elo_only": [], "elo_platt": [], "lineup": []}
    ys, coef_log, cover = [], [], {"rows": 0, "both_known": 0, "any_standin": 0}
    for start, end, tr, te in folds:
        fit_part, val_part = bt.chrono_split(tr, 0.8)
        Xf, Xv, Xtr, Xte = (design(p, hist, oracle) for p in (fit_part, val_part, tr, te))
        yf, yv, ytr, yte = ([wf.label(r) for r in p] for p in (fit_part, val_part, tr, te))
        entry = {"block": start.isoformat()}
        for name, prior in MODELS.items():
            lam = min(LAMBDAS, key=lambda l: bt.log_loss(predict(Xv[name], fit(Xf[name], yf, l, prior)), yv))
            c = fit(Xtr[name], ytr, lam, prior)
            oof[name] += predict(Xte[name], c)
            entry[name] = {"lambda": lam, "coef": [round(v, 4) for v in c]}
        oof["elo_only"] += [r["meta"]["p_elo"] for r in te]
        ys += yte
        coef_log.append(entry)
        for r in te:
            d = r["match"]["date"]
            if oracle:
                d = (date.fromisoformat(d) + timedelta(days=1)).isoformat()
            fa = LF.features(hist.get(r["match"]["team_a"], []), d)
            fb = LF.features(hist.get(r["match"]["team_b"], []), d)
            cover["rows"] += 1
            cover["both_known"] += fa["known"] and fb["known"]
            cover["any_standin"] += bool(fa["standin"] or fb["standin"])
    res = {name: wf.metrics(ps, ys) for name, ps in oof.items()}
    paired = {"lineup - elo_only": wf.paired(oof["lineup"], oof["elo_only"], ys),
              "lineup - elo_platt": wf.paired(oof["lineup"], oof["elo_platt"], ys)}
    return {"n": len(ys), "coverage": cover, "metrics": res, "paired": paired, "folds": coef_log}


def show(title, rep):
    c = rep["coverage"]
    print(f"\n== {title}  (OOF n={rep['n']}; both lineups known {c['both_known']}, "
          f"rows with a stand-in flag {c['any_standin']})")
    print("  model       acc    Brier   log loss")
    for name, b in rep["metrics"].items():
        print(f"  {name:10s} {b['accuracy']:.3f}  {b['brier']:.4f}  {b['log_loss']:.4f}")
    for name, d in rep["paired"].items():
        ll = d["log_loss"]
        print(f"  {name:20s} log loss {ll['diff']:+.4f} [{ll['ci95'][0]:+.4f}, {ll['ci95'][1]:+.4f}]  {ll['verdict']}")
    print("  last fold coefficients:", {k: v for k, v in rep["folds"][-1].items() if k != "block"})


def main():
    strict = run(oracle=False)
    oracle = run(oracle=True)
    show("STRICT (lineups dated < D) -- the acceptance test", strict)
    show("ORACLE (also day D's own lineups; violates the cutoff, information only)", oracle)
    d = strict["paired"]["lineup - elo_only"]["log_loss"]
    verdict = "PASS" if d["ci95"][1] < 0 else "FAIL"
    print(f"\nACCEPTANCE (lineup - elo_only, CI must lie below 0): {verdict}")
    with open(os.path.join(HERE, "data", "lineup_eval_report.json"), "w", encoding="utf-8") as f:
        json.dump({"strict": strict, "oracle": oracle, "verdict": verdict}, f, indent=1)
    print("wrote data/lineup_eval_report.json")


if __name__ == "__main__":
    main()
