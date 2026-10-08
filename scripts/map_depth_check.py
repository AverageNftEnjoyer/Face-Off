#!/usr/bin/env python3
"""
map_depth_check.py -- does map-pool depth add held-out signal beyond strength?

PRE-REGISTERED (2026-10-08, before any depth number was computed)
-----------------------------------------------------------------
All map features are point-in-time: per-map results of the 90 days before
the match day (backtest.History), the active pool of that day.

  depth(team)    PRIMARY. Number of active-pool maps the team played >= 3
                 times in 90 days with a plain win rate >= 50%.
  depth_resid    same count, but the per-map rate is the Elo-residual rate
                 (0.5 + mean(won - Elo-expected)), so strength is not
                 counted twice.
  breadth        number of active-pool maps played >= 3 times (any rate).
  veto_comfort   on the simulated BO3 veto (picks + decider, predictor's
                 simulate_veto), how many of the three maps the team played
                 >= 3 times with a plain win rate >= 50%.
  map_veto       the engine's existing simulated-veto signal, capped
                 logit(veto-only series P(A)).
Signal = team_a value - team_b value (map_veto is already A's view).

Rows and folds: as walkforward.py (BO3, both teams >= 5 prior series, 91-day
blocks from eval row 400, expanding window). L = the shipped engine's
pre-temperature log-odds (total_logodds). Per fold, fitted on the rows
before the block, unpenalised logistic regression without intercept:
    base     logit p = a L
    +signal  logit p = a L + b S
SHIP RULE (primary only): the +depth model beats base on OOF log loss AND the
paired-bootstrap 95% CI of the difference lies entirely below 0. Then depth
enters the engine as a factor with its weight fitted on all rows. The four
other signals are reported with their own CIs; with five looks, one passing
by chance is likely, so none of them ships on this test alone.
Also reported: the sign and size of b per fold, favourite-view calibration.

USAGE (from D:/Face-Off):  python scripts/map_depth_check.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import fit_weights as fw  # noqa: E402
import predictor as pr  # noqa: E402
import walkforward as wf  # noqa: E402

MIN_N = 3
MIN_WR = 0.5
SIGNALS = ["depth", "depth_resid", "breadth", "veto_comfort", "map_veto"]
OUT = os.path.join(ROOT, "data", "map_depth_report.json")


def count(maps, pool, wr=MIN_WR, n=MIN_N):
    return sum(1 for mp in pool if mp in maps and maps[mp][1] >= n and maps[mp][0] >= wr)


def signals(inp, raw_a, raw_b):
    pool = inp["map_pool"]
    veto = pr.simulate_veto(dict(inp), 3)
    vm = veto["maps"]
    return {
        "depth": count(raw_a, pool) - count(raw_b, pool),
        "depth_resid": count(inp["maps_a"], pool) - count(inp["maps_b"], pool),
        "breadth": count(raw_a, pool, wr=0.0) - count(raw_b, pool, wr=0.0),
        "veto_comfort": count(raw_a, vm) - count(raw_b, vm),
        "map_veto": pr._cap(pr.logit(veto["p_series_a"])),
    }


def lab(r):
    return 1 if r["match"]["winner"] == r["match"]["team_a"] else 0


def main():
    rows = bt.build_dataset(bt.load_matches())
    saved = bt.MAP_RATES
    bt.MAP_RATES = "raw"
    try:
        raw = {r["match"]["id"]: r["input"] for r in bt.build_dataset(bt.load_matches())}
    finally:
        bt.MAP_RATES = saved
    ev, _ = bt.select_eval(rows, bt.MIN_HISTORY)
    L = [pr.total_logodds(dict(r["input"])) for r in ev]
    S = [signals(r["input"], raw[r["match"]["id"]]["maps_a"], raw[r["match"]["id"]]["maps_b"]) for r in ev]
    y = [lab(r) for r in ev]
    dates = [r["match"]["date"] for r in ev]
    oof = {"base": []}
    oof.update({s: [] for s in SIGNALS})
    coefs = {s: [] for s in SIGNALS}
    yo = []
    for start, end, _, _ in wf.blocks(ev, 400, 91):
        st, en = start.isoformat(), end.isoformat()
        tr = [i for i, d in enumerate(dates) if d < st]
        te = [i for i, d in enumerate(dates) if st <= d < en]
        ytr = [y[i] for i in tr]
        a = fw.fit_logistic([[L[i]] for i in tr], [0.0] * len(tr), ytr, 0.0, prior=[1.0])
        oof["base"] += [bt.sigmoid(a[0] * L[i]) for i in te]
        for s in SIGNALS:
            c = fw.fit_logistic([[L[i], S[i][s]] for i in tr], [0.0] * len(tr), ytr, 0.0, prior=[1.0, 0.0])
            coefs[s].append(round(c[1], 4))
            oof[s] += [bt.sigmoid(c[0] * L[i] + c[1] * S[i][s]) for i in te]
        yo += [y[i] for i in te]
    rep = {"n_oof": len(yo), "coef_by_fold": coefs,
           "metrics": {m: wf.metrics(ps, yo) for m, ps in oof.items()},
           "paired_vs_base": {s: wf.paired(oof[s], oof["base"], yo) for s in SIGNALS},
           "signal_spread": {s: sorted({S[i][s] for i in range(len(S))})[:3] + ["..."] for s in SIGNALS[:4]}}
    # all-rows fit (information; the live weight if the primary passes)
    rep["all_rows_coef"] = {s: fw.fit_logistic([[L[i], S[i][s]] for i in range(len(L))], [0.0] * len(L), y, 0.0,
                                               prior=[1.0, 0.0]) for s in SIGNALS}
    # how often does depth disagree with strength, and who wins then
    dis = [i for i in range(len(L)) if S[i]["depth"] * L[i] < 0 and abs(S[i]["depth"]) >= 2]
    rep["depth_disagrees_with_strength"] = {
        "n": len(dis),
        "deeper_team_won": sum(1 for i in dis if (S[i]["depth"] > 0) == (y[i] == 1)) / len(dis) if dis else None,
        "engine_said_deeper_team": sum(bt.sigmoid(L[i]) if S[i]["depth"] > 0 else 1 - bt.sigmoid(L[i]) for i in dis) / len(dis) if dis else None}
    p = rep["paired_vs_base"]["depth"]["log_loss"]
    rep["ship_primary"] = p["diff"] < 0 and p["ci95"][1] < 0
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(f"OOF n={len(yo)}")
    for m, b in rep["metrics"].items():
        print(f"  {m:13s} acc {b['accuracy']:.3f} Brier {b['brier']:.4f} logloss {b['log_loss']:.4f} ECEfav {b['ece10_fav']:.4f}")
    for s, d in rep["paired_vs_base"].items():
        ll = d["log_loss"]
        print(f"  {s:13s} - base: logloss {ll['diff']:+.4f} [{ll['ci95'][0]:+.4f},{ll['ci95'][1]:+.4f}] {ll['verdict']};"
              f" coef by fold {coefs[s]}; all rows {[round(x, 4) for x in rep['all_rows_coef'][s]]}")
    d = rep["depth_disagrees_with_strength"]
    if d["n"]:
        print(f"  depth gap >= 2 against the strength lean: n={d['n']}, deeper team won {d['deeper_team_won']:.3f}, "
              f"engine gave the deeper team {d['engine_said_deeper_team']:.3f}")
    print("SHIP depth factor:", "YES" if rep["ship_primary"] else "NO")


if __name__ == "__main__":
    main()
