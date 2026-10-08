#!/usr/bin/env python3
"""
mismatch_check.py -- are big favourites under-predicted, and what fixes it?

PRE-REGISTERED (written 2026-10-08 before any candidate was scored)
-----------------------------------------------------------------
Question: the owner finds the engine timid in mismatches (Spirit vs Kaleido
Gaming ~86% Bo5). Three suspected causes, each a candidate knob:
  (a) the base-strength signal is capped at signal_cap = 2 units, i.e. at a
      200-Elo gap (rating gap 0.10): every gap beyond 200 Elo gets the same
      ~1.59 log-odds. Knob: CONFIG["base_cap"] in {2 (shipped), 3, 4, 8},
      with one temperature T (scales the whole log-odds) fitted with it.
  (b) newcomers to S/A start at Elo 1500 (= an average S/A team) and move
      slowly. Knobs: backtest.NEWCOMER_OFFSET in {0, 100, 200} and a
      provisional K (x2 for a team's first 10 series) on or off.
  (c) the uncertainty shrink k pulls p toward 50%. Knob: CONFIG["vol_sd"]
      in {1.0 (shipped), 0.5, 0.0} and CONFIG["weight_cv"] in {0.25, 0}.

Rows, folds: identical to walkforward.py -- BO3 series where both teams have
>= 5 prior series; expanding window; 91-day test blocks starting at eval row
400. Rows are matched by series id across feature builds.

PRIMARY MODEL ("selected"): for each fold, every grid config gets its own T
fitted on that fold's training rows (golden section on log loss); the
config with the lowest TRAINING log loss is used for the block. Nothing from
the block is seen. Its out-of-fold (OOF) predictions are compared with:
  shipped     predictor.py as shipped (T = 1)
  shipped_T   shipped config, T fitted per fold (what backtest.py does)
  elo_only    point-in-time Elo expectation

ACCEPTANCE (all must hold to ship the config chosen on all rows):
  B1  OOF log loss of selected < shipped (point estimate; the paired
      bootstrap 95% CI is reported and an improvement is only CLAIMED as
      significant when it excludes 0).
  B2  favourite-view buckets 50-60 .. 90-100 with n >= 30: |actual - stated|
      <= 10pp and the Wilson 95% interval contains the stated mean.
  B3  ECE (favourite view) <= shipped + 0.005.
  B4  no overshoot at the top: for all calls with p_fav >= 0.80 pooled, the
      Wilson 95% interval contains the stated mean.
Also reported, not part of the test: newcomer rows (BO3, a team with 0-4
prior series) scored OOF the same way, Elo-gap buckets, each fixed config's
walk-forward OOF log loss (with T fitted per fold).

USAGE (from D:/Face-Off):  python scripts/mismatch_check.py
Stdlib only, deterministic.
"""
import copy
import itertools
import json
import os
import sys
from datetime import timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402
import walkforward as wf  # noqa: E402

CAPS = [2.0, 3.0, 4.0, 8.0]
SHRINK = {"ship": (1.0, 0.25), "vol0.5": (0.5, 0.25), "vol0": (0.0, 0.25), "none": (0.0, 0.0)}
ELO = {"init": (0.0, 0, 1.0), "off100": (100.0, 0, 1.0), "off200": (200.0, 0, 1.0),
       "provK": (0.0, 10, 2.0), "off100+provK": (100.0, 10, 2.0)}
SHIPPED = ("init", 2.0, "ship")
CFG0 = copy.deepcopy(pr.CONFIG)
OUT = os.path.join(ROOT, "data", "mismatch_report.json")


def build(elo_key):
    off, n, mult = ELO[elo_key]
    saved = (bt.NEWCOMER_OFFSET, bt.PROVISIONAL_N, bt.PROVISIONAL_K_MULT)
    bt.NEWCOMER_OFFSET, bt.PROVISIONAL_N, bt.PROVISIONAL_K_MULT = off, n, mult
    try:
        return bt.build_dataset(bt.load_matches())
    finally:
        bt.NEWCOMER_OFFSET, bt.PROVISIONAL_N, bt.PROVISIONAL_K_MULT = saved


def logits(rows, cap, shrink):
    vol, wcv = SHRINK[shrink]
    cfg = copy.deepcopy(CFG0)
    cfg.update({"base_cap": cap, "vol_sd": vol, "weight_cv": wcv, "temperature": 1.0})
    return wf.with_config(cfg, lambda: [pr.total_logodds(dict(r["input"])) for r in rows])


def lab(r):
    return 1 if r["match"]["winner"] == r["match"]["team_a"] else 0


def main():
    builds = {k: build(k) for k in ELO}
    base_rows = builds["init"]
    ev, _ = bt.select_eval(base_rows, bt.MIN_HISTORY)
    ids = [r["match"]["id"] for r in ev]
    newc = [r for r in base_rows if r["match"].get("best_of") == 3
            and min(r["meta"]["hist_a"], r["meta"]["hist_b"]) < bt.MIN_HISTORY]
    nids = [r["match"]["id"] for r in newc]
    by_id = {k: {r["match"]["id"]: r for r in rows} for k, rows in builds.items()}
    configs = list(itertools.product(ELO, CAPS, SHRINK))
    L, LN = {}, {}
    for c in configs:
        rows = [by_id[c[0]][i] for i in ids]
        L[c] = logits(rows, c[1], c[2])
        LN[c] = logits([by_id[c[0]][i] for i in nids], c[1], c[2])
    y = [lab(r) for r in ev]
    yn = [lab(r) for r in newc]
    dates = [r["match"]["date"] for r in ev]
    ndates = [r["match"]["date"] for r in newc]
    folds = wf.blocks(ev, 400, 91)
    oof = {"selected": [], "shipped": [], "shipped_T": [], "elo_only": []}
    oofn = {"selected": [], "shipped": [], "shipped_T": [], "elo_only": []}
    fixed = {c: [] for c in configs}
    idx_oof, idx_n, fold_log = [], [], []
    for start, end, tr, te in folds:
        s, e = start.isoformat(), end.isoformat()
        itr = [i for i, d in enumerate(dates) if d < s]
        ite = [i for i, d in enumerate(dates) if s <= d < e]
        inn = [i for i, d in enumerate(ndates) if s <= d < e]
        ytr = [y[i] for i in itr]
        fits = {}
        for c in configs:
            T = bt.fit_temperature([L[c][i] for i in itr], ytr)
            fits[c] = (bt.log_loss([bt.sigmoid(T * L[c][i]) for i in itr], ytr), T)
            fixed[c] += [bt.sigmoid(T * L[c][i]) for i in ite]
        best = min(configs, key=lambda c: fits[c][0])
        Tb = fits[best][1]
        oof["selected"] += [bt.sigmoid(Tb * L[best][i]) for i in ite]
        oofn["selected"] += [bt.sigmoid(Tb * LN[best][i]) for i in inn]
        Ts = fits[SHIPPED][1]
        oof["shipped_T"] += [bt.sigmoid(Ts * L[SHIPPED][i]) for i in ite]
        oofn["shipped_T"] += [bt.sigmoid(Ts * LN[SHIPPED][i]) for i in inn]
        oof["shipped"] += [bt.sigmoid(L[SHIPPED][i]) for i in ite]
        oofn["shipped"] += [bt.sigmoid(LN[SHIPPED][i]) for i in inn]
        oof["elo_only"] += [ev[i]["meta"]["p_elo"] for i in ite]
        oofn["elo_only"] += [newc[i]["meta"]["p_elo"] for i in inn]
        idx_oof += ite
        idx_n += inn
        fold_log.append({"block": s, "n_train": len(itr), "n_test": len(ite), "n_newcomer": len(inn),
                         "selected": list(best), "T": round(Tb, 4), "train_ll": round(fits[best][0], 5),
                         "shipped_T": round(Ts, 4)})
        print(f"{s} train {len(itr):4d} test {len(ite):3d} newc {len(inn):3d}  selected {best} T={Tb:.3f}"
              f"  (shipped T={Ts:.3f})")
    yo = [y[i] for i in idx_oof]
    yno = [yn[i] for i in idx_n]
    rep = {"n_oof": len(yo), "n_newcomer_oof": len(yno), "folds": fold_log,
           "oof": {m: wf.metrics(ps, yo) for m, ps in oof.items()},
           "paired": {f"{m} - shipped": wf.paired(oof[m], oof["shipped"], yo)
                      for m in ("selected", "shipped_T", "elo_only")},
           "newcomer_oof": {m: wf.metrics(ps, yno) for m, ps in oofn.items()} if yno else {},
           "newcomer_paired": {"selected - shipped": wf.paired(oofn["selected"], oofn["shipped"], yno)} if yno else {},
           "fixed_configs_oof_log_loss": sorted(
               ([list(c), bt.log_loss(ps, yo)] for c, ps in fixed.items()), key=lambda t: t[1])}
    # top-bucket and Elo-gap tables
    def top(ps, ys, lo=0.8):
        b = [(max(p, 1 - p), y if p >= 0.5 else 1 - y) for p, y in zip(ps, ys) if max(p, 1 - p) >= lo]
        if not b:
            return {"n": 0}
        k = sum(t[1] for t in b)
        st = sum(t[0] for t in b) / len(b)
        w = bt.wilson(k, len(b))
        return {"n": len(b), "stated": st, "actual": k / len(b), "wilson95": list(w),
                "pass": w[0] <= st <= w[1]}
    rep["top80"] = {m: top(ps, yo) for m, ps in oof.items()}
    rep["top75"] = {m: top(ps, yo, 0.75) for m, ps in oof.items()}
    gaps = [abs(ev[i]["meta"]["elo_a"] - ev[i]["meta"]["elo_b"]) for i in idx_oof]
    fav_a = [ev[i]["meta"]["elo_a"] >= ev[i]["meta"]["elo_b"] for i in idx_oof]
    gt = []
    for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 2000)):
        ii = [j for j, g in enumerate(gaps) if lo <= g < hi]
        if not ii:
            continue
        yy = [yo[j] if fav_a[j] else 1 - yo[j] for j in ii]
        row = {"gap": f"{lo}-{hi}", "n": len(ii), "actual": sum(yy) / len(ii),
               "wilson95": list(bt.wilson(sum(yy), len(ii)))}
        for m, ps in oof.items():
            row[m] = sum(ps[j] if fav_a[j] else 1 - ps[j] for j in ii) / len(ii)
        gt.append(row)
    rep["elo_gap_table"] = gt
    o = rep["oof"]
    rep["acceptance"] = {
        "B1_ll_lower": o["selected"]["log_loss"] < o["shipped"]["log_loss"],
        "B2_buckets": all(b.get("pass", True) for b in o["selected"]["fav_buckets"]),
        "B3_ece": o["selected"]["ece10_fav"] <= o["shipped"]["ece10_fav"] + 0.005,
        "B4_top80": rep["top80"]["selected"].get("pass", True),
    }
    rep["acceptance"]["ALL"] = all(rep["acceptance"].values())
    # live choice: the config + T with the lowest log loss on ALL eval rows
    allfit = {}
    for c in configs:
        T = bt.fit_temperature(L[c], y)
        allfit[c] = (bt.log_loss([bt.sigmoid(T * x) for x in L[c]], y), T)
    bc = min(configs, key=lambda c: allfit[c][0])
    rep["live_choice"] = {"config": list(bc), "T": allfit[bc][1], "log_loss_all": allfit[bc][0],
                          "shipped_T_all": allfit[SHIPPED][1], "shipped_ll_all_T1": bt.log_loss(
                              [bt.sigmoid(x) for x in L[SHIPPED]], y)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    render(rep)


def render(rep):
    print(f"\nOOF n={rep['n_oof']}  model         acc    Brier   logloss  ECEfav")
    for m, b in rep["oof"].items():
        print(f"  {m:12s} {b['accuracy']:.3f}  {b['brier']:.4f}  {b['log_loss']:.4f}  {b['ece10_fav']:.4f}")
    for k, d in rep["paired"].items():
        ll, br = d["log_loss"], d["brier"]
        print(f"  {k:22s} logloss {ll['diff']:+.4f} [{ll['ci95'][0]:+.4f},{ll['ci95'][1]:+.4f}] {ll['verdict']};"
              f" Brier {br['diff']:+.4f} [{br['ci95'][0]:+.4f},{br['ci95'][1]:+.4f}]")
    for m in ("shipped", "shipped_T", "selected", "elo_only"):
        print(f"  favourite buckets: {m}")
        for b in rep["oof"][m]["fav_buckets"]:
            if b["n"]:
                print(f"    {b['bucket']:9s} n={b['n']:4d} stated {b['stated']:.3f} actual {b['actual']:.3f} "
                      f"[{b['wilson95'][0]:.3f},{b['wilson95'][1]:.3f}] {'PASS' if b['pass'] else 'FAIL'}")
        t = rep["top80"][m]
        if t["n"]:
            print(f"    >=80% pooled n={t['n']} stated {t['stated']:.3f} actual {t['actual']:.3f} "
                  f"[{t['wilson95'][0]:.3f},{t['wilson95'][1]:.3f}]")
    print("  Elo-gap buckets (favourite = higher Elo): stated by model vs actual")
    for r in rep["elo_gap_table"]:
        print(f"    {r['gap']:9s} n={r['n']:4d} actual {r['actual']:.3f} [{r['wilson95'][0]:.3f},{r['wilson95'][1]:.3f}]"
              f"  shipped {r['shipped']:.3f}  shipped_T {r['shipped_T']:.3f}  selected {r['selected']:.3f}  elo {r['elo_only']:.3f}")
    if rep["newcomer_oof"]:
        print(f"  newcomer rows (a team with 0-4 prior series), OOF n={rep['n_newcomer_oof']}:")
        for m, b in rep["newcomer_oof"].items():
            print(f"    {m:12s} acc {b['accuracy']:.3f} Brier {b['brier']:.4f} logloss {b['log_loss']:.4f}")
        d = rep["newcomer_paired"]["selected - shipped"]["log_loss"]
        print(f"    selected - shipped logloss {d['diff']:+.4f} [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] {d['verdict']}")
    print("  best fixed configs (walk-forward T), OOF log loss:")
    for c, v in rep["fixed_configs_oof_log_loss"][:8]:
        print(f"    {c} {v:.4f}")
    sh = [v for c, v in rep["fixed_configs_oof_log_loss"] if c == list(SHIPPED)]
    print(f"    shipped config with walk-forward T: {sh[0]:.4f}")
    print("Acceptance: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in rep["acceptance"].items()))
    print(f"Live choice (all rows): {rep['live_choice']}")


if __name__ == "__main__":
    main()
