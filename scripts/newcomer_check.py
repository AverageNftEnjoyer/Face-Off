#!/usr/bin/env python3
"""
newcomer_check.py -- should teams new to S/A tier start below Elo 1500?

WHY: scripts/mismatch_check.py found no overall gain from sharper base
strength, but its newcomer rows (a team with < 5 prior S/A series) scored
WORSE than a coin flip. Teams that enter the S/A data later mostly come up
from lower tiers, yet start at 1500 = an average S/A team. Kaleido Gaming
(4 series, all at one A-tier Asian event) is one of them.

PRE-REGISTERED (2026-10-08). Disclosure: before this test was written, a
first-half / second-half sweep of the offset on newcomer rows with plain Elo
(no engine, no fitting) was looked at; it suggested 100-200. The grid below
was fixed before the walk-forward run.

  Knob: backtest.NEWCOMER_OFFSET in {0, 50, 100, 150, 200, 250, 300}; a team
  whose first S/A series is more than 90 days after the start of the data
  starts at 1500 - offset. Teams present in the first 90 days start at 1500.
  Engine: predictor.py as shipped (T = 1), only the Elo feature build changes.
  Newcomer rows: BO3 series dated more than 90 days after the start of the
  data where either team has < 5 prior series.
  Folds: the walk-forward block starts of walkforward.py (91-day blocks from
  eval row 400). Per fold, the offset with the lowest engine log loss on the
  NEWCOMER rows dated before the block is used for the block.
  N1  newcomer OOF log loss below offset 0, and the paired-bootstrap 95% CI
      of the difference excludes 0.
  N2  the main eval rows (both teams >= 5 prior series) are not hurt: OOF
      log loss with the chosen offsets <= offset 0 + 0.001.
  N3  newcomer favourite-view buckets with n >= 30 pass the walkforward.py
      rule (within 10pp and the Wilson 95% interval holds the stated mean).
  Ship the offset chosen on ALL newcomer rows only if N1-N3 all hold.

SECOND VARIANT, pre-registered after the first run (same grid, same folds,
same N1-N3 and the same shipping rule), --mode feature: the Elo table is
left alone and only the rating fed to the engine for a late entrant is
lowered, by offset x (5 - prior series) / 5 (backtest.NEWCOMER_MODE). Reason:
in the first run the start offset also moved established teams' Elo, and
the all-rows log loss of the graded rows (both teams >= 5 series) rose from
0.6262 (offset 0) to 0.6279 (150) / 0.6305 (200). In feature mode those
rows are unchanged by construction.

USAGE (from D:/Face-Off):  python scripts/newcomer_check.py [--mode start|feature]
"""
import json
import os
import sys
from datetime import timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402
import walkforward as wf  # noqa: E402

OFFSETS = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]
OUT = os.path.join(ROOT, "data", "newcomer_report{}.json")


MODE = "feature" if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "feature" else "start"


def build(off):
    saved = (bt.NEWCOMER_OFFSET, bt.NEWCOMER_MODE, bt.NEWCOMER_FEATURE_OFFSET)
    if MODE == "start":
        bt.NEWCOMER_OFFSET, bt.NEWCOMER_MODE, bt.NEWCOMER_FEATURE_OFFSET = off, "start", 0.0
    else:
        bt.NEWCOMER_OFFSET, bt.NEWCOMER_MODE, bt.NEWCOMER_FEATURE_OFFSET = 0.0, "feature", off
    try:
        return bt.build_dataset(bt.load_matches())
    finally:
        bt.NEWCOMER_OFFSET, bt.NEWCOMER_MODE, bt.NEWCOMER_FEATURE_OFFSET = saved


def lab(r):
    return 1 if r["match"]["winner"] == r["match"]["team_a"] else 0


def main():
    builds = {o: build(o) for o in OFFSETS}
    base = builds[0.0]
    first = bt._d(base[0]["match"]["date"])
    cut = (first + timedelta(days=bt.NEWCOMER_AFTER_DAYS)).isoformat()
    ev, _ = bt.select_eval(base, bt.MIN_HISTORY)
    nc = [r for r in base if r["match"].get("best_of") == 3 and r["match"]["date"] > cut
          and min(r["meta"]["hist_a"], r["meta"]["hist_b"]) < bt.MIN_HISTORY]
    ev_ids = [r["match"]["id"] for r in ev]
    nc_ids = [r["match"]["id"] for r in nc]
    P_ev, P_nc = {}, {}
    with_t1 = dict(pr.CONFIG, temperature=1.0)
    for o, rows in builds.items():
        by = {r["match"]["id"]: r for r in rows}
        P_ev[o] = wf.with_config(with_t1, lambda: [pr.predict_match(dict(by[i]["input"]))["p_a_exact"] for i in ev_ids])
        P_nc[o] = wf.with_config(with_t1, lambda: [pr.predict_match(dict(by[i]["input"]))["p_a_exact"] for i in nc_ids])
    y_ev, y_nc = [lab(r) for r in ev], [lab(r) for r in nc]
    d_ev = [r["match"]["date"] for r in ev]
    d_nc = [r["match"]["date"] for r in nc]
    oof = {"chosen": [], "off0": []}
    oofe = {"chosen": [], "off0": []}
    ynn, yee, log = [], [], []
    for start, end, _, _ in wf.blocks(ev, 400, 91):
        s, e = start.isoformat(), end.isoformat()
        tr = [i for i, d in enumerate(d_nc) if d < s]
        te = [i for i, d in enumerate(d_nc) if s <= d < e]
        tee = [i for i, d in enumerate(d_ev) if s <= d < e]
        if not tr:
            continue
        best = min(OFFSETS, key=lambda o: (bt.log_loss([P_nc[o][i] for i in tr], [y_nc[i] for i in tr]), o))
        oof["chosen"] += [P_nc[best][i] for i in te]
        oof["off0"] += [P_nc[0.0][i] for i in te]
        oofe["chosen"] += [P_ev[best][i] for i in tee]
        oofe["off0"] += [P_ev[0.0][i] for i in tee]
        ynn += [y_nc[i] for i in te]
        yee += [y_ev[i] for i in tee]
        log.append({"block": s, "n_train_newcomer": len(tr), "n_test_newcomer": len(te), "offset": best})
        print(f"{s}: newcomer train {len(tr):3d} test {len(te):3d} -> offset {best:.0f}")
    rep = {"n_newcomer_oof": len(ynn), "n_eval_oof": len(yee), "folds": log,
           "newcomer": {m: wf.metrics(ps, ynn) for m, ps in oof.items()},
           "eval": {m: wf.metrics(ps, yee) for m, ps in oofe.items()},
           "paired_newcomer": wf.paired(oof["chosen"], oof["off0"], ynn),
           "paired_eval": wf.paired(oofe["chosen"], oofe["off0"], yee),
           "all_rows_ll_by_offset": {str(o): {"newcomer": bt.log_loss(P_nc[o], y_nc),
                                              "eval": bt.log_loss(P_ev[o], y_ev)} for o in OFFSETS}}
    n = rep["newcomer"]
    rep["acceptance"] = {
        "N1_newcomer_better_sig": rep["paired_newcomer"]["log_loss"]["ci95"][1] < 0,
        "N2_eval_not_hurt": rep["eval"]["chosen"]["log_loss"] <= rep["eval"]["off0"]["log_loss"] + 0.001,
        "N3_buckets": all(b.get("pass", True) for b in n["chosen"]["fav_buckets"]),
    }
    rep["acceptance"]["ALL"] = all(rep["acceptance"].values())
    rep["live_offset"] = min(OFFSETS, key=lambda o: bt.log_loss(P_nc[o], y_nc))
    rep["mode"] = MODE
    with open(OUT.format("" if MODE == "start" else "_feature"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    for view, key in (("newcomer rows", "newcomer"), ("main eval rows", "eval")):
        print(f"\n{view} (OOF n={len(ynn) if key == 'newcomer' else len(yee)}):")
        for m, b in rep[key].items():
            print(f"  {m:7s} acc {b['accuracy']:.3f} Brier {b['brier']:.4f} logloss {b['log_loss']:.4f} ECEfav {b['ece10_fav']:.4f}")
        d = rep["paired_" + key]
        for k in ("log_loss", "brier"):
            print(f"  chosen - off0 {k}: {d[k]['diff']:+.4f} [{d[k]['ci95'][0]:+.4f},{d[k]['ci95'][1]:+.4f}] {d[k]['verdict']}")
        for m in ("off0", "chosen"):
            print(f"  favourite buckets {m}:")
            for b in rep[key][m]["fav_buckets"]:
                if b["n"]:
                    print(f"    {b['bucket']:9s} n={b['n']:4d} stated {b['stated']:.3f} actual {b['actual']:.3f} "
                          f"[{b['wilson95'][0]:.3f},{b['wilson95'][1]:.3f}] {'PASS' if b['pass'] else 'FAIL'}")
    print("\nall-rows log loss by offset (in-sample, information only):")
    for o, v in rep["all_rows_ll_by_offset"].items():
        print(f"  {o:>5s}: newcomer {v['newcomer']:.4f}  eval {v['eval']:.4f}")
    print("Acceptance: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in rep["acceptance"].items()))
    print(f"Live offset (all newcomer rows): {rep['live_offset']:.0f}")


if __name__ == "__main__":
    main()
