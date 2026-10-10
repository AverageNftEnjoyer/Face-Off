"""Scoreline acceptance test: flat per-map chances vs map-specific chances.

The series win probability p_a is the same in every variant; only the split
of p_a across 2-0 / 2-1 (and 1-2 / 0-2) changes. CONFIG["map_shape"] sets how
much of the veto's map-to-map gap reaches the three map win chances
(0 = flat, the same chance on every map).

Metric: 4-way scoreline log loss over BO3 series with a known scoreline.
map_shape is chosen on TRAIN only and then scored once on TEST (same
chronological split as backtest.py). Also prints the mean predicted vs actual
scoreline distribution and a calibration table per scoreline.

Run from the repo root:  python scripts/scoreline_eval.py
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402

KEYS = ["p_2_0", "p_2_1", "p_1_2", "p_0_2"]
LABELS = ["2-0", "2-1", "1-2", "0-2"]
SHAPES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
BINS = [(0.0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 1.01)]


def dists(rows, shape):
    saved = pr.CONFIG["map_shape"]
    pr.CONFIG["map_shape"] = shape
    try:
        return [[pr.predict_match(dict(r["input"]))["series_probs_exact"][k] for k in KEYS] for r in rows]
    finally:
        pr.CONFIG["map_shape"] = saved


def logloss(ds, ys):
    return -sum(math.log(max(1e-15, d[y])) for d, y in zip(ds, ys)) / len(ys)


def paired_ci(d1, d2, ys, n_boot=2000, seed=12345):
    import random
    rng = random.Random(seed)
    per = [-math.log(max(1e-15, a[y])) + math.log(max(1e-15, b[y])) for a, b, y in zip(d1, d2, ys)]
    n = len(per)
    vals = sorted(sum(per[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(per) / n, vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def fmt(xs):
    return "/".join(f"{x:.2f}" for x in pr.display_percent(xs))


def main():
    rows = bt.build_dataset(bt.load_matches())
    ev, _ = bt.select_eval(rows, bt.MIN_HISTORY)
    train, test = bt.chrono_split(ev, 0.6)
    keep = lambda rs: [r for r in rs if bt.actual_scoreline(r["match"])]
    train, test = keep(train), keep(test)
    y_tr = [LABELS.index(bt.actual_scoreline(r["match"]) or "") for r in train]
    y_te = [LABELS.index(bt.actual_scoreline(r["match"]) or "") for r in test]

    print(f"train {len(train)}  test {len(test)}  (BO3 with a known scoreline)")
    print("map_shape  train scoreline log loss")
    tr_scores = {s: logloss(dists(train, s), y_tr) for s in SHAPES}
    for s in SHAPES:
        print(f"  {s:4.2f}     {tr_scores[s]:.6f}")
    best = min(tr_scores, key=lambda s: tr_scores[s])
    print(f"chosen on train: map_shape {best}")

    flat, chosen, shipped = dists(test, 0.0), dists(test, best), dists(test, pr.CONFIG["map_shape"])
    print(f"\nTEST scoreline log loss: flat {logloss(flat, y_te):.6f}  "
          f"map_shape {best} {logloss(chosen, y_te):.6f}  "
          f"shipped {pr.CONFIG['map_shape']} {logloss(shipped, y_te):.6f}")
    for name, d in ((f"chosen {best}", chosen), (f"shipped {pr.CONFIG['map_shape']}", shipped)):
        m, lo, hi = paired_ci(d, flat, y_te)
        print(f"  {name} - flat: {m:+.6f} [{lo:+.6f}, {hi:+.6f}]")

    n = len(y_te)
    actual = [sum(1 for y in y_te if y == i) / n for i in range(4)]
    print(f"\nTEST scoreline distribution (2-0/2-1/1-2/0-2, team_a view)")
    print(f"  actual          {fmt(actual)}")
    for name, d in (("flat", flat), (f"map_shape {best}", chosen)):
        print(f"  {name:15s} {fmt([sum(x[i] for x in d) / n for i in range(4)])}")
    modal = lambda d: [max(range(4), key=lambda i: x[i]) for x in d]
    for name, d in (("flat", flat), (f"map_shape {best}", chosen)):
        md = modal(d)
        print(f"  modal {name:9s} " + "  ".join(f"{LABELS[i]} {md.count(i) / n:.1%}" for i in range(4))
              + f"   hit rate {sum(1 for a, b in zip(md, y_te) if a == b) / n:.1%}")

    print(f"\nTEST scoreline calibration (map_shape {best}): stated -> observed, n")
    for i, lab in enumerate(LABELS):
        cells = []
        for lo, hi in BINS:
            idx = [j for j, x in enumerate(chosen) if lo <= x[i] < hi]
            if idx:
                st = sum(chosen[j][i] for j in idx) / len(idx)
                ob = sum(1 for j in idx if y_te[j] == i) / len(idx)
                cells.append(f"{st * 100:5.2f}->{ob * 100:5.2f} ({len(idx)})")
        print(f"  {lab}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
