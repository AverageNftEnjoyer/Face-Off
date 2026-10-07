"""BO5 scoreline check on every real BO5 in data/matches.json.

No BO5 is used to fit anything, so all of them are held out. For each series
(both teams >= 5 prior series), the engine's p_a is turned into BO5
scorelines three ways, and scored by 6-way scoreline log loss:
  veto_bo5   the BO5 veto's per-map chances (map_shape), shifted so BO5
             P(A) = p_a (what predictor.py outputs as series_probs_bo5)
  flat_bo5   one per-map chance on all five maps, set so BO5 P(A) = p_a
             (the BO5 output before the BO5 veto)
  flat_bo3   one per-map chance from inverting the BO3 formula (BO5 P(A) > p_a
             for the favourite)
  lengths    winner from p_a, then the leave-one-out historical shares of
             3-0 / 3-1 / 3-2 (ignores how close the match is)
Also prints how often 3-1 is the favourite's most likely score (rule: with a
flat per-map chance q, 3-1 beats 3-0 exactly when q < 2/3) and the actual vs
mean predicted scoreline mix.

Run from the repo root:  python scripts/bo5_check.py
"""
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402

LABELS = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]


def idx(a, b):
    """Position of the final score (a maps for team_a, b for team_b) in LABELS."""
    return b if a == 3 else 5 - a


def main():
    rows = bt.build_dataset(bt.load_matches())
    bo5 = [r for r in rows if r["match"].get("best_of") == 5
           and min(r["meta"]["hist_a"], r["meta"]["hist_b"]) >= 5]
    recs = []
    for r in bo5:
        m = r["match"]
        a = sum(x["winner"] == m["team_a"] for x in m["maps"])
        b = sum(x["winner"] == m["team_b"] for x in m["maps"])
        if max(a, b) != 3:
            continue
        out = pr.predict_match(dict(r["input"]))
        recs.append((out["p_a_exact"], idx(a, b),
                     [out["series_probs_bo5_exact"][k] for k in pr.BO5_KEYS]))
    n = len(recs)

    def loo_lengths(i):
        c = [1, 1, 1]                      # +1 smoothing per length
        for j, (_, y, _) in enumerate(recs):
            if j != i:
                c[min(y, 5 - y)] += 1
        return [x / sum(c) for x in c]

    losses = {"veto_bo5": [], "flat_bo5": [], "flat_bo3": [], "lengths": []}
    modal31, modal_v, mean = 0, 0, [0.0] * 6
    for i, (p, y, dv) in enumerate(recs):
        d5 = pr.series_scorelines([pr.flat_map_prob(p, 5)] * 5)
        d3 = pr.series_scorelines([pr.flat_map_prob(p, 3)] * 5)
        sh = loo_lengths(i)
        dl = [p * s for s in sh] + [(1 - p) * s for s in reversed(sh)]
        for name, d in (("veto_bo5", dv), ("flat_bo5", d5), ("flat_bo3", d3), ("lengths", dl)):
            losses[name].append(-math.log(d[y]))
        fav = d5[:3] if p >= 0.5 else d5[3:][::-1]
        modal31 += max(range(3), key=lambda k: fav[k]) == 1
        modal_v += max(range(6), key=lambda k: dv[k]) in (1, 2, 3, 4)
        mean = [s + x / n for s, x in zip(mean, d5)]

    print(f"BO5 series: {n}")
    for name, ls in losses.items():
        print(f"  {name:9s} scoreline log loss {sum(ls) / n:.4f}")
    print(f"  uniform   scoreline log loss {math.log(6):.4f}")
    rng = random.Random(12345)
    d = [a - b for a, b in zip(losses["flat_bo5"], losses["lengths"])]
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
    print(f"  flat_bo5 - lengths: {sum(d) / n:+.4f} [{boots[100]:+.4f}, {boots[3899]:+.4f}] (95% paired bootstrap)")
    d = [a - b for a, b in zip(losses["veto_bo5"], losses["flat_bo5"])]
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
    print(f"  veto_bo5 - flat_bo5: {sum(d) / n:+.4f} [{boots[100]:+.4f}, {boots[3899]:+.4f}] (95% paired bootstrap)")
    print(f"flat: 3-1 is the favourite's most likely score in {modal31}/{n} series")
    print(f"veto: a 3-1 / 3-2 (either side) is the most likely score in {modal_v}/{n} series")
    actual = [sum(1 for _, y, _ in recs if y == k) / n for k in range(6)]
    print("actual         " + "  ".join(f"{l} {v:.2f}" for l, v in zip(LABELS, pr.display_percent(actual))))
    mean_v = [sum(r[2][k] for r in recs) / n for k in range(6)]
    print("mean veto_bo5  " + "  ".join(f"{l} {v:.2f}" for l, v in zip(LABELS, pr.display_percent(mean_v))))
    print("mean flat_bo5  " + "  ".join(f"{l} {v:.2f}" for l, v in zip(LABELS, pr.display_percent(mean))))


if __name__ == "__main__":
    main()
