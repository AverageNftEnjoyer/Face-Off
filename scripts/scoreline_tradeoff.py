"""How often does the engine call a 2-1 (or 3-1 / 3-2), and what does it cost?

With one flat per-map chance q > 0.5 the favourite's 2-0 (q^2) always beats
its 2-1 (2 q^2 (1 - q)), so a 2-1 can only be the most likely score when the
three map chances differ -- the veto's map shape (CONFIG["map_shape"]) is
what makes them differ. This script measures, for a range of map_shape
values, on the same chronological TRAIN / TEST split as
scripts/scoreline_eval.py (BO3 series with a known scoreline):

  modal 3-map   share of series whose most likely scoreline is a 2-1 / 1-2
  hit           share where that single most likely exact score happened
  hit|3-map     the same, only over the series where the modal score is 2-1/1-2
  logloss       4-way scoreline log loss (lower is better)
  pred 3-map    mean predicted chance the series goes to a third map
  act 3-map     actual share of three-map series
  d vs flat     TEST log loss minus flat (map_shape 0), paired bootstrap 95% CI

and the same idea for every real BO5 (both teams >= 5 prior series; none is
used to fit anything) with the BO5 veto: share of series whose most likely
score is a 3-1 / 3-2 (either side), hit rate, 6-way log loss, mean predicted
vs actual share of 3-0 / 0-3. The BO5 sample is tiny (44 series): read it as
a sanity check, not evidence.

Both map-rate inputs are measured: backtest.MAP_RATES = "residual" (the
shipped Elo-residual 90-day map rates) and "raw" (plain 90-day win rates).
p_a itself does not depend on either (w_veto is 0); only the map shape does.

map_scale is not swept: with w_veto = 0 the per-map logits used for the
scoreline are map_shape * map_scale * edge (times one shared factor), and the
simulated bans and picks do not depend on the scale, so map_scale and
map_shape are interchangeable here (checked at start-up).

Nothing is changed in CONFIG permanently. Run from the repo root:
    python scripts/scoreline_tradeoff.py
"""
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402

SHAPES = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
K3 = ["p_2_0", "p_2_1", "p_1_2", "p_0_2"]
L3 = ["2-0", "2-1", "1-2", "0-2"]
L5 = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]


def with_config(changes, fn):
    saved = {k: pr.CONFIG[k] for k in changes}
    pr.CONFIG.update(changes)
    try:
        return fn()
    finally:
        pr.CONFIG.update(saved)


def check_scale_equivalence(rows):
    """map_scale x map_shape is one knob for the scoreline (see docstring)."""
    for r in rows[:40]:
        a = with_config({"map_scale": 6.0, "map_shape": 1.5}, lambda: pr.predict_match(dict(r["input"])))
        b = with_config({"map_scale": 12.0, "map_shape": 0.75}, lambda: pr.predict_match(dict(r["input"])))
        for k in K3:
            assert abs(a["series_probs_exact"][k] - b["series_probs_exact"][k]) < 1e-9
        assert a["veto_log"] == b["veto_log"] and a["veto_bo5"]["veto_log"] == b["veto_bo5"]["veto_log"]
        assert all(abs(x - y) < 1e-9 for x, y in zip(a["veto_bo5"]["map_probs_exact"],
                                                      b["veto_bo5"]["map_probs_exact"]))


def data(mode):
    bt.MAP_RATES = mode
    try:
        rows = bt.build_dataset(bt.load_matches())
    finally:
        bt.MAP_RATES = "residual"
    ev, _ = bt.select_eval(rows, bt.MIN_HISTORY)
    train, test = bt.chrono_split(ev, 0.6)
    keep = lambda rs: [r for r in rs if bt.actual_scoreline(r["match"])]
    train, test = keep(train), keep(test)
    y = lambda rs: [L3.index(bt.actual_scoreline(r["match"]) or "") for r in rs]
    bo5, y5 = [], []
    for r in rows:
        m = r["match"]
        if m.get("best_of") != 5 or min(r["meta"]["hist_a"], r["meta"]["hist_b"]) < 5:
            continue
        a = sum(x["winner"] == m["team_a"] for x in m["maps"])
        b = sum(x["winner"] == m["team_b"] for x in m["maps"])
        if max(a, b) != 3:
            continue
        bo5.append(r)
        y5.append(b if a == 3 else 5 - a)
    return {"train": (train, y(train)), "test": (test, y(test)), "bo5": (bo5, y5)}


def predict_all(rows, shape):
    def run():
        out3, out5 = [], []
        for r in rows:
            o = pr.predict_match(dict(r["input"]))
            out3.append([o["series_probs_exact"][k] for k in K3])
            out5.append([o["series_probs_bo5_exact"][k] for k in pr.BO5_KEYS])
        return out3, out5
    return with_config({"map_shape": shape}, run)


def modal(d):
    return max(range(len(d)), key=lambda i: d[i])


def logloss(ds, ys):
    return sum(-math.log(max(1e-15, d[y])) for d, y in zip(ds, ys)) / len(ys)


def paired_ci(d1, d0, ys, n_boot=2000, seed=12345):
    rng = random.Random(seed)
    per = [math.log(max(1e-15, b[y])) - math.log(max(1e-15, a[y])) for a, b, y in zip(d1, d0, ys)]
    n = len(per)
    vals = sorted(sum(per[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(per) / n, vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def bo3_row(ds, ys, flat=None):
    n = len(ys)
    md = [modal(d) for d in ds]
    three = [i for i, k in enumerate(md) if k in (1, 2)]
    hit = sum(1 for k, y in zip(md, ys) if k == y) / n
    hit3 = (sum(1 for i in three if md[i] == ys[i]) / len(three)) if three else float("nan")
    went3 = (sum(1 for i in three if ys[i] in (1, 2)) / len(three)) if three else float("nan")
    pred3 = sum(d[1] + d[2] for d in ds) / n
    act3 = sum(1 for y in ys if y in (1, 2)) / n
    cell = (f"{len(three) / n:7.1%} ({len(three):3d})  {hit:6.1%}  "
            f"{'   -  ' if not three else f'{hit3:6.1%}'}  {'   -  ' if not three else f'{went3:6.1%}'}  "
            f"{logloss(ds, ys):.4f}  {pred3:6.1%}  {act3:6.1%}")
    if flat is not None:
        m, lo, hi = paired_ci(ds, flat, ys)
        cell += f"  {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    return cell


def bo5_row(ds, ys):
    n = len(ys)
    md = [modal(d) for d in ds]
    mid = sum(1 for k in md if k in (1, 2, 3, 4))
    hit = sum(1 for k, y in zip(md, ys) if k == y) / n
    sweep_p = sum(d[0] + d[5] for d in ds) / n
    sweep_a = sum(1 for y in ys if y in (0, 5)) / n
    return (f"{mid / n:7.1%} ({mid:2d})  {hit:6.1%}  {logloss(ds, ys):.4f}  "
            f"{sweep_p:6.1%}  {sweep_a:6.1%}")


def main():
    for stream in (sys.stdout,):
        reconfigure = getattr(stream, "reconfigure", None)   # absent on a redirected stream
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
    print(f"shipped CONFIG: map_shape {pr.CONFIG['map_shape']}, map_scale {pr.CONFIG['map_scale']}, "
          f"w_veto {pr.CONFIG['w_veto']}")
    for mode in ("residual", "raw"):
        D = data(mode)
        if mode == "residual":
            check_scale_equivalence(D["train"][0])
        print(f"\n=== map rates: {mode} ===  train {len(D['train'][1])}  test {len(D['test'][1])}  "
              f"BO5 {len(D['bo5'][1])}")
        preds = {}
        for split in ("train", "test", "bo5"):
            rows = D[split][0]
            preds[split] = {s: predict_all(rows, s) for s in SHAPES}
        for split in ("train", "test"):
            ys = D[split][1]
            print(f"\nBO3 {split.upper()} (n={len(ys)})")
            print("shape  modal 3-map    hit    hit|3m  went3|3m  logloss  pred3m  act3m"
                  + ("  d logloss vs flat [95% CI]" if split == "test" else ""))
            flat = preds[split][0.0][0]
            for s in SHAPES:
                print(f"{s:5.2f}  " + bo3_row(preds[split][s][0], ys, flat if split == "test" and s else None))
        ys = D["bo5"][1]
        print(f"\nBO5 (all {len(ys)} real BO5s, none used for fitting; SMALL SAMPLE)")
        print("shape  modal 3-1/3-2   hit   logloss  pred sweep  act sweep")
        for s in SHAPES:
            print(f"{s:5.2f}  " + bo5_row(preds["bo5"][s][1], ys))


if __name__ == "__main__":
    main()
