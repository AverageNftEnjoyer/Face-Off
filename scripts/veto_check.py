"""How well does the simulated veto predict the maps actually played?

For every BO3 series in the backtest (point-in-time inputs), compare the
veto's two picks with the two maps actually played first (map 1 and map 2 are
the two teams' picks in a standard BO3 veto) and its decider with map 3 when
a third map was played. The order of the picks is ignored (the data does not
say which team picked which), so the score is the overlap of the two sets.

Tunes CONFIG["veto_comfort"] (how much a team's own map experience counts in
its picks and bans) on the first 60% of series by date and reports the last
40% (held out). Also reports how often the simulation has a team pick a map
it has not played in the last 90 days.

Run from the repo root:  python scripts/veto_check.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402

GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]


def rows():
    out = []
    for r in bt.build_dataset(bt.load_matches()):
        m = r["match"]
        if m.get("best_of") != 3 or len(m.get("maps") or []) < 2:
            continue
        pool = pr._pool(r["input"])
        played = [x["map"] for x in m["maps"]]
        if any(p not in pool for p in played):
            continue          # map pool changed around this series; not comparable
        out.append((r["input"], played))
    return out


def score(data, w):
    saved = pr.CONFIG["veto_comfort"]
    pr.CONFIG["veto_comfort"] = w
    try:
        overlap = dec_hit = dec_n = unplayed = picks = 0
        for inp, played in data:
            v = pr.simulate_veto(inp, 3)
            pk = set(v["picks"])
            overlap += len(pk & set(played[:2])) / 2
            if len(played) >= 3:
                dec_n += 1
                dec_hit += v["decider"] == played[2]
            for line in v["veto_log"]:
                if " picks " in line:
                    side, mp = line.split(" picks ")
                    raw = inp.get("maps_a" if side == "A" else "maps_b") or {}
                    e = raw.get(mp)
                    picks += 1
                    unplayed += not (e and e[1])
        return overlap / len(data), dec_hit / max(1, dec_n), unplayed / max(1, picks)
    finally:
        pr.CONFIG["veto_comfort"] = saved


def main():
    data = rows()
    k = int(len(data) * 0.6)
    train, test = data[:k], data[k:]
    print(f"BO3 series with known maps: {len(data)} (tune on {len(train)}, report on {len(test)})")
    print("weight   train picks-hit  decider  picked-unplayed | test picks-hit  decider  picked-unplayed")
    best = None
    for w in GRID:
        a, b = score(train, w), score(test, w)
        print(f"{w:6.2f}   {a[0]:.3f}          {a[1]:.3f}    {a[2]:.3f}           | {b[0]:.3f}          {b[1]:.3f}    {b[2]:.3f}")
        if best is None or a[0] + a[1] > best[1]:
            best = (w, a[0] + a[1])
    print(f"chosen on train (picks-hit + decider-hit): veto_comfort = {best[0]}")
    # chance level: a random two of the seven maps would hit about 2/7 per pick slot
    print("chance level: picks-hit ~0.286, decider ~0.143")


if __name__ == "__main__":
    main()
