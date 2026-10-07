"""How much should lower-tier results move the ratings?

Scores the engine (CONFIG as shipped, nothing refitted) on S/A-tier BO3
series under different Elo weights for B/C-tier matches, and with B/C-tier
matches left out of the history entirely. Same rows and chronological split
in every variant: the last 40% (by date) of S/A-tier BO3 series where both
teams have >= 5 prior S/A-tier series. B-tier BO3 series are scored too
(where both teams have >= 5 prior series of any tier), for the cost on them.

Run from the repo root:  python scripts/tier_ratings.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402

ALL = bt.load_matches()
TOP = {"S", "A"}


def score(rows):
    ys = [1 if r["match"]["winner"] == r["match"]["team_a"] else 0 for r in rows]
    ps = [pr.predict_match(dict(r["input"]))["p_a_exact"] for r in rows]
    return len(ys), bt.accuracy(ps, ys), bt.brier(ps, ys), bt.log_loss(ps, ys), ps, ys


def run(k_low, drop_low=False):
    bt.ELO_K_BY_TIER.update({"B": k_low, "C": k_low})
    ms = [m for m in ALL if m.get("tier") in TOP] if drop_low else ALL
    return bt.build_dataset(ms)


def eval_rows(rows):
    """S/A rows fixed by a history-independent rule so every variant scores the same series."""
    byid = {r["match"]["id"]: r for r in rows}
    out_sa, out_b = [], []
    prior = {}
    day_buf = []
    ms = sorted(ALL, key=lambda m: (m["date"], m["id"]))
    i = 0
    while i < len(ms):
        j = i
        while j < len(ms) and ms[j]["date"] == ms[i]["date"]:
            j += 1
        for m in ms[i:j]:
            if m["id"] in byid and m.get("best_of") == 3:
                ok = min(prior.get((m["team_a"], "top"), 0), prior.get((m["team_b"], "top"), 0)) >= 5
                okb = min(prior.get((m["team_a"], "any"), 0), prior.get((m["team_b"], "any"), 0)) >= 5
                if m.get("tier") in TOP and ok:
                    out_sa.append(byid[m["id"]])
                elif m.get("tier") not in TOP and okb:
                    out_b.append(byid[m["id"]])
        for m in ms[i:j]:
            for t in (m["team_a"], m["team_b"]):
                prior[(t, "any")] = prior.get((t, "any"), 0) + 1
                if m.get("tier") in TOP:
                    prior[(t, "top")] = prior.get((t, "top"), 0) + 1
        i = j
    cut_sa = int(len(out_sa) * 0.6)
    cut_b = int(len(out_b) * 0.6)
    return out_sa[cut_sa:], out_b[cut_b:]


def main():
    base = None
    print("variant                      S/A test: n   acc    Brier   logloss | B test: n   acc    Brier   logloss")
    for name, k, drop in (("B/C full weight (now)", 1.0, False), ("B/C half weight", 0.5, False),
                          ("B/C quarter weight", 0.25, False), ("B/C no Elo weight", 0.0, False),
                          ("B/C left out entirely", 0.0, True)):
        rows = run(k, drop)
        sa, b = eval_rows(rows)
        s = score(sa)
        bb = score(b) if b else (0, 0, 0, 0, [], [])
        if base is None:
            base = s
        d = bt.paired_bootstrap_diff(bt.log_loss, s[4], base[4], s[5]) if s[5] == base[5] else (float("nan"),) * 2
        print(f"{name:28s} {s[0]:5d} {s[1]:.3f} {s[2]:.4f} {s[3]:.4f} | {bb[0]:5d} {bb[1]:.3f} {bb[2]:.4f} {bb[3]:.4f}"
              f"   S/A logloss vs now {s[3] - base[3]:+.4f} [{d[0]:+.4f}, {d[1]:+.4f}]")
    bt.ELO_K_BY_TIER.update({"B": 1.0, "C": 1.0})


if __name__ == "__main__":
    main()
