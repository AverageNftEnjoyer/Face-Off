"""Does series length change the odds? Checks the engine's BO3 win chance
against real BO1 and BO5 results, as is and converted by series-length maths
(q = the per-map chance that gives the BO3 number; BO1 = q, BO5 = win 3 of 5
at q), plus a logit scale fitted on the first half of each format.

Run from the repo root:  python scripts/series_length_check.py [matches.json]
"""
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402

rows = bt.build_dataset(bt.load_matches(sys.argv[1] if len(sys.argv) > 1 else None))
def ev(bo):
    out = []
    for r in rows:
        m = r["match"]
        if m.get("best_of") != bo or min(r["meta"]["hist_a"], r["meta"]["hist_b"]) < 5: continue
        p = pr.predict_match(dict(r["input"]))["p_a_exact"]
        out.append((m["date"], p, 1 if m["winner"] == m["team_a"] else 0))
    return sorted(out)
def ll(ps, ys): return bt.log_loss(ps, ys)
def bootdiff(a, b, ys, n=4000):
    rng = random.Random(1); N = len(ys)
    la = [-(y*math.log(p)+(1-y)*math.log(1-p)) for p,y in zip(a,ys)]
    lb = [-(y*math.log(p)+(1-y)*math.log(1-p)) for p,y in zip(b,ys)]
    d = [x-y for x,y in zip(la,lb)]
    v = sorted(sum(d[rng.randrange(N)] for _ in range(N))/N for _ in range(n))
    return sum(d)/N, v[int(.025*n)], v[int(.975*n)-1]
def amp(p, bo):
    # per-map chance implied by the BO3 call, then that chance played as a best-of-`bo`
    q = bt.map_prob_from_series(p, 3)
    if bo == 1: return q
    return sum(pr.series_scorelines([q]*5)[:3])
for bo in (1, 5):
    d = ev(bo); ps = [x[1] for x in d]; ys = [x[2] for x in d]
    fav = [max(p,1-p) for p in ps]
    print(f"\n=== Bo{bo}: n={len(d)} ({d[0][0]}..{d[-1][0]})")
    cands = {"same as Bo3 (now)": ps, "series-length maths": [amp(p, bo) for p in ps]}
    for k, v in cands.items():
        print(f"  {k:24s} acc {bt.accuracy(v,ys):.3f}  Brier {bt.brier(v,ys):.4f}  logloss {ll(v,ys):.4f}  mean fav {sum(max(p,1-p) for p in v)/len(v):.3f}")
    print(f"  actual favourite win rate: {sum(1 for p,y in zip(ps,ys) if (p>=.5)==(y==1))/len(ys):.3f} (fav = engine's pick)")
    m, lo, hi = bootdiff(cands["series-length maths"], ps, ys)
    print(f"  series-length minus now: logloss {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    # fitted scale on logit: fit on first half, test on second half
    h = len(d)//2
    xs = [bt.logit(p) for p in ps]
    c = bt.fit_temperature(xs[:h], ys[:h])
    test = [bt.sigmoid(c*x) for x in xs[h:]]
    print(f"  fitted logit scale on first half: c={c:.3f} (1 = same as Bo3); 2nd-half logloss fitted {ll(test, ys[h:]):.4f} vs now {ll(ps[h:], ys[h:]):.4f} vs maths {ll([amp(p,bo) for p in ps[h:]], ys[h:]):.4f}")
    cfull = bt.fit_temperature(xs, ys); print(f"  fitted scale on all Bo{bo}: c={cfull:.3f}")
# Bo3 reference: how far are fav rates by bucket
d3 = ev(3); ps3=[x[1] for x in d3]; ys3=[x[2] for x in d3]
print(f"\nBo3 reference n={len(d3)}: fitted scale c={bt.fit_temperature([bt.logit(p) for p in ps3], ys3):.3f}, acc {bt.accuracy(ps3,ys3):.3f}")
# implied amplification check: per-map logit scale
print("maths: Bo1 logit scale for a 65% Bo3 call:", round(bt.logit(amp(.65,1))/bt.logit(.65),3), " Bo5:", round(bt.logit(amp(.65,5))/bt.logit(.65),3))
