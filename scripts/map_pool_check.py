"""Check that every team's map pool on the built site matches the results.

Recomputes, straight from data/matches.json and without the engine code, each
team's plain map record over the 90 days before the data date (wins / plays per
map) and the live map pool (maps with >= 3 plays in the prior 60 days, the 7
most recently played), then compares both with the DATA blob in a built
index.html. Any map a team has started (or stopped) playing shows up here.

    python scripts/map_pool_check.py [path/to/index.html]
"""
import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def site_data(path):
    html = open(path, encoding="utf-8").read()
    i = html.index("const DATA = ") + len("const DATA = ")
    return json.JSONDecoder().raw_decode(html[i:].replace("<\\/", "</"))[0]


def main(path):
    ms = json.load(open(os.path.join(ROOT, "data", "matches.json"), encoding="utf-8"))
    D = site_data(path)
    as_of = date.fromisoformat(D["as_of"])

    rec = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    seen = defaultdict(list)
    for m in ms:
        d = date.fromisoformat(m["date"])
        if d >= as_of:
            continue
        for x in m.get("maps", []):
            if not x.get("map"):
                continue
            if d >= as_of - timedelta(days=60):
                seen[x["map"]].append(d)
            if d >= as_of - timedelta(days=90):
                for t in (m["team_a"], m["team_b"]):
                    rec[t][x["map"]][0] += x["winner"] == t
                    rec[t][x["map"]][1] += 1

    cand = sorted(((max(ds), len(ds), mp) for mp, ds in seen.items() if len(ds) >= 3), reverse=True)
    pool = sorted(mp for _, _, mp in cand[:7])
    bad = 0
    print(f"data as of {as_of}; live pool: {', '.join(pool)}")
    if D.get("pool") != pool:
        bad += 1
        print(f"  POOL MISMATCH: site has {D.get('pool')}")
    out_of_pool = {mp for mp in seen if mp not in pool}
    if out_of_pool:
        print(f"  maps played in the last 60 days but not in the pool: {', '.join(sorted(out_of_pool))}")

    for t, info in sorted(D["teams"].items()):
        want = {mp: [round(w / n, 4), n] for mp, (w, n) in sorted(rec[t].items())}
        have = info.get("maps") or {}
        if want != have:
            bad += 1
            print(f"  {t}: site {have}\n  {' ' * len(t)}  data {want}")
        extra = sorted(mp for mp in want if mp not in pool)
        if extra:
            print(f"  {t} has 90-day plays on maps outside the pool: {', '.join(extra)}")
    print(f"{len(D['teams'])} teams checked, {bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "index.html")))
