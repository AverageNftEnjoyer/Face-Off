#!/usr/bin/env python3
"""
Per-match five-man lineups from Valve's Regional Standings repository
(github.com/ValveSoftware/counter-strike_regional_standings).

Every live/<year>/details/<snapshot>/<n>--<team>--<players>.md page lists the
matches that fed one roster's standing over the previous six months, one row
per match: date, opponent, W/L and the five players who played it (a
stand-in shows up as a different name in that row). Snapshots are monthly, so
fetching one snapshot every SNAPSHOT_STEP months covers every match since
~February 2024 with overlap.

Join to data/matches.json: for a series (date, team_a, team_b, winner), the
row on team_a's pages with the same opponent, a W/L that agrees with the
winner and a date within one day (Liquipedia dates are event-local, Valve's
may not be); the closest date wins. Same for team_b. A side with no matching
row gets no lineup -- nothing is guessed.

Team names: Valve file slug / "Team Name" header -> Liquipedia name by the
same normalisation the hub uses (viewer/build_viewer._norm) plus SLUG_ALIAS.

Output: data/lineups.json
    {"source": ..., "snapshots": [...],
     "lineups": {"<date>|<team_a>|<team_b>": {"a": [5 ids] | null, "b": [...] | null}}}
Player ids are lower-cased.

Raw files are cached in data/raw/valve/details/ (gitignored); --offline uses
only the cache. Requests: one cached tree + raw.githubusercontent.com files,
spaced REQUEST_GAP seconds.

USAGE (from the repo root):  python data/lineups.py [--offline]
"""
import json
import os
import re
import sys
import time
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "viewer"))

import vrs  # noqa: E402

OFFLINE = "--offline" in sys.argv
TREE = os.path.join(HERE, "raw", "valve", "valve_tree.json")
CACHE = os.path.join(HERE, "raw", "valve", "details")
OUT = os.path.join(HERE, "lineups.json")
SNAPSHOT_STEP = 2          # every 2nd monthly snapshot; pages look back 6 months
REQUEST_GAP = 0.25
# Valve slug (normalised) -> Liquipedia team name, where the names differ
SLUG_ALIAS = {}

_DROP = {"team", "esports", "esport", "gaming", "clan", "club", "gg"}


def norm(name):
    words = re.findall(r"[a-z0-9]+", name.lower())
    return "".join(w for w in words if w not in _DROP) or "".join(words)


def tree_paths():
    if os.path.exists(TREE):
        with open(TREE, encoding="utf-8") as f:
            t = json.load(f)
    elif OFFLINE:
        raise SystemExit("offline: no cached Valve tree (data/raw/valve/valve_tree.json)")
    else:
        t = json.loads(vrs._get(vrs.TREE_URL, api=True))
        os.makedirs(os.path.dirname(TREE), exist_ok=True)
        with open(TREE, "w", encoding="utf-8") as f:
            json.dump(t, f)
    return [x["path"] for x in (t["tree"] if isinstance(t, dict) else t)]


def pick_snapshots(paths):
    snaps = sorted({p.split("/")[3] for p in paths
                    if p.startswith("live/") and p.count("/") == 4 and "/details/" in p})
    # one per calendar month (the first), then every SNAPSHOT_STEP-th, always the latest
    monthly, seen = [], set()
    for s in snaps:
        if s[:7] not in seen:
            seen.add(s[:7])
            monthly.append(s)
    chosen = monthly[::SNAPSHOT_STEP]
    if monthly and monthly[-1] not in chosen:
        chosen.append(monthly[-1])
    return chosen


def fetch(path):
    local = os.path.join(CACHE, path.split("/details/", 1)[1])
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            return f.read()
    if OFFLINE:
        raise SystemExit(f"offline: {path} not cached")
    time.sleep(REQUEST_GAP)
    txt = vrs._get(vrs.RAW_URL + path)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


ROW = re.compile(r"^\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]*?)\s*\|\s*([WL])\s*\|(.*)$")


def parse_detail(txt):
    """(team name, [(date, opponent, won, [5 ids])])"""
    m = re.search(r"Team Name:\s*(.*?)<br", txt)
    team = m.group(1).strip() if m else None
    rows = []
    for line in txt.splitlines():
        r = ROW.match(line.strip())
        if not r:
            continue
        cells = [c.strip() for c in r.group(4).strip().strip("|").split("|")]
        players = [p.strip().lower() for p in cells[-1].split(",") if p.strip()]
        if len(players) == 5:
            rows.append((r.group(1), r.group(2), r.group(3) == "W", players))
    return team, rows


def main():
    with open(os.path.join(HERE, "matches.json"), encoding="utf-8") as f:
        matches = json.load(f)
    ours = {}
    for m in matches:
        for t in (m["team_a"], m["team_b"]):
            ours.setdefault(norm(t), t)
    ours.update({k: v for k, v in SLUG_ALIAS.items()})

    paths = tree_paths()
    snaps = pick_snapshots(paths)
    wanted = [p for p in paths if p.startswith("live/") and "/details/" in p and p.endswith(".md")
              and p.split("/")[3] in snaps
              and norm(p.rsplit("/", 1)[1].split("--")[1].replace("_", " ")) in ours]
    print(f"{len(snaps)} snapshots ({snaps[0]} .. {snaps[-1]}), {len(wanted)} detail pages for our teams")

    # team -> {(date, opponent_valve_name, won): lineup}; Valve display name -> Liquipedia name
    rows_by_team, display = {}, {}
    for i, p in enumerate(wanted, 1):
        lp = ours[norm(p.rsplit("/", 1)[1].split("--")[1].replace("_", " "))]
        name, rows = parse_detail(fetch(p))
        if name:
            display[name] = lp
        store = rows_by_team.setdefault(lp, {})
        for d, opp, won, five in rows:
            store[(d, opp, won)] = five
        if i % 100 == 0:
            print(f"  {i}/{len(wanted)}")

    def find(team, opp, won, d0):
        best = None
        for (d, o, w), five in rows_by_team.get(team, {}).items():
            if w != won or display.get(o, ours.get(norm(o))) != opp:
                continue
            gap = abs((date.fromisoformat(d) - d0).days)
            if gap <= 1 and (best is None or gap < best[0]):
                best = (gap, five)
        return best[1] if best else None

    out, both = {}, 0
    for m in matches:
        d0 = date.fromisoformat(m["date"])
        a, b = m["team_a"], m["team_b"]
        la = find(a, b, m["winner"] == a, d0)
        lb = find(b, a, m["winner"] == b, d0)
        if la or lb:
            out[f"{m['date']}|{a}|{b}"] = {"a": la, "b": lb}
            both += bool(la and lb)
    covered = [m for m in matches if m["date"] >= "2024-02-01"]
    print(f"lineups: {len(out)} series with at least one side, {both} with both "
          f"(of {len(covered)} series since 2024-02-01)")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": f"https://github.com/{vrs.REPO}", "snapshots": snaps,
                   "lineups": out}, f, indent=0, ensure_ascii=False, sort_keys=True)
    print("wrote data/lineups.json")


if __name__ == "__main__":
    main()
