#!/usr/bin/env python3
"""
Valve Regional Standings (VRS) from Valve's public repository
github.com/ValveSoftware/counter-strike_regional_standings.

Finds the newest `live/<year>/standings_global_<date>.md` (and the Europe,
Americas and Asia files of the same date), parses their tables (standing,
points, team name, roster) and writes data/vrs.json. Raw files are cached in
data/raw/valve/standings/; --offline uses only the cache.

Requests: one GitHub API call (repository tree) and up to four raw-file
downloads. In GitHub Actions the workflow's GITHUB_TOKEN is sent to the API
to avoid the shared unauthenticated rate limit.

USAGE (from the repo root):  python data/vrs.py [--offline]
"""

import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "ValveSoftware/counter-strike_regional_standings"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/main?recursive=1"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/"
CACHE = os.path.join(HERE, "raw", "valve", "standings")
UA = "FaceOff-CS2-insights/0.2 (non-commercial fan analytics; https://github.com/AverageNftEnjoyer/Face-Off)"
REGIONS = ("global", "europe", "americas", "asia")
OFFLINE = "--offline" in sys.argv


def _get(url, api=False):
    headers = {"User-Agent": UA}
    if api and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read().decode("utf-8")


def latest_paths():
    tree = json.loads(_get(TREE_URL, api=True))
    files = [t["path"] for t in tree.get("tree", []) if t["path"].startswith("live/")]
    dated = {}
    for p in files:
        m = re.search(r"standings_(global|europe|americas|asia)_(\d{4}_\d{2}_\d{2})\.md$", p)
        if m:
            dated.setdefault(m.group(2), {})[m.group(1)] = p
    full = [d for d, v in dated.items() if "global" in v]
    if not full:
        raise SystemExit("no global standings found in the Valve repository")
    d = max(full)
    return d, dated[d]


def parse_table(md):
    rows = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        rows.append({"rank": int(cells[0]), "points": int(cells[1]), "name": cells[2],
                     "roster": [p.strip() for p in cells[3].split(",") if p.strip()]})
    return rows


def main():
    os.makedirs(CACHE, exist_ok=True)
    if OFFLINE:
        cached = sorted(f for f in os.listdir(CACHE) if f.startswith("standings_global_"))
        if not cached:
            raise SystemExit("offline: no cached VRS standings")
        d = cached[-1][len("standings_global_"):-3]
        paths = {r: f"standings_{r}_{d}.md" for r in REGIONS if os.path.exists(os.path.join(CACHE, f"standings_{r}_{d}.md"))}
    else:
        d, paths = latest_paths()
    out = {"date": d.replace("_", "-"), "source": f"https://github.com/{REPO}", "standings": {}}
    for region, path in paths.items():
        local = os.path.join(CACHE, os.path.basename(path))
        if not os.path.exists(local) and not OFFLINE:
            with open(local, "w", encoding="utf-8") as f:
                f.write(_get(RAW_URL + path))
        with open(local, encoding="utf-8") as f:
            out["standings"][region] = parse_table(f.read())
    with open(os.path.join(HERE, "vrs.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"VRS {out['date']}: " + ", ".join(f"{r} {len(v)} teams" for r, v in out["standings"].items()))


if __name__ == "__main__":
    main()
