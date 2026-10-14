#!/usr/bin/env python3
"""
Valve Regional Standings (VRS) from Valve's public repository
github.com/ValveSoftware/counter-strike_regional_standings.

Finds the newest `live/<year>/standings_global_<date>.md` (and the Europe,
Americas and Asia files of the same date), parses their tables (standing,
points, team name, roster) and writes data/vrs.json. Raw files are cached in
data/raw/valve/standings/; --offline uses only the cache.

The newest standings are chosen by DATE among every file in the tree on every
run. A failed fetch or an empty parse keeps the last good vrs.json and is
recorded in data/vrs_status.json (ok=false, error): never silent. vrs.json is
rewritten only when the standings change, so an unchanged refresh commits
nothing. `--check` is an offline freshness test (exit 1 when the stored
standings are stale or the last refresh failed).

Requests: one GitHub API call (repository tree) and up to four raw-file
downloads. In GitHub Actions the workflow's GITHUB_TOKEN is sent to the API
to avoid the shared unauthenticated rate limit.

USAGE (from the repo root):  python collectors/vrs.py [--offline | --check]
"""

import json
import os
import re
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (puts every code folder on sys.path)
DATA = paths.DATA
REPO = "ValveSoftware/counter-strike_regional_standings"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/main?recursive=1"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/"
CACHE = os.path.join(DATA, "raw", "valve", "standings")
UA = "FaceOff-CS2-insights/0.2 (non-commercial fan analytics; https://github.com/AverageNftEnjoyer/Face-Off)"
REGIONS = ("global", "europe", "americas", "asia")
OFFLINE = "--offline" in sys.argv
# Freshness model, measured from Valve's own publication dates (see cadence()):
# since 2025-03 Valve publishes on the first Monday of each month (28-35 day
# gaps; weekly before that). A snapshot older than the longest recent gap plus
# GRACE_DAYS is stale. DEFAULT_STALE_AFTER is only the fallback with no history.
GRACE_DAYS = 7
MIN_GAPS = 4
DEFAULT_STALE_AFTER = 42


def _get(url, api=False):
    headers = {"User-Agent": UA}
    if api and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read().decode("utf-8")


_TREE = []


def tree():
    """The repository tree (one GitHub API call per process; collectors/lineups.py
    reuses it when both run in the same daily refresh)."""
    if not _TREE:
        t = json.loads(_get(TREE_URL, api=True))
        if t.get("truncated"):
            raise ValueError("GitHub returned a truncated repository tree")
        _TREE.append(t)
    return _TREE[0]


def published(paths):
    """{ISO date: {region: path}} for every complete (global) standings file in
    `paths`; dates that are not real calendar dates are ignored."""
    dated = {}
    for p in paths:
        m = re.search(r"standings_(global|europe|americas|asia)_(\d{4})_(\d{2})_(\d{2})\.md$", p)
        if not (m and p.startswith("live/")):
            continue
        try:
            d = date(int(m.group(2)), int(m.group(3)), int(m.group(4))).isoformat()
        except ValueError:
            continue
        dated.setdefault(d, {})[m.group(1)] = p
    return {d: v for d, v in dated.items() if "global" in v}


def newest_published(paths):
    """(ISO date, {region: path}) of the newest complete standings, compared as
    dates and never by file-name text."""
    full = published(paths)
    if not full:
        raise ValueError("no global standings found in the Valve repository")
    d = max(full)
    return d, full[d]


def latest_paths():
    """(underscore date, {region: path}); kept for older callers."""
    d, files = newest_published([t["path"] for t in tree().get("tree", [])])
    return d.replace("-", "_"), files


def cadence(dates, recent=12):
    """Valve's publication rhythm measured from the dates it published (the
    repository's own history), in days: typical = median gap, max = longest of
    the last `recent` gaps, stale_after = max + GRACE_DAYS."""
    ds = sorted(date.fromisoformat(d) for d in dates)
    gaps = [(b - a).days for a, b in zip(ds, ds[1:])][-recent:]
    if len(gaps) < MIN_GAPS:
        return {"typical": None, "max": None, "stale_after": DEFAULT_STALE_AFTER}
    s = sorted(gaps)
    return {"typical": s[len(s) // 2], "max": max(gaps), "stale_after": max(gaps) + GRACE_DAYS}


def assess(stored, status, today):
    """Freshness problems (list of strings; empty = fine) from the stored
    standings date, the status recorded by the last refresh, and today. Pure:
    no network, so the data gate and the tests can call it."""
    status = status or {}
    if not stored:
        return ["no stored Valve standings (data/vrs.json)"]
    out = []
    pub = status.get("published")
    if pub and pub > stored:
        out.append(f"stored standings are {stored} but Valve has published {pub}")
    limit = status.get("stale_after_days") or DEFAULT_STALE_AFTER
    age = (today - date.fromisoformat(stored)).days
    if age > limit:
        who = ("Valve has published nothing newer, so Valve itself is overdue"
               if pub and pub <= stored else "newer standings may exist")
        out.append(f"stored standings are {age} days old (Valve's cadence allows {limit}); {who}")
    if status.get("ok") is False:
        out.append(f"the last Valve standings refresh failed: {status.get('error')}")
    return out


def parse_table(md):
    rows = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        rows.append({"rank": int(cells[0]), "points": int(cells[1]), "name": cells[2],
                     "roster": [p.strip() for p in cells[3].split(",") if p.strip()]})
    return rows


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _dump(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False)


def _write_if_changed(path, text):
    """Write only when the bytes differ, so an unchanged refresh commits nothing."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            if f.read() == text:
                return False
    except OSError:
        pass
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return True


def fetch_standings(get=None, paths=None, tree_obj=None):
    """(vrs dict, cadence info, newest published date) for the newest published
    standings; raises ValueError when nothing usable parsed. A raw file is
    cached only after it parsed to a non-empty table."""
    get = get or _get
    if paths is not None:
        tree_obj = {"tree": [{"path": p} for p in paths]}
    elif tree_obj is None:
        tree_obj = tree()
    allp = [t["path"] for t in tree_obj.get("tree", [])]
    d, files = newest_published(allp)
    out = {"date": d, "source": f"https://github.com/{REPO}",
           "tree_sha": tree_obj.get("sha"), "standings": {}}
    for region, path in files.items():
        local = os.path.join(CACHE, os.path.basename(path))
        cached = os.path.exists(local)
        if cached:
            with open(local, encoding="utf-8") as f:
                txt = f.read()
        else:
            txt = get(RAW_URL + path)
        rows = parse_table(txt)
        if not rows:
            raise ValueError(f"{path} parsed to no teams")
        if not cached:
            os.makedirs(CACHE, exist_ok=True)
            with open(local, "w", encoding="utf-8") as f:
                f.write(txt)
        out["standings"][region] = rows
    if out["standings"]["global"][0]["rank"] != 1:
        raise ValueError("global standings do not start at rank 1")
    pub = published(allp)
    return out, cadence(pub), max(pub)


def refresh(get=None, paths=None, now=None):
    """One refresh. Never raises on a fetch or parse problem: the last good
    data/vrs.json stays and data/vrs_status.json records the failure. Returns
    the status dict. vrs.json is rewritten only when the standings changed and
    the status file only when its content changed (no per-run timestamp)."""
    now = now or datetime.now(timezone.utc)
    vpath, spath = os.path.join(DATA, "vrs.json"), os.path.join(DATA, "vrs_status.json")
    prev = _read_json(vpath) or {}
    old_status = _read_json(spath) or {}
    try:
        out, cad, newest = fetch_standings(get, paths)
        if prev.get("date") and out["date"] < prev["date"]:
            raise ValueError(f"newest published standings {out['date']} are older than the stored "
                             f"{prev['date']}; keeping the stored ones")
        same = prev.get("date") == out["date"] and prev.get("standings") == out["standings"]
        if not same:
            _write_if_changed(vpath, _dump(out))
        nxt = ((date.fromisoformat(newest) + timedelta(days=cad["typical"])).isoformat()
               if cad["typical"] else None)
        status = {"ok": True, "stored": prev["date"] if same else out["date"], "published": newest,
                  "typical_gap_days": cad["typical"], "stale_after_days": cad["stale_after"],
                  "expected_next": nxt, "error": None, "failing_since": None}
    except Exception as e:   # noqa: BLE001  (network, HTTP, parse: keep last good, say so)
        since = old_status.get("failing_since") if old_status.get("ok") is False else None
        status = dict(old_status)
        status.update({"ok": False, "stored": prev.get("date"), "error": f"{type(e).__name__}: {e}",
                       "failing_since": since or now.strftime("%Y-%m-%dT%H:%M:%SZ")})
        status.setdefault("published", None)
        status.setdefault("stale_after_days", DEFAULT_STALE_AFTER)
        print(f"WARNING VRS refresh failed ({status['error']}); keeping the stored standings "
              f"({prev.get('date')})", file=sys.stderr)
    _write_if_changed(spath, _dump(status) + "\n")
    return status


def main():
    os.makedirs(CACHE, exist_ok=True)
    if "--check" in sys.argv:
        stored = (_read_json(os.path.join(DATA, "vrs.json")) or {}).get("date")
        problems = assess(stored, _read_json(os.path.join(DATA, "vrs_status.json")),
                          datetime.now(timezone.utc).date())
        for p in problems:
            print("VRS FRESHNESS:", p, file=sys.stderr)
        print("VRS standings are current" if not problems else f"{len(problems)} problem(s)")
        sys.exit(1 if problems else 0)
    if OFFLINE:
        cached = sorted(f for f in os.listdir(CACHE) if f.startswith("standings_global_"))
        if not cached:
            raise SystemExit("offline: no cached VRS standings")
        d = cached[-1][len("standings_global_"):-3]
        paths = {r: f"standings_{r}_{d}.md" for r in REGIONS
                 if os.path.exists(os.path.join(CACHE, f"standings_{r}_{d}.md"))}
        prev = _read_json(os.path.join(DATA, "vrs.json")) or {}
        out = {"date": d.replace("_", "-"), "source": f"https://github.com/{REPO}",
               "tree_sha": prev.get("tree_sha"), "standings": {}}
        for region, path in paths.items():
            with open(os.path.join(CACHE, path), encoding="utf-8") as f:
                out["standings"][region] = parse_table(f.read())
        _write_if_changed(os.path.join(DATA, "vrs.json"), _dump(out))
        print(f"VRS {out['date']}: " + ", ".join(f"{r} {len(v)} teams" for r, v in out["standings"].items()))
        return
    status = refresh()
    print(f"VRS stored {status.get('stored')}, newest published by Valve {status.get('published')}"
          + ("" if status["ok"] else f" (REFRESH FAILED: {status['error']})"))
    if not status["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
