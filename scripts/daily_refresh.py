#!/usr/bin/env python3
"""
Daily data refresh for the Faceoff hub (run by .github/workflows/daily-refresh.yml).

1. Discover new tournament pages: Liquipedia `allpages` listings for the
   tracked series, this year and next; main events and their stage pages are
   appended to data/raw/lp_titles_selected.txt (qualifiers and regional
   sub-events are skipped).
2. Mark pages that can still change as stale for this run: every live or
   upcoming tournament, anything that ended in the last REFRESH_DAYS days,
   their stage pages, and the team pages of every team in them (rosters,
   stand-ins, logos). Everything else keeps being served from the cache.
3. Rebuild, in order: data/matches.json (collect_liquipedia), data/
   roster_events.json (rosters), viewer/assets.json + new images
   (fetch_assets), data/vrs.json (Valve Regional Standings, from GitHub).

The site itself is built from these files on each deploy
(python viewer/build_viewer.py public/index.html, see vercel.json).

All requests go through data/lpfetch.py: descriptive User-Agent, gzip, at most
one request every 2.5 s, no action=parse. A typical day is 15-40 requests.

USAGE (from the repo root):  python scripts/daily_refresh.py [--dry-run]
"""

import os
import re
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "viewer"))

import lpfetch  # noqa: E402

REFRESH_DAYS = 3
SERIES = ["Intel Extreme Masters/{y}", "BLAST/Premier/{y}", "BLAST/Open/{y}", "BLAST/Rivals/{y}",
          "BLAST/Bounty/{y}", "PGL/{y}", "StarLadder/StarSeries/{y}", "Thunderpick/World Championship/{y}",
          "Esports World Cup/{y}", "Perfect World/{y}"]
ESL_PL = "ESL/Pro League/Season "
# regional qualifier sub-pages ("PGL/2026/Astana/Europe"); country-named
# events such as "IEM China" are NOT skipped
SKIP_SEGMENTS = {"Qualifier", "Qualifiers", "Open", "Asia", "Europe", "North America", "South America",
                 "Oceania", "East Asia", "West Asia", "Americas", "Global"}
TITLES_FILE = os.path.join(ROOT, "data", "raw", "lp_titles_selected.txt")


def wanted(title, series_root):
    """Main events and their stage pages; not qualifiers or regional sub-events."""
    rest = title[len(series_root):].strip("/")
    segs = [s for s in rest.split("/") if s]
    if not segs:
        return False
    if any("Qualifier" in s for s in segs):
        return False
    # regions can be the first segment when the year is the event itself
    # ("Thunderpick/World Championship/2026/North America")
    if any(s in SKIP_SEGMENTS for s in segs) or any(s.isdigit() for s in segs[1:]):
        return False
    return len(segs) <= 2          # "<event>" or "<event>/<stage>"


def discover(today):
    selected = [l.strip() for l in open(TITLES_FILE, encoding="utf-8") if l.strip()]
    have = set(selected)
    new = []
    for y in (today.year, today.year + 1):
        for pat in SERIES:
            root = pat.format(y=y)
            for t in lpfetch.prefix(root, refresh=True):
                if t not in have and wanted(t, root):
                    new.append(t)
                    have.add(t)
    # ESL Pro League seasons: only seasons after the newest one already tracked
    seasons = [int(m.group(1)) for t in selected for m in [re.match(re.escape(ESL_PL) + r"(\d+)$", t)] if m]
    newest = max(seasons) if seasons else 0
    for t in lpfetch.prefix(ESL_PL, refresh=True):
        m = re.match(re.escape(ESL_PL) + r"(\d+)(/.*)?$", t)
        if m and int(m.group(1)) > newest and t not in have:
            sub = (m.group(2) or "").strip("/")
            if not sub or (sub.count("/") == 0 and sub not in SKIP_SEGMENTS and "Qualifier" not in sub):
                new.append(t)
                have.add(t)
    return selected, new


def stale_titles(today):
    """Event pages and team pages that may have changed since they were cached."""
    import events as E
    evs = E.discover()
    cutoff = (today - timedelta(days=REFRESH_DAYS)).isoformat()
    titles, teams = set(), set()
    for e in evs:
        if e["end"] >= cutoff:
            titles.add(e["title"])
            teams |= set(e["participants"])
            teams |= {m["t1"] for m in e["matches"] if m["t1"]} | {m["t2"] for m in e["matches"] if m["t2"]}
    selected = [l.strip() for l in open(TITLES_FILE, encoding="utf-8") if l.strip()]
    titles |= {s for s in selected for t in list(titles) if s.startswith(t + "/")}
    return titles, teams


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dry = "--dry-run" in argv
    today = date.today()

    # stale pages come from the CURRENT cache, before new titles are appended
    titles, teams = stale_titles(today)
    selected, new = discover(today)
    print(f"tracked tournament pages: {len(selected)}; new: {len(new)}")
    for t in new:
        print("  +", t)
    if new and not dry:
        with open(TITLES_FILE, "a", encoding="utf-8") as f:
            f.write("".join(t + "\n" for t in new))
    titles |= set(new)
    print(f"refreshing {len(titles)} tournament pages and {len(teams)} team pages")
    if dry:
        for t in sorted(titles):
            print("  ~", t)
        return
    lpfetch.FRESH_TITLES |= titles | teams

    import collect_liquipedia
    collect_liquipedia.OFFLINE = False
    collect_liquipedia.main()

    import rosters
    rosters.OFFLINE = False
    rosters.main()

    import fetch_assets
    fetch_assets.OFFLINE = False
    fetch_assets.main()

    import vrs          # Valve Regional Standings (GitHub; new file roughly monthly)
    vrs.OFFLINE = False
    vrs.main()
    print("refresh done")


if __name__ == "__main__":
    main()
