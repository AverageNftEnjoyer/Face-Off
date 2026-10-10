#!/usr/bin/env python3
"""Review and promote the roster candidate.

The daily refresh never rewrites data/roster_events.json (stand-ins and missing
IGLs feed every prediction, so a misread page must not go live unreviewed). It
builds a candidate, diffs it against the live file and opens a "roster-review"
issue. The candidate itself is not committed (.gitignore); this script rebuilds
it from the committed Liquipedia cache, shows the diff, and promotes it on
request.

USAGE (repo root):
  python scripts/promote_rosters.py            show the diff only (changes nothing)
  python scripts/promote_rosters.py --yes      promote: candidate -> data/roster_events.json
  python scripts/promote_rosters.py --keep-candidate   leave the candidate file in data/

Run it after `git pull`, review the diff, then commit data/roster_events.json.
Offline: it reads only the cache already in the repo. Stdlib only.
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, DATA)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

LIVE = os.path.join(DATA, "roster_events.json")
CAND = os.path.join(DATA, "roster_events.candidate.json")


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    import rosters
    import roster_watch
    rosters.OFFLINE = True
    rosters.CANDIDATE = True
    rosters.main()
    live, cand = _load(LIVE), _load(CAND)
    if cand is None:
        print("could not build a candidate from the cache", file=sys.stderr)
        return 2
    diffs = roster_watch.diff_events(live, cand)
    if not diffs:
        print("candidate matches the live roster file: nothing to promote")
        return 0
    print(f"{len(diffs)} difference(s) between the live file and the candidate:")
    for kind, team, tour, detail in diffs:
        print(f"  {kind:8} {team} @ {tour}: {detail}")
    if "--yes" not in argv:
        print("\nreview the list above, then run again with --yes to promote it")
        return 1
    shutil.copyfile(CAND, LIVE)
    if "--keep-candidate" not in argv:
        os.remove(CAND)
    print(f"\npromoted: {os.path.relpath(LIVE, ROOT)} updated; commit it to publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
