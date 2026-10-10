#!/usr/bin/env python3
"""Supervised roster diff: compare the candidate roster events
(data/roster_events.candidate.json, built by `python data/rosters.py --candidate`)
against the live, human-confirmed data/roster_events.json.

Roster changes are never auto-applied: a stand-in the scraper misreads would
corrupt every downstream prediction. This script reports the diff; the workflow
opens a GitHub issue with the report, and a human promotes the candidate only
after review (python scripts/promote_rosters.py --yes).

USAGE (repo root):  python scripts/roster_watch.py [--report PATH]
Exit 0 = candidate matches live (or no live file yet and none needed).
Exit 1 = differences found; report written to --report (default
         data/roster_diff.md).
Stdlib only. Never touches the network.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LIVE = os.path.join(DATA, "roster_events.json")
CAND = os.path.join(DATA, "roster_events.candidate.json")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def diff_events(live, cand):
    """[(kind, team, tournament, detail)] — kind in added/removed/changed."""
    live = live or {}
    cand = cand or {}
    out = []
    for team in sorted(set(live) | set(cand)):
        lt, ct = live.get(team, {}), cand.get(team, {})
        for tour in sorted(set(lt) | set(ct)):
            l, c = lt.get(tour), ct.get(tour)
            if l is None:
                out.append(("added", team, tour, fmt(c)))
            elif c is None:
                out.append(("removed", team, tour, fmt(l)))
            elif l != c:
                out.append(("changed", team, tour, f"{fmt(l)}  ->  {fmt(c)}"))
    return out


def fmt(ev):
    if not ev:
        return "-"
    players = ", ".join(f"{s} for {r}" for s, r in ev.get("players", []))
    igl = " (covers IGL)" if ev.get("missing_igl") else ""
    return f"stand-in: {players}{igl}" if players else "stand-in recorded"


def report_md(diffs):
    lines = ["# Roster watch: candidate differs from live",
             "",
             "The roster scraper found changes. **Do not auto-apply.** Review each",
             "entry, then promote with: `python scripts/promote_rosters.py --yes`",
             "(it rebuilds the candidate from the committed cache), and commit",
             "data/roster_events.json.",
             "",
             f"{len(diffs)} difference(s):", ""]
    for kind, team, tour, detail in diffs:
        lines.append(f"- **{kind}** `{team}` @ `{tour}`: {detail}")
    lines += ["",
              "Candidate file (not committed): `data/roster_events.candidate.json`",
              "Live file: `data/roster_events.json`"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    report_path = argv[argv.index("--report") + 1] if "--report" in argv else \
        os.path.join(DATA, "roster_diff.md")
    live, cand = load(LIVE), load(CAND)
    if cand is None:
        print("no candidate file; run `python data/rosters.py --candidate` first",
              file=sys.stderr)
        return 2
    if live is None:
        print("no live roster_events.json yet; candidate becomes the baseline on first promote")
        return 0
    diffs = diff_events(live, cand)
    if not diffs:
        print("roster candidate matches live: no changes to review")
        if os.path.exists(report_path):
            os.remove(report_path)
        return 0
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md(diffs))
    print(f"{len(diffs)} roster difference(s); report at {report_path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
