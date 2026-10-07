#!/usr/bin/env python3
"""
lineup_features.py -- point-in-time lineup features from data/lineups.json.

CUTOFF: a series on day D sees only lineups of series dated strictly before
D (the same rule as every other backtest feature). A stand-in who first
plays on day D is invisible to day D's prediction.

Per team, from its lineups before D (newest first):
  last        the five of its most recent series
  core        the five players with the most appearances in its last CORE_N
              lineups (ties: most recent appearance first)
  standin     1 if `last` contains a player outside `core`, else 0
  continuity  share of `last` who played in >= half of the team's series in
              the CONT_DAYS days before D (1.0 = the same five all along)
No lineup history -> standin 0, continuity 1.0, known False (neutral).
"""
import json
import os
from collections import defaultdict
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_N = 8
CONT_DAYS = 90


def load(path=None):
    path = path or os.path.join(HERE, "data", "lineups.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("lineups", {})


def team_history(matches, lineups):
    """team -> [(date, five)] in date order, from series that have a lineup."""
    hist = defaultdict(list)
    for m in sorted(matches, key=lambda m: m["date"]):
        rec = lineups.get(f"{m['date']}|{m['team_a']}|{m['team_b']}")
        if not rec:
            continue
        for side, team in (("a", m["team_a"]), ("b", m["team_b"])):
            if rec.get(side):
                hist[team].append((m["date"], tuple(rec[side])))
    return hist


def features(hist_rows, day):
    """hist_rows: [(date_str, five)] for ONE team; uses only dates < day."""
    prior = [(d, f) for d, f in hist_rows if d < day]
    if not prior:
        return {"standin": 0, "continuity": 1.0, "known": False}
    last = prior[-1][1]
    recent = prior[-CORE_N:]
    count, seen = defaultdict(int), {}
    for i, (_, five) in enumerate(recent):
        for p in five:
            count[p] += 1
            seen[p] = i
    core = set(sorted(count, key=lambda p: (-count[p], -seen[p]))[:5])
    lo = (date.fromisoformat(day) - timedelta(days=CONT_DAYS)).isoformat()
    window = [f for d, f in prior if d >= lo] or [last]
    share = {p: sum(p in f for f in window) / len(window) for p in last}
    return {"standin": int(any(p not in core for p in last)),
            "continuity": sum(1 for p in last if share[p] >= 0.5) / 5.0,
            "known": True}
