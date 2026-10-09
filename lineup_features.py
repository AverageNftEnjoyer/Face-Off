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

MANUAL OVERRIDES (data/roster_overrides.json, see load_overrides): entries
{team, from, players[5], kind, [until], [note]} replace Valve's lineup for that
team on series dated >= `from` (and < `until` when given), and add one row at
`from` itself so the move is known before the team next plays. Point-in-time:
a row dated F is only ever read by days D > F, so an override dated at or
after D cannot change anything on day D. kind:
  roster_change  permanent; the new five is ordinary history (core and
                 continuity adapt as the old five ages out).
  stand_in       temporary; the row is marked, left out of `core`, so
                 standin is 1 and continuity drops for its players.
No team or player names live in this code; everything is in the file.
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


KINDS = ("roster_change", "stand_in")
OVERRIDES_PATH = os.path.join(HERE, "data", "roster_overrides.json")


class Five(tuple):
    """A lineup tuple; `temporary` marks a stand_in override row."""
    temporary = False


def _norm(name):
    return str(name).strip().lower()


def parse_override(e):
    """One raw entry -> normalised dict, or None when it is malformed
    (scripts/roster_overrides_check.py reports why)."""
    try:
        if e["kind"] not in KINDS or not str(e["team"]).strip():
            return None
        frm = date.fromisoformat(e["from"]).isoformat()
        until = e.get("until")
        until = date.fromisoformat(until).isoformat() if until else None
        players = [_norm(p) for p in e["players"]]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if len(players) != 5 or len(set(players)) != 5 or "" in players:
        return None
    if until and until <= frm:
        return None
    return {"team": e["team"], "from": frm, "until": until, "players": players,
            "kind": e["kind"], "note": e.get("note", "")}


def load_overrides(path=None):
    """Valid entries of data/roster_overrides.json (a JSON list), by date."""
    path = path or OVERRIDES_PATH
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = [o for o in (parse_override(e) for e in raw) if o]
    return sorted(out, key=lambda o: (o["from"], o["team"]))


def _governing(ovs, day):
    """Latest override with from <= day < until (None if none)."""
    hit = None
    for o in ovs:  # sorted by `from`
        if o["from"] <= day and (not o["until"] or day < o["until"]):
            hit = o
    return hit


def _five(o):
    f = Five(o["players"])
    f.temporary = o["kind"] == "stand_in"
    return f


def team_history(matches, lineups, overrides=None):
    """team -> [(date, five)] in date order, from series that have a lineup.

    overrides: list from load_overrides(); None reads the file, [] disables.
    """
    if overrides is None:
        overrides = load_overrides()
    by_team = defaultdict(list)
    for o in overrides:
        by_team[o["team"]].append(o)
    hist = defaultdict(list)
    for m in sorted(matches, key=lambda m: m["date"]):
        rec = lineups.get(f"{m['date']}|{m['team_a']}|{m['team_b']}") or {}
        for side, team in (("a", m["team_a"]), ("b", m["team_b"])):
            o = _governing(by_team.get(team, ()), m["date"])
            if o:  # override wins over Valve for this team on this date
                hist[team].append((m["date"], _five(o)))
            elif rec.get(side):
                hist[team].append((m["date"], tuple(rec[side])))
    for team, ovs in by_team.items():
        for o in ovs:  # the move itself, known before the team next plays
            if not any(d == o["from"] for d, _ in hist[team]):
                hist[team].append((o["from"], _five(o)))
        hist[team].sort(key=lambda r: r[0])
    return hist


def features(hist_rows, day):
    """hist_rows: [(date_str, five)] for ONE team; uses only dates < day."""
    prior = [(d, f) for d, f in hist_rows if d < day]
    if not prior:
        return {"standin": 0, "continuity": 1.0, "known": False}
    last = prior[-1][1]
    recent = prior[-CORE_N:]
    count, seen = defaultdict(int), {}
    base = [r for r in recent if not getattr(r[1], "temporary", False)] or recent
    for i, (_, five) in enumerate(base):
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
