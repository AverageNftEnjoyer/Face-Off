#!/usr/bin/env python3
"""
roster_overrides_check.py -- validate data/roster_overrides.json.

Checks every entry (team, from, players[5], kind, optional until/note) and LISTS
every player or team name it cannot find in the data, with the closest known
spellings, instead of ignoring it. Known players come from data/lineups.json,
data/event_lineups.json (squads and pages) and data/roster_events.json (lower-
cased, as lineups store them); known teams from data/matches.json, lineups keys
and data/team_aliases.json.

Usage:  python scripts/roster_overrides_check.py [--file PATH]
Exit 0 clean, 1 structural errors, 2 only unknown names (confirm the spelling,
or that the player is genuinely new, then re-run).
"""
import argparse
import difflib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import lineup_features as LF  # noqa: E402

DATA = os.path.join(ROOT, "data")


def _json(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def known_players():
    out = set()
    for rec in ((_json("lineups.json") or {}).get("lineups") or {}).values():
        for side in ("a", "b"):
            out.update(LF._norm(p) for p in (rec.get(side) or []))
    ev = _json("event_lineups.json") or {}
    for squad in (ev.get("squads") or {}).values():
        out.update(LF._norm(r[0]) for r in squad)
    for page in (ev.get("pages") or {}).values():
        for five in page.values():
            out.update(LF._norm(p) for p in five)
    for events in (_json("roster_events.json") or {}).values():
        for e in events.values():
            out.update(LF._norm(p[0]) for p in e.get("players", []))
    return out


def known_teams():
    out = {m["team_a"] for m in (_json("matches.json") or [])}
    out |= {m["team_b"] for m in (_json("matches.json") or [])}
    for k in ((_json("lineups.json") or {}).get("lineups") or {}):
        out.update(k.split("|")[1:])
    out |= set((_json("team_aliases.json") or {}).get("aliases", {}))
    return out


def check(raw, players, teams):
    """-> (errors, unknown): lists of strings."""
    errors, unknown = [], []
    if not isinstance(raw, list):
        return ["file must be a JSON list"], unknown
    seen = {}
    for i, e in enumerate(raw):
        tag = f"entry {i}"
        if not isinstance(e, dict):
            errors.append(f"{tag}: not an object")
            continue
        tag += f" ({e.get('team')!r} from {e.get('from')!r})"
        miss = [k for k in ("team", "from", "players", "kind") if k not in e]
        if miss:
            errors.append(f"{tag}: missing {miss}")
            continue
        if e["kind"] not in LF.KINDS:
            errors.append(f"{tag}: kind {e['kind']!r} not in {list(LF.KINDS)}")
        o = LF.parse_override(e)
        if o is None:
            if e["kind"] in LF.KINDS:
                errors.append(f"{tag}: malformed (need ISO dates, until after from, "
                              f"exactly 5 distinct non-empty player names)")
            continue
        if (o["team"], o["from"]) in seen:
            errors.append(f"{tag}: duplicate of entry {seen[(o['team'], o['from'])]}")
        seen[(o["team"], o["from"])] = i
        if teams and o["team"] not in teams:
            near = difflib.get_close_matches(o["team"], sorted(teams), n=3, cutoff=0.6)
            unknown.append(f"{tag}: unknown team {o['team']!r}; closest: {near or 'none'}")
        for p in o["players"]:
            if players and p not in players:
                near = difflib.get_close_matches(p, sorted(players), n=3, cutoff=0.6)
                unknown.append(f"{tag}: unknown player {p!r}; closest: {near or 'none'}")
    return errors, unknown


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=LF.OVERRIDES_PATH)
    a = ap.parse_args(argv)
    try:
        with open(a.file, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as ex:
        print(f"ERROR cannot read {a.file}: {ex}")
        return 1
    errors, unknown = check(raw, known_players(), known_teams())
    n = len(raw) if isinstance(raw, list) else 0
    print(f"{a.file}: {n} entries")
    for s in errors:
        print("ERROR  ", s)
    for s in unknown:
        print("UNKNOWN", s)
    print(f"{len(errors)} error(s), {len(unknown)} unknown name(s)")
    return 1 if errors else (2 if unknown else 0)


if __name__ == "__main__":
    sys.exit(main())
