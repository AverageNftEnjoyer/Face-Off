#!/usr/bin/env python3
"""
notes_data.py -- the bundle's `notes` section (docs/bundle-schema.md, "notes"):
the inputs the notes engine (model/notes.py) reads, the web facts, the
freshness signal, and the notes pre-rendered for every team page and every
event match. Called by build_bundle.py; reads data files, so it is the I/O
half of the notes feature (notes.py itself reads nothing).

Inputs (all point-in-time filtered by the engine, never here):
  series    data/matches.json (S/A), one row per series with its stage
  lineups   data/lineups.json (who played each series) + Liquipedia squads
            (data/event_lineups.json) for tenure.py
  overrides data/roster_overrides.json (the engine reads them only for the present)
  facts     data/notes_facts.json (collectors/notes_fetch.py), cleaned again

The page reads only the pre-rendered notes; the tool (notes.get_notes) reads
the inputs. Both go through the same engine functions.
"""
import json
import os
import re
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import backtest as bt  # noqa: E402
import lineup_features as LF  # noqa: E402
import notes as NT  # noqa: E402
import notes_facts as NF  # noqa: E402

FACTS_PATH = os.path.join(paths.DATA, "notes_facts.json")
PAGE_KEYS = ("id", "rule", "scope", "team_scope", "kind", "headline", "evidence", "source", "source_url",
             "observed_at", "age_days")


def _load(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def series_rows(matches, events):
    """[date, event, stage, team_a, team_b, winner, score] for every S/A
    series; the stage comes from the tracked event listing the series (2026
    events; older series have none)."""
    stage_of = {}
    for e in events:
        for m in e["matches"]:
            if m.get("t1") and m.get("t2") and m.get("day"):
                stage_of.setdefault((m["day"], frozenset((m["t1"], m["t2"]))), []).append((e["name"], m.get("stage")))
    rows = []
    for m in sorted(matches, key=lambda m: (m["date"], m["team_a"], m["team_b"])):
        got = stage_of.get((m["date"], frozenset((m["team_a"], m["team_b"]))), [])
        st = next((s for name, s in got if name == m["event"]), got[0][1] if len(got) == 1 else None)
        rows.append([m["date"], m["event"], st, m["team_a"], m["team_b"], m["winner"], bt.actual_scoreline(m)])
    return rows


def live_cutoff(as_of, facts):
    """The information cutoff of 'now': the start of as_of, or just after the
    last fact run when that is later (facts observed at a run are visible)."""
    c = NF.cutoff_of(as_of.isoformat())
    up = facts.get("updated_at")
    if up:
        c = max(c, NF.parse_ts(up) + timedelta(seconds=1))
    return NF.fmt_ts(c)


def page_note(n):
    return {k: n[k] for k in PAGE_KEYS}


def section(as_of, matches, events, data_teams, players, fixtures, lineups, squads, overrides, facts,
            aliases=None):
    """The engine half of the `notes` section from already-loaded inputs (no
    I/O; evaluation/leakage_check.py rebuilds it from scrambled copies)."""
    teams = sorted(set(data_teams) | {t for m in matches for t in (m["team_a"], m["team_b"])})
    names = {}
    for pid, p in sorted(players.items()):
        names.setdefault(_norm(p.get("id") or pid), p.get("id") or pid)
    form, news, bad = NF.clean(facts, set(teams))
    if bad:
        print(f"notes: {len(bad)} malformed facts dropped: {bad[:3]}")
    return {
        "schema": {"note": NT.NOTE_SCHEMA, "fact": NF.FACT_SCHEMA, "inputs": NT.INPUTS_SCHEMA},
        "tool_version": NT.TOOL_VERSION,
        "as_of": as_of.isoformat(),
        "live_cutoff": live_cutoff(as_of, facts),
        "config": dict(NT.CONFIG),
        "rules": list(NT.RULES),
        "inputs": {
            "series": series_rows(matches, events),
            "lineups": {k: [v.get("a"), v.get("b")] for k, v in sorted(lineups.items())},
            "squads": squads, "overrides": overrides, "teams": teams, "names": names,
            # short names a caller may use ("nip", "9z"): alias -> team, from data/team_aliases.json
            "aliases": {k: t for t, ks in sorted((aliases or {}).items()) if t in teams for k in ks},
            "fixtures": [[f["day"], f["event"], f["team_a"], f["team_b"]] for f in fixtures if f.get("day")],
        },
        "facts": {"fact_schema": NF.FACT_SCHEMA, "updated_at": facts.get("updated_at"), "form": form, "news": news},
    }


def load_inputs(facts_path=None):
    """(matches, lineups, squads, overrides, facts) from data/."""
    matches = bt.load_matches()
    lineups = _load(os.path.join(paths.DATA, "lineups.json"), {}).get("lineups", {})
    squads = _load(os.path.join(paths.DATA, "event_lineups.json"), {}).get("squads", {})
    overrides = [{"team": o["team"], "kind": o["kind"], "from": str(o["from"]),
                  "players": list(o["players"])} for o in LF.load_overrides()]
    facts = _load(facts_path or FACTS_PATH, {"fact_schema": NF.FACT_SCHEMA, "updated_at": None,
                                             "form": [], "news": []})
    return matches, lineups, squads, overrides, facts


def build(data, fixtures, facts_path=None):
    """The bundle's `notes` section. `data`/`fixtures` are the bundle's own."""
    as_of = bt._d(data["as_of"])
    matches, lineups, squads, overrides, facts = load_inputs(facts_path)
    aliases = _load(os.path.join(paths.DATA, "team_aliases.json"), {}).get("aliases", {})
    sec = section(as_of, matches, data["events"], data["teams"], data["players"], fixtures,
                  lineups, squads, overrides, facts, aliases)
    fx = sec["inputs"]["fixtures"]
    b = {"notes": sec}
    inp = NT.inputs_of(b)
    cut = sec["live_cutoff"]
    sec["freshness"] = NT.freshness(inp, cut)
    # team pages: as of now, rest counted to the team's next fixture within the scope window
    nxt = {}
    hi = (as_of + timedelta(days=NT.CONFIG["next_match_days"])).isoformat()
    for d, ev, a, bb in sorted(fx):
        if as_of.isoformat() <= d <= hi:
            nxt.setdefault(a, {"day": d, "event": ev, "opp": bb})
            nxt.setdefault(bb, {"day": d, "event": ev, "opp": a})
    tn = {}
    for t in sorted(data["teams"]):
        n = nxt.get(t)
        r = NT.team_notes(t, cut, b, n["day"] if n else None, n["event"] if n else None)
        tn[t] = {"notes": r["notes"], "omitted": r["omitted"], "next": n}
    sec["teams"] = tn
    # every event match: a played one as of its own day, an unplayed one as of now
    mn = {}
    for e in data["events"]:
        for m in e["matches"]:
            if not (m.get("t1") and m.get("t2") and m.get("day")) or m["t1"] == m["t2"]:
                continue
            asof = m["day"] if m.get("finished") else cut
            r = NT.get_notes(m["t1"], m["t2"], asof, b, match_day=m["day"], event=e["name"])
            if r["ok"]:
                mn[m["id"]] = {"asof": asof, "notes": r["notes"], "omitted": r["omitted"]}
    sec["matches"] = mn
    return sec


def page_payload(section):
    """The compact copy the page inlines (NOTES): no inputs, no facts."""
    return {
        "as_of": section["as_of"], "freshness": section["freshness"],
        "teams": {t: {"n": [page_note(n) for n in v["notes"]], "next": v["next"]}
                  for t, v in section["teams"].items()},
        "matches": {k: [page_note(n) for n in v["notes"]] for k, v in section["matches"].items()},
    }
