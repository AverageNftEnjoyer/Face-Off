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

Nothing is pre-rendered: the page asks api/notes_api.py for the notes it
shows, and that endpoint runs notes.get_notes on this snapshot (written to
viewer/notes-inputs.json, versioned by inputs_version; decision record:
docs/notes-contracts.md section 4).
"""
import hashlib
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
SUMMARIES_PATH = os.path.join(paths.DATA, "notes_summaries.json")
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
    # team pages count rest to the team's next fixture within next_match_days
    nxt = {}
    hi = (as_of + timedelta(days=NT.CONFIG["next_match_days"])).isoformat()
    for d, ev, a, bb in sorted(fx):
        if as_of.isoformat() <= d <= hi:
            nxt.setdefault(a, {"day": d, "event": ev, "opp": bb})
            nxt.setdefault(bb, {"day": d, "event": ev, "opp": a})
    sec["next"] = {t: nxt[t] for t in sorted(nxt) if t in data["teams"]}
    # AI summaries of the model's calls (collectors/notes_summarize.py), for this build's matches
    ids = {m["id"] for e in data["events"] for m in e["matches"]}
    sums = (_load(SUMMARIES_PATH, {}) or {}).get("summaries") or {}
    sec["summaries"] = {k: {f: v[f] for f in ("headline", "summary", "favourite", "win_prob_a", "team_a", "team_b",
                                               "model", "generated_at")}
                        for k, v in sorted(sums.items()) if k in ids}
    sec["inputs_version"] = inputs_version(sec)
    return sec


def inputs_version(sec):
    """sha1-12 of the snapshot's canonical JSON (its id in every notes URL)."""
    body = {k: v for k, v in sec.items() if k != "inputs_version"}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def write_inputs(sec, path):
    """The snapshot api/notes_api.py serves from (canonical JSON, one line)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(sec, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def page_payload(section):
    """What the page inlines (NOTES): no notes, only what it needs to ask
    the endpoint for them (snapshot version, freshness, next fixtures)."""
    return {"v": section["inputs_version"], "as_of": section["as_of"], "freshness": section["freshness"],
            "next": section.get("next") or {}, "api": "/api/notes_api", "sum": section.get("summaries") or {}}
