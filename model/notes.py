#!/usr/bin/env python3
"""
notes.py -- the matchup notes engine: deterministic, evidence-carrying notes
on a team or a matchup ("short rest", "five together 214 days", "playing
hot"), and the tool interface (get_notes / get_notes_batch / team_notes) an
LLM or the site calls. Contracts: docs/notes-contracts.md. Stdlib only.

CONTEXT, NOT SIGNAL
  No note moves a probability: nothing here is read by predictor.py,
  backtest.py or any fit, and nothing here reads them. (The tenure lesson:
  roster tenure failed its held-out test, so it is display context only.)

PURE
  Every function below takes the bundle's `notes` section and returns JSON;
  no file, network or clock is read (the CLI at the bottom reads the bundle
  file and nothing else). Same bundle in, same notes out, byte for byte.

POINT IN TIME (evaluation/leakage_check.py, test 8)
  asof is the information cutoff: a series is visible only when dated before
  the cutoff's day, a lineup row likewise, a web fact only when observed
  before the cutoff instant. The cutoff is never later than the start of
  match_day. Hand-entered roster overrides (entered with
  hindsight) are read only for cutoffs on or after the bundle's as_of, i.e.
  for the present, never for history.

PER TEAM, THEN ASSEMBLED
  Team notes are computed per team (O(teams)); a matchup is team A's notes +
  the head-to-head notes + team B's notes, ranked and capped (O(1)).

Thresholds live in CONFIG; notes_eval.py reports how often each rule fires.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

import notes_facts as NF  # noqa: E402
import tenure as TN  # noqa: E402

TOOL_NAME = "faceoff_get_notes"
TOOL_VERSION = "1.0.0"
NOTE_SCHEMA = 1
INPUTS_SCHEMA = 1

# ---- thresholds (notes_eval.py reports each rule's hit rate) ---------------
CONFIG = {
    "short_rest_days": 4,        # SHORT_REST: last official at most this many days before the match
    "heavy_window_days": 14,     # HEAVY_SCHEDULE window
    "heavy_min_series": 8,       # HEAVY_SCHEDULE: at least this many officials in the window
    "cold_rest_days": 21,        # COLD_REST: no official for at least this long
    "tenure_stable_days": 180,   # TENURE_STABLE: the five together at least this long
    "tenure_new_days": 90,       # TENURE_NEW window (tenure.GELL_DAYS); TENURE_STABLE needs no arrival in it
    "new_addition_days": 45,     # NEW_ADDITION: a single arrival this recent
    "tenure_new_min": 2,         # TENURE_NEW: at least this many of the five joined in the window
    "form_delta": 0.10,          # HOT_PLAYER / COLD_PLAYER: |rating_last30 - baseline| at least this
    "form_min_maps": 5,          # ... over at least this many maps
    "streak_min": 4,             # STREAK: at least this many straight officials won or lost
    "revenge_max_days": 90,      # REVENGE: the last meeting at most this long ago
    "max_matchup_notes": 6,
    "max_team_notes": 6,
    "max_per_rule_team": 2,      # at most this many notes of one rule about one team
    "scope_days": 3,             # freshness: teams with a fixture within this many days (notes_fetch.SCOPE_DAYS)
    "next_match_days": 7,        # team pages: rest counted to the next fixture within this many days
    "stale_after_days": 7,       # freshness: oldest fact older than this -> stale
}

# fixed importance order (docs/notes-contracts.md); rank = position + 1
RULES = ("STAND_IN", "SHORT_REST", "COLD_REST", "TENURE_NEW", "NEW_ADDITION", "REVENGE",
         "HOT_PLAYER", "COLD_PLAYER", "STREAK", "HEAVY_SCHEDULE", "TENURE_STABLE", "ROSTER_RUMOR")
RANK = {r: i + 1 for i, r in enumerate(RULES)}

REQUIRED = ("notes.schema", "notes.as_of", "notes.live_cutoff", "notes.inputs.series",
            "notes.inputs.lineups", "notes.inputs.squads", "notes.inputs.teams",
            "notes.inputs.names", "notes.facts.form", "notes.facts.news")


class NotesError(Exception):
    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra

    def payload(self):
        return {"code": self.code, "message": self.message, **self.extra}


def _d(s):
    return s if isinstance(s, date) else date.fromisoformat(s)


def _days(n):
    return "today" if n == 0 else ("1 day ago" if n == 1 else f"{n} days ago")


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


# ---- the bundle's notes section, indexed once ------------------------------
class Inputs:
    """Index over bundle["notes"]: series by team, head-to-head, event dates,
    tenure (with and without the hand-entered overrides) and clean facts."""

    def __init__(self, notes):
        inp = notes["inputs"]
        self.as_of = _d(notes["as_of"])
        self.live_cutoff = notes["live_cutoff"]
        self.teams = list(inp["teams"])
        self.team_set = set(self.teams)
        self.names = dict(inp["names"])
        self.aliases = dict(inp.get("aliases") or {})
        self.by_team = defaultdict(list)      # team -> [(date, event, stage, opp, won, score)]
        self.h2h = defaultdict(list)          # (x, y) sorted -> [(date, event, stage, winner, score_x_y)]
        self.event_days = defaultdict(list)   # event -> sorted series dates
        for dt, ev, stage, a, b, winner, score in sorted(inp["series"], key=lambda r: (r[0], r[3], r[4])):
            d = _d(dt)
            rev = "-".join(reversed(score.split("-"))) if score else score
            self.by_team[a].append((d, ev, stage, b, winner == a, score))
            self.by_team[b].append((d, ev, stage, a, winner == b, rev))
            x, y = sorted((a, b))
            self.h2h[(x, y)].append((d, ev, stage, winner, score if x == a else rev))
            self.event_days[ev].append(d)
        self.dates = {t: [r[0] for r in rows] for t, rows in self.by_team.items()}
        for v in self.event_days.values():
            v.sort()
        lineups = {k: {"a": v[0], "b": v[1]} for k, v in inp["lineups"].items()}
        self.tenure_hist = TN.Tenure(lineups, inp["squads"], None, live=True, first_seen_censored=True)
        self.tenure_live = TN.Tenure(lineups, inp["squads"], inp.get("overrides") or [], live=True,
                                     first_seen_censored=True)
        teams = self.team_set
        self.form, self.news, self.rejected = NF.clean(notes.get("facts"), teams)
        self.facts_updated_at = (notes.get("facts") or {}).get("updated_at")
        self.fixtures = [tuple(f) for f in inp.get("fixtures") or []]

    def prior(self, team, day):
        """The team's series dated before `day`, oldest first."""
        return self.by_team.get(team, [])[:bisect_left(self.dates.get(team, []), day)]

    def player(self, norm):
        return self.names.get(norm, norm)


_CACHE = {}


def inputs_of(bundle):
    """Validated, cached Inputs for a bundle (dict) or its notes section."""
    notes = bundle.get("notes") if isinstance(bundle, dict) and "notes" in bundle else None
    missing = []
    for path in REQUIRED:
        cur = bundle
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                missing.append(path)
                break
            cur = cur[part]
    if missing:
        raise NotesError("MISSING_FIELDS", "The bundle lacks the notes fields: " + ", ".join(missing) + ".",
                         fields=missing)
    sch = notes["schema"]
    want = {"note": NOTE_SCHEMA, "fact": NF.FACT_SCHEMA, "inputs": INPUTS_SCHEMA}
    if any(sch.get(k) != v for k, v in want.items()):
        raise NotesError("BAD_SCHEMA", "The bundle's notes schema is not one this engine reads.",
                         found=sch, supported=want)
    hit = _CACHE.get(id(notes))
    if hit and hit[0] is notes:
        return hit[1]
    _CACHE.clear()                      # one bundle at a time; keeps memory flat
    inp = Inputs(notes)
    _CACHE[id(notes)] = (notes, inp)
    return inp


# ---- notes ----------------------------------------------------------------
def _note(rule, team, headline, evidence, values, observed, ref_day, source="matches",
          url=None, scope="team", kind="fact", mag=0.0):
    obs = _d(observed)
    nid = hashlib.sha1(json.dumps([rule, team, values], sort_keys=True, ensure_ascii=False)
                       .encode("utf-8")).hexdigest()[:12]
    return {"id": nid, "rule": rule, "scope": scope, "team_scope": team, "kind": kind,
            "headline": headline, "evidence": evidence, "values": values, "source": source,
            "source_url": url, "observed_at": obs.isoformat(), "age_days": max((ref_day - obs).days, 0),
            "rank": RANK[rule], "_mag": mag}


def _stage(ev, stage):
    return f"{ev} ({stage})" if stage else ev


def schedule_notes(inp, team, cutoff_day, match_day, event, ref_day):
    rows = inp.prior(team, cutoff_day)
    if not rows:
        return []
    out = []
    last = rows[-1]
    d, ev, stage = last[0], last[1], last[2]
    days = (match_day - d).days
    win = CONFIG["heavy_window_days"]
    n14 = sum(1 for r in rows if (match_day - r[0]).days <= win)
    ev_dates = [x for x in inp.event_days.get(ev, []) if x < cutoff_day]
    ev_end = max(ev_dates) if ev_dates else d
    ev_note = f"{ev} last series {ev_end.isoformat()}" if ev_end != d else ""
    base = {"last_date": d.isoformat(), "event": ev, "stage": stage, "days": days, "n14": n14,
            "event_last_series": ev_end.isoformat()}
    ev_line = " · ".join(x for x in (f"Last official: {d.isoformat()} vs {last[3]}",
                                     f"{_plural(n14, 'official')} in the last {win} days", ev_note) if x)
    if 0 <= days <= CONFIG["short_rest_days"] and (event is None or ev != event):
        out.append(_note("SHORT_REST", team, f"Short rest: last played {_stage(ev, stage)} {_days(days)}",
                         ev_line, base, d, ref_day, mag=-days))
    if days >= CONFIG["cold_rest_days"]:
        out.append(_note("COLD_REST", team, f"No officials in {days} days: possible rust",
                         f"Last official: {d.isoformat()} at {_stage(ev, stage)} vs {last[3]}",
                         base, d, ref_day, mag=days))
    if n14 >= CONFIG["heavy_min_series"]:
        first = min(r[0] for r in rows if (match_day - r[0]).days <= win)
        out.append(_note("HEAVY_SCHEDULE", team, f"Heavy schedule: {n14} officials in {win} days",
                         f"{n14} series from {first.isoformat()} to {d.isoformat()}",
                         base, d, ref_day, mag=n14))
    # STREAK: newest backwards while the result repeats
    won = rows[-1][4]
    n = 0
    for r in reversed(rows):
        if r[4] != won:
            break
        n += 1
    if n >= CONFIG["streak_min"]:
        seg = rows[-n:]
        opps = ", ".join(r[3] for r in reversed(seg[-3:]))
        out.append(_note("STREAK", team, f"{'Won' if won else 'Lost'} {n} straight officials",
                         f"Since {seg[0][0].isoformat()} · latest: {opps}",
                         {"result": "W" if won else "L", "n": n, "since": seg[0][0].isoformat(),
                          "last_date": d.isoformat()}, d, ref_day, mag=n))
    return out


def tenure_notes(inp, team, cutoff_day, ref_day):
    tn = inp.tenure_live if cutoff_day >= inp.as_of else inp.tenure_hist
    s = tn.state(team, cutoff_day)
    if not s["known"] or not s["five"]:
        return []
    out = []
    thru = s["thru"]
    five = [inp.player(p) for p in s["five"]]
    starts = {p: v[0] for p, v in s["starts"].items() if v[1] and v[0]}
    win = CONFIG["tenure_new_days"]
    new = sorted((p for p, st in starts.items() if (cutoff_day - _d(st)).days <= win),
                 key=lambda p: (starts[p], p))
    src = "lineups"
    if len(new) >= CONFIG["tenure_new_min"]:
        names = [f"{inp.player(p)} ({starts[p]})" for p in new]
        out.append(_note("TENURE_NEW", team,
                         f"New roster: {len(new)} players joined in the last {win} days",
                         "Joined: " + ", ".join(names) + f" · lineups through {thru}",
                         {"n": len(new), "players": [inp.player(p) for p in new],
                          "joined": [starts[p] for p in new], "lineups_thru": thru},
                         thru, ref_day, source=src, mag=len(new)))
    else:
        for p in new:
            n = (cutoff_day - _d(starts[p])).days
            if n > CONFIG["new_addition_days"]:
                continue
            out.append(_note("NEW_ADDITION", team, f"{inp.player(p)} joined {n} days ago",
                             f"First series with {team}'s current five: {starts[p]} · lineups through {thru}",
                             {"player": inp.player(p), "joined": starts[p], "days": n, "lineups_thru": thru},
                             thru, ref_day, source=src, mag=-n))
    since, broke, at_least = five_run(tn, team, cutoff_day, set(s["five"]))
    dt = (cutoff_day - since).days if since else None
    if dt is not None and dt >= CONFIG["tenure_stable_days"] and not new:
        lead = "At least " if at_least else ""
        why = (f"first series as this five after {', '.join(inp.player(p) for p in broke)}'s last"
               if broke else "as this five since the first lineup in the data")
        out.append(_note("TENURE_STABLE", team, f"{lead}{dt} days with the current five",
                         "Five: " + ", ".join(five) + f" · together since {since.isoformat()} ({why})"
                         + f" · no arrival in the last {win} days · lineups through {thru}",
                         {"days": dt, "five": five, "since": since.isoformat(), "at_least": at_least,
                          "lineups_thru": thru},
                         thru, ref_day, source=src, mag=dt))
    return out


def five_run(tn, team, day, five):
    """Start of the current run of `five` as a unit: going back through the
    team's readable lineups, the run ends at a lineup with a non-member who
    played at least tenure.CONFIRM_N series for the team (a regular, not a
    one-off stand-in). Players' own join dates are not used: a benched player
    keeps his old squad spell, so per-player tenure overstates the unit.
    -> (since, [players who broke it], at_least: run reaches the data start)."""
    rows = tn.readable(team, day)
    count = defaultdict(int)
    for r in rows:
        for p in r[2]:
            count[p] += 1
    since = None
    for r in reversed(rows):
        broke = sorted(p for p in r[2] - five if count[p] >= TN.CONFIRM_N)
        if broke:
            return since, broke, False
        since = r[0]
    return since, [], True


def fact_notes(inp, team, cutoff, ref_day):
    out = []
    for (t, player), f in sorted(NF.live_form(inp.form, cutoff).items()):
        if t != team or f["maps_last30"] < CONFIG["form_min_maps"]:
            continue
        delta = round(f["rating_last30"] - f["rating_baseline"], 2)
        if abs(delta) < CONFIG["form_delta"] - 1e-9:
            continue
        rule = "HOT_PLAYER" if delta > 0 else "COLD_PLAYER"
        base = "last 12 months" if f["baseline"] == "last12m" else "career"
        out.append(_note(rule, team,
                         f"{player}: {f['rating_last30']:.2f} rating over the last 30 days, "
                         f"{delta:+.2f} vs baseline ({'playing hot' if delta > 0 else 'cold spell'})",
                         f"{f['rating_last30']:.2f} over {f['maps_last30']} maps · baseline {f['rating_baseline']:.2f} "
                         f"({base}) · {f['source']}",
                         {"player": player, "rating_last30": f["rating_last30"], "rating_baseline": f["rating_baseline"],
                          "baseline": f["baseline"], "delta": delta, "maps_last30": f["maps_last30"]},
                         f["observed_at"][:10], ref_day, source=f["source"], url=f["source_url"], mag=abs(delta)))
    for f in NF.live_news(inp.news, cutoff):
        if f["team"] != team:
            continue
        if f["type"] == "stand_in":
            head = f"Stand-in: {f['player']}" + (f" in for {f['replaced']}" if f.get("replaced") else "")
            ev = " · ".join(x for x in (f"From {f['date']}", f.get("until") and f"until {f['until']}",
                                        f.get("event") and f"for {f['event']}", f["source"]) if x)
            out.append(_note("STAND_IN", team, head, ev,
                             {k: f.get(k) for k in ("player", "replaced", "date", "until", "event")},
                             f["observed_at"][:10], ref_day, source=f["source"], url=f["source_url"]))
        else:
            other = f.get("other_team")
            if f.get("move") == "in":
                head = f"Rumor: {f['player']} linked with joining {team}"
            else:
                head = f"Rumor: {f['player']} linked with a move" + (f" to {other}" if other else "")
            out.append(_note("ROSTER_RUMOR", team, head,
                             f"Reported {f['date']} by {f['source']} · unconfirmed",
                             {k: f.get(k) for k in ("player", "date", "move", "other_team")},
                             f["observed_at"][:10], ref_day, source=f["source"], url=f["source_url"],
                             kind="rumor"))
    return out


def h2h_notes(inp, a, b, cutoff_day, ref_day, match_day=None):
    x, y = sorted((a, b))
    rows = [r for r in inp.h2h.get((x, y), []) if r[0] < cutoff_day]
    if not rows:
        return []
    d, ev, stage, winner, score_xy = rows[-1]
    days = ((match_day or ref_day) - d).days
    if days > CONFIG["revenge_max_days"]:
        return []
    loser = y if winner == x else x
    # score from the loser's side
    sc = score_xy if loser == x else ("-".join(reversed(score_xy.split("-"))) if score_xy else score_xy)
    wins = defaultdict(int)
    for r in rows:
        wins[r[3]] += 1
    tally = f"{x} {wins[x]}-{wins[y]} {y}"
    return [_note("REVENGE", loser,
                  f"Lost the last meeting {sc + ' ' if sc else ''}({_days(days)})",
                  f"{d.isoformat()} at {_stage(ev, stage)} · winner {winner} · all meetings in the data: {tally}",
                  {"date": d.isoformat(), "event": ev, "stage": stage, "winner": winner, "score": sc,
                   "days": days, "meetings": len(rows), "tally": {x: wins[x], y: wins[y]}},
                  d, ref_day, scope="matchup", mag=-days)]


def _team_candidates(inp, team, cutoff, match_day, event):
    c = NF.cutoff_of(cutoff)
    cday = c.date()
    ref = cday
    return (schedule_notes(inp, team, cday, match_day or cday, event, ref)
            + tenure_notes(inp, team, cday, ref) + fact_notes(inp, team, c, ref))


def _rank(cands, side_of):
    return sorted(cands, key=lambda n: (n["rank"], side_of(n), -n["_mag"], n["id"]))


def _select(cands, cap, side_of):
    out, per, seen = [], defaultdict(int), set()
    for n in _rank(cands, side_of):
        if n["id"] in seen:
            raise AssertionError(f"duplicate note {n['id']} ({n['rule']}, {n['team_scope']})")
        seen.add(n["id"])
        k = (n["rule"], n["team_scope"])
        if per[k] >= CONFIG["max_per_rule_team"]:
            continue
        per[k] += 1
        out.append(n)
    kept = out[:cap]
    for n in kept:
        n.pop("_mag", None)
    return kept, len(out) - len(kept)


def freshness(inp, cutoff, scope=None):
    """notes_freshness over the scope teams (default: teams with a fixture in
    the next CONFIG['scope_days'] days of the cutoff)."""
    c = NF.cutoff_of(cutoff)
    day = c.date()
    if scope is None:
        hi = day + timedelta(days=CONFIG["scope_days"])
        scope = sorted({t for d, _, a, b in inp.fixtures if day <= _d(d) <= hi for t in (a, b)})
    newest = {}
    for f in inp.form + inp.news:
        t = NF.parse_ts(f["observed_at"])
        if f["team"] in scope and t < c and (f["team"] not in newest or t > newest[f["team"]]):
            newest[f["team"]] = t
    ages = [(day - t.date()).days for t in newest.values()]
    oldest = max(ages) if ages else None
    live = len(NF.live_form(inp.form, c)) + len(NF.live_news(inp.news, c))
    return {"as_of": day.isoformat(), "facts_updated_at": inp.facts_updated_at,
            "oldest_fact_age_days": oldest,
            "stale": inp.facts_updated_at is None or oldest is None or oldest > CONFIG["stale_after_days"],
            "stale_after_days": CONFIG["stale_after_days"], "scope_teams": len(scope),
            "teams_with_no_facts": sorted(t for t in scope if t not in newest), "facts_live": live}


# ---- the tool interface ----------------------------------------------------
def _resolve(inp, team):
    if team in inp.team_set:
        return team
    low = {t.lower(): t for t in inp.teams}
    if str(team).lower() in low:
        return low[str(team).lower()]
    if str(team).lower().strip() in inp.aliases:
        return inp.aliases[str(team).lower().strip()]
    norm = {re.sub(r"[^a-z0-9]", "", t.lower()): t for t in inp.teams}
    k = re.sub(r"[^a-z0-9]", "", str(team).lower())
    if k in norm:
        return norm[k]
    close = difflib.get_close_matches(str(team), inp.teams, n=5, cutoff=0.5)
    close += [t for t in inp.teams if k and k in re.sub(r"[^a-z0-9]", "", t.lower()) and t not in close]
    raise NotesError("UNKNOWN_TEAM", f"No team named {team!r} in the bundle.", team=team,
                     close_matches=close[:5])


def _clamp(cutoff, md):
    """A series played on md sees nothing from md on, whatever asof says
    (an unplayed fixture dated in the past must not read later results)."""
    if md is not None and md < NF.cutoff_of(cutoff).date():
        return md.isoformat()
    return cutoff


def _cutoff(inp, asof):
    if asof == "latest":
        return inp.live_cutoff
    try:
        NF.cutoff_of(asof)
    except (ValueError, TypeError):
        raise NotesError("BAD_DATE", "asof_date must be YYYY-MM-DD, YYYY-MM-DDTHH:MM:SSZ or 'latest'.",
                         field="asof_date", value=asof)
    return asof


def _day(value, field):
    if value is None:
        return None
    try:
        return _d(value)
    except (ValueError, TypeError):
        raise NotesError("BAD_DATE", f"{field} must be YYYY-MM-DD.", field=field, value=value)


def _envelope(ok=True, **kw):
    return {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "note_schema": NOTE_SCHEMA, "ok": ok, **kw}


def _fail(e):
    return _envelope(False, error=e.payload())


def _matchup(inp, team_a, team_b, asof, match_day=None, event=None):
    a, b = _resolve(inp, team_a), _resolve(inp, team_b)
    if a == b:
        raise NotesError("SAME_TEAM", "team_a and team_b are the same team.", team=a)
    cutoff = _cutoff(inp, asof)
    cday = NF.cutoff_of(cutoff).date()
    md = _day(match_day, "match_day") or cday
    cutoff = _clamp(cutoff, md)
    cday = NF.cutoff_of(cutoff).date()
    cands = (_team_candidates(inp, a, cutoff, md, event) + h2h_notes(inp, a, b, cday, cday, md)
             + _team_candidates(inp, b, cutoff, md, event))
    side = {a: 0, b: 2}
    notes, omitted = _select(cands, CONFIG["max_matchup_notes"],
                             lambda n: 1 if n["scope"] == "matchup" else side[n["team_scope"]])
    return _envelope(team_a=a, team_b=b, asof_date=asof, match_day=md.isoformat(), event=event,
                     notes=notes, omitted=omitted, freshness=freshness(inp, cutoff, [a, b]),
                     counted_in_win_probability=False)


def get_notes(team_a, team_b, asof_date, bundle, match_day=None, event=None):
    """Notes for one matchup (contract: docs/notes-contracts.md section 3)."""
    try:
        return _matchup(inputs_of(bundle), team_a, team_b, asof_date, match_day, event)
    except NotesError as e:
        return _fail(e)


def get_notes_batch(matchups, asof_date, bundle):
    """get_notes for each {"team_a", "team_b", "match_day"?, "event"?}; one bad
    matchup is an error in its slot, the rest still return."""
    try:
        inp = inputs_of(bundle)
    except NotesError as e:
        return _fail(e)
    results = []
    for m in matchups or []:
        try:
            results.append(_matchup(inp, m.get("team_a"), m.get("team_b"), m.get("asof_date", asof_date),
                                    m.get("match_day"), m.get("event")))
        except NotesError as e:
            results.append(_fail(e))
    return _envelope(results=results)


def team_notes(team, asof_date, bundle, match_day=None, event=None):
    """Team-scoped notes (the team page): schedule, tenure, form, news."""
    try:
        inp = inputs_of(bundle)
        t = _resolve(inp, team)
        cutoff = _cutoff(inp, asof_date)
        cday = NF.cutoff_of(cutoff).date()
        md = _day(match_day, "match_day") or cday
        cutoff = _clamp(cutoff, md)
        cday = NF.cutoff_of(cutoff).date()
        notes, omitted = _select(_team_candidates(inp, t, cutoff, md, event), CONFIG["max_team_notes"],
                                 lambda n: 0)
        return _envelope(team=t, asof_date=asof_date, match_day=md.isoformat(), event=event, notes=notes,
                         omitted=omitted, freshness=freshness(inp, cutoff, [t]),
                         counted_in_win_probability=False)
    except NotesError as e:
        return _fail(e)


def dumps(obj):
    """The canonical serialisation (what the determinism test diffs)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Matchup notes for two teams (or one with --team-only).")
    ap.add_argument("--a", required=True, help="team A (bundle name; close matches are suggested)")
    ap.add_argument("--b", help="team B")
    ap.add_argument("--date", default="latest", help="asof: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SSZ or latest")
    ap.add_argument("--match-day", help="day the series is played (default: the asof day)")
    ap.add_argument("--event", help="the series' event name (suppresses same-event short rest)")
    ap.add_argument("--bundle", default=os.path.join(paths.VIEWER, "data-bundle.json"))
    ap.add_argument("--team-only", action="store_true", help="team notes for --a only")
    args = ap.parse_args(argv)
    if not os.path.exists(args.bundle):
        print(dumps(_fail(NotesError("MISSING_FIELDS", f"No bundle at {args.bundle}; build it with "
                                     "python viewer/build_bundle.py viewer/data-bundle.json", fields=["notes"]))))
        return 2
    with open(args.bundle, encoding="utf-8") as f:
        bundle = json.load(f)
    if args.team_only or not args.b:
        out = team_notes(args.a, args.date, bundle, args.match_day, args.event)
    else:
        out = get_notes(args.a, args.b, args.date, bundle, args.match_day, args.event)
    print(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
