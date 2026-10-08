#!/usr/bin/env python3
"""
Per-team, per-event lineups (the players each team brought to each event),
from the tournament pages already cached for matches.json and
matches_lower.json. No network.

SOURCES, in order of preference, for team T at a series on page P dated D:
  1. the participant section of P ({{TeamParticipants}} / {{Opponent ...
     players=}} or the older {{TeamCard}}, parsed by
     viewer/events.parse_participants), else of the nearest ancestor page
     that lists T (a stage page "<event>/Stage 1" usually has none; the
     event page has them). Players marked as unused subs are left out;
     coaches too. Needs at least 5 players.
  2. else T's Liquipedia team page squad history (cached for the teams in
     matches.json): players of the player squads (not staff, not coaches)
     with joindate <= D and no inactivedate / leavedate <= D. Used only when
     that gives exactly 5 or 6 players.
  Otherwise the lineup is unknown (None): nothing is guessed.

Participant lists are announced before an event and amended by Liquipedia
editors when a stand-in plays, so a lineup is a pre-match fact about that
event (the cutoff rule in leakage_check.py). Squad dates are historical
facts (a join or leave dated <= D).

Player ids are Liquipedia page names, lower-cased.

Output: data/event_lineups.json
  {"pages": {page: {team: [ids]}},          # participant sections
   "squads": {team: [[id, joindate, enddate or ""]]}}

USAGE (from D:/Face-Off):  python data/event_lineups.py
Library: load(); lineup(store, team, page, day) -> sorted ids or None.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "event_lineups.json")
DATE = re.compile(r"(\d{4})-(\d{2}|\?\?)-(\d{2}|\?\?)")


def _alias():
    import collect_liquipedia as C
    alias = C.cached_aliases()
    names = set(alias.values())
    for src, dst in C.cached_page_redirects().items():
        if src not in names:
            alias.setdefault(src, dst)
    for n in sorted(names):
        alias.setdefault(n.lower(), n)
    return alias


def _date(s):
    """First YYYY-MM-DD in a squad date field ('??' month/day -> 01 / 28 is
    NOT assumed: an unknown month gives None, an unknown day the 1st)."""
    m = DATE.search(s or "")
    if not m or m.group(2) == "??":
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3) if m.group(3) != '??' else '01'}"


def squad_rows(txt):
    import collect_liquipedia as C
    rows = []
    for start, body in C.find_templates(txt, "Squad"):
        _, nm = C.split_params(body.split("\n", 1)[0])
        if nm.get("type", "").strip().lower() == "staff":
            continue
        for _, pb in C.find_templates(body, "Person"):
            _, p = C.split_params(pb)
            pid = p.get("id", "").strip()
            role = p.get("role", "").strip().lower()
            if not pid or role:          # coaches, analysts, managers carry a role
                continue
            j = _date(p.get("joindate"))
            end = min([d for d in (_date(p.get("inactivedate")), _date(p.get("leavedate"))) if d] or [""])
            if j:
                rows.append([pid.lower(), j, end])
    return rows


def build():
    import lpfetch
    sys.path.insert(0, os.path.join(ROOT, "viewer"))
    import events as E
    idx = lpfetch.cached_pages()
    alias = _alias()
    titles = []
    for fn in ("lp_titles_selected.txt", "lp_titles_lower.txt"):
        p = os.path.join(HERE, "raw", fn)
        if os.path.exists(p):
            titles += [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    pages = {}
    # every page and its ancestors that are cached (event pages hold the lists)
    want = set()
    for t in titles:
        parts = t.split("/")
        want |= {"/".join(parts[:i]) for i in range(1, len(parts) + 1)}
    for t in sorted(want):
        if t not in idx:
            continue
        got = {}
        for team, rec in E.parse_participants(idx[t][1], alias).items():
            ps = [p.lower() for p in dict.fromkeys(rec["players"]) if p]
            if len(ps) >= 5:
                got[team] = ps
        if got:
            pages[t] = got
    teams = set()
    for fn in ("matches.json", "matches_lower.json"):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            for m in json.load(open(p, encoding="utf-8")):
                teams |= {m["team_a"], m["team_b"]}
    squads = {}
    for team in sorted(teams):
        if team in idx:
            rows = squad_rows(idx[team][1])
            if rows:
                squads[team] = rows
    return {"pages": pages, "squads": squads}


_CACHE = {}


def load(path=OUT):
    if path not in _CACHE:
        _CACHE[path] = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"pages": {}, "squads": {}}
    return _CACHE[path]


def page_of(m):
    src = m.get("source") or ""
    return src[len("liquipedia:"):].split(" (hltv")[0] if src.startswith("liquipedia:") else m.get("event_title")


def lineup(store, team, page, day):
    """Sorted player ids of `team` at the series on `page` dated `day`
    (YYYY-MM-DD), or None. See the module docstring for the sources."""
    t = page
    while t:
        got = store["pages"].get(t, {}).get(team)
        if got:
            return sorted(got)
        t = t.rsplit("/", 1)[0] if "/" in t else None
    rows = store["squads"].get(team)
    if rows:
        act = sorted({p for p, j, e in rows if j <= day and (not e or e > day)})
        if 5 <= len(act) <= 6:
            return act
    return None


def main():
    out = build()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=0, ensure_ascii=False, sort_keys=True)
    # coverage report on the series data
    store = out
    for fn in ("matches.json", "matches_lower.json"):
        ms = json.load(open(os.path.join(HERE, fn), encoding="utf-8"))
        n = known = from_page = 0
        for m in ms:
            for team in (m["team_a"], m["team_b"]):
                n += 1
                lu = lineup(store, team, page_of(m), m["date"])
                if lu:
                    known += 1
                    t = page_of(m)
                    while t and team not in store["pages"].get(t, {}):
                        t = t.rsplit("/", 1)[0] if "/" in t else None
                    from_page += bool(t)
        print(f"{fn}: team-series {n}, lineup known {known} ({known / max(1, n):.1%}; "
              f"{from_page} from participant sections, {known - from_page} from squad dates)")
    print(f"wrote data/event_lineups.json: {len(out['pages'])} pages, {len(out['squads'])} team squads")


if __name__ == "__main__":
    main()
