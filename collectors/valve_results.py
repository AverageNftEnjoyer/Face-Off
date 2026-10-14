#!/usr/bin/env python3
"""
Second, independent source of series RESULTS: Valve's public repository
github.com/ValveSoftware/counter-strike_regional_standings (the same repository
collectors/vrs.py and collectors/lineups.py read).

Every live/<year>/details/<snapshot>/<n>--<team>--<players>.md page lists the
series that fed one roster's standing over the six months before <snapshot>:
a snapshot-local match id, the date, the opponent's display name and W/L. The
same match id appears on both rosters' pages. This module joins those rows
into one record per series, in the matches.json record shape, with every
field the source does not carry set to null (never guessed):

    match_id     stable_match_id(date, team_a, team_b, SOURCE_TAG, "" | seq)
    date, team_a, team_b, winner     from the rows (team names mapped to ours)
    event, event_page, best_of, tier, ext_match_id   null (not in the source)
    maps         [] (not in the source)
    source       SOURCE_TAG
    snapshot     the newest snapshot read for it (its reading wins)
    first_seen   the oldest snapshot that listed it (measured lag)
    valve_names  [Valve's name for team_a, for team_b];  seq: 1, or 2+ for a
                 further meeting of the same pair on the same day

What it is for: a read-only cross-check of data/matches.json
(checks/cross_source_check.py and the cross_source gate in
checks/data_gates.py). It never writes matches.json.

What it is NOT: a fresh source. Valve publishes a snapshot about once a month
(the first Monday) and lists an event once it has finished, so a series
reaches this file weeks after it was played (cross_source_report.json has the
measured lag). It has no maps, best-of, score or event name.

Team names are DERIVED, never listed by hand (see name_index): our canonical
names from matches.json / matches_lower.json, every Liquipedia code in
team_aliases.json, the team slug and "Team Name:" header of each Valve page,
all compared after data/lineups.norm (lower-case, punctuation and filler words
such as "team" / "esports" dropped). A normalised key that would point at two
different teams is dropped as ambiguous, so it can never map wrongly. A Valve
name that still matches nothing is inferred from dates alone (infer_aliases,
thresholds in the tolerance block) and recorded in the file's inferred_names.

Raw pages are shared with collectors/lineups.py (data/raw/valve/details/,
gitignored) and fetched through lineups.fetch (cached, spaced, User-Agent).

USAGE (repo root):
  python collectors/valve_results.py [--offline] [--incremental]
    --offline      read only cached pages (a page that is not cached is
                   skipped and counted, not fetched)
    --incremental  keep data/valve_results.json and add only snapshots newer
                   than the newest it records (what pipeline/daily_refresh.py
                   runs; nothing is rewritten when Valve published nothing new)
Output: data/valve_results.json
"""
import json
import os
import re
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (puts every code folder on sys.path)
DATA = paths.DATA

import lineups  # noqa: E402  (norm, fetch, tree_paths: one fetcher, one cache)
from collect_liquipedia import stable_match_id  # noqa: E402

OUT = os.path.join(DATA, "valve_results.json")
SOURCE_TAG = "valve-vrs"

# ---------------------------------------------------------------- tolerances
# The ONE place the cross-source tolerances live (the gate and the report
# import them from here).
# Liquipedia dates are event-local, Valve's are not: a late-evening series can
# land on the next calendar day, so the same series may differ by one day.
DATE_WINDOW_DAYS = 1
# A Valve name that normalises to none of ours is mapped by evidence only: on
# the days a mapped team met it, that team had exactly one primary series that
# no mapped Valve row already explains, and it was against the same team of
# ours. Two coincidences on busy event days happen; three independent series
# that all point the same way do not.
ALIAS_MIN_VOTES = 3
# ...and at least this share of that name's usable rows must agree (a rename
# mid-window, a stand-in team or a typo leave a few strays, not a split).
ALIAS_MIN_SHARE = 0.8
# The cross_source gate (checks/data_gates.py) fails when an update ADDS more
# than this many winner disagreements or identity mismatches with the second
# source (relative to the published data, against the same second-source file).
# Measured on 2,191 matched series (2024-02 .. 2026-10): 0 winner
# disagreements, so any new one is worth a human look before publishing.
GATE_NEW_DISAGREE_MAX = 0
# ---------------------------------------------------------------------------

# | played | match id | date | opponent | W/L | ...
ROW = re.compile(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]*?)\s*\|\s*([WL])\s*\|")
HEADER = re.compile(r"Team Name:\s*(.*?)<br")


def parse_detail(txt):
    """One Valve roster page -> (header team name or None, [(local_id, date, opponent, won)])."""
    m = HEADER.search(txt)
    team = m.group(1).strip() if m else None
    rows = []
    for line in txt.splitlines():
        r = ROW.match(line.strip())
        if r:
            rows.append((int(r.group(1)), r.group(2), r.group(3), r.group(4) == "W"))
    return team, rows


def slug_name(path):
    """live/2026/details/2026_10_05/0003--vitality--apex-....md -> 'vitality'"""
    return path.rsplit("/", 1)[-1].split("--")[1].replace("_", " ")


def snapshot_of(path):
    return path.split("/")[3]


def name_index(canonical_names, aliases=None):
    """normalised key -> our canonical team name, derived from the names we
    already use plus every alias / code recorded for them. A key claimed by
    two different canonical names is ambiguous and left out."""
    claims = {}
    for name in canonical_names:
        claims.setdefault(lineups.norm(name), set()).add(name)
    for name, codes in (aliases or {}).items():
        if name not in canonical_names:
            continue
        for c in codes:
            claims.setdefault(lineups.norm(c), set()).add(name)
    return {k: next(iter(v)) for k, v in claims.items() if k and len(v) == 1}


def load_names():
    """(primary team names, all canonical names, alias table) from the
    committed data files. Primary = teams in matches.json (the series kept);
    all = primary + matches_lower.json, so a lower-tier team's Valve name can
    never be mistaken for a tracked team (it claims its own key)."""
    primary, names = set(), set()
    for fn in ("matches.json", "matches_lower.json"):
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for m in json.load(f):
                    names.update((m["team_a"], m["team_b"]))
                    if fn == "matches.json":
                        primary.update((m["team_a"], m["team_b"]))
    aliases = {}
    p = os.path.join(DATA, "team_aliases.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            aliases = json.load(f).get("aliases", {})
    return primary, names, aliases


def infer_aliases(pages, index, primary, window=None):
    """{Valve name: our name} for Valve names the normalised index cannot
    place, inferred from the dates alone (never from who won, so the
    agreement measured later is not inflated). primary: matches.json records.
    Only names that map to NO team of ours are considered: a Valve name that
    already resolves to a different team is an identity mismatch for the
    cross-check to report, never something to re-map silently."""
    window = DATE_WINDOW_DAYS if window is None else window
    by_team = {}
    for m in primary:
        for t, o in ((m["team_a"], m["team_b"]), (m["team_b"], m["team_a"])):
            by_team.setdefault(t, []).append((m["date"], o))
    targets = set(by_team)
    parsed = []
    explained = {}                         # our team -> {(date, our opponent)} Valve already shows
    for path, txt in pages:
        team, rows = parse_detail(txt)
        own = index.get(lineups.norm(slug_name(path))) or (team and index.get(lineups.norm(team)))
        if own not in targets:
            continue
        parsed.append((own, rows))
        for lid, d, opp, won in rows:
            o = index.get(lineups.norm(opp))
            if o is not None:
                explained.setdefault(own, set()).add((d, o))
    votes, usable, seen = {}, {}, set()
    for own, rows in parsed:
        for lid, d, opp, won in rows:
            if index.get(lineups.norm(opp)) is not None or (own, d, opp) in seen:
                continue
            seen.add((own, d, opp))            # the same row in two snapshots counts once
            # primary series of `own` near that date that no mapped Valve row explains
            near = {o for dd, o in by_team[own] if days_apart(dd, d) <= window
                    and not any(days_apart(dd, ed) <= window and eo == o
                                for ed, eo in explained.get(own, ()))}
            if len(near) != 1:
                continue
            usable[opp] = usable.get(opp, 0) + 1
            x = near.pop()
            votes.setdefault(opp, {}).setdefault(x, 0)
            votes[opp][x] += 1
    out = {}
    for opp, vs in votes.items():
        best, n = max(vs.items(), key=lambda x: (x[1], x[0]))
        if n >= ALIAS_MIN_VOTES and n >= ALIAS_MIN_SHARE * usable[opp] and best != opp:
            out[opp] = best
    return out


def build_series(pages, index, keep=None, inferred=None):
    """pages: [(path, text)] of Valve detail pages.
    Returns (records, spans, stats). One record per real series:
      * within a snapshot, a series is its local match id (seen on both
        rosters' pages; both pages must agree on who won, else it is dropped
        and counted as inconsistent);
      * across snapshots the same series is (date, team pair) and the NEWEST
        snapshot's reading wins (multiplicity per day kept from that snapshot).
    Only series whose two teams both map to our names are kept; the rest are
    counted (unmapped Valve names are reported, most frequent first).
    spans: {our team: [[first row date, snapshot date], ...]}, the dates its
    Valve pages cover (where a missing series counts as missing).
    keep: if given, only series whose two teams are both in it are returned.
    inferred: extra {Valve name: our name} from infer_aliases."""
    inferred = inferred or {}
    display = {}                       # Valve display name -> our name
    page_rows = []
    for path, txt in pages:
        team, rows = parse_detail(txt)
        own = index.get(lineups.norm(slug_name(path)))
        if team and own is None:
            own = index.get(lineups.norm(team)) or inferred.get(team)
        if team and own:
            display[team] = own
        page_rows.append((path, team, own, rows))

    def resolve(valve_name):
        return (display.get(valve_name) or inferred.get(valve_name)
                or index.get(lineups.norm(valve_name)))

    # snapshot -> local match id -> {"date": d, "rows": [one row per page that lists it]}
    by_snap = {}
    unmapped = {}
    for path, team, own, rows in page_rows:
        snap = snapshot_of(path)
        for lid, d, opp, won in rows:
            o = resolve(opp)
            if own is None or o is None:
                for nm, ok in ((team or slug_name(path), own), (opp, o)):
                    if ok is None:
                        unmapped[nm] = unmapped.get(nm, 0) + 1
                continue
            e = by_snap.setdefault(snap, {}).setdefault(lid, {"date": d, "rows": []})
            e["rows"].append((own, o, won, team, opp, d))

    stats = {"inconsistent": 0, "self_or_odd": 0}
    per_snap = {}                       # snapshot -> {(date, a, b): [records]}
    for snap, ids in by_snap.items():
        out = per_snap.setdefault(snap, {})
        for lid, e in sorted(ids.items()):
            winners, names, dates = set(), {}, set()
            for own, o, won, vt, vo, d in e["rows"]:
                if own == o:
                    winners.add(None)
                    continue
                winners.add(own if won else o)
                names[own], names[o] = vt or own, vo
                dates.add(d)
            if None in winners or len(names) != 2 or len(dates) != 1:
                stats["self_or_odd"] += 1
                continue
            if len(winners) != 1:
                stats["inconsistent"] += 1
                continue
            a, b = sorted(names)
            d = dates.pop()
            out.setdefault((d, a, b), []).append({
                "date": d, "team_a": a, "team_b": b, "winner": winners.pop(),
                "valve_names": [names[a], names[b]], "snapshot": snap})

    final, first = {}, {}
    for snap in sorted(per_snap):       # newest snapshot last -> wins
        for key, recs in per_snap[snap].items():
            final[key] = recs
            first.setdefault(key, snap)
    records = []
    for (d, a, b), recs in sorted(final.items()):
        if keep is not None and not (a in keep and b in keep):
            continue
        for i, r in enumerate(recs, 1):    # i > 1: another meeting the same day
            records.append(make_record(d, a, b, r["winner"], r["snapshot"], first[(d, a, b)],
                                       r["valve_names"], i))
    # per-team coverage spans: (first row date, snapshot date) of each page
    spans = {}
    for path, team, own, rows in page_rows:
        if own and rows:
            snap = snapshot_of(path).replace("_", "-")
            spans.setdefault(own, []).append([min(r[1] for r in rows), snap])
    for t in spans:
        spans[t] = merge_spans(spans[t])
    stats["unmapped_names"] = sorted(unmapped.items(), key=lambda x: (-x[1], x[0]))
    return records, spans, stats


def merge_spans(spans):
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def covered(spans, team, d):
    return any(lo <= d <= hi for lo, hi in spans.get(team, []))


def shift(d, days):
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def days_apart(d1, d2):
    return abs((date.fromisoformat(d1) - date.fromisoformat(d2)).days)


# --------------------------------------------------------------------- I/O

def detail_paths(paths, snaps):
    return [p for p in paths if p.startswith("live/") and "/details/" in p and p.endswith(".md")
            and p.count("/") == 4 and snapshot_of(p) in snaps]


def read_pages(paths, offline):
    """[(path, text)] for the wanted pages; offline skips what is not cached."""
    pages, missing = [], 0
    for p in paths:
        local = os.path.join(lineups.CACHE, p.split("/details/", 1)[1])
        if offline and not os.path.exists(local):
            missing += 1
            continue
        pages.append((p, lineups.fetch(p)))
    return pages, missing


def make_record(d, a, b, winner, snapshot, first_seen, valve_names, seq=1):
    """A second-source series in the matches.json record shape. Fields the
    source does not carry are null / empty, never guessed. seq > 1 marks a
    further meeting of the same pair on the same day (its own match_id).
    snapshot: the newest snapshot read for it; first_seen: the oldest one that
    listed it (how late the source was, measured)."""
    return {"match_id": stable_match_id(d, a, b, SOURCE_TAG, "" if seq == 1 else str(seq)),
            "date": d, "event": None, "event_page": None, "team_a": a, "team_b": b,
            "winner": winner, "best_of": None, "maps": [], "tier": None, "ext_match_id": None,
            "source": SOURCE_TAG, "snapshot": snapshot, "first_seen": first_seen,
            "valve_names": list(valve_names), "seq": seq}


# The file stores one compact row per series (ROW_FIELDS); load() expands each
# back to the full matches.json-shaped record, so thousands of rows stay small.
ROW_FIELDS = ["date", "team_a", "team_b", "winner", "snapshot", "first_seen", "valve_a", "valve_b", "seq"]


def to_row(r):
    return [r["date"], r["team_a"], r["team_b"], r["winner"], r["snapshot"], r["first_seen"],
            r["valve_names"][0], r["valve_names"][1], r["seq"]]


def from_row(row):
    d, a, b, w, snap, first, va, vb, seq = row
    return make_record(d, a, b, w, snap, first, (va, vb), seq)


def write(out, path):
    """Atomic write, one series per line (small daily diffs)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("{\n")
        for k, v in out.items():
            if k != "series":
                f.write(f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False, sort_keys=True)},\n")
        f.write('"series": [\n')
        f.write(",\n".join(json.dumps(to_row(r), ensure_ascii=False) for r in out["series"]))
        f.write("\n]}\n")
    os.replace(tmp, path)


def load(path=OUT):
    """The second-source file with full records, or None when it is absent."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw["series"] = [from_row(r) for r in raw["series"]]
    return raw


def cached_snapshots():
    if not os.path.isdir(lineups.CACHE):
        return []
    return sorted(d for d in os.listdir(lineups.CACHE) if os.path.isdir(os.path.join(lineups.CACHE, d)))


def main(offline=None, incremental=None):
    offline = ("--offline" in sys.argv) if offline is None else offline
    incremental = ("--incremental" in sys.argv) if incremental is None else incremental
    lineups.OFFLINE = offline
    primary, names, aliases = load_names()
    index = name_index(names, aliases)

    old = None
    if incremental:
        old = load(OUT)
    paths = lineups.tree_paths()
    all_snaps = sorted({snapshot_of(p) for p in paths if p.startswith("live/") and "/details/" in p
                        and p.count("/") == 4})
    if old is not None:
        newest = max(old.get("snapshots") or [""])
        snaps = [s for s in all_snaps if s > newest][-1:]   # the newest covers the last 6 months
        if not snaps:
            print(f"valve_results: no Valve snapshot newer than {newest or '-'}; "
                  f"data/valve_results.json unchanged ({len(old.get('series', []))} series)")
            return 0
    elif offline:
        snaps = [s for s in cached_snapshots() if s in set(all_snaps)] or cached_snapshots()
    else:
        snaps = lineups.pick_snapshots(paths)
    wanted = [p for p in detail_paths(paths, set(snaps)) if index.get(lineups.norm(slug_name(p))) in primary]
    pages, missing = read_pages(wanted, offline)
    with open(os.path.join(DATA, "matches.json"), encoding="utf-8") as f:
        inferred = {**((old or {}).get("inferred_names") or {}), **infer_aliases(pages, index, json.load(f))}
    recs, spans, stats = build_series(pages, index, keep=primary, inferred=inferred)

    if old is not None:
        # new snapshot readings replace the old ones for the same (date, pair);
        # older series outside the new window are kept as they were
        new_keys = {(r["date"], r["team_a"], r["team_b"]) for r in recs}
        first = {}
        for r in old["series"]:
            k = (r["date"], r["team_a"], r["team_b"])
            first[k] = min(first.get(k, r["first_seen"]), r["first_seen"])
        for r in recs:                     # keep the oldest sighting
            r["first_seen"] = min(r["first_seen"], first.get((r["date"], r["team_a"], r["team_b"]), r["first_seen"]))
        recs = [r for r in old["series"] if (r["date"], r["team_a"], r["team_b"]) not in new_keys] + recs
        recs.sort(key=lambda r: (r["date"], r["team_a"], r["team_b"], r["match_id"]))
        for t, s in (old.get("spans") or {}).items():
            spans[t] = merge_spans(s + spans.get(t, []))
        snaps = sorted(set(old.get("snapshots") or []) | set(snaps))
    out = {"source": f"https://github.com/{lineups.vrs.REPO}", "snapshots": snaps,
           "row_fields": ROW_FIELDS,
           "date_window_days": DATE_WINDOW_DAYS,
           "inferred_names": dict(sorted(inferred.items())),
           "note": "second-source series (winner only); read-only cross-check of matches.json",
           "spans": {t: spans[t] for t in sorted(spans)},
           "series": recs}
    write(out, OUT)
    top = ", ".join(f"{n} ({c})" for n, c in stats["unmapped_names"][:8])
    print(f"valve_results: {len(snaps)} snapshots, {len(pages)} pages read"
          f"{f' ({missing} not cached, skipped)' if missing else ''}, {len(recs)} series; "
          f"{stats['inconsistent']} dropped (the two pages disagree), "
          f"{stats['self_or_odd']} dropped (malformed); most frequent unmapped names: {top or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
