#!/usr/bin/env python3
"""
build_db.py -- data/faceoff.db, a read-only SQLite copy of the data for querying.

data/matches.json stays the source of truth (it is what the pipeline ingests
and what the gates check); this database is rebuilt from it and from the other
data/*.json files on every refresh, and committed. Nothing in the pipeline
reads it. Every table and column is described in the table `schema_doc`, so
`SELECT * FROM schema_doc` is the place to start.

Elo columns are the engine's own point-in-time series Elo (backtest.History):
`elo_before` uses only series dated before the match day, exactly what the
predictions see.

The file is built in a temp file and replaces data/faceoff.db only when its
bytes differ, so an unchanged input never produces a commit. Builds are
byte-identical for identical input with the same SQLite library version.

USAGE (repo root):  python scripts/build_db.py [--out data/faceoff.db]
Stdlib only.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402

SCHEMA_VERSION = 1

# table -> [(column, SQL type, description)]; the table's own description is under column ""
SCHEMA = {
    "meta": [
        ("", "", "One row per fact about this build: schema_version, data_through, matches_hash, counts."),
        ("key", "TEXT PRIMARY KEY", "Name of the fact."),
        ("value", "TEXT", "Its value."),
    ],
    "matches": [
        ("", "", "Every finished series of the tracked S/A-tier events (data/matches.json). "
                 "This is the set the ratings and predictions are built from."),
        ("match_id", "TEXT PRIMARY KEY", "Stable 12-hex id of the series."),
        ("date", "TEXT", "Match day, YYYY-MM-DD (event-local; no start times)."),
        ("event", "TEXT", "Event name."),
        ("event_page", "TEXT", "Event page path the series was read from (stage pages included)."),
        ("tier", "TEXT", "Event tier, S or A."),
        ("best_of", "INTEGER", "1, 3 or 5."),
        ("team_a", "TEXT", "First team as listed (not a home side)."),
        ("team_b", "TEXT", "Second team."),
        ("winner", "TEXT", "Series winner (team_a or team_b)."),
        ("maps_a", "INTEGER", "Maps won by team_a (NULL when no map list was recorded)."),
        ("maps_b", "INTEGER", "Maps won by team_b."),
        ("elo_a_before", "REAL", "team_a's series Elo from series dated before this day."),
        ("elo_b_before", "REAL", "team_b's series Elo from series dated before this day."),
        ("p_a_elo", "REAL", "Elo expectation that team_a wins the series (no other factors)."),
        ("ext_match_id", "TEXT", "External match number printed on the event page, if any."),
    ],
    "maps": [
        ("", "", "One row per map played in a series of `matches`."),
        ("match_id", "TEXT", "Series (matches.match_id)."),
        ("map_no", "INTEGER", "1-based order within the series."),
        ("map", "TEXT", "Map name."),
        ("winner", "TEXT", "Team that won the map."),
    ],
    "elo_history": [
        ("", "", "Each team's series Elo before and after every series in `matches`, in date order."),
        ("match_id", "TEXT", "Series (matches.match_id)."),
        ("date", "TEXT", "Match day."),
        ("team", "TEXT", "Team."),
        ("opponent", "TEXT", "Opponent."),
        ("won", "INTEGER", "1 if team won the series."),
        ("expected", "REAL", "Elo expectation that team wins, from elo_before of both sides."),
        ("elo_before", "REAL", "Elo from series dated before this day."),
        ("elo_after", "REAL", "Elo after this series is folded in."),
    ],
    "teams": [
        ("", "", "One row per team in `matches`, with totals and the current Elo."),
        ("team", "TEXT PRIMARY KEY", "Team name."),
        ("first_date", "TEXT", "First series in the data."),
        ("last_date", "TEXT", "Most recent series."),
        ("series", "INTEGER", "Series played."),
        ("series_won", "INTEGER", "Series won."),
        ("maps_won", "INTEGER", "Maps won."),
        ("maps_lost", "INTEGER", "Maps lost."),
        ("elo", "REAL", "Series Elo after every series in the data."),
        ("elo_rank", "INTEGER", "Rank by elo among teams with a series in the 90 days before data_through."),
    ],
    "lineups": [
        ("", "", "Five-man lineups per series where known (data/lineups.json, matched from Valve's "
                 "regional standings files). One row per player."),
        ("match_id", "TEXT", "Series (matches.match_id); NULL when the lineup's series is not in `matches`."),
        ("date", "TEXT", "Match day."),
        ("team", "TEXT", "Team."),
        ("player", "TEXT", "Player handle."),
    ],
    "standins": [
        ("", "", "Stand-ins and missing in-game leaders recorded per team and event (data/roster_events.json). "
                 "Not dated within an event."),
        ("team", "TEXT", "Team."),
        ("event_page", "TEXT", "Event page."),
        ("standin", "INTEGER", "1 if the team played with a stand-in."),
        ("missing_igl", "INTEGER", "1 if the in-game leader was missing."),
        ("player_out", "TEXT", "Player replaced (NULL if not recorded)."),
        ("player_in", "TEXT", "Stand-in (NULL if not recorded)."),
    ],
    "vrs_standings": [
        ("", "", "Valve Regional Standings, latest snapshot in data/vrs.json."),
        ("snapshot_date", "TEXT", "Snapshot date."),
        ("region", "TEXT", "Region of the table."),
        ("rank", "INTEGER", "Rank within the region."),
        ("points", "INTEGER", "VRS points."),
        ("team", "TEXT", "Team name as Valve lists it."),
        ("roster", "TEXT", "Players, comma-separated."),
    ],
    "lower_matches": [
        ("", "", "Lower-tier and qualifier series (data/matches_lower.json). NOT used by the ratings."),
        ("match_id", "TEXT", "Stable 12-hex id."),
        ("date", "TEXT", "Match day."),
        ("event", "TEXT", "Event name."),
        ("event_page", "TEXT", "Event page path."),
        ("tier", "TEXT", "Event tier (S, A, B, C)."),
        ("kind", "TEXT", "main, qualifier or sa_gap."),
        ("best_of", "INTEGER", "1, 3 or 5."),
        ("team_a", "TEXT", "First team."),
        ("team_b", "TEXT", "Second team."),
        ("winner", "TEXT", "Series winner."),
        ("maps_a", "INTEGER", "Maps won by team_a."),
        ("maps_b", "INTEGER", "Maps won by team_b."),
    ],
    "lower_maps": [
        ("", "", "Maps of `lower_matches`."),
        ("match_id", "TEXT", "Series (lower_matches.match_id)."),
        ("map_no", "INTEGER", "1-based order."),
        ("map", "TEXT", "Map name."),
        ("winner", "TEXT", "Map winner."),
    ],
    "schema_doc": [
        ("", "", "Describes every table, view and column. Start here."),
        ("table_name", "TEXT", "Table or view."),
        ("column_name", "TEXT", "Column ('' = the table itself)."),
        ("sql_type", "TEXT", "Declared type."),
        ("description", "TEXT", "What it holds."),
    ],
}

VIEWS = {
    "team_matches": (
        "One row per team per series in `matches` (both sides), for per-team questions.",
        """SELECT m.match_id, m.date, m.event, m.tier, m.best_of,
                  m.team_a AS team, m.team_b AS opponent, (m.winner = m.team_a) AS won,
                  m.maps_a AS maps_for, m.maps_b AS maps_against,
                  m.elo_a_before AS elo_before, m.elo_b_before AS opp_elo_before, m.p_a_elo AS p_win_elo
           FROM matches m
           UNION ALL
           SELECT m.match_id, m.date, m.event, m.tier, m.best_of,
                  m.team_b, m.team_a, (m.winner = m.team_b),
                  m.maps_b, m.maps_a,
                  m.elo_b_before, m.elo_a_before, 1.0 - m.p_a_elo
           FROM matches m"""),
    "team_map_stats": (
        "Per team and map: maps played and won in `matches`.",
        """SELECT t.team, mp.map, COUNT(*) AS played, SUM(mp.winner = t.team) AS won
           FROM maps mp JOIN team_matches t ON t.match_id = mp.match_id
           GROUP BY t.team, mp.map"""),
}

INDEXES = [
    "CREATE INDEX idx_matches_date ON matches(date)",
    "CREATE INDEX idx_matches_team_a ON matches(team_a)",
    "CREATE INDEX idx_matches_team_b ON matches(team_b)",
    "CREATE INDEX idx_maps_match ON maps(match_id)",
    "CREATE INDEX idx_elo_team ON elo_history(team, date)",
    "CREATE INDEX idx_lineups_team ON lineups(team, date)",
    "CREATE INDEX idx_lower_date ON lower_matches(date)",
    "CREATE INDEX idx_lower_maps_match ON lower_maps(match_id)",
]


def load(name, default=None):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def map_counts(m):
    maps = m.get("maps") or []
    if not maps:
        return None, None
    return (sum(1 for x in maps if x["winner"] == m["team_a"]),
            sum(1 for x in maps if x["winner"] == m["team_b"]))


def elo_rows(matches):
    """(per-match pre-match Elo, elo_history rows, final History). Days are
    folded in like backtest.build_dataset: every series of day D sees only
    days < D."""
    h = bt.History()
    pre, hist = {}, []
    i = 0
    while i < len(matches):
        j = i
        while j < len(matches) and matches[j]["date"] == matches[i]["date"]:
            j += 1
        day = matches[i:j]
        h.clock = bt._d(day[0]["date"])
        for m in day:
            pre[m["match_id"]] = (h.feature_elo(m["team_a"]), h.feature_elo(m["team_b"]))
        for m in day:
            a, b = m["team_a"], m["team_b"]
            ea, eb = pre[m["match_id"]]
            h.add(m)
            e = bt.elo_expected(ea, eb)
            hist.append((m["match_id"], m["date"], a, b, int(m["winner"] == a), e, ea, h.elo[a]))
            hist.append((m["match_id"], m["date"], b, a, int(m["winner"] == b), 1 - e, eb, h.elo[b]))
        i = j
    return pre, hist, h


def build(path):
    matches = bt.load_matches()
    for m in matches:
        if not m.get("match_id"):
            raise SystemExit(f"series without match_id: {m['date']} {m['team_a']} vs {m['team_b']}")
    pre, hist, h = elo_rows(matches)
    with open(os.path.join(DATA, "matches.json"), "rb") as f:
        mhash = hashlib.sha1(f.read().replace(b"\r\n", b"\n")).hexdigest()[:12]

    con = sqlite3.connect(path)
    cur = con.cursor()
    for t, cols in SCHEMA.items():
        cur.execute(f"CREATE TABLE {t} (" + ", ".join(f"{c} {ty}" for c, ty, _ in cols if c) + ")")

    for m in matches:
        ma, mb = map_counts(m)
        ea, eb = pre[m["match_id"]]
        cur.execute("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (m["match_id"], m["date"], m["event"], m.get("event_page"), m.get("tier"), m.get("best_of"),
                     m["team_a"], m["team_b"], m["winner"], ma, mb, ea, eb, bt.elo_expected(ea, eb),
                     m.get("ext_match_id")))
        for k, x in enumerate(m.get("maps") or [], 1):
            cur.execute("INSERT INTO maps VALUES (?,?,?,?)", (m["match_id"], k, x["map"], x["winner"]))
    cur.executemany("INSERT INTO elo_history VALUES (?,?,?,?,?,?,?,?)", hist)

    # teams
    agg = {}
    for m in matches:
        ma, mb = map_counts(m)
        for t, won, mf, mag in ((m["team_a"], m["winner"] == m["team_a"], ma, mb),
                                (m["team_b"], m["winner"] == m["team_b"], mb, ma)):
            r = agg.setdefault(t, {"first": m["date"], "last": m["date"], "n": 0, "w": 0, "mw": 0, "ml": 0})
            r["last"] = m["date"]
            r["n"] += 1
            r["w"] += won
            r["mw"] += mf or 0
            r["ml"] += mag or 0
    through = matches[-1]["date"] if matches else None
    if through:
        lo = (bt._d(through) - bt.timedelta(days=90)).isoformat()
        active = sorted((t for t, r in agg.items() if r["last"] >= lo), key=lambda t: (-h.elo[t], t))
        rank = {t: k for k, t in enumerate(active, 1)}
    else:
        rank = {}
    for t in sorted(agg):
        r = agg[t]
        cur.execute("INSERT INTO teams VALUES (?,?,?,?,?,?,?,?,?)",
                    (t, r["first"], r["last"], r["n"], r["w"], r["mw"], r["ml"], h.elo[t], rank.get(t)))

    # lineups
    ids = {(m["date"], m["team_a"], m["team_b"]): m["match_id"] for m in matches}
    lu = (load("lineups.json", {}) or {}).get("lineups", {})
    for key in sorted(lu):
        d, a, b = key.split("|", 2)
        for side, team in (("a", a), ("b", b)):
            for p in sorted(lu[key].get(side) or []):
                cur.execute("INSERT INTO lineups VALUES (?,?,?,?)", (ids.get((d, a, b)), d, team, p))

    # stand-ins
    ros = load("roster_events.json", {}) or {}
    for team in sorted(ros):
        for page in sorted(ros[team]):
            ev = ros[team][page]
            pairs = ev.get("players") or [[None, None]]
            for pair in pairs:
                out_p, in_p = (list(pair) + [None, None])[:2]
                cur.execute("INSERT INTO standins VALUES (?,?,?,?,?,?)",
                            (team, page, int(bool(ev.get("standin"))), int(bool(ev.get("missing_igl"))), out_p, in_p))

    # VRS
    vrs = load("vrs.json", {}) or {}
    for region in sorted(vrs.get("standings") or {}):
        for r in vrs["standings"][region]:
            cur.execute("INSERT INTO vrs_standings VALUES (?,?,?,?,?,?)",
                        (vrs.get("date"), region, r.get("rank"), r.get("points"), r.get("name"),
                         ", ".join(r.get("roster") or [])))

    # lower tier
    for m in load("matches_lower.json", []) or []:
        ma, mb = map_counts(m)
        cur.execute("INSERT INTO lower_matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (m.get("match_id"), m["date"], m.get("event"), m.get("event_page"), m.get("tier"),
                     m.get("kind"), m.get("best_of"), m["team_a"], m["team_b"], m["winner"], ma, mb))
        for k, x in enumerate(m.get("maps") or [], 1):
            cur.execute("INSERT INTO lower_maps VALUES (?,?,?,?)", (m.get("match_id"), k, x["map"], x["winner"]))

    for name, (_, sql) in VIEWS.items():
        cur.execute(f"CREATE VIEW {name} AS {sql}")
    for sql in INDEXES:
        cur.execute(sql)

    doc = []
    for t, cols in SCHEMA.items():
        for c, ty, d in cols:
            doc.append((t, c, ty.replace(" PRIMARY KEY", ""), d))
    for name, (d, _) in VIEWS.items():
        doc.append((name, "", "VIEW", d))
    cur.executemany("INSERT INTO schema_doc VALUES (?,?,?,?)", doc)

    meta = {"schema_version": SCHEMA_VERSION, "data_through": through, "matches_hash": mhash,
            "matches": len(matches), "lower_matches": cur.execute("SELECT COUNT(*) FROM lower_matches").fetchone()[0],
            "elo": "series Elo, init 1500, K 32 (backtest.py)"}
    cur.executemany("INSERT INTO meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
    con.commit()
    cur.execute("VACUUM")
    con.close()
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="build data/faceoff.db from data/*.json")
    ap.add_argument("--out", default=os.path.join(DATA, "faceoff.db"))
    a = ap.parse_args(argv)
    fd, tmp = tempfile.mkstemp(suffix=".db", dir=os.path.dirname(os.path.abspath(a.out)))
    os.close(fd)
    os.remove(tmp)
    try:
        meta = build(tmp)
        with open(tmp, "rb") as f:
            new = f.read()
        old = None
        if os.path.exists(a.out):
            with open(a.out, "rb") as f:
                old = f.read()
        if new == old:
            print(f"{a.out}: unchanged ({meta['matches']} series, data through {meta['data_through']})")
            return 0
        os.replace(tmp, a.out)
        print(f"wrote {a.out}: {meta['matches']} series, {meta['lower_matches']} lower-tier, "
              f"data through {meta['data_through']}, {len(new) // 1024} KB")
        return 0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    sys.exit(main())
