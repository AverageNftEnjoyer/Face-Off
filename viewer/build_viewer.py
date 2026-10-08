#!/usr/bin/env python3
"""
Build the Faceoff CS2 hub: one static HTML page with

  * every 2026 tournament on the cached Liquipedia pages: info, prize pool,
    map pool, participants with their event lineups, schedule and results
    with per-map round scores, group tables and playoff brackets
  * the engine's PRE-MATCH call for every match with two known teams
    (features from series dated strictly before that match day)
  * a matchup builder inside each tournament: every ordered pair of its
    participants. Finished events use ratings as of the event's first day;
    live and upcoming events use today's ratings.
  * a global head-to-head comparer, team pages, and the held-out track record

Inputs: data/matches.json, cached event pages (viewer/events.py),
viewer/assets.json (python viewer/fetch_assets.py) and predictor.py.
Nothing is estimated by hand.

Output, next to OUT.html (all names content-hashed, so they can be cached forever):
  * OUT_DIR/img/<hash>.<ext>      every logo, map image and flag the page uses
                                  (stale files in OUT_DIR/img are removed)

USAGE:
    python viewer/build_viewer.py OUT.html
"""

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, HERE)

import backtest as bt  # noqa: E402
import events as E  # noqa: E402
import predictor as pr  # noqa: E402
import team_colors as TC  # noqa: E402

N_RECENT_CALLS = 20
N_TEAM_RESULTS = 10
FACTORS = ["base_strength", "form_30d", "form_last5", "head_to_head", "map_veto", "roster", "stakes"]


IMG_DIR = "img"                   # beside the output html


def _short_hash(raw):
    return hashlib.sha256(raw).hexdigest()[:16]


class ImageWriter:
    """Copies each image the page uses to OUT_DIR/img/<content hash>.<ext> and
    hands back its relative URL. Same bytes -> same file, so a logo used twice
    is written once, and a changed logo gets a new name (safe to cache forever)."""

    EXTS = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}

    def __init__(self, out_dir):
        self.dir = os.path.join(out_dir, IMG_DIR)
        os.makedirs(self.dir, exist_ok=True)
        self.names = set()
        self.bytes = 0

    def url(self, rel):
        if not rel:
            return None
        path = os.path.join(ROOT, rel)
        ext = os.path.splitext(path)[1].lower()
        ext = ext if ext in self.EXTS else ".png"
        with open(path, "rb") as f:
            raw = f.read()
        name = _short_hash(raw) + ext
        if name not in self.names:
            dst = os.path.join(self.dir, name)
            if not (os.path.exists(dst) and os.path.getsize(dst) == len(raw)):
                with open(dst, "wb") as f:
                    f.write(raw)
            self.names.add(name)
            self.bytes += len(raw)
        return f"{IMG_DIR}/{name}"

    def prune(self):
        """Remove files in OUT_DIR/img this build did not write (only that folder)."""
        n = 0
        for fn in os.listdir(self.dir):
            p = os.path.join(self.dir, fn)
            if fn not in self.names and os.path.isfile(p):
                os.remove(p)
                n += 1
        return n


def round_floats(o, n):
    """Copy of `o` with every float rounded to n decimals (page payload only)."""
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, list):
        return [round_floats(x, n) for x in o]
    if isinstance(o, dict):
        return {k: round_floats(v, n) for k, v in o.items()}
    return o


# Decimals kept for the probabilities in the page payload. The page shows at
# most two decimals of a percentage (1e-4 of a probability) via largest
# remainder; 9 decimals moves a value by at most 5e-10, so a shown figure can
# only change if the exact value sits within 5e-10 of a rounding edge. It cuts
# about 0.5 MB of compressed transfer. None keeps full precision.
PAGE_DECIMALS = 9


_DROP = {"team", "esports", "esport", "gaming", "clan", "club", "gg"}


def _norm(name):
    words = re.findall(r"[a-z0-9]+", name.lower())
    return "".join(w for w in words if w not in _DROP) or "".join(words)


def vrs_lookup(path):
    """(date, assign, info) from data/vrs.py output.

    assign({team: roster}) -> {team: entry} matches every hub team to a global
    VRS entry in one pass, so no entry is given to two teams: first the same
    normalised name ("Team Vitality" == "Vitality"), then one name starting
    the other ("Betclic Apogee Esports" / "Betclic"), then roster overlap
    (>= 3 shared players, for renamed teams). Each pass only uses entries no
    earlier pass claimed. info(entry) -> {"rank", "points", "name", "region",
    "region_rank"} for the page."""
    if not os.path.exists(path):
        return None, lambda rosters: {}, lambda r: None, []
    vrs = json.load(open(path, encoding="utf-8"))
    glob = vrs["standings"].get("global", [])
    regional = {}
    for region, rows in vrs["standings"].items():
        if region == "global":
            continue
        for r in rows:
            regional[(r["name"], tuple(sorted(p.lower() for p in r["roster"])))] = (region, r["rank"])

    def info(best):
        region, rrank = regional.get((best["name"], tuple(sorted(p.lower() for p in best["roster"]))), (None, None))
        return {"rank": best["rank"], "points": best["points"], "name": best["name"],
                "region": region, "region_rank": rrank}

    def assign(rosters):
        rosters = {t: {p.lower() for p in r} for t, r in rosters.items()}
        overlap = lambda t, r: len(rosters[t] & {p.lower() for p in r["roster"]})
        out, taken = {}, set()

        def claim(t, cands):
            cands = [r for r in cands if id(r) not in taken]
            if cands:
                best = max(cands, key=lambda r: (overlap(t, r), -r["rank"]))
                out[t] = best
                taken.add(id(best))

        for t in sorted(rosters):
            claim(t, [r for r in glob if _norm(r["name"]) == _norm(t)])
        for t in sorted(x for x in rosters if x not in out):
            k = _norm(t)
            if len(k) >= 4:
                claim(t, [r for r in glob if len(_norm(r["name"])) >= 4
                          and (k.startswith(_norm(r["name"])) or _norm(r["name"]).startswith(k))])
        for t in sorted((x for x in rosters if x not in out), key=lambda x: -len(rosters[x])):
            claim(t, [r for r in glob if overlap(t, r) >= 3])
        return out

    return vrs["date"], assign, info, glob


VMAPS = []                    # map names referenced by the coded BO1 / BO5 vetoes
_VCODE = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_VSTEP = {("A", "bans"): "a", ("B", "bans"): "b", ("A", "picks"): "A", ("B", "picks"): "B"}


def veto_code(log):
    """A veto log as two characters per step: the step kind (a / b = A / B ban,
    x / y = A / B permaban, A / B = A / B pick, D = decider) then the map as
    one character indexing DATA.vmaps. 'A bans Dust2', ... -> 'a0b1A2...D6'."""
    out = []
    for step in log:
        if step.startswith("decider: "):
            kind, mp = "D", step[len("decider: "):]
        else:
            side, verb, mp = step.split(" ", 2)
            kind = _VSTEP[(side, verb)]
            if mp.endswith(" (permaban)"):
                kind, mp = ("x" if side == "A" else "y"), mp[:-len(" (permaban)")]
        if mp not in VMAPS:
            VMAPS.append(mp)
        out.append(kind + _VCODE[VMAPS.index(mp)])
    return "".join(out)


def compact(r, bo=3):
    """Engine output trimmed to what the page draws.

    Probabilities stay at full precision, except the BO5 per-map chances
    (p5, 5 decimals, page payload only). The page formats a set (the two
    teams, a map, the four scorelines) with largest-remainder rounding so the
    labels add to 100.00. v1 / v5 are the BO1 / BO5 vetoes in veto_code form;
    the BO1 map chance is p, and the BO5 maps are v5's picks and decider.
    """
    s = r["series_probs_exact"]
    fac = {f["factor"]: f["marginal_pp"] for f in r["factor_breakdown"]}
    v5 = r.get("veto_bo5")
    return {
        "p": r["p_a_exact"],
        "b": r["confidence_interval"],
        "r": r["reliability"][0],
        "s": [s["p_2_0"], s["p_2_1"], s["p_1_2"], s["p_0_2"]],
        "vm": r["veto_maps"], "vp": r["map_probs_exact"],
        "v": [re.sub(r"^(A|B) (bans|picks) ", lambda m: m.group(1) + ("-" if m.group(2) == "bans" else "+"), x)
              .replace("decider: ", "D") for x in r["veto_log"]],
        "f": [round(fac.get(k, 0.0), 1) for k in FACTORS],
        "w": r["warnings"],
        "v1": veto_code(r["veto_bo1"]["veto_log"]),
        # win chance and likely range by series length: [BO1, BO5] (p / b above are BO3)
        "pf": [round(r["p_a_bo1_exact"], 6), round(r["p_a_bo5_exact"], 6)],
        "bf": [r["confidence_interval_bo1"], r["confidence_interval_bo5"]],
        **({"v5": veto_code(v5["veto_log"]), "p5": [round(p, 5) for p in v5["map_probs_exact"]]} if v5 else {}),
        # best-of-five matches also carry the exact BO5 scorelines (3-0 .. 0-3)
        **({"s5": [r["series_probs_bo5_exact"][k] for k in pr.BO5_KEYS]} if bo == 5 else {}),
    }


class Timeline:
    """Fold series day by day so a prediction for day D sees only days < D."""

    def __init__(self, matches):
        self.ms = sorted(matches, key=lambda m: m["date"])
        self.h = bt.History()
        self.i = 0
        self.day = None

    def at(self, day):
        assert self.day is None or day >= self.day, "days must be requested in order"
        while self.i < len(self.ms) and bt._d(self.ms[self.i]["date"]) < day:
            self.h.add(self.ms[self.i])
            self.i += 1
        self.day = day
        return self.h


def predict(h, a, b, day, title=None, bo=3, elo=False):
    """Engine call with point-in-time features; `title` (the tournament page)
    lets the feature builder attach stand-in / missing-IGL flags."""
    inp, _ = h.features({"team_a": a, "team_b": b, "date": day.isoformat(), "event_title": title})
    out = compact(pr.predict_match(inp), bo)
    if elo:   # pre-match Elo of both teams, so the track record can tell favourite from underdog
        out["e"] = [round(bt.ELO_INIT + (inp[k] - 1.0) * bt.ELO_PER_RATING) for k in ("rating_a", "rating_b")]
    return out


def main(out_path):
    assets = json.load(open(os.path.join(HERE, "assets.json"), encoding="utf-8"))
    all_matches = sorted(bt.load_matches(), key=lambda m: m["date"])
    events = sorted(E.discover(), key=lambda e: e["start"])
    last_day = max(bt._d(m["date"]) for m in all_matches)
    as_of = last_day + timedelta(days=1)

    # ---- status, champion
    for e in events:
        s, en = date.fromisoformat(e["start"]), date.fromisoformat(e["end"])
        e["status"] = "finished" if en < as_of else ("upcoming" if s > as_of else "live")
        done = [m for m in e["matches"] if m["finished"] and m["t1"] and m["t2"]]
        gf = [m for m in done if m["stage"] == "Grand final"]
        last = (gf or sorted(done, key=lambda m: (m["day"] or "", m["when"] or "")))[-1:] if done else []
        e["champion"] = None
        e["runner_up"] = None
        if e["status"] == "finished" and last:
            m = last[0]
            e["champion"] = m["t1"] if m["w1"] > m["w2"] else m["t2"]
            e["runner_up"] = m["t2"] if e["champion"] == m["t1"] else m["t1"]
        # each team's own finish, derived from the results and the prize slots
        e["placements"] = E.placements(e, e["status"])

    # ---- teams: every participant, every team in a match, plus the top active teams
    h_now = bt.History()
    for m in all_matches:
        h_now.add(m)
    active = [t for t, g in h_now.games.items() if g and g[-1]["date"] >= as_of - timedelta(days=90)]
    rank = {t: i + 1 for i, t in enumerate(sorted(active, key=lambda t: -h_now.elo[t]))}
    top = [t for t in sorted(active, key=lambda t: -h_now.elo[t]) if len(h_now.games[t]) >= 15][:16]
    # the matchup comparer's pool: every event team, the top active teams and
    # the VRS top VRS_TOP_N
    pool = set(E.event_teams(events)) | set(top)
    names = sorted(set(E.event_teams(events)) | set(top))

    trace = defaultdict(list)
    hh = bt.History()
    for m in all_matches:
        hh.add(m)
        for t in (m["team_a"], m["team_b"]):
            trace[t].append([m["date"], round(hh.elo[t])])

    players = {}
    for t, rec in assets["teams"].items():
        for p in rec["roster"]:
            players.setdefault(p["id"].lower(), {"id": p["id"], "name": p["name"], "flag": p["flag"], "igl": p["igl"]})
    for e in events:
        for v in e["participants"].values():
            for pid, cc in v["flags"].items():
                players.setdefault(pid.lower(), {"id": pid, "name": "", "flag": cc, "igl": False})
                if not players[pid.lower()]["flag"]:
                    players[pid.lower()]["flag"] = cc

    colors = TC.all_team_colors(assets, ROOT)
    vrs_date, vrs_assign, vrs_info, vrs_glob = vrs_lookup(os.path.join(ROOT, "data", "vrs.json"))
    latest_lineup = {}
    for e in events:                     # newest event lineup per team (events are date-sorted)
        for t, v in e["participants"].items():
            if v["players"]:
                latest_lineup[t] = v["players"]
    # VRS ranks for every team with series in the data, assigned in one pass so
    # no rank appears on two cards. Every global top-VRS_TOP_N team gets a card:
    # the data team it matched, else a card under its VRS name (no series yet).
    # Player-overlap matching only for teams with their own Liquipedia squad: a
    # national selection (e.g. Team Brazil at the Nations Cup) shares players with a
    # club and must not inherit that club's rank. Others match by name only.
    def own_squad(t):
        return [p["id"] for p in assets["teams"].get(t, {"roster": []})["roster"] if p["role"].lower() != "coach"]
    rosters = {t: own_squad(t) + (latest_lineup.get(t, []) if own_squad(t) else [])
               for t in set(names) | {t for t, g in h_now.games.items() if g}}
    vrs_of = vrs_assign(rosters)
    pool |= {t for t, r in vrs_of.items() if r["rank"] <= E.VRS_TOP_N}
    names = sorted(set(names) | {t for t, r in vrs_of.items() if r["rank"] <= E.VRS_TOP_N})
    claimed = {id(r) for r in vrs_of.values()}
    for r in vrs_glob:
        if r["rank"] <= E.VRS_TOP_N and id(r) not in claimed and r["name"] not in vrs_of:
            vrs_of[r["name"]] = r
            names.append(r["name"])
            pool.add(r["name"])
    # Only teams with a current lineup (at least one player, not just staff, on
    # their Liquipedia team page) get a card, a page and a place in the matchup
    # pickers. Others (national selections, a team whose players all left)
    # keep their results in the rating history and show as plain names.
    def has_lineup(t):
        # a team needs its own Liquipedia team page (logo, roster) with at least one
        # active player; teams with no page at all (e.g. BBL Esports, Haunted House)
        # or no active players (3DMAX after its players left) are left off the site
        return t in assets["teams"] and bool(own_squad(t))
    dropped = sorted(t for t in set(names) if not has_lineup(t))
    names = sorted(t for t in set(names) if has_lineup(t))
    pool = {t for t in pool if has_lineup(t)}
    if dropped:
        print(f"teams without a lineup left off the site ({len(dropped)}): {', '.join(dropped)}")
    teams = {}
    for t in names:
        rec = assets["teams"].get(t, {"roster": [], "location": "", "region": ""})
        f = h_now.team_feats(t, as_of)
        results = []
        for m in reversed(all_matches):
            if t not in (m["team_a"], m["team_b"]):
                continue
            opp = m["team_b"] if m["team_a"] == t else m["team_a"]
            sc = bt.actual_scoreline(m)
            if sc and m["team_a"] != t:
                sc = "-".join(reversed(sc.split("-")))
            results.append({"date": m["date"], "opp": opp, "won": m["winner"] == t,
                            "score": sc or ("W" if m["winner"] == t else "L"), "event": m["event"]})
            if len(results) >= N_TEAM_RESULTS:
                break
        roster = [{"id": p["id"], "coach": p["role"].lower() == "coach"} for p in rec["roster"]]
        teams[t] = {
            # no series in the data -> no Elo (it would only be the 1500 starting value)
            "slug": E.slug(t), "elo": round(h_now.elo[t]) if h_now.games[t] else None, "rank": rank.get(t),
            "color": colors.get(t, [None, None]),
            "vrs": vrs_info(vrs_of[t]) if t in vrs_of else None,
            "series": len(h_now.games[t]),
            # in the all-pairs matchup grid (DATA.pairs)
            "cmp": t in pool and bool(h_now.games[t]),
            "location": rec.get("location", ""), "region": rec.get("region", ""),
            "liquipedia": "https://liquipedia.net/counterstrike/" + t.replace(" ", "_"),
            "roster": roster,
            "form30": f.get("form30"), "n30": f.get("n30", 0), "vol": f["volatility"],
            # plain 90-day map win rates for display (the engine reads Elo-adjusted ones)
            "maps": f["maps_raw"], "results": results, "trace": trace[t][-40:],
            "events": [e["slug"] for e in events if t in e["participants"]
                       or any(t in (m["t1"], m["t2"]) for m in e["matches"])],
        }

    # ---- predictions, in date order so each sees only earlier days
    jobs = defaultdict(list)            # day -> [callable(h, day)]
    for e in events:
        for m in e["matches"]:
            # every match between two teams with results gets its pre-match call,
            # including teams left off the site for having no current lineup, so
            # the record on Stats is not filtered by which teams still exist
            if m["t1"] and m["t2"] and h_now.games.get(m["t1"]) and h_now.games.get(m["t2"]) and m["day"]:
                d = min(date.fromisoformat(m["day"]), as_of)
                jobs[d].append(("match", (m, e["title"])))
        if e["status"] == "finished":
            jobs[date.fromisoformat(e["start"])].append(("event", e))
    epairs = {}
    tl = Timeline(all_matches)
    for d in sorted(jobs):
        h = tl.at(d)
        for kind, obj in jobs[d]:
            if kind == "match":
                m, title = obj
                m["pred"] = predict(h, m["t1"], m["t2"], d, title, m.get("bo") or 3, elo=True)
            else:
                ps = [t for t in obj["participants"] if t in teams]
                epairs[obj["slug"]] = {f"{a}|{b}": predict(h, a, b, d, obj["title"]) for a in ps for b in ps if a != b}
    h_today = tl.at(as_of)
    rated = [t for t in names if teams[t]["cmp"]]   # a team with no series has nothing to predict from
    pairs = {f"{a}|{b}": predict(h_today, a, b, as_of) for a in rated for b in rated if a != b}
    # live and upcoming tournaments get their own matchup grid too, so that
    # announced stand-ins for that tournament are applied
    for e in events:
        if e["status"] != "finished":
            ps = [t for t in e["participants"] if t in teams]
            epairs[e["slug"]] = {f"{a}|{b}": predict(h_today, a, b, as_of, e["title"])
                                 for a in ps for b in ps if a != b}

    h2h = defaultdict(list)
    for m in reversed(all_matches):
        a, b = m["team_a"], m["team_b"]
        if a in teams and b in teams:
            key = "|".join(sorted([a, b]))
            if len(h2h[key]) < 6:
                h2h[key].append({"date": m["date"], "event": m["event"], "a": a, "b": b,
                                 "winner": m["winner"], "score": bt.actual_scoreline(m)})

    rows = bt.build_dataset(all_matches)
    ev_rows, _ = bt.select_eval(rows, bt.MIN_HISTORY)
    _, test = bt.chrono_split(ev_rows, 0.6)
    recent = []
    for r in test[-N_RECENT_CALLS:][::-1]:
        res = pr.predict_match(dict(r["input"]))
        m = r["match"]
        recent.append({"date": m["date"], "event": m["event"], "a": m["team_a"], "b": m["team_b"],
                       "winner": m["winner"], "p": round(res["p_a_exact"], 3)})
    full = json.load(open(os.path.join(ROOT, "data", "backtest_report.json"), encoding="utf-8"))
    tm = full["test_metrics"]
    report = {
        "data": {k: full["data"][k] for k in ("matches_total", "eval_series", "train", "test",
                                              "train_range", "test_range")},
        "models": {k: {"accuracy": tm[k]["accuracy"], "brier": tm[k]["brier"],
                       "calibration": tm[k].get("reliability_table", [])}
                   for k in ("new_engine", "elo_only", "old_engine_v1")},
    }

    ev_out = []
    for e in events:
        ev_out.append({k: e[k] for k in ("slug", "name", "start", "end", "tier", "tl", "type", "city", "country",
                                         "venue", "prize", "organizer", "swiss", "format", "prizes", "pool",
                                         "status", "champion", "runner_up", "placements", "page")}
                      | {"participants": [{"team": t, **v} for t, v in e["participants"].items()],
                         "matches": e["matches"]})

    out_dir = os.path.dirname(os.path.abspath(out_path))
    imw = ImageWriter(out_dir)
    images = {
        "logos": {t: imw.url(assets["teams"][t]["logo_dark_path"]) for t in names
                  if t in assets["teams"] and assets["teams"][t].get("logo_dark_path")},
        "maps": {mp: imw.url(v["path"]) for mp, v in assets["maps"].items()},
        "flags": {c: imw.url(p) for c, p in assets["flags"].items() if p},
        "events": {s: imw.url(p) for s, p in assets["event_logos"].items() if p},
    }
    pruned = imw.prune()
    if PAGE_DECIMALS is not None:
        pairs = round_floats(pairs, PAGE_DECIMALS)
        epairs = round_floats(epairs, PAGE_DECIMALS)
        for e in ev_out:
            e["matches"] = [m | {"pred": round_floats(m["pred"], PAGE_DECIMALS)} if "pred" in m else m
                            for m in e["matches"]]
    data = {
        "as_of": as_of.isoformat(), "events": ev_out, "teams": teams, "players": players,
        "pairs": pairs, "epairs": epairs, "h2h": h2h, "recent": recent, "report": report,
        "map_info": {mp: {"location": v.get("location", "")} for mp, v in assets["maps"].items()},
        "factors": FACTORS, "vrs_date": vrs_date, "vmaps": VMAPS,
    }
    tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    img = json.dumps(images, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/null", blob).replace("/*__IMAGES__*/null", img)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    n_m = sum(len(e["matches"]) for e in events)
    n_p = sum(1 for e in events for m in e["matches"] if "pred" in m)
    print(f"wrote {out_path}: {len(events)} events, {len(teams)} teams, {n_m} matches "
          f"({n_p} pre-match calls), {len(pairs)} + {sum(len(v) for v in epairs.values())} matchups, "
          f"{len(html.encode('utf-8')) // 1024} KB (data {len(blob.encode('utf-8')) // 1024} KB); "
          f"{len(imw.names)} images in {IMG_DIR}/ ({imw.bytes // 1024} KB, {pruned} stale removed)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "faceoff_viewer.html"))
