#!/usr/bin/env python3
"""
Roster events (stand-ins, missing IGL) per team per tournament, from the
"Stand-ins" table on each Liquipedia team page.

Each {{Stand-in}} row names the stand-in player, the player they replaced
(`for`), whether the stand-in was for a coach (`role=Coach`, ignored here),
and the tournament(s) where it happened. A row becomes a roster event for
every listed tournament page. `missing_igl` is set when the replaced player
is marked `igl=y` anywhere in that team's squad history (active, inactive or
former entries), i.e. the stand-in covered the in-game leader.

Output: data/roster_events.json
    {"<team>": {"<tournament page title>": {"standin": true, "missing_igl": bool,
                                           "players": [[standin, replaced], ...]}}}

Limits (also in data/SOURCES.md):
  * a stand-in is applied to every match the team played at that tournament,
    even if they only covered some of them;
  * stand-ins that Liquipedia editors never recorded are missed;
  * the IGL check uses squad-history igl flags, which mark the player's role
    on that team, not necessarily on the exact date.

USAGE (from D:/Face-Off):  python data/rosters.py [--offline]
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import collect_liquipedia as C  # noqa: E402
import lpfetch  # noqa: E402

OFFLINE = "--offline" in sys.argv
LINK = re.compile(r"\[\[([^|\]]+)(?:\|[^\]]*)?\]\]")


def team_pages(teams):
    texts = {}
    for i in range(0, len(teams), 8):
        texts.update({k: v for k, v in lpfetch.wikitext(teams[i:i + 8], offline=OFFLINE).items() if v})
    return texts


def igl_ids(txt):
    out = set()
    for _, body in C.find_templates(txt, "Person"):
        _, nm = C.split_params(body)
        if nm.get("igl", "").strip().lower() in ("y", "yes", "true") and nm.get("id"):
            out.add(nm["id"].strip().lower())
    ib = re.search(r"\|\s*igl\s*=([^\n]*)", txt)
    if ib:
        for pid in LINK.findall(ib.group(1)):
            out.add(pid.strip().lower())
    return out


def standin_rows(txt):
    rows = []
    for _, body in C.find_templates(txt, "Stand-in"):
        _, nm = C.split_params(body)
        if nm.get("role", "").strip().lower() == "coach":
            continue
        sid, rep = nm.get("id", "").strip(), nm.get("for", "").strip()
        # links are written both "ESL/Pro_League/Season_24" and "ESL/Pro League/Season 24"
        tours = [t.strip().replace("_", " ") for t in LINK.findall(nm.get("tournament", ""))
                 if not t.startswith(("File:", "Category:"))]
        if sid and tours:
            rows.append((sid, rep, tours))
    return rows


def build(teams):
    texts = team_pages(teams)
    out = {}
    for team, txt in texts.items():
        igls = igl_ids(txt)
        for sid, rep, tours in standin_rows(txt):
            for tour in tours:
                ev = out.setdefault(team, {}).setdefault(tour, {"standin": True, "missing_igl": False, "players": []})
                ev["players"].append([sid, rep])
                if rep and rep.lower() in igls:
                    ev["missing_igl"] = True
    return out, texts


def main():
    ms = json.load(open(os.path.join(HERE, "matches.json"), encoding="utf-8"))
    teams = sorted({m["team_a"] for m in ms} | {m["team_b"] for m in ms})
    out, texts = build(teams)
    with open(os.path.join(HERE, "roster_events.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, sort_keys=True)
    n_rows = sum(len(v) for v in out.values())
    n_igl = sum(1 for v in out.values() for e in v.values() if e["missing_igl"])
    print(f"team pages {len(texts)}/{len(teams)}; teams with stand-ins {len(out)}; "
          f"team-tournament stand-in events {n_rows} ({n_igl} covering the IGL)")


if __name__ == "__main__":
    main()
