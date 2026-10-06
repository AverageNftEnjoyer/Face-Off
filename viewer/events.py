"""
Tournament discovery and parsing for the Faceoff hub, from the Liquipedia
pages already cached by data/collect_liquipedia.py (no network).

An event is a top-level tournament page (plus its stage subpages, e.g.
".../Cologne/Stage 1"). For each one we read the infobox, format, prize pool,
map pool, participants with their event lineups ({{TeamParticipants}} or the
older {{TeamCard}}), and every {{Match}} with per-map round scores.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))

import collect_liquipedia as C  # noqa: E402
import lpfetch  # noqa: E402

YEARS = ("2026",)
MAX_DAYS = 45          # longer "events" are season circuits, not tournaments
TZ_OFFSETS = {"CEST": "+02:00", "CET": "+01:00", "UTC": "+00:00", "EEST": "+03:00",
              "BST": "+01:00", "EDT": "-04:00", "EST": "-05:00", "CDT": "-05:00",
              "PDT": "-07:00", "BRT": "-03:00", "SGT": "+08:00", "CST": "+08:00",
              "MSK": "+03:00", "AST": "+03:00", "GST": "+04:00", "KST": "+09:00",
              "AEST": "+10:00", "ALMT": "+05:00"}


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def strip_markup(s):
    s = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", s or "", flags=re.S)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"\{\{abbr\|([^|}]*)\|[^}]*\}\}", r"\1", s, flags=re.I)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\S+\s+([^\]]+)\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    return s.replace("&nbsp;", " ").replace("'''", "").replace("''", "").strip()


def infobox(txt, name):
    for variant in (name, name[:8] + name[8:].capitalize(), name[:8] + name[8:].lower(), "HiddenDataBox"):
        for _, body in C.find_templates(txt or "", variant):
            return C.split_params(body)[1]
    return {}


def load_texts():
    titles = [l.strip() for l in open(os.path.join(ROOT, "data", "raw", "lp_titles_selected.txt"),
                                      encoding="utf-8") if l.strip()]
    texts = {}
    for b in batched(titles, 8):
        texts.update({k: v for k, v in lpfetch.wikitext(b, offline=True).items() if v})
    return titles, texts


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ------------------------------------------------------------------ matches
def parse_when(s):
    d = C.parse_date(s)
    tm = re.search(r"-\s*(\d{1,2}):(\d{2})", s or "")
    tz = re.search(r"Abbr/([A-Z]+)", s or "")
    if not d or not tm:
        return d, None
    off = TZ_OFFSETS.get(tz.group(1) if tz else "UTC", "+00:00")
    return d, f"{d}T{int(tm.group(1)):02d}:{tm.group(2)}:00{off}"


def _rounds(nm, team):
    sides, ot = {"ct": 0, "t": 0}, 0
    for k, v in nm.items():
        mm = re.fullmatch(r"(o\d+)?t%d(t|ct)" % team, k)
        n = C._int(v)
        if not mm or n is None:
            continue
        if mm.group(1):
            ot += n
        else:
            sides[mm.group(2)] += n
    return sides["ct"] + sides["t"] + ot, sides


def parse_map(body):
    for _, mb in C.find_templates(body, "Map"):
        _, nm = C.split_params(mb)
        name = nm.get("map", "").strip()
        if not name:
            return None
        name = C.MAP_NORMALIZE.get(name.lower(), name)
        s1, side1 = _rounds(nm, 1)
        s2, side2 = _rounds(nm, 2)
        if not (C._finished(nm.get("finished", "")) and s1 + s2 > 0 and s1 != s2):
            w = C._int(nm.get("winner"))
            if w not in (1, 2) or s1 + s2 > 0:
                return None
            # map decided but no round data on the page
            return {"map": name, "s1": None, "s2": None, "w": w, "side1": side1, "side2": side2,
                    "ot": 0, "first1": None, "vod": None}
        return {"map": name, "s1": s1, "s2": s2, "w": 1 if s1 > s2 else 2,
                "side1": side1, "side2": side2,
                "ot": (s1 + s2) - sum(side1.values()) - sum(side2.values()),
                "first1": nm.get("t1firstside", "").strip().lower() or None,
                "vod": nm.get("vod", "").strip() or None}
    return None


ELIM_NAMES = ["Grand final", "Semifinal", "Quarterfinal", "Round of 16", "Round of 32"]


def headings(txt):
    return [(m.start(), len(m.group(1)), strip_markup(re.sub(r"\{\{Stage\|([^}]*)\}\}", r"\1", m.group(2))).strip())
            for m in re.finditer(r"^(=+)([^=\n]+)\1\s*$", txt, re.M)]


def section_label(heads, pos):
    path = {}
    for hpos, level, title in heads:
        if hpos > pos:
            break
        if level <= 2:
            path = {}
            continue
        path = {k: v for k, v in path.items() if k < level}
        path[level] = title
    parts = [v for k, v in sorted(path.items()) if v not in ("Results", "Detailed Results", "Overview")]
    return " ".join(parts)


def parse_matches(title, txt, alias, stage_prefix=""):
    heads = headings(txt)
    brackets = [(m.start(), m.group(1)) for m in re.finditer(r"\{\{\s*Bracket\s*\|\s*Bracket/([^|\s}]+)", txt)]
    raw = []
    show = C.showmatch_spans(txt)
    for start, body in C.find_templates(txt, "Match"):
        if any(a <= start < b for a, b in show):
            continue      # exhibition matches are not part of the tournament
        _, nm = C.split_params(body)
        k1, c1, _ = C.parse_opponent(nm.get("opponent1"))
        k2, c2, _ = C.parse_opponent(nm.get("opponent2"))
        t1 = alias.get(c1, c1) if (k1 == "TeamOpponent" and c1) else None
        t2 = alias.get(c2, c2) if (k2 == "TeamOpponent" and c2) else None
        if C.is_showmatch(t1 or "") or C.is_showmatch(t2 or ""):
            continue
        day, when = parse_when(nm.get("date"))
        sec = section_label(heads, start)
        key = re.search(r"\|(R\w+?)M(\w+)=\s*$", txt[max(0, start - 40):start])
        bkt = [b for p, b in brackets if p < start]
        slots = sorted([k for k in nm if re.fullmatch(r"map\d+", k)], key=lambda k: int(k[3:]))
        maps = [pm for pm in (parse_map(nm[k]) for k in slots) if pm]
        w1 = sum(1 for m in maps if m["w"] == 1)
        w2 = sum(1 for m in maps if m["w"] == 2)
        bo = C._int(nm.get("bestof")) or (len(slots) if len(slots) in (1, 3, 5) else 3)
        need = bo // 2 + 1
        finished = max(w1, w2) >= need
        if not finished and C._finished(nm.get("finished", "")):
            mw = C._int(nm.get("winner"))
            if mw in (1, 2):           # decided without full map data (e.g. forfeit)
                finished = True
                w1, w2 = (need, w2) if mw == 1 else (w1, need)
        raw.append({"t1": t1, "t2": t2, "day": day, "when": when, "sec": sec,
                    "rkey": key.group(1) if key else None, "mkey": key.group(2) if key else None,
                    "bracket": bkt[-1] if bkt else None,
                    "bo": bo, "finished": finished, "w1": w1, "w2": w2, "maps": maps,
                    "hltv": nm.get("hltv", "").strip() or None})
    # name bracket rounds: single-elimination brackets ("Bracket/8") count back from the final
    by_bracket = {}
    for m in raw:
        if m["rkey"] and m["rkey"].startswith("R") and m["rkey"][1:].isdigit():
            by_bracket.setdefault((m["sec"], m["bracket"]), []).append(int(m["rkey"][1:]))
    for m in raw:
        base = (stage_prefix + " " + m["sec"]).strip() or "Matches"
        if m["mkey"] == "TP":
            m["stage"], m["kind"] = "Third-place match", "playoffs"
        elif m["rkey"] and m["rkey"][1:].isdigit():
            r = int(m["rkey"][1:])
            top = max(by_bracket[(m["sec"], m["bracket"])])
            single = bool(m["bracket"]) and m["bracket"].isdigit()
            if single and top - r < len(ELIM_NAMES):
                m["stage"] = ELIM_NAMES[top - r]
            else:
                m["stage"] = f"{base}, round {r}"
            m["kind"] = "playoffs"
        else:
            m["stage"], m["kind"] = base, "groups"
        for k in ("rkey", "mkey", "bracket", "sec"):
            m.pop(k)
    return raw


# ------------------------------------------------------------------ participants
def parse_participants(txt, alias):
    """{team: {"players": [ids], "coach": [ids], "subs": [ids], "flags": {id: cc}}}"""
    out = {}
    show = C.showmatch_spans(txt)
    for start, body in C.find_templates(txt, "Opponent"):
        if any(a <= start < b for a, b in show):
            continue
        pos, nm = C.split_params(body)
        team = (pos[0] if pos else "").strip()
        # team entries carry a players= list (possibly still empty); player-type
        # opponents elsewhere on a page (awards, solo brackets) do not
        if (not team or "players" not in nm or team.lower() in ("tbd", "tba") or "{" in team
                or C.is_showmatch(team)):
            continue
        rec = {"players": [], "coach": [], "subs": [], "flags": {}}
        for _, pb in C.find_templates(nm.get("players", ""), "Person"):
            ppos, pnm = C.split_params(pb)
            pid = (ppos[0] if ppos else pnm.get("id", "")).strip()
            if not pid:
                continue
            if pnm.get("flag"):
                rec["flags"][pid] = pnm["flag"].strip().lower()
            role = pnm.get("role", "").lower()
            if role == "coach":
                rec["coach"].append(pid)
            elif pnm.get("status", "").lower() == "sub" and pnm.get("played", "").lower() in ("false", "no", "n"):
                rec["subs"].append(pid)
            else:
                rec["players"].append(pid)
        out[team] = rec
    for start, body in C.find_templates(txt, "TeamCard"):
        if any(a <= start < b for a, b in show):
            continue
        _, nm = C.split_params(body)
        team = strip_markup(nm.get("team", ""))
        if not team:
            continue
        team = alias.get(team.lower(), team)
        rec = {"players": [], "coach": [], "subs": [], "flags": {}}
        for i in range(1, 8):
            pid = strip_markup(nm.get(f"p{i}", ""))
            if pid:
                rec["players"].append(pid)
                if nm.get(f"p{i}flag"):
                    rec["flags"][pid] = nm[f"p{i}flag"].strip().lower()
        for k in ("c", "c1", "coach"):
            if strip_markup(nm.get(k, "")):
                rec["coach"].append(strip_markup(nm[k]))
        out.setdefault(team, rec)
    return out


# ------------------------------------------------------------------ events
def discover(alias=None):
    alias = alias if alias is not None else C.cached_aliases()
    titles, texts = load_texts()
    tops = [t for t in titles if not any(t != o and t.startswith(o + "/") for o in titles)]
    events, seen = [], set()
    for t in tops:
        txt = texts.get(t, "")
        ib = infobox(txt, "Infobox league")
        name = strip_markup(ib.get("name", "")) or t
        sd, ed = ib.get("sdate", "")[:10], ib.get("edate", "")[:10]
        if not sd or not ed or (name, sd) in seen or sd[:4] not in YEARS:
            continue
        try:
            from datetime import date
            if (date.fromisoformat(ed) - date.fromisoformat(sd)).days > MAX_DAYS:
                continue
        except ValueError:
            continue
        subs = [s for s in titles if s.startswith(t + "/") and texts.get(s)]
        matches = parse_matches(t, txt, alias)
        parts = parse_participants(txt, alias)
        for s in subs:
            sib = infobox(texts[s], "Infobox league")
            prefix = s[len(t) + 1:]
            matches += parse_matches(s, texts[s], alias, stage_prefix=prefix)
            for k, v in parse_participants(texts[s], alias).items():
                parts.setdefault(k, v)
        named = {m["t1"] for m in matches if m["t1"]} | {m["t2"] for m in matches if m["t2"]}
        if not parts and not named and sd < C.TODAY:
            continue          # past page with no teams at all: an unfilled placeholder
        seen.add((name, sd))
        fmt = re.search(r"===\s*Format\s*===\n(.*?)\n==", txt, re.S)
        fmt_lines = []
        for line in (fmt.group(1).splitlines() if fmt else []):
            if not line.startswith("*") or "Explained" in line:
                continue
            depth = len(line) - len(line.lstrip("*"))
            text = re.sub(r"\{\{Abbr/Bo(\d)\}\}", r"Bo\1", line.lstrip("* "))
            text = strip_markup(text)
            if text:
                fmt_lines.append([depth, text])
        prizes, place = [], 1
        for _, body in C.find_templates(txt, "Slot"):
            _, nm = C.split_params(body)
            cnt = C._int(nm.get("count")) or 1
            usd = C._int(re.sub(r"[^\d]", "", nm.get("usdprize", "")) or None)
            prizes.append({"from": place, "to": place + cnt - 1, "usd": usd})
            place += cnt
        pool = []
        for i in range(1, 12):
            mp = ib.get(f"map{i}", "").strip()
            if mp:
                pool.append(C.MAP_NORMALIZE.get(mp.lower(), mp))
        for i, m in enumerate(sorted(matches, key=lambda m: (m["day"] or "9999", m["when"] or ""))):
            m["id"] = f"{slug(name)[:24]}-{i + 1}"
        events.append({
            "title": t, "name": name, "slug": slug(name), "start": sd, "end": ed,
            "tier": ib.get("liquipediatier", "").strip(), "type": ib.get("type", "").strip(),
            "city": strip_markup(ib.get("city", "")), "country": strip_markup(ib.get("country", "")),
            "venue": strip_markup(ib.get("venue", "")), "prize": ib.get("prizepoolusd", "").strip(),
            "organizer": strip_markup(ib.get("organizer", "")),
            "logo_file": (ib.get("imagedark") or ib.get("image") or "").strip(),
            "swiss": "swiss" in (fmt.group(1).lower() if fmt else "") or "SwissTable" in txt,
            "format": fmt_lines, "prizes": prizes, "pool": pool,
            "participants": parts, "matches": sorted(matches, key=lambda m: m["id"]),
            "page": "https://liquipedia.net/counterstrike/" + t.replace(" ", "_"),
        })
    return events


def event_teams(events):
    out = set()
    for e in events:
        out |= set(e["participants"])
        out |= {m["t1"] for m in e["matches"] if m["t1"]} | {m["t2"] for m in e["matches"] if m["t2"]}
    return sorted(out)
