"""Build data/matches.json from cached Liquipedia wikitext (real results only).

Usage (from D:/Face-Off):  python data/collect_liquipedia.py [--offline]

Pipeline
  1. read page titles from data/raw/lp_titles_selected.txt
  2. fetch raw wikitext (action=query&prop=revisions, cached in
     data/raw/liquipedia/, polite rate limit -- see lpfetch.py)
  3. parse every {{Match ...}} template: opponents, date, per-map results
  4. resolve Liquipedia team-template codes (e.g. "tl") to team names via
     the Template:TeamShort/<code> redirects (also cached)
  5. de-duplicate (redirected pages appear under several titles), keep
     finished matches only, write data/matches.json and data/team_aliases.json

Nothing is imputed: a map winner is recorded only when the page gives a
winner or round scores; a series without parsable maps keeps maps=[] and
its winner comes from the series score / winner field, else it is dropped.
"""
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lpfetch  # noqa: E402

OFFLINE = "--offline" in sys.argv
# Series dated after today are excluded; only decided series are parsed, so
# today's finished matches count. Override
# with FACEOFF_TODAY=YYYY-MM-DD to reproduce an earlier snapshot exactly.
TODAY = os.environ.get("FACEOFF_TODAY") or datetime.utcnow().strftime("%Y-%m-%d")
CS2_START = "2023-09-27"   # CS2 official release; earlier matches are CS:GO

MAP_NORMALIZE = {"dust ii": "Dust2", "dust2": "Dust2", "de_dust2": "Dust2",
                 "mirage": "Mirage", "inferno": "Inferno", "nuke": "Nuke",
                 "ancient": "Ancient", "anubis": "Anubis", "vertigo": "Vertigo",
                 "overpass": "Overpass", "train": "Train", "cache": "Cache",
                 "cobblestone": "Cobblestone"}

# Exhibition / showmatch line-ups (national all-star teams, streamer teams,
# legends teams) appear as {{Match}} entries on event pages but are not
# competitive results; they are dropped.
SHOWMATCH_TEAMS = {
    "Team Australia", "Team United Kingdom", "Team Denmark", "Team India",
    "Team Pakistan", "Team Germany", "Team Poland", "Team United States",
    "BLAST Dream Team", "Danish Squad", "Team DD", "Team captainMo", "Team HAI",
    "Team PGL", "Team Winline", "Stream Team", "Team GOAT", "Team VACO",
    "Team StRoGo", "Team buster",
}


_SHOW_LC = {t.lower() for t in SHOWMATCH_TEAMS}


def is_showmatch(name):
    """Applied to the RESOLVED name (case-insensitive) in both online and offline runs."""
    n = name.lower()
    return n in _SHOW_LC or "showmatch" in n


# ---------------------------------------------------------------- parsing
def find_templates(text, name):
    """Yield (start, body) of {{name ...}} templates (exact name), brace-balanced."""
    pat = re.compile(r"\{\{\s*" + re.escape(name) + r"\s*(?=[|\n}])")
    for m in pat.finditer(text):
        i, depth = m.start(), 0
        j = i
        n = len(text)
        while j < n:
            if text.startswith("{{", j):
                depth += 1
                j += 2
            elif text.startswith("}}", j):
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        yield i, text[m.end():j - 2]


def split_params(body):
    """Split a template body at top-level '|' -> (positional list, named dict)."""
    parts, depth_c, depth_s, cur = [], 0, 0, []
    i = 0
    while i < len(body):
        if body.startswith("{{", i):
            depth_c += 1; cur.append("{{"); i += 2; continue
        if body.startswith("}}", i):
            depth_c -= 1; cur.append("}}"); i += 2; continue
        if body.startswith("[[", i):
            depth_s += 1; cur.append("[["); i += 2; continue
        if body.startswith("]]", i):
            depth_s -= 1; cur.append("]]"); i += 2; continue
        ch = body[i]
        if ch == "|" and depth_c == 0 and depth_s == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    pos, named = [], {}
    for p in parts[1:] if parts and parts[0].strip() == "" else parts:
        p2 = re.sub(r"<!--.*?-->", "", p, flags=re.S)
        if "=" in p2 and re.match(r"^\s*[\w ]+\s*=", p2):
            k, v = p2.split("=", 1)
            named[k.strip()] = v.strip()
        else:
            pos.append(p2.strip())
    return pos, named


def parse_opponent(s):
    """Return (kind, code, score) from '{{TeamOpponent|tl|score=2}}' etc."""
    if not s:
        return None, None, None
    for kind in ("TeamOpponent", "LiteralOpponent", "TBDOpponent"):
        for _, body in find_templates(s, kind):
            pos, named = split_params(body)
            code = named.get("template") or (pos[0] if pos else None)
            sc = named.get("score")
            return kind, (code.strip().lower() if code else None), sc
    return None, None, None


def _int(x):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def _finished(fin):
    """Liquipedia's `finished` flag, tolerant of editor typos ("t", "trur", "trie",
    "y", "reyw" occur in the raw cache). Anything not explicitly unplayed counts;
    callers still require a decided winner (winner field or round scores)."""
    return fin.strip().lower() not in ("", "skip", "false", "f", "no", "n", "0")


def parse_map(s):
    """Return dict(map, winner in {1,2}) or None if not a played/decided map."""
    for _, body in find_templates(s, "Map"):
        pos, nm = split_params(body)
        fin = nm.get("finished", "").lower()
        mname = nm.get("map", "").strip()
        if fin in ("skip", "") and not nm.get("winner"):
            return None
        if not mname:
            return None
        mapn = MAP_NORMALIZE.get(mname.lower(), mname)
        w = _int(nm.get("winner"))
        if w not in (1, 2):
            s1 = _int(nm.get("score1"))
            s2 = _int(nm.get("score2"))
            if s1 is None or s2 is None:
                keys1 = [k for k in nm if re.fullmatch(r"(o\d+)?t1(t|ct)", k)]
                keys2 = [k for k in nm if re.fullmatch(r"(o\d+)?t2(t|ct)", k)]
                if not keys1 or not keys2:
                    return None
                v1 = [_int(nm[k]) for k in keys1]
                v2 = [_int(nm[k]) for k in keys2]
                if None in v1 or None in v2:
                    return None
                s1, s2 = sum(v1), sum(v2)
            if s1 == s2:
                return None
            w = 1 if s1 > s2 else 2
        if not _finished(fin) and not nm.get("winner"):
            return None
        return {"map": mapn, "w": w}
    return None


def parse_date(s):
    if not s:
        return None
    s = re.sub(r"\{\{.*?\}\}", "", s)
    s = s.split(" - ")[0].strip()
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def showmatch_spans(text):
    """(start, end) spans of showmatch sections: from a '{{show match' infobox or a
    '{{Stage|Showmatch}}' heading to the next level-2 heading (or end of page)."""
    spans = []
    for m in re.finditer(r"\{\{\s*show ?match\b|\{\{\s*Stage\s*\|\s*Showmatch\s*\}\}", text, flags=re.I):
        nxt = re.compile(r"\n==[^=]").search(text, m.end())
        spans.append((m.start(), nxt.start() if nxt else len(text)))
    return spans


def parse_page(title, text):
    out = []
    spans = showmatch_spans(text)
    for start, body in find_templates(text, "Match"):
        if any(a <= start < b for a, b in spans):
            continue   # exhibition match, not a competitive result
        pos, nm = split_params(body)
        k1, c1, s1 = parse_opponent(nm.get("opponent1"))
        k2, c2, s2 = parse_opponent(nm.get("opponent2"))
        if k1 != "TeamOpponent" or k2 != "TeamOpponent" or not c1 or not c2:
            continue
        date = parse_date(nm.get("date"))
        maps_raw = sorted([k for k in nm if re.fullmatch(r"map\d+", k)], key=lambda k: int(k[3:]))
        maps = []
        for k in maps_raw:
            pm = parse_map(nm[k])
            if pm:
                maps.append(pm)
        n_slots = len(maps_raw)
        bo = _int(nm.get("bestof"))
        if bo is None:
            bo = n_slots if n_slots in (1, 3, 5) else (3 if n_slots == 2 else None)
        if nm.get("walkover") or "walkover" in body.lower() and not maps:
            continue
        w1 = sum(1 for m in maps if m["w"] == 1)
        w2 = sum(1 for m in maps if m["w"] == 2)
        need = (bo // 2 + 1) if bo else None
        winner = None
        if need and (w1 >= need or w2 >= need):
            winner = 1 if w1 > w2 else 2
        else:
            # fall back to series score / winner field given on the page
            mw = _int(nm.get("winner"))
            a, b = _int(s1), _int(s2)
            if mw in (1, 2) and _finished(nm.get("finished", "")):
                winner = mw
            elif a is not None and b is not None and need and max(a, b) >= need:
                winner = 1 if a > b else 2
            if winner and maps and (w1 + w2) < need:
                maps_complete = False
        if not winner or not date:
            continue
        out.append({"date": date, "page": title, "c1": c1, "c2": c2, "winner": winner,
                    "best_of": bo, "maps": maps, "hltv": nm.get("hltv")})
    return out


def _parse_expansion(wikitext, alias):
    for line in wikitext.split(chr(10)):
        m = re.match(r"@@(.*?)@@(.*)", line)
        if not m:
            continue
        link = re.search(r'team-template-text">\[\[([^|\]]+)', m.group(2))
        if link:
            alias.setdefault(m.group(1), link.group(1).strip())


def cached_aliases():
    """Code -> team page name from EVERY cached expandtemplates response.
    Independent of how codes were batched, so offline rebuilds do not depend on
    reproducing the exact request text. data/team_aliases.json is an OUTPUT
    only (never read back), so a bad run cannot poison later runs."""
    alias = {}
    if os.path.isdir(lpfetch.CACHE):
        for fn in sorted(os.listdir(lpfetch.CACHE)):
            with open(os.path.join(lpfetch.CACHE, fn), encoding="utf-8") as f:
                resp = json.load(f).get("response", {})
            if "expandtemplates" in resp:
                _parse_expansion(resp["expandtemplates"].get("wikitext", ""), alias)
    return alias


def cached_page_redirects():
    """Plain page redirects seen in the cache, e.g. a renamed team's old or new
    page name -> the page Liquipedia serves ("Inner Circle Esports" ->
    "IC Esports"). Both the exact and the lower-case source are keys, matching
    how codes and page names are looked up. Namespaced (Template:, File:) and
    sub-page redirects (maps, tournaments) are left out."""
    out = {}
    if os.path.isdir(lpfetch.CACHE):
        for fn in sorted(os.listdir(lpfetch.CACHE)):
            with open(os.path.join(lpfetch.CACHE, fn), encoding="utf-8") as f:
                q = json.load(f).get("response", {}).get("query", {})
            for r in q.get("redirects", []):
                src, dst = r.get("from", ""), r.get("to", "")
                if src and dst and not any(ch in src + dst for ch in ":/"):
                    out.setdefault(src, dst)
                    out.setdefault(src.lower(), dst)
    return out


def resolve_codes(codes):
    """Map team-template codes -> Liquipedia team page name.

    Known codes come from the cache (cached_aliases). Online, unknown codes are
    resolved with action=expandtemplates on '{{Team|code}}' (100 codes per
    request, spaced >= 30 s like a parse request, cached). Offline, unknown codes
    are an error: we refuse to write a half-resolved dataset.
    """
    import time
    alias = cached_aliases()
    unknown = [c for c in codes if c not in alias]
    if unknown and OFFLINE:
        raise SystemExit("offline: %d team codes not in cache: %s" % (len(unknown), unknown[:20]))
    for i in range(0, len(unknown), 100):
        if i > 0:
            time.sleep(30)
        batch = unknown[i:i + 100]
        text = chr(10).join("@@%s@@{{Team|%s}}" % (c, c) for c in batch)
        resp = lpfetch.query({"action": "expandtemplates", "text": text, "prop": "wikitext"}) or {}
        _parse_expansion(resp.get("expandtemplates", {}).get("wikitext", ""), alias)
    return {c: alias.get(c, c) for c in codes}


def event_name(title, texts):
    t = title
    while True:
        txt = texts.get(t)
        if txt:
            m = re.search(r"\{\{Infobox league.*?\|\s*name\s*=\s*([^\n|]+)", txt, flags=re.S)
            if m:
                return m.group(1).replace('&nbsp;', ' ').strip()
        if "/" not in t:
            return title
        t = t.rsplit("/", 1)[0]


def main():
    titles = [l.strip() for l in open(os.path.join(HERE, "raw", "lp_titles_selected.txt"),
                                      encoding="utf-8") if l.strip()]
    texts = {}
    for i in range(0, len(titles), 8):
        batch = titles[i:i + 8]
        res = lpfetch.wikitext(batch, offline=OFFLINE)
        texts.update({k: v for k, v in res.items() if v})
    # parent pages that are not in the selected list (for event names) are optional
    raw = []
    for t, txt in texts.items():
        for m in parse_page(t, txt):
            m["event"] = event_name(t, texts)
            raw.append(m)
    # resolve team codes
    codes = sorted({m["c1"] for m in raw} | {m["c2"] for m in raw})
    alias = resolve_codes(codes)

    def disp(code):
        return alias.get(code) or code

    # dedupe and normalize
    seen = {}
    for m in raw:
        a, b = disp(m["c1"]), disp(m["c2"])
        if a == b or is_showmatch(a) or is_showmatch(b):
            continue
        key = (m["date"], tuple(sorted([a, b])), m["hltv"] or "")
        if m["date"] < CS2_START or m["date"] > TODAY:   # only decided series reach here, so today's finished matches are kept
            continue
        rec = {"date": m["date"], "event": m["event"], "team_a": a, "team_b": b,
               "winner": a if m["winner"] == 1 else b, "best_of": m["best_of"],
               "maps": [{"map": x["map"], "winner": a if x["w"] == 1 else b} for x in m["maps"]],
               "source": "liquipedia:" + m["page"] + (f" (hltv match {m['hltv']})" if m["hltv"] else "")}
        k2 = (m["date"], tuple(sorted([a, b])))
        # same pair + same date + same hltv id (or no id) -> duplicate via redirect
        dup = None
        for kk in list(seen):
            if kk[:2] == k2 and (kk[2] == key[2] or not kk[2] or not key[2]):
                if len(seen[kk]["maps"]) == len(rec["maps"]):
                    dup = kk
                    break
        if dup:
            continue
        seen[key] = rec
    matches = sorted(seen.values(), key=lambda r: (r["date"], r["event"], r["team_a"]))
    with open(os.path.join(HERE, "matches.json"), "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=1, ensure_ascii=False)
    used = {}
    for c in codes:
        used.setdefault(disp(c), set()).add(c)
    with open(os.path.join(HERE, "team_aliases.json"), "w", encoding="utf-8") as f:
        json.dump({"_note": "canonical name -> Liquipedia team-template codes seen; "
                            "resolved via Template:TeamShort/<code> redirects",
                   "aliases": {k: sorted(v) for k, v in sorted(used.items())}},
                  f, indent=1, ensure_ascii=False)
    bo = {}
    for m in matches:
        bo[m["best_of"]] = bo.get(m["best_of"], 0) + 1
    print(f"pages parsed: {len(texts)}  raw match records: {len(raw)}  unique finished: {len(matches)}")
    print("best_of counts:", bo)
    print("date range:", matches[0]["date"], "->", matches[-1]["date"])
    print("teams:", len({m['team_a'] for m in matches} | {m['team_b'] for m in matches}))


if __name__ == "__main__":
    main()
