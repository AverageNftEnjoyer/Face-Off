#!/usr/bin/env python3
"""
backfill_lower.py -- slow, resumable, month-by-month Liquipedia backfill of
2025 lower-tier results (rating DATA only: never shown on the site, never
given model predictions).

WHAT IS FETCHED
  * B-tier main events of 2025 and their stage pages. Discovery: the cached
    `Category:B-Tier Tournaments` listing (all 2,709 members, read on
    2026-10-07). A title is a candidate when it names 2025, or names no year
    and was added to the category on or after 2024-01-01. Each candidate's
    own infobox decides, with the same rules daily_refresh.py uses for S/A
    (tier B, tier type not Qualifier / Showmatch / Weekly / Monthly / Misc /
    Points, sdate in 2025, at most events.MAX_DAYS = 45 days long; a
    sub-page that is a stage of a listed page or a regional sub-page is not
    an event of its own). Stage pages: one `allpages` listing per parent
    path (most are cached already), filtered by daily_refresh.wanted /
    events.is_stage.
  * Closed regional qualifiers of S/A events (2025 and 2026 events,
    qualifier dated 2025 or 2026): sub-pages of the S/A main events in
    lp_titles_selected.txt whose remaining path is a region / "Qualifier"
    name ("PGL/2025/Astana/Europe", ".../Closed Qualifier",
    "Thunderpick/World Championship/2025/Europe/1"). Open qualifiers
    (a segment "Open" below the event) are left out: hundreds of amateur
    teams, mostly BO1. Sub-pages come from the cached `allpages` listings;
    an event no cached listing covers gets one listing of its own.
  * S/A main events of 2025 that lp_titles_selected.txt does not track
    ("gap" events: Liquipedia lists them as S/A but the hand-picked list
    missed them). They are recorded with kind "sa_gap" in the lower file,
    never in matches.json.
  * C-tier: sized only (see `plan`); not fetched (rationale in the plan
    output and data/SOURCES.md).

POLITENESS
  All requests go through data/lpfetch.py (cached, never re-fetched;
  action=parse never used). This job sets lpfetch.MIN_GAP = 5.0 s (twice the
  daily refresh's gap), reads pages 20 per request, and stops when
  Liquipedia still answers 429 after lpfetch's backoff (30/90/180 s, or the
  server's Retry-After). Everything fetched is cached, so a re-run continues
  where the last one stopped. Progress: data/raw/backfill_2025.json.

USAGE (from D:/Face-Off)
  python scripts/backfill_lower.py plan      # discovery + per-month table
  python scripts/backfill_lower.py fetch     # Dec 2025 -> Jan 2025, newest first
  python scripts/backfill_lower.py titles    # (re)write data/raw/lp_titles_lower.txt
Then: python data/collect_liquipedia.py --lower   (or --offline --lower)
Stdlib only.
"""
import json
import math
import os
import re
import sys
import urllib.error
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "viewer"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import lpfetch  # noqa: E402
import events as E  # noqa: E402
import daily_refresh as DR  # noqa: E402

lpfetch.MIN_GAP = 5.0
STATE = os.path.join(ROOT, "data", "raw", "backfill_2025.json")
LOWER_TITLES = os.path.join(ROOT, "data", "raw", "lp_titles_lower.txt")
SELECTED = os.path.join(ROOT, "data", "raw", "lp_titles_selected.txt")
YEAR = "2025"
SINCE = "2024-01-01"
BATCH = 20
# B-tier league seasons (ESL Challenger League regions run ~110 days) are real
# results; for rating data they are kept up to this length (the site's 45-day
# rule is about what counts as one tournament on a page, not about the data)
LEAGUE_MAX_DAYS = 150
REGION = DR.SKIP_SEGMENTS - {"Open"} | {
    "Closed Qualifier", "China", "MENA", "Rest of Asia", "Asia-Pacific", "Oceania and Southeast Asia",
    "Mongolia and West Asia", "Middle East", "Southeast Asia", "CIS", "Eastern Europe", "Western Europe",
    "Northern Europe", "Turkey", "Brazil", "Africa", "Mongolia", "India", "Japan", "Korea"}


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"months": {}, "requests": 0, "http_retries": 0, "log": []}


def save_state(st):
    st["requests"] = st.get("requests_before", 0) + lpfetch.NETWORK_REQUESTS[0]
    st["http_retries"] = st.get("retries_before", 0) + lpfetch.HTTP_RETRIES[0]
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)


def category_members(cat):
    """Every cached categorymembers entry for `cat` (no request)."""
    out = {}
    for fn in sorted(os.listdir(lpfetch.CACHE)):
        with open(os.path.join(lpfetch.CACHE, fn), encoding="utf-8") as f:
            rec = json.load(f)
        if "cmtitle=" + cat.replace(" ", "+").replace(":", "%3A") not in rec["url"]:
            continue
        for x in rec["response"].get("query", {}).get("categorymembers", []):
            out[x["title"]] = x.get("timestamp", "")
    return out


def cached_listings():
    """{apprefix: [titles]} of every cached allpages listing (no request)."""
    import urllib.parse as up
    out = {}
    for fn in sorted(os.listdir(lpfetch.CACHE)):
        with open(os.path.join(lpfetch.CACHE, fn), encoding="utf-8") as f:
            rec = json.load(f)
        q = rec["response"].get("query", {})
        if "allpages" in q:
            m = re.search(r"apprefix=([^&]*)", rec["url"])
            assert m is not None
            out[up.unquote_plus(m.group(1))] = [p["title"] for p in q["allpages"]]
    return out


def texts_for(titles):
    return DR.page_texts(list(titles), batch=BATCH) if titles else {}


def main_event_ok(info, tiers, max_days=None):
    tier, ttype, sd, ed = info
    if tier not in tiers or ttype.lower() in E.SKIP_TIERTYPES or not sd.startswith(YEAR):
        return False
    try:
        return not (ed and (date.fromisoformat(ed[:10]) - date.fromisoformat(sd[:10])).days
                    > (max_days or E.MAX_DAYS))
    except ValueError:
        return False


def discover_b(have):
    members = category_members("Category:B-Tier Tournaments")
    cands = [t for t in DR.tier_candidates([("B", t, ts) for t, ts in members.items()], (YEAR,), SINCE)
             if t not in have]
    texts = texts_for(cands)
    mains = [t for t in cands if texts.get(t) and main_event_ok(E.event_info(texts[t]), ("B",), LEAGUE_MAX_DAYS)]
    idx = lpfetch.cached_pages()
    # only pages we track or keep can own a sub-page: a season overview that is
    # not kept itself ("CCT/Season 2/Europe") must not swallow its series
    parents = have | set(mains)
    keep = []
    for t in mains:
        anc = [t[:i] for i in range(len(t) - 1, 0, -1) if t[i] == "/" and t[:i] in parents]
        if anc:
            tail = t[len(anc[0]) + 1:].split("/")
            if any(x in DR.SKIP_SEGMENTS for x in tail) and all(x in DR.SKIP_SEGMENTS or x.isdigit() for x in tail):
                continue
            ptxt = texts.get(anc[0]) or (idx[anc[0]][1] if anc[0] in idx else "")
            pinfo = E.event_info(ptxt)
            if pinfo[2] and E.is_stage(E.event_info(texts[t]), pinfo):
                continue
        keep.append(t)
    return keep, {t: E.event_info(texts[t]) for t in keep}, len(cands)


def sa_events():
    """S/A main events tracked in lp_titles_selected.txt, with their infobox."""
    sel = [l.strip() for l in open(SELECTED, encoding="utf-8") if l.strip()]
    idx = lpfetch.cached_pages()
    texts = {t: idx[t][1] for t in sel if t in idx}
    own = E.owners(sel, texts)
    return [t for t in sel if own[t] == t], texts


def is_qualifier_path(event, title):
    rest = [s for s in title[len(event):].strip("/").split("/") if s]
    if not rest or "Open" in rest or any("Open" in s.split() for s in rest):
        return False
    if any("Showmatch" in s or "Stage" in s or "Playoff" in s for s in rest):
        return False
    return all(s in REGION or "Qualifier" in s or s.isdigit() for s in rest) and not all(s.isdigit() for s in rest)


def discover_qualifiers(have):
    mains, texts = sa_events()
    events = [t for t in mains if E.event_info(texts.get(t))[2][:4] in ("2025", "2026")]
    lst = cached_listings()
    found = set()
    new_listings = 0
    for e in events:
        cover = [p for p, ts in lst.items() if (e + "/").startswith(p) and len(ts) < 500 and p]
        if not cover:
            before = lpfetch.NETWORK_REQUESTS[0]
            lst[e + "/"] = lpfetch.prefix(e + "/")
            new_listings += lpfetch.NETWORK_REQUESTS[0] - before
        for p, ts in lst.items():
            if (e + "/").startswith(p) or p.startswith(e + "/"):
                found |= {t for t in ts if t.startswith(e + "/") and is_qualifier_path(e, t)}
    found -= have
    # a qualifier owned by a deeper tracked event (".../Stage 1/Europe") belongs to that one
    qtexts = texts_for(sorted(found))
    keep = {}
    for t in sorted(found):
        info = E.event_info(qtexts.get(t))
        if not qtexts.get(t) or "{{Match" not in qtexts[t]:
            continue
        if info[1].lower() not in ("qualifier", "") or not info[2][:4] in ("2025", "2026"):
            continue
        keep[t] = info
    return keep, new_listings


def discover_gaps(have):
    """S/A main events of 2025 that the S/A tier categories list but the
    hand-picked list does not track."""
    out, cands = {}, []
    for tier, cat in (("S", "Category:S-Tier Tournaments"), ("A", "Category:A-Tier Tournaments")):
        mem = category_members(cat)
        cands += [t for t in DR.tier_candidates([(tier, t, ts) for t, ts in mem.items()], (YEAR,), SINCE)
                  if t not in have]
    texts = texts_for(sorted(set(cands)))
    for t in sorted(set(cands)):
        if texts.get(t) and main_event_ok(E.event_info(texts[t]), ("S", "A")):
            out[t] = E.event_info(texts[t])
    # sub-pages of a tracked event (regional finals / stages) are not gaps
    return {t: v for t, v in out.items() if not any(t.startswith(h + "/") for h in have)}


def c_tier_probe():
    """One listing request: the 500 most recently added C-tier pages, to
    estimate what a C-tier backfill would cost."""
    resp = lpfetch.query({"action": "query", "list": "categorymembers", "cmtitle": "Category:C-Tier Tournaments",
                          "cmnamespace": "0", "cmprop": "title|timestamp", "cmsort": "timestamp",
                          "cmdir": "desc", "cmlimit": "500"}) or {}
    rows = resp.get("query", {}).get("categorymembers", [])
    return rows


def plan(argv):
    st = load_state()
    st["requests_before"] = st.get("requests", 0)
    st["retries_before"] = st.get("http_retries", 0)
    lower_have = {l.strip() for l in open(LOWER_TITLES, encoding="utf-8") if l.strip()}
    sel = {l.strip() for l in open(SELECTED, encoding="utf-8") if l.strip()}
    have = lower_have | sel
    r0 = lpfetch.NETWORK_REQUESTS[0]
    b_mains, b_info, n_cand = discover_b(have)
    r_b = lpfetch.NETWORK_REQUESTS[0] - r0
    r0 = lpfetch.NETWORK_REQUESTS[0]
    quals, q_list = discover_qualifiers(have | set(b_mains))
    r_q = lpfetch.NETWORK_REQUESTS[0] - r0
    r0 = lpfetch.NETWORK_REQUESTS[0]
    gaps = discover_gaps(have | set(b_mains) | set(quals))
    r_g = lpfetch.NETWORK_REQUESTS[0] - r0
    r0 = lpfetch.NETWORK_REQUESTS[0]
    crow = c_tier_probe()
    r_c = lpfetch.NETWORK_REQUESTS[0] - r0
    # C-tier estimate from the newest 500 additions
    c25 = [r for r in crow if "2025" in r["title"]]
    span = (crow[0]["timestamp"][:10], crow[-1]["timestamp"][:10]) if crow else ("", "")
    lst = cached_listings()
    months = {}
    for kind, d in (("B", b_info), ("Q", quals), ("G", gaps)):
        for t, info in d.items():
            mo = info[2][:7]
            months.setdefault(mo, {"B": [], "Q": [], "G": []})[kind].append(t)
    # estimated requests per month: stage listings not cached + stage reads
    table = []
    for mo in sorted(months, reverse=True):
        evs = months[mo]["B"] + months[mo]["G"]
        parents = sorted({e.rsplit("/", 1)[0] + "/" if "/" in e else e + "/" for e in evs})
        uncached = [p for p in parents if p not in lst]
        est = len(uncached) + math.ceil(len(evs) * 0.3 / BATCH)   # ~0.3 stage pages per event (2026 lower list)
        table.append({"month": mo, "b_events": len(months[mo]["B"]), "qualifiers": len(months[mo]["Q"]),
                      "sa_gap_events": len(months[mo]["G"]), "stage_listings_needed": len(uncached),
                      "est_requests": est})
    if "plan" in st and not (r_b or r_q or r_g or r_c):
        disc = st["plan"]["discovery_requests"]      # a re-plan from the cache keeps the first run's counts
    else:
        disc = {"b": r_b, "qualifiers": r_q, "qualifier_listings": q_list, "gaps": r_g, "c_probe": r_c}
    st["plan"] = {"b_candidates_read": n_cand, "b_events": b_info, "qualifiers": quals, "sa_gaps": gaps,
                  "discovery_requests": disc,
                  "c_probe": {"newest_500_added": span, "titles_naming_2025": len(c25)},
                  "table": table}
    for mo in months:
        st["months"].setdefault(mo, {"done": False})
    save_state(st)
    print(f"discovery requests: B {r_b} (candidates read {n_cand}), qualifiers {r_q} "
          f"(new listings {q_list}), S/A gaps {r_g}, C probe {r_c}")
    print(f"C-tier probe: newest 500 additions span {span[1]} .. {span[0]}; {len(c25)} of them name 2025")
    print("month    B-events  S/A-qual  S/A-gap  stage-listings  est.requests")
    for r in table:
        print(f"{r['month']}  {r['b_events']:8d}  {r['qualifiers']:8d}  {r['sa_gap_events']:7d}  "
              f"{r['stage_listings_needed']:14d}  {r['est_requests']:12d}")
    print("S/A gap events:", ", ".join(sorted(gaps)) or "none")


def fetch(argv):
    st = load_state()
    st["requests_before"] = st.get("requests", 0)
    st["retries_before"] = st.get("http_retries", 0)
    p = st.get("plan")
    if not p:
        raise SystemExit("run `plan` first")
    lower_have = {l.strip() for l in open(LOWER_TITLES, encoding="utf-8") if l.strip()}
    sel = {l.strip() for l in open(SELECTED, encoding="utf-8") if l.strip()}
    for mo in sorted(st["months"], reverse=True):
        rec = st["months"][mo]
        if rec.get("done"):
            continue
        evs = [t for t, i in p["b_events"].items() if i[2][:7] == mo] + \
              [t for t, i in p["sa_gaps"].items() if i[2][:7] == mo]
        quals = [t for t, i in p["qualifiers"].items() if i[2][:7] == mo]
        r0, q0 = lpfetch.NETWORK_REQUESTS[0], lpfetch.HTTP_RETRIES[0]
        try:
            stages = DR.stage_pages(evs, lower_have | sel | set(evs) | set(quals), refresh=False)
            qstages = []          # nested qualifier pages were already picked up by discovery
            texts_for(evs + quals + stages)     # main pages: cached by discovery already
        except urllib.error.HTTPError as e:
            rec.update({"done": False, "error": f"HTTP {e.code}"})
            st["log"].append(f"{mo}: stopped, Liquipedia answered {e.code} after the backoff")
            save_state(st)
            raise SystemExit(f"{mo}: Liquipedia keeps answering {e.code}; stopped (re-run to resume)")
        rec.update({"done": True, "events": sorted(evs), "qualifiers": sorted(quals),
                    "stages": sorted(stages + qstages), "requests": lpfetch.NETWORK_REQUESTS[0] - r0,
                    "http_retries": lpfetch.HTTP_RETRIES[0] - q0})
        save_state(st)
        print(f"{mo}: {len(evs)} events, {len(quals)} qualifiers, {len(stages) + len(qstages)} stage pages; "
              f"{rec['requests']} requests ({rec['http_retries']} retries); total {st['requests']}")
    write_titles(st)


def write_titles(st=None):
    """data/raw/lp_titles_lower.txt = the 2026 list + every 2025 page fetched
    (events, S/A gaps, qualifiers, stages), in a stable order."""
    st = st or load_state()
    old = [l.strip() for l in open(LOWER_TITLES, encoding="utf-8") if l.strip()]
    new = []
    for mo in sorted(st["months"]):
        r = st["months"][mo]
        if r.get("done"):
            new += r.get("events", []) + r.get("qualifiers", []) + r.get("stages", [])
    out = sorted(set(old) | set(new))
    with open(LOWER_TITLES, "w", encoding="utf-8") as f:
        f.write("".join(t + "\n" for t in out))
    # kinds for the builder: qualifier pages and S/A gap events
    p = st.get("plan", {})
    kinds = {"qualifier": sorted(p.get("qualifiers", {})), "sa_gap": sorted(p.get("sa_gaps", {}))}
    with open(os.path.join(ROOT, "data", "raw", "lp_titles_lower_kinds.json"), "w", encoding="utf-8") as f:
        json.dump(kinds, f, indent=1, ensure_ascii=False)
    print(f"lp_titles_lower.txt: {len(old)} -> {len(out)} titles")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    {"plan": plan, "fetch": fetch, "titles": lambda a: write_titles()}[cmd](sys.argv[2:])
