#!/usr/bin/env python3
"""
Daily data refresh for the Faceoff hub (run by .github/workflows/daily-refresh.yml).

1. Discover new tournament pages. Main source: Liquipedia's tier categories
   ("S-Tier Tournaments", "A-Tier Tournaments"), newest additions first (one
   request per tier per run). B-tier and lower are not tracked. A candidate is
   kept only when its own infobox says S or A tier (`liquipediatier`), it is not a
   qualifier / showmatch / weekly / monthly / misc / points-circuit page
   (`liquipediatiertype`), it starts this year or next and it lasts at most
   events.MAX_DAYS days. A tiered sub-page of another event that is dated
   apart from it (ESL Challenger League cups, Thunderpick regional events)
   feeds that event and is skipped like a qualifier. Stage sub-pages ("<event>/Stage 1") come from one
   `allpages` listing per parent path. Second source (unchanged): `allpages`
   listings for the tracked series (a page there whose own infobox names a
   tier below A is dropped). Main events and their stage pages are
   appended to data/raw/lp_titles_selected.txt.
2. Mark pages that can still change as stale for this run: every live
   tournament, anything that ended in the last REFRESH_DAYS days or starts in
   the next AHEAD_DAYS days, their stage pages, and the team pages of every
   team in them (rosters, stand-ins, logos). Everything else keeps being
   served from the cache.
3. Rebuild, in order: data/matches.json (collect_liquipedia), data/
   roster_events.json (rosters), viewer/assets.json + new images
   (fetch_assets), data/vrs.json (Valve Regional Standings, from GitHub),
   data/lineups.json (per-series five-man lineups from Valve's standings
   detail pages, incremental: only snapshots newer than the file's newest
   one are read, so most runs download nothing; a failure here is reported
   and never stops the refresh).

The site itself is built from these files on each deploy
(python viewer/build_viewer.py public/index.html, see vercel.json).

All requests go through data/lpfetch.py: descriptive User-Agent, gzip, at most
one request every 2.5 s, no action=parse. A typical run is 15-50 requests.

LIVE MODE (python scripts/daily_refresh.py --live, run every 10 minutes by
.github/workflows/live-refresh.yml): only while a tracked tournament is on (its
dates, one day either side) the event's pages are read again and matches.json is
rebuilt, 1-3 requests and no assets. Outside a tournament it exits at once.

A page that was meant to be refreshed but could only be served from the cache
(Liquipedia refused us) is retried after a pause and, if it stays stale, listed
in data/refresh_status.json so the workflow fails loudly instead of reporting a
green run with old results.

USAGE (from the repo root):  python scripts/daily_refresh.py [--dry-run] [--live]
"""

import os
import re
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "viewer"))

import lpfetch  # noqa: E402
import events as E  # noqa: E402

event_info, is_stage, SKIP_TIERTYPES = E.event_info, E.is_stage, E.SKIP_TIERTYPES

REFRESH_DAYS = 3
AHEAD_DAYS = 14        # upcoming events are re-read once they are this close
# only S- and A-tier tournaments are tracked
TIERS = ("S", "A")
TIER_CATEGORIES = {"S": "Category:S-Tier Tournaments", "A": "Category:A-Tier Tournaments"}
YEAR_RE = re.compile(r"(?<!\d)(19|20)\d\d(?!\d)")
SERIES = ["Intel Extreme Masters/{y}", "BLAST/Premier/{y}", "BLAST/Open/{y}", "BLAST/Rivals/{y}",
          "BLAST/Bounty/{y}", "PGL/{y}", "StarLadder/StarSeries/{y}", "Thunderpick/World Championship/{y}",
          "Esports World Cup/{y}", "Perfect World/{y}"]
ESL_PL = "ESL/Pro League/Season "
# regional qualifier sub-pages ("PGL/2026/Astana/Europe"); country-named
# events such as "IEM China" are NOT skipped
SKIP_SEGMENTS = {"Qualifier", "Qualifiers", "Open", "Asia", "Europe", "North America", "South America",
                 "Oceania", "East Asia", "West Asia", "Americas", "Global"}
TITLES_FILE = os.path.join(ROOT, "data", "raw", "lp_titles_selected.txt")


def wanted(title, series_root, stage=False):
    """Main events and their stage pages; not qualifiers or regional sub-events.
    stage=True: `series_root` is the event itself, so only its stage pages
    ("<event>/Stage 1", "<event>/Online Stage/Stage 2") are wanted."""
    rest = title[len(series_root):].strip("/")
    segs = [s for s in rest.split("/") if s]
    if not segs:
        return False
    if stage:
        return (len(segs) <= 2 and not any("Qualifier" in s or "Showmatch" in s or s in SKIP_SEGMENTS
                                           or s.isdigit() for s in segs))
    if any("Qualifier" in s for s in segs):
        return False
    # regions can be the first segment when the year is the event itself
    # ("Thunderpick/World Championship/2026/North America")
    if any(s in SKIP_SEGMENTS for s in segs) or any(s.isdigit() for s in segs[1:]):
        return False
    return len(segs) <= 2          # "<event>" or "<event>/<stage>"


def tier_members(full=False, refresh=True):
    """[(tier letter, title, added timestamp)] from the S/A tier categories,
    newest additions first. full=False reads one page (500) per tier, enough
    for everything added since the previous run; full=True pages through the
    whole category (used once to backfill)."""
    out = []
    for tier, cat in TIER_CATEGORIES.items():
        cont = None
        while True:
            p = {"action": "query", "list": "categorymembers", "cmtitle": cat, "cmnamespace": "0",
                 "cmprop": "title|timestamp", "cmsort": "timestamp", "cmdir": "desc", "cmlimit": "500"}
            if cont:
                p["cmcontinue"] = cont
            resp = lpfetch.query(p, refresh=refresh) or {}
            out += [(tier, r["title"], r.get("timestamp", ""))
                    for r in resp.get("query", {}).get("categorymembers", [])]
            cont = resp.get("continue", {}).get("cmcontinue")
            if not full or not cont:
                break
    return out


def tier_candidates(members, years, since):
    """Titles worth reading: the title names one of `years`, or names no year at
    all ("ESL/Pro League/Season 25", "Stake Ranked/Episode 4") and was added to
    the category on or after `since`. Qualifier pages are dropped by title."""
    out = []
    for _, t, ts in members:
        segs = t.split("/")
        if any("Qualifier" in x for x in segs) or t in out:
            continue
        found = {m.group(0) for m in YEAR_RE.finditer(t)}
        if found & set(years) or (not found and ts >= since):
            out.append(t)
    return out


def is_main_event(info, years):
    """S/A-tier, not a qualifier / showmatch / weekly / monthly / misc /
    points page, starting in one of `years` and no longer than MAX_DAYS.
    Undated pages wait (they are read again on later runs)."""
    tier, ttype, sd, ed = info
    if tier not in TIERS or ttype.lower() in SKIP_TIERTYPES:
        return False
    if not sd or sd[:4] not in years:
        return False
    try:
        if sd and ed and (date.fromisoformat(ed) - date.fromisoformat(sd)).days > E.MAX_DAYS:
            return False      # a season-long league / circuit, not a tournament
    except ValueError:
        return False
    return True


def page_texts(titles, batch=20):
    """{title: wikitext} through lpfetch (already cached pages cost nothing);
    a page left out of a large batch response is asked for on its own."""
    out = {}
    for i in range(0, len(titles), batch):
        out.update(lpfetch.wikitext(titles[i:i + batch], strict=False))
    for t in [t for t in titles if out.get(t) is None]:
        out.update(lpfetch.wikitext([t], strict=False))
    return out


def stage_pages(events, have, refresh=True):
    """Stage sub-pages ("<event>/<stage>", "<event>/<stage>/<part>") of the
    given main events: one allpages listing per parent path, then the same
    filter as the series discovery. Sub-pages with their own qualifier /
    showmatch infobox are dropped once read."""
    found = []
    for parent in sorted({e.rsplit("/", 1)[0] + "/" if "/" in e else e + "/" for e in events}):
        listing = lpfetch.prefix(parent, refresh=refresh)
        for e in events:
            if not (e + "/").startswith(parent):
                continue
            # listing cut off at 500: ask for this event's own pages instead
            own = lpfetch.prefix(e + "/", refresh=refresh) if len(listing) >= 500 else listing
            for t in own:
                if t.startswith(e + "/") and t not in have and t not in found and wanted(t, e, stage=True):
                    found.append(t)
    if refresh:
        # a stage page cached before its bracket was filled in is read again
        idx = lpfetch.cached_pages()
        lpfetch.FRESH_TITLES |= {t for t in found if t in idx and "{{Match" not in idx[t][1]}
    texts = page_texts(found + [e for e in events])
    out = []
    for t in found:
        e = max((e for e in events if t.startswith(e + "/")), key=len)
        if texts.get(t) and is_stage(event_info(texts[t]), event_info(texts.get(e)), texts[t]):
            out.append(t)
    return out


def discover_tiers(have, today, full=False):
    """New S/A-tier main events (this year and next) and their stage pages."""
    years = (str(today.year), str(today.year + 1))
    # pages are often created a year or more ahead: the backfill reads every
    # undated-title page added since the start of the year before last
    since = f"{today.year - 2}-01-01" if full else (today - timedelta(days=120)).isoformat()
    members = tier_members(full=full, refresh=not full)
    listed = {t for _, t, _ in members}
    cands = [t for t in tier_candidates(members, years, since) if t not in have]
    # a candidate cached while still undated is read again
    idx = lpfetch.cached_pages()
    if not full:
        lpfetch.FRESH_TITLES |= {t for t in cands if t in idx and not event_info(idx[t][1])[2]}
    texts = page_texts(cands)
    mains = [t for t in cands if texts.get(t) and is_main_event(event_info(texts[t]), years)]
    # sub-pages of another tiered, tracked or kept page: a stage of it is
    # picked up with its stages below; a regional sub-page ("/North America",
    # "/Europe/2") is a regional qualifier; anything else is its own event
    parents = listed | have | set(mains)
    keep = []
    for t in mains:
        anc = [t[:i] for i in range(len(t) - 1, 0, -1) if t[i] == "/" and t[:i] in parents]
        if anc:
            tail = [x for x in t[len(anc[0]) + 1:].split("/")]
            if any(x in SKIP_SEGMENTS for x in tail) and all(x in SKIP_SEGMENTS or x.isdigit() for x in tail):
                continue
            pinfo = event_info(texts.get(anc[0]) or (idx[anc[0]][1] if anc[0] in idx else ""))
            # an undated parent is a series overview page ("Esports World Cup"), not an event
            if pinfo[2] and is_stage(event_info(texts[t]), pinfo):
                continue
        keep.append(t)
    mains = keep
    return mains, stage_pages(mains, have | set(mains), refresh=not full)


def discover(today, soon=()):
    """(tracked titles, new titles). `soon`: tracked main events that are live
    or start within AHEAD_DAYS; their stage pages are listed again, since
    organisers often create them only shortly before the event."""
    selected = [l.strip() for l in open(TITLES_FILE, encoding="utf-8") if l.strip()]
    have = set(selected)
    mains, stages = discover_tiers(have, today)
    stages += [t for t in stage_pages(list(soon), have | set(mains) | set(stages)) if t not in stages]
    new = []
    for t in mains + stages:
        new.append(t)
        have.add(t)
    for y in (today.year, today.year + 1):
        for pat in SERIES:
            root = pat.format(y=y)
            for t in lpfetch.prefix(root, refresh=True):
                if t not in have and wanted(t, root):
                    new.append(t)
                    have.add(t)
    # ESL Pro League seasons: only seasons after the newest one already tracked
    seasons = [int(m.group(1)) for t in selected for m in [re.match(re.escape(ESL_PL) + r"(\d+)$", t)] if m]
    newest = max(seasons) if seasons else 0
    for t in lpfetch.prefix(ESL_PL, refresh=True):
        m = re.match(re.escape(ESL_PL) + r"(\d+)(/.*)?$", t)
        if m and int(m.group(1)) > newest and t not in have:
            sub = (m.group(2) or "").strip("/")
            if not sub or (sub.count("/") == 0 and sub not in SKIP_SEGMENTS and "Qualifier" not in sub):
                new.append(t)
                have.add(t)
    # series listings are not tier-filtered: drop a page whose own infobox
    # names a tier below A (a page with no tier yet, or a stage page, stays)
    found = new[len(mains) + len(stages):]
    texts = page_texts(found) if found else {}
    low = {t for t in found if event_info(texts.get(t))[0] not in ("",) + TIERS}
    return selected, [t for t in new if t not in low]


def stale_titles(today):
    """Event pages and team pages that may have changed since they were cached,
    and the main events that are live or start within AHEAD_DAYS."""
    evs = E.discover()
    cutoff = (today - timedelta(days=REFRESH_DAYS)).isoformat()
    ahead = (today + timedelta(days=AHEAD_DAYS)).isoformat()
    titles, teams = set(), set()
    for e in evs:
        if e["end"] >= cutoff and e["start"] <= ahead:
            titles.add(e["title"])
            teams |= set(e["participants"])
            teams |= {m["t1"] for m in e["matches"] if m["t1"]} | {m["t2"] for m in e["matches"] if m["t2"]}
    selected = [l.strip() for l in open(TITLES_FILE, encoding="utf-8") if l.strip()]
    soon = sorted(titles)
    titles |= {s for s in selected for t in list(titles) if s.startswith(t + "/")}
    return titles, teams, soon


STATUS_FILE = os.path.join(ROOT, "data", "refresh_status.json")
LIVE_WINDOW_DAYS = 1       # a tournament counts as live this many days before it starts / after it ends
RETRY_PAUSES = (120, 240)  # seconds to wait before asking Liquipedia again for pages it refused us


def live_titles(today):
    """Pages of tracked tournaments that are on now (+/- LIVE_WINDOW_DAYS), with
    their stage pages."""
    lo = (today - timedelta(days=LIVE_WINDOW_DAYS)).isoformat()
    hi = (today + timedelta(days=LIVE_WINDOW_DAYS)).isoformat()
    titles = {e["title"] for e in E.discover() if e["start"] <= hi and e["end"] >= lo}
    selected = [l.strip() for l in open(TITLES_FILE, encoding="utf-8") if l.strip()]
    return titles | {s for s in selected for t in list(titles) if s.startswith(t + "/")}


def stale_live(live):
    """Live pages that a refresh asked for and only the cache could answer."""
    served = set()
    for entry in lpfetch.STALE_SERVED:
        served |= set(str(entry).split("|"))
    return sorted(t for t in live if t in served)


def refresh_live(live, collect):
    """Refresh the live pages, rebuild matches.json, and retry pages Liquipedia
    refused; returns the live pages still stale afterwards."""
    import time
    import collect_liquipedia
    lpfetch.FRESH_TITLES |= live
    lpfetch.STALE_SERVED.clear()
    collect()
    stale = stale_live(live)
    for pause in RETRY_PAUSES:
        if not stale:
            break
        print(f"live pages still stale after the refresh: {stale}; trying again in {pause}s")
        time.sleep(pause)
        lpfetch._THROTTLED[0] = False
        lpfetch.STALE_SERVED.clear()
        lpfetch.RUN_STARTED = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # a copy fetched now counts as fresh
        collect()
        stale = stale_live(live)
    return stale


def write_status(stale, live):
    """data/refresh_status.json: written only when it changes, so quiet runs
    commit nothing. The workflow fails the run when `stale_pages` is not empty."""
    import json
    new = {"ok": not stale, "stale_pages": stale}
    try:
        old = json.load(open(STATUS_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        old = None
    if old != new:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(new, f, indent=1)
            f.write("\n")
    if stale:
        print(f"STALE: could not refresh {len(stale)} live tournament page(s): {stale}", file=sys.stderr)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dry = "--dry-run" in argv
    today = date.today()

    if "--live" in argv:
        live = live_titles(today)
        if not live:
            print("no tracked tournament is live: nothing to do")
            return
        print(f"live tournament pages: {sorted(live)}")
        import collect_liquipedia
        collect_liquipedia.OFFLINE = False
        stale = refresh_live(live, collect_liquipedia.main)
        write_status(stale, live)
        print(f"live refresh done; Liquipedia API requests sent: {lpfetch.NETWORK_REQUESTS[0]}")
        return

    # stale pages come from the CURRENT cache, before new titles are appended
    titles, teams, soon = stale_titles(today)
    selected, new = discover(today, soon)
    print(f"tracked tournament pages: {len(selected)}; new: {len(new)}")
    for t in new:
        print("  +", t)
    if new and not dry:
        with open(TITLES_FILE, "a", encoding="utf-8") as f:
            f.write("".join(t + "\n" for t in new))
    titles |= set(new)
    print(f"refreshing {len(titles)} tournament pages and {len(teams)} team pages")
    if dry:
        for t in sorted(titles):
            print("  ~", t)
        print(f"Liquipedia requests sent: {lpfetch.NETWORK_REQUESTS[0]}")
        return
    lpfetch.FRESH_TITLES |= titles | teams

    import collect_liquipedia
    collect_liquipedia.OFFLINE = False
    live = live_titles(today)
    stale = refresh_live(live, collect_liquipedia.main) if live else []
    if not live:
        collect_liquipedia.main()
    write_status(stale, live)

    import rosters
    rosters.OFFLINE = False
    rosters.main()

    import fetch_assets
    fetch_assets.OFFLINE = False
    fetch_assets.main()

    import vrs          # Valve Regional Standings (GitHub; new file roughly monthly)
    vrs.OFFLINE = False
    vrs.main()

    # lineups are optional input (roster-core features): a failure here must
    # never break the data refresh, so it is reported and the old file is kept
    try:
        import lineups
        lineups.OFFLINE = False
        lineups.main(incremental=True)
    except (Exception, SystemExit) as e:   # noqa: BLE001
        print(f"lineups: refresh skipped ({type(e).__name__}: {e}); data/lineups.json unchanged",
              file=sys.stderr)
    print(f"refresh done; Liquipedia API requests sent: {lpfetch.NETWORK_REQUESTS[0]}")


if __name__ == "__main__":
    main()
