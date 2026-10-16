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
   roster_events.candidate.json (rosters --candidate: SUPERVISED, a human
   promotes it to roster_events.json after reviewing pipeline/roster_watch.py's
   diff), viewer/assets.json + new images
   (fetch_assets), data/vrs.json (Valve Regional Standings, from GitHub),
   data/lineups.json (per-series five-man lineups from Valve's standings
   detail pages, incremental: only snapshots newer than the file's newest
   one are read, so most runs download nothing; a failure here is reported
   and never stops the refresh), data/valve_results.json (the second
   results source for the cross_source gate: Valve's standings match rows,
   incremental the same way, a failure only warns), and last
   data/matches_lower.json (pipeline/lower_refresh.py: B-tier results for team
   pages and rating data, never events; own request budget; a failure only
   warns and never blocks the S/A data).

The site itself is built from these files on each deploy
(python viewer/build_viewer.py public/index.html, see vercel.json).

All requests go through collectors/lpfetch.py: descriptive User-Agent, gzip, at most
one request every 2.5 s, no action=parse. A typical run is 15-50 requests.

LIVE MODE (python pipeline/daily_refresh.py --live, run every 10 minutes by
.github/workflows/live-refresh.yml): only while a tracked tournament is on (its
dates, one day either side) the event's pages are read again and matches.json is
rebuilt, 1-3 requests and no assets. Outside a tournament it exits at once.

A page that was meant to be refreshed but could only be served from the cache
(Liquipedia refused us) is retried after a pause and, if it stays stale, listed
in data/refresh_status.json so the workflow fails loudly instead of reporting a
green run with old results.

USAGE (from the repo root):  python pipeline/daily_refresh.py [--dry-run] [--live]
"""

import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

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
RETRY_PAUSES = (120,)      # full daily run: wait this long (seconds) before asking Liquipedia again for refused pages
# Live runs fire every 10 minutes, so the next run IS the retry: one short try,
# no waiting loop. (A patient loop held the shared queue for 22 minutes when
# Liquipedia refused the runner's address.)
LIVE_RETRY_BACKOFF = [20]
LIVE_MAX_RETRY_AFTER = 60
STALE_FAIL_MINUTES = 30    # live runs turn red only after pages have been stale this long


MATCH_LEAD = timedelta(minutes=30)   # start asking this long before a match is due
# ...and keep asking until the match is finished, up to this long after its SCHEDULED start.
# Scheduled times slip (earlier series run long, tech pauses, overtime): on 2026-10-15 Legacy-FURIA
# was scheduled 20:50 CEST, was still on map 3 past the old 4 h tail, and its result reached the
# site 4.5 hours late -- with the next match already showing Legacy as the opponent. A match with
# a map already on the page is certainly under way, so it keeps the longer tail.
MATCH_TAIL = timedelta(hours=12)
MATCH_TAIL_STARTED = timedelta(hours=24)
OVERDUE_ALERT = timedelta(hours=12)  # unfinished this long after its start: the data is behind, not the match slow
OVERDUE_GIVE_UP = timedelta(days=2)  # ...and older than this it is a cancelled / unreported match, not a refresh problem


def _match_start(m):
    when = m.get("when")
    if not when:
        return None
    try:
        start = datetime.fromisoformat(when)
    except ValueError:
        return None
    return start if start.tzinfo else start.replace(tzinfo=timezone.utc)


def match_window_active(ev, now):
    """True while one of the event's unfinished matches is about to start, is under
    way, or is overdue (see MATCH_TAIL). A match that has a date but no start time
    yet counts for that whole day. Outside these windows nothing on the page can
    change, so there is nothing to ask Liquipedia."""
    today = now.date().isoformat()
    for m in ev.get("matches", []):
        if m.get("finished"):
            continue
        start = _match_start(m)
        if start:
            tail = MATCH_TAIL_STARTED if m.get("maps") else MATCH_TAIL
            if start - MATCH_LEAD <= now <= start + tail:
                return True
        elif not m.get("when") and m.get("day") == today:
            return True
    return False


def overdue_matches(now=None, events=None):
    """Unfinished matches of tracked events that started more than OVERDUE_ALERT ago
    (and less than OVERDUE_GIVE_UP): "event: A vs B (scheduled ...)". A series
    does not take 12 hours, so each one is a result the site is missing."""
    now = now or _now()
    out = []
    for e in (events if events is not None else E.discover()):
        for m in e.get("matches", []):
            start = _match_start(m)
            if m.get("finished") or not start or not (m.get("t1") and m.get("t2")):
                continue
            if OVERDUE_ALERT < now - start < OVERDUE_GIVE_UP:
                out.append(f"{e['name']}: {m['t1']} vs {m['t2']} (scheduled {m['when']})")
    return sorted(out)


def live_titles(today, now=None, force=False):
    """Pages of tracked tournaments with a match due, under way or overdue right now
    (see match_window_active), with their stage pages. `force` skips the match-time
    test and takes every tournament within LIVE_WINDOW_DAYS of today."""
    now = now or _now()
    lo = (today - timedelta(days=LIVE_WINDOW_DAYS)).isoformat()
    hi = (today + timedelta(days=LIVE_WINDOW_DAYS)).isoformat()
    titles = {e["title"] for e in E.discover()
              if e["start"] <= hi and e["end"] >= lo and (force or match_window_active(e, now))}
    selected = [l.strip() for l in open(TITLES_FILE, encoding="utf-8") if l.strip()]
    return titles | {s for s in selected for t in list(titles) if s.startswith(t + "/")}


def stale_live(live):
    """Live pages that a refresh asked for and only the cache could answer."""
    served = set()
    for entry in lpfetch.STALE_SERVED:
        served |= set(str(entry).split("|"))
    return sorted(t for t in live if t in served)


def refresh_live(live, collect, pauses=RETRY_PAUSES):
    """Refresh the live pages, rebuild matches.json, and retry pages Liquipedia
    refused after each pause in `pauses`; returns the live pages still stale."""
    import time
    import collect_liquipedia
    lpfetch.FRESH_TITLES |= live
    lpfetch.STALE_SERVED.clear()
    collect()
    stale = stale_live(live)
    for pause in pauses:
        if not stale:
            break
        print(f"live pages still stale after the refresh: {stale}; trying again in {pause}s")
        time.sleep(pause)
        lpfetch._THROTTLED[0] = False
        lpfetch.STALE_SERVED.clear()
        lpfetch.RUN_STARTED = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # a copy fetched now counts as fresh
        lpfetch._CONFIRMED.clear()
        collect()
        stale = stale_live(live)
    return stale


def _now():
    return datetime.now(timezone.utc)


def _load_status():
    import json
    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_status(stale, live):
    """data/refresh_status.json: {ok, stale_pages, stale_since}. `stale_since` is
    when the pages FIRST went stale and stays put while they are stale, so the
    file changes (and commits) only when an outage starts or ends, never on every
    run. `--check-status` turns a long outage into a red run."""
    import json
    old = _load_status() or {}
    if stale:
        since = old.get("stale_since") if old.get("stale_pages") else None
        new = {"ok": False, "stale_pages": stale,
               "stale_since": since or _now().strftime("%Y-%m-%dT%H:%M:%SZ")}
    else:
        new = {"ok": True, "stale_pages": [], "stale_since": None}
    if old != new:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(new, f, indent=1)
            f.write("\n")
    if stale:
        print(f"STALE: could not refresh {len(stale)} live tournament page(s): {stale}", file=sys.stderr)


def check_status(max_stale_minutes, now=None):
    """Exit code for the workflows' last step: 0 when current, or stale for less
    than `max_stale_minutes` (a warning only: the next run retries); 1 once the
    pages have been stale that long. 0 minutes = fail on any staleness."""
    st = _load_status() or {}
    if not st.get("stale_pages"):
        print("live pages are current")
        return 0
    since = st.get("stale_since")
    try:
        age = ((now or _now()) - datetime.strptime(str(since), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds() / 60
    except (TypeError, ValueError):
        age = float("inf")     # no usable timestamp: treat as long stale
    msg = f"Liquipedia refused these live pages {age:.0f} min ago or more: {st['stale_pages']}; the site shows older results"
    if age >= max_stale_minutes:
        print(msg, file=sys.stderr)
        return 1
    print(f"::warning::{msg} (will fail after {max_stale_minutes} min; the next run retries)")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dry = "--dry-run" in argv
    today = date.today()

    if "--check-status" in argv:
        mins = float(argv[argv.index("--max-stale-minutes") + 1]) if "--max-stale-minutes" in argv else 0.0
        sys.exit(check_status(mins))

    if "--live" in argv:
        live = live_titles(today, force="--force" in argv)
        if not live:
            print("no match is due, under way or overdue: nothing to ask Liquipedia (--force overrides)")
            return
        print(f"live tournament pages: {sorted(live)}")
        import collect_liquipedia
        collect_liquipedia.OFFLINE = False
        # one short retry, no waiting loop: the next scheduled run is the retry
        lpfetch.RETRY_BACKOFF[:] = LIVE_RETRY_BACKOFF
        lpfetch.MAX_RETRY_AFTER = LIVE_MAX_RETRY_AFTER
        stale = refresh_live(live, collect_liquipedia.main, pauses=())
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
    import collect_liquipedia
    collect_liquipedia.OFFLINE = False
    live = live_titles(today)
    # 1) the live tournament pages alone, first, while Liquipedia's rate budget is
    #    untouched: on 2026-10-14 the bulk refresh got rate-limited and the live page,
    #    fetched in the same batch, was refused with it
    stale = refresh_live(live, collect_liquipedia.main, pauses=()) if live else []
    # 2) then every other tracked tournament and team page
    lpfetch.FRESH_TITLES |= titles | teams
    lpfetch._THROTTLED[0] = False
    collect_liquipedia.main()
    # 3) a live page refused in step 1 gets the full retry (pause, ask again) at the end
    if stale:
        lpfetch._THROTTLED[0] = False
        stale = refresh_live(set(stale), collect_liquipedia.main)
    write_status(stale, live)

    import rosters
    rosters.OFFLINE = False
    # SUPERVISED roster flow: the refresh only stages a candidate; a human
    # promotes it after reviewing the roster_watch diff. Auto-applying roster
    # changes corrupts every downstream prediction.
    rosters.CANDIDATE = True
    rosters.main()

    import roster_watch
    rc = roster_watch.main()
    if rc == 1:
        print("roster changes need human review: see data/roster_diff.md "
              "(the live roster_events.json is unchanged)", file=sys.stderr)
    elif rc == 2:
        print("roster_watch found no candidate to diff", file=sys.stderr)

    import fetch_assets
    fetch_assets.OFFLINE = False
    fetch_assets.main()

    import vrs          # Valve Regional Standings (GitHub; new file roughly monthly)
    vrs.OFFLINE = False
    # refresh() never raises: on a failure the last good vrs.json stays and
    # data/vrs_status.json says so (ok=false). The run goes on so S/A data still
    # publishes, and the workflow's `vrs.py --check` step plus the data gate
    # turn it into a red run and an issue: never silent.
    vstat = vrs.refresh()
    print(f"VRS stored {vstat.get('stored')}, newest published by Valve {vstat.get('published')}"
          + ("" if vstat["ok"] else f" (REFRESH FAILED: {vstat['error']})"))

    # lineups are optional input (roster-core features): a failure here must
    # never break the data refresh, so it is reported and the old file is kept
    try:
        import lineups
        lineups.OFFLINE = False
        lineups.main(incremental=True)
    except (Exception, SystemExit) as e:   # noqa: BLE001
        print(f"lineups: refresh skipped ({type(e).__name__}: {e}); data/lineups.json unchanged",
              file=sys.stderr)

    # second results source (Valve's standings match rows) for the cross_source
    # gate: read-only cross-check, so an outage only warns; the committed file
    # stays and the gate keeps comparing against it
    try:
        import valve_results
        valve_results.main(offline=False, incremental=True)
    except (Exception, SystemExit) as e:   # noqa: BLE001
        print(f"WARNING valve_results: refresh skipped ({type(e).__name__}: {e}); "
              "data/valve_results.json unchanged", file=sys.stderr)
    refresh_lower()
    print(f"refresh done; Liquipedia API requests sent: {lpfetch.NETWORK_REQUESTS[0]}")


def refresh_lower(run=None):
    """Lower-tier results (data/matches_lower.json: Elo data and team-page
    results, never events). Runs LAST in the daily run (never in --live), so it
    can neither delay nor block the S/A data, with its own request budget
    (pipeline/lower_refresh.py). Any failure is a WARNING: the committed file
    stays and the run goes on. Returns the step's report, or None on failure."""
    s0 = lpfetch.NETWORK_REQUESTS[0]
    try:
        import lower_refresh
        rep = (run or lower_refresh.run)()
        print(lower_refresh.report_line(rep))
        for w in rep.get("warnings", []):
            print(f"WARNING lower tier: {w}", file=sys.stderr)
        return rep
    except (Exception, SystemExit) as e:   # noqa: BLE001
        print(f"WARNING lower tier: refresh skipped ({type(e).__name__}: {e}); data/matches_lower.json "
              f"unchanged; {lpfetch.NETWORK_REQUESTS[0] - s0} Liquipedia requests sent by this step",
              file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
