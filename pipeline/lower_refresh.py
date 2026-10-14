#!/usr/bin/env python3
"""
lower_refresh.py -- daily refresh of data/matches_lower.json.

Lower-tier results (B-tier main events and their stages, closed qualifiers of
tracked S/A events) are RATING DATA and team-page results only: they never
become tournaments, fixtures or predictions on the site (viewer/events.py
reads lp_titles_selected.txt, never the lower-tier list). This keeps the file
current; backtest.LOWER_TIER_ELO_WEIGHT decides whether it moves Elo.

Run by pipeline/daily_refresh.py at the END of the daily run (never in --live),
after every S/A step, inside a try/except: any failure here is a WARNING and the
committed file stays as it was. All requests go through collectors/lpfetch.py (cache,
2.5 s spacing, backoff); this step stops asking once REQUEST_BUDGET requests were
sent (what it did not get to is picked up by the next run: everything fetched is
cached).

Steps
  1. Discovery: the newest 500 additions to each tier category in LOWER_TIERS
     (one request per tier). A title is a candidate under the same rule as the
     S/A discovery (daily_refresh.tier_candidates: names this year or next, or
     no year and added in the last NEW_SINCE_DAYS days). Its own infobox
     decides: tier in LOWER_TIERS, not a qualifier / showmatch / weekly /
     monthly / misc / points page, starts this year or next, at most
     LOWER_MAX_DAYS long. A sub-page that is a stage of a listed page or a
     regional sub-page is not an event of its own. Stage pages of new events:
     one `allpages` listing per parent path.
  2. Closed qualifiers of tracked S/A events: sub-pages found in the cached
     `allpages` listings (the S/A run refreshes the series listings every run,
     so this costs no listing request) whose path is a region / "Qualifier"
     (backfill_lower.is_qualifier_path); open qualifiers are left out.
  3. Re-read the pages of lower-tier events that are on now or ended in the
     last daily_refresh.REFRESH_DAYS days (20 per request). Their stage
     listing is read again when the cached one is older than RELIST_DAYS.
  4. Rebuild with collect_liquipedia.build_lower (from the cache; new team
     codes cost one expandtemplates request per 100). The new file is written
     only when it passes the schema check (data_gates.gate_lower rules) and
     keeps at least MIN_KEEP_FRAC of the old series; otherwise the old file
     stays and a warning is printed.

USAGE (repo root):  python pipeline/lower_refresh.py [--dry-run]
Stdlib only.
"""
import json
import os
import sys
import time
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

import lpfetch  # noqa: E402
import events as E  # noqa: E402
import daily_refresh as DR  # noqa: E402

# B-tier only: the 2026-10-08 test (lower_tier_check, removed 2026-10-13, in git history) covers B-tier
# events and closed qualifiers; C-tier was sized and left out (about 115 new
# pages a month, mostly amateur teams that rarely meet S/A teams).
LOWER_TIERS = ("B",)
LOWER_TIER_CATEGORIES = {t: f"Category:{t}-Tier Tournaments" for t in LOWER_TIERS}
# league seasons (about 110 days) are real results; for rating data they are
# kept up to this length (the site's 45-day rule is about what counts as one
# tournament on a page). Same value as the 2025 backfill.
LOWER_MAX_DAYS = 150
# undated-title candidates: added to the category this recently (same window
# as the S/A run's daily discovery)
NEW_SINCE_DAYS = 120
# requests this step may send per run (a typical run needs 2-10)
REQUEST_BUDGET = 40
# a running event's stage listing is read again when the cached one is older
RELIST_DAYS = 3
# a rebuilt file with fewer series than this share of the old one is not written
MIN_KEEP_FRAC = 0.98
# backoff for this step: one short retry, then the cache (never the S/A run's long waits)
LOWER_RETRY_BACKOFF = [30]

TITLES = os.path.join(ROOT, "data", "raw", "lp_titles_lower.txt")
KINDS = os.path.join(ROOT, "data", "raw", "lp_titles_lower_kinds.json")
OUT = os.path.join(ROOT, "data", "matches_lower.json")


class Budget:
    """Requests sent by this step, by purpose; `left()` against REQUEST_BUDGET."""

    def __init__(self, cap=REQUEST_BUDGET):
        self.cap, self.start, self.by = cap, lpfetch.NETWORK_REQUESTS[0], {}
        self._mark = self.start

    def used(self):
        return lpfetch.NETWORK_REQUESTS[0] - self.start

    def left(self):
        return self.cap - self.used()

    def note(self, what):
        n = lpfetch.NETWORK_REQUESTS[0] - self._mark
        self._mark = lpfetch.NETWORK_REQUESTS[0]
        if n:
            self.by[what] = self.by.get(what, 0) + n


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def lower_main_ok(info, years):
    """A lower-tier main event: tier in LOWER_TIERS, not a skipped tier type,
    starts in one of `years`, at most LOWER_MAX_DAYS long."""
    tier, ttype, sd, ed = info
    if tier not in LOWER_TIERS or ttype.lower() in E.SKIP_TIERTYPES or not sd or sd[:4] not in years:
        return False
    try:
        return not (ed and (date.fromisoformat(ed[:10]) - date.fromisoformat(sd[:10])).days > LOWER_MAX_DAYS)
    except ValueError:
        return False


def texts_within(titles, budget, batch=20):
    """{title: wikitext} for as many titles as the budget allows (cached pages
    cost nothing; a batch that needs the network is skipped once the budget
    is spent)."""
    out = {}
    idx = lpfetch.cached_pages()
    for i in range(0, len(titles), batch):
        part = titles[i:i + batch]
        need = any(t in lpfetch.FRESH_TITLES or (t not in idx and t not in lpfetch._MISSING) for t in part)
        if need and budget.left() <= 0:
            break
        out.update(DR.page_texts(part, batch=batch))
    return out


def discover_events(today, have, budget):
    """New lower-tier main events (and their stage pages) from the tier categories."""
    years = (str(today.year), str(today.year + 1))
    since = (today - timedelta(days=NEW_SINCE_DAYS)).isoformat()
    members = []
    for tier, cat in LOWER_TIER_CATEGORIES.items():
        if budget.left() <= 0:
            break
        resp = lpfetch.query({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmnamespace": "0",
                              "cmprop": "title|timestamp", "cmsort": "timestamp", "cmdir": "desc",
                              "cmlimit": "500"}, refresh=True) or {}
        members += [(tier, r["title"], r.get("timestamp", ""))
                    for r in resp.get("query", {}).get("categorymembers", [])]
    budget.note("category listings")
    cands = [t for t in DR.tier_candidates(members, years, since) if t not in have]
    idx = lpfetch.cached_pages()
    # a candidate cached while still undated is read again (it may have dates now)
    lpfetch.FRESH_TITLES |= {t for t in cands if t in idx and not E.event_info(idx[t][1])[2]}
    texts = texts_within(cands, budget)
    budget.note("candidate infoboxes")
    mains = [t for t in cands if texts.get(t) and lower_main_ok(E.event_info(texts[t]), years)]
    parents = have | set(mains) | {t for _, t, _ in members}
    keep = []
    for t in mains:
        anc = [t[:i] for i in range(len(t) - 1, 0, -1) if t[i] == "/" and t[:i] in parents]
        if anc:
            tail = t[len(anc[0]) + 1:].split("/")
            if any(x in DR.SKIP_SEGMENTS for x in tail) and all(x in DR.SKIP_SEGMENTS or x.isdigit() for x in tail):
                continue          # a regional sub-page: a qualifier of its parent
            ptxt = texts.get(anc[0]) or (idx[anc[0]][1] if anc[0] in idx else "")
            pinfo = E.event_info(ptxt)
            if pinfo[2] and E.is_stage(E.event_info(texts[t]), pinfo):
                continue          # a stage: picked up with its parent's stage pages
        keep.append(t)
    stages = []
    if keep and budget.left() > 0:
        stages = DR.stage_pages(keep, have | set(keep), refresh=True)
    budget.note("stage listings + pages")
    return keep, stages


def discover_qualifiers(today, have, budget):
    """Closed qualifier pages of tracked S/A main events, from the cached
    allpages listings (no listing request)."""
    gap = lpfetch.MIN_GAP
    import backfill_lower as BL      # its import sets the backfill's own 5 s gap: keep the run's
    lpfetch.MIN_GAP = gap
    years = (str(today.year), str(today.year + 1))
    mains, texts = BL.sa_events()
    events = [t for t in mains if E.event_info(texts.get(t))[2][:4] in years]
    listings = BL.cached_listings()
    found = set()
    for e in events:
        for p, ts in listings.items():
            if (e + "/").startswith(p) or p.startswith(e + "/"):
                found |= {t for t in ts if t.startswith(e + "/") and BL.is_qualifier_path(e, t)}
    found = sorted(found - have)
    qtexts = texts_within(found, budget)
    budget.note("qualifier pages")
    keep = []
    for t in found:
        txt = qtexts.get(t)
        info = E.event_info(txt)
        if not txt or "{{Match" not in txt:
            continue          # not played yet: found again on a later run
        if info[1].lower() not in ("qualifier", "") or info[2][:4] not in years:
            continue
        keep.append(t)
    return keep


def page_dates(titles):
    """{title: (sdate, edate)} from each page's own infobox, else its nearest
    listed ancestor's (stage pages often have no dates)."""
    idx = lpfetch.cached_pages()
    have = set(titles)
    out = {}
    for t in sorted(titles, key=lambda t: t.count("/")):
        info = E.event_info(idx[t][1]) if t in idx else ("", "", "", "")
        if info[2] and info[3]:
            out[t] = (info[2], info[3])
            continue
        anc = [t[:i] for i in range(len(t) - 1, 0, -1) if t[i] == "/" and t[:i] in have]
        out[t] = out.get(anc[0], ("", "")) if anc else ("", "")
    return out


def live_lower(titles, today):
    """Lower-tier pages whose event is on now or ended in the last REFRESH_DAYS days."""
    lo = (today - timedelta(days=DR.REFRESH_DAYS)).isoformat()
    hi = (today + timedelta(days=1)).isoformat()
    return sorted(t for t, (sd, ed) in page_dates(titles).items() if sd and ed and sd <= hi and ed >= lo)


def listing_age_days(prefix, today):
    """Days since the cached allpages listing of `prefix` was fetched (None: not cached)."""
    k, _ = lpfetch._key({"action": "query", "list": "allpages", "apprefix": prefix, "apnamespace": "0",
                          "aplimit": "500", "format": "json"})
    path = os.path.join(lpfetch.CACHE, k + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            fetched = json.load(f).get("fetched", "")[:10]
        return (today - date.fromisoformat(fetched)).days
    except (OSError, ValueError):
        return None


def refresh_live(titles, today, have, budget):
    """Re-read the live lower-tier pages; list a live event's stage pages again
    when its cached listing is older than RELIST_DAYS. Returns new stage pages."""
    live = live_lower(titles, today)
    tops = [t for t in live if not any(t.startswith(o + "/") for o in live if o != t)]
    new_stages = []
    for e in tops:
        if budget.left() <= 0:
            break
        parent = e.rsplit("/", 1)[0] + "/" if "/" in e else e + "/"
        age = listing_age_days(parent, today)
        if age is None or age >= RELIST_DAYS:
            new_stages += [s for s in DR.stage_pages([e], have | set(new_stages), refresh=True)
                           if s not in new_stages]
    budget.note("stage listings (running events)")
    want = sorted(set(live) | set(new_stages))
    stale = set()
    if want and budget.left() > 0:
        lpfetch.FRESH_TITLES |= set(want)
        served0 = len(lpfetch.STALE_SERVED)
        texts_within(want, budget)
        stale = {t for e in lpfetch.STALE_SERVED[served0:] for t in str(e).split("|")} & set(want)
        # read once this run: the builder's own batches must not ask again
        lpfetch.FRESH_TITLES -= set(want)
    budget.note("page refresh (running events)")
    return live, new_stages, sorted(stale)


def check(records):
    """Schema problems of a rebuilt list (data_gates.gate_lower rules)."""
    import data_gates as G
    bad = G.gate_schema(records)
    bad += [f"record {i}: unknown kind {m.get('kind')!r}" for i, m in enumerate(records)
            if m.get("kind") not in G.LOWER_KINDS]
    return bad


def write_titles(add, qualifiers):
    old = read_lines(TITLES)
    out = sorted(set(old) | set(add))
    if out != old:
        with open(TITLES, "w", encoding="utf-8") as f:
            f.write("".join(t + "\n" for t in out))
    if qualifiers:
        kinds = json.load(open(KINDS, encoding="utf-8")) if os.path.exists(KINDS) else {}
        q = sorted(set(kinds.get("qualifier", [])) | set(qualifiers))
        if q != kinds.get("qualifier"):
            kinds["qualifier"] = q
            with open(KINDS, "w", encoding="utf-8") as f:
                json.dump(kinds, f, indent=1, ensure_ascii=False)
    return len(out) - len(old)


def run(today=None, dry=False, out_path=OUT):
    """One refresh. Returns a report dict; raises nothing it can catch (the
    caller still wraps it). The old file stays unless the rebuild passes."""
    import collect_liquipedia as C
    today = today or date.today()
    rep = {"requests": 0, "by_purpose": {}, "warnings": []}
    budget = Budget()
    saved = (list(lpfetch.RETRY_BACKOFF), lpfetch.MAX_RETRY_AFTER)
    lpfetch.RETRY_BACKOFF[:] = LOWER_RETRY_BACKOFF
    lpfetch.MAX_RETRY_AFTER = 60
    try:
        if lpfetch._THROTTLED[0]:
            rep["warnings"].append("Liquipedia throttled this run: lower-tier pages were not asked for")
            have_titles = read_lines(TITLES)
            new_mains, stages, quals, live, live_stages, stale = [], [], [], [], [], []
        else:
            have_titles = read_lines(TITLES)
            have = set(have_titles) | set(read_lines(DR.TITLES_FILE))
            new_mains, stages = discover_events(today, have, budget)
            have |= set(new_mains) | set(stages)
            quals = discover_qualifiers(today, have, budget) if budget.left() > 0 else []
            have |= set(quals)
            live, live_stages, stale = refresh_live(sorted(set(have_titles) | set(new_mains) | set(stages)
                                                           | set(quals)), today, have, budget)
            if stale:
                rep["warnings"].append(f"{len(stale)} running lower-tier page(s) served from the cache")
            if budget.left() <= 0:
                rep["warnings"].append(f"request budget ({REQUEST_BUDGET}) reached; the rest waits for the next run")
        add = new_mains + stages + quals + live_stages
        rep.update(new_events=new_mains, new_stages=stages + live_stages, new_qualifiers=quals,
                   refreshed_pages=len(live))
        if dry:
            return rep
        rep["titles_added"] = write_titles(add, quals)
        # every lower-tier page that needed reading was read above: the
        # builder's own batches (8 titles) must not ask again
        lpfetch.FRESH_TITLES -= set(read_lines(TITLES))
        C.OFFLINE = False
        records = C.build_lower() or []
        budget.note("builder (team codes)")
        old = []
        if os.path.exists(out_path):
            try:
                with open(out_path, encoding="utf-8") as f:
                    old = json.load(f)
            except (OSError, ValueError):
                old = []
        bad = check(records)
        if bad:
            rep["warnings"].append(f"rebuilt lower-tier data fails the schema check ({len(bad)} issues, "
                                   f"e.g. {bad[0]}); data/matches_lower.json unchanged")
        elif old and len(records) < MIN_KEEP_FRAC * len(old):
            rep["warnings"].append(f"rebuilt lower-tier data has {len(records)} series, the published file "
                                   f"{len(old)}; data/matches_lower.json unchanged")
        elif records != old:
            C.write_lower(records, out_path)
            rep["written"] = True
        rep["series"] = len(records)
        rep["newest"] = records[-1]["date"] if records else None
        rep["added_series"] = len(records) - len(old)
    finally:
        lpfetch.RETRY_BACKOFF[:], lpfetch.MAX_RETRY_AFTER = saved
        rep["requests"] = budget.used()
        rep["by_purpose"] = budget.by
    return rep


def report_line(rep):
    return (f"lower tier: {len(rep.get('new_events', []))} new events, {len(rep.get('new_stages', []))} stage "
            f"pages, {len(rep.get('new_qualifiers', []))} qualifiers, {rep.get('refreshed_pages', 0)} running "
            f"pages re-read; {rep.get('series', '?')} series (newest {rep.get('newest')}, "
            f"{rep.get('added_series', 0):+d}){' written' if rep.get('written') else ''}; "
            f"Liquipedia requests {rep['requests']} {rep.get('by_purpose') or ''}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for s in (sys.stdout, sys.stderr):      # page titles are not all ASCII (a Windows console is cp1252)
        try:
            s.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    t0 = time.time()
    rep = run(dry="--dry-run" in argv)
    print(report_line(rep) + f" ({time.time() - t0:.0f}s)")
    for w in rep["warnings"]:
        print(f"WARNING lower tier: {w}", file=sys.stderr)
    for t in rep.get("new_events", []) + rep.get("new_qualifiers", []):
        print("  +", t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
