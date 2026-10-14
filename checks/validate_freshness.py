#!/usr/bin/env python3
"""
Freshness check for the Faceoff site, run on its own schedule by
.github/workflows/validate.yml (separate from the refresh that updates the data).

The refresh job cannot be the only thing that notices it failed: on 2026-10-09
GitHub skipped scheduled runs for nine hours and one run reported success with
old data. This script looks from the outside and answers three questions.

  A. LIVE PAGES  Does Liquipedia hold newer content for a live tournament than
                 the repo's cache? (one request per run, only while a tracked
                 tournament is on; none otherwise)
  B. REFRESH     Did the last refresh say it could not update a live page?
                 (data/refresh_status.json)
  D. VALVE       Are the stored Valve Regional Standings (data/vrs.json) older
                 than the newest file in Valve's repository, or older than
                 Valve's measured cadence allows? (one GitHub API call; the
                 verdict starts the DAILY refresh, which fetches standings)
  C. SITE        Is the deployed site built from the repo's current
                 data/matches.json? (its page carries a data_hash; a mismatch
                 right after a push is a deploy in progress, not a failure)

Exit code 0 = everything current. 1 = the data is behind Liquipedia or the
refresh reported stale pages (the workflow then starts the live refresh and
fails the run). 2 = only the deployed site is behind the repo. When running in
GitHub Actions the verdict is also written to $GITHUB_OUTPUT.

USAGE (repo root):  python checks/validate_freshness.py [--site URL] [--no-liquipedia]
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import date, datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

SITE = "https://cs2faceoffanalytics.vercel.app/"
RECENT_REFRESH_MIN = 25      # a data commit younger than this means the refresh is working; skip the Liquipedia request
DEPLOY_GRACE_MIN = 20      # a data commit younger than this may still be building on the host
STALE_FAIL_MIN = 30        # a live page refused this long fails the run; before that the live refresh is started (heal)


def local_hash():
    with open(os.path.join(ROOT, "data", "matches.json"), "rb") as f:
        return hashlib.sha1(f.read().replace(b"\r\n", b"\n")).hexdigest()[:12]


def site_hash(url):
    """data_hash of the page at `url`, '' when the page predates the stamp."""
    req = urllib.request.Request(url, headers={"User-Agent": "FaceOff-freshness-check/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read(40_000_000).decode("utf-8", "replace")
    m = re.search(r'"data_hash":"([0-9a-f]{12})"', html)
    return m.group(1) if m else ""


def minutes_since_data_commit(path="data/matches.json"):
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ct", "--", path],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return (time.time() - int(out)) / 60.0
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 1e9


def live_pages_behind(live):
    """Live pages whose Liquipedia content differs from the repo's cached copy.
    Fetched into a scratch cache so the repo's cache is untouched."""
    import lpfetch
    have = lpfetch.cached_pages()
    saved = lpfetch.CACHE
    behind = []
    with tempfile.TemporaryDirectory() as tmp:
        lpfetch.CACHE = tmp
        lpfetch.FRESH_TITLES |= set(live)
        try:
            titles = sorted(live)
            for i in range(0, len(titles), 20):
                got = lpfetch.wikitext(titles[i:i + 20], strict=False)
                for t, txt in got.items():
                    if txt is not None and (t not in have or have[t][1] != txt):
                        behind.append(t)
        finally:
            lpfetch.CACHE = saved
    if lpfetch.STALE_SERVED:
        print("could not reach Liquipedia for a fresh read; treating the live pages as unverified",
              file=sys.stderr)
    return sorted(behind)


def vrs_problems(tree_obj=None, today=None):
    """Stored standings vs the newest Valve published (live tree read) and vs
    Valve's cadence. A tree that cannot be read falls back to the status the
    last refresh recorded (a warning is printed)."""
    import vrs
    stored = None
    try:
        with open(os.path.join(ROOT, "data", "vrs.json"), encoding="utf-8") as f:
            stored = json.load(f).get("date")
    except (OSError, ValueError):
        pass
    try:
        paths = [t["path"] for t in (tree_obj or vrs.tree()).get("tree", [])]
        pub = vrs.published(paths)
        status = {"published": max(pub), "stale_after_days": vrs.cadence(pub)["stale_after"]}
    except Exception as e:   # noqa: BLE001
        print(f"could not read Valve's repository tree ({type(e).__name__}: {e}); "
              "checking against the status the last refresh recorded", file=sys.stderr)
        try:
            with open(os.path.join(ROOT, "data", "vrs_status.json"), encoding="utf-8") as f:
                status = json.load(f)
        except (OSError, ValueError):
            status = {}
        status = {k: status.get(k) for k in ("published", "stale_after_days")}
    return vrs.assess(stored, status, today or date.today())


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    url = argv[argv.index("--site") + 1] if "--site" in argv else SITE
    problems, site_only = [], []
    heal = False      # B: a fresh refusal -> start the live refresh without failing the run

    # A. live pages against Liquipedia
    if "--no-liquipedia" not in argv:
        import daily_refresh as dr
        live = dr.live_titles(date.today())
        recent = minutes_since_data_commit("data")
        if live and recent < RECENT_REFRESH_MIN:
            # the live refresh just ran and committed: asking Liquipedia again adds a request and tells us nothing
            print(f"data was refreshed {recent:.0f} min ago: not asking Liquipedia again")
        elif live:
            behind = live_pages_behind(live)
            if behind:
                problems.append(f"Liquipedia has newer content than the repo for: {behind}")
            else:
                print(f"live pages current with Liquipedia: {sorted(live)}")
        else:
            print("no match is due, under way or overdue: nothing to ask Liquipedia")

    # B. what the last refresh reported
    try:
        with open(os.path.join(ROOT, "data", "refresh_status.json"), encoding="utf-8") as f:
            st = json.load(f)
        if st.get("stale_pages"):
            # one refusal is healed by starting the live refresh at once; only a page that
            # has stayed stale for STALE_FAIL_MIN (retries keep failing) fails the run
            try:
                since = datetime.strptime(str(st.get("stale_since")), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - since).total_seconds() / 60
            except (TypeError, ValueError):
                age = float("inf")
            if age >= STALE_FAIL_MIN:
                problems.append(f"the last refresh could not update {st['stale_pages']} for {age:.0f} min")
            else:
                heal = True
                print(f"::warning::Liquipedia refused {st['stale_pages']} {age:.0f} min ago; starting the live "
                      f"refresh now (fails only after {STALE_FAIL_MIN} min)")
    except (OSError, ValueError):
        pass

    # D. Valve standings against Valve's repository
    vrs_bad = []
    if "--no-valve" not in argv:
        vrs_bad = vrs_problems()
        problems.extend(vrs_bad)

    # C. the deployed site against the repo
    try:
        theirs, mine = site_hash(url), local_hash()
        if theirs == mine:
            print(f"site is built from the repo's data ({mine})")
        elif minutes_since_data_commit() < DEPLOY_GRACE_MIN:
            print(f"site shows {theirs or 'no stamp'}, repo has {mine}: data was committed in the last "
                  f"{DEPLOY_GRACE_MIN} minutes, so a deploy is probably still running")
        else:
            site_only.append(f"the site is built from data {theirs or '(no stamp)'} but the repo has {mine}; "
                             f"check the Vercel deployment")
    except (OSError, ValueError) as e:
        site_only.append(f"could not read the site at {url}: {e}")

    for p in problems + site_only:
        print("FAIL:", p, file=sys.stderr)
    code = 1 if problems else 2 if site_only else 0
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"code={code}\nneeds_refresh={'true' if heal or len(problems) > len(vrs_bad) else 'false'}\n")
            f.write(f"needs_vrs_refresh={'true' if vrs_bad else 'false'}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
