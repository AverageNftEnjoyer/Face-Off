"""Polite, cached fetcher for the Liquipedia Counter-Strike MediaWiki API.

Terms followed (https://liquipedia.net/api-terms-of-use):
  * descriptive User-Agent, gzip accepted
  * <= 1 request per 2 seconds for action=query (we use 2.5 s)
  * action=parse is NOT used (it has a 30 s limit); we read raw wikitext
    via action=query&prop=revisions instead.
  * every response is cached under data/raw/liquipedia/ and never re-fetched.
Stdlib only.
"""
import gzip
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://liquipedia.net/counterstrike/api.php"
UA = ("FaceOff-CS2-insights/0.2 (non-commercial fan analytics; "
      "https://github.com/AverageNftEnjoyer/Face-Off; python-urllib; low volume, cached)")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "liquipedia")
MIN_GAP = 2.5
_last = [0.0]
# Daily refresh (scripts/daily_refresh.py): pages named here are re-fetched
# ONCE per run even if cached; everything else is still served from the cache.
# A cached response fetched before RUN_STARTED counts as stale for them.
FRESH_TITLES = set()
FRESH_ALL = False          # set True to refresh every request made this run (prefix listings)
RUN_STARTED = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
NETWORK_REQUESTS = [0]     # requests actually sent this process (cache hits excluded)


def _key(params):
    s = urllib.parse.urlencode(sorted(params.items()))
    h = s if API.endswith("/counterstrike/api.php") else API + "?" + s
    return hashlib.sha1(h.encode()).hexdigest()[:16], s


def query(params, offline=False, refresh=False):
    params = dict(params)
    params.setdefault("format", "json")
    assert params.get("action") != "parse", "parse is rate limited to 1/30s; not used"
    os.makedirs(CACHE, exist_ok=True)
    k, qs = _key(params)
    path = os.path.join(CACHE, k + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        stale = (refresh or FRESH_ALL) and cached.get("fetched", "") < RUN_STARTED
        if offline or not stale:
            return cached["response"]
    if offline:
        return None
    wait = MIN_GAP - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(API + "?" + qs, headers={
        "User-Agent": UA, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    _last[0] = time.time()
    resp = json.loads(raw.decode("utf-8"))
    rec = {"url": API + "?" + qs,
           "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "response": resp}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    if _PAGE_INDEX is not None:
        _index_add(_PAGE_INDEX, rec)
    NETWORK_REQUESTS[0] += 1
    return resp


_PAGE_INDEX = None
_MISSING = {}              # title -> fetch time of a response saying the page does not exist


def cached_pages():
    """{requested or resolved title: (fetched, wikitext)} across EVERY cached
    revisions response, newest fetch wins. Lets offline reads find a page no
    matter which batch it was fetched in (adding titles to a list changes the
    batch, and so the cache key, of the last batch)."""
    global _PAGE_INDEX
    if _PAGE_INDEX is None:
        idx = {}
        _MISSING.clear()
        if os.path.isdir(CACHE):
            for fn in sorted(os.listdir(CACHE)):
                with open(os.path.join(CACHE, fn), encoding="utf-8") as f:
                    _index_add(idx, json.load(f))
        _PAGE_INDEX = idx
    return _PAGE_INDEX


def _index_add(idx, rec):
    q = rec.get("response", {}).get("query", {})
    pages = {p.get("title"): p for p in q.get("pages", {}).values()}
    alias = {r["from"]: r["to"] for r in q.get("normalized", []) + q.get("redirects", [])}
    for title, p in pages.items():
        revs = p.get("revisions")
        names = [title] + [k for k, v in alias.items() if v == title]
        names += [k for k, v in alias.items() if v in names]
        if "missing" in p:
            for n in names:
                _MISSING[n] = max(_MISSING.get(n, ""), rec.get("fetched", ""))
            continue
        if not revs:
            continue
        text = revs[0]["slots"]["main"]["*"]
        for n in names:
            if n not in idx or idx[n][0] < rec.get("fetched", ""):
                idx[n] = (rec.get("fetched", ""), text)


def wikitext(titles, offline=False, strict=True):
    """Return {title: wikitext or None} for up to 50 titles (one request).
    offline=True never touches the network. On an offline miss of the exact
    batch, pages are looked up individually in the whole cache; a title that is
    nowhere in the cache raises (strict) or comes back None (strict=False).

    Online, a batch whose pages are ALL already cached (in any earlier batch)
    and not marked fresh is served from the cache too, so re-batching a title
    list (discovery fetches 20 per request, the collectors 8) never re-fetches
    a page. Titles a cached response reported as missing (teams without a
    Liquipedia page) count as cached and come back None."""
    refresh = any(t in FRESH_TITLES for t in titles)
    if not offline and not refresh and not FRESH_ALL:
        idx = cached_pages()
        if all(t in idx or t in _MISSING for t in titles):
            return {t: (idx[t][1] if t in idx else None) for t in titles}
    resp = query({"action": "query", "prop": "revisions", "rvprop": "content",
                  "rvslots": "main", "redirects": "1", "titles": "|".join(titles)},
                 offline=offline, refresh=refresh)
    if resp is None:
        idx = cached_pages()
        missing = [t for t in titles if t not in idx and t not in _MISSING]
        if missing and strict:
            raise RuntimeError("offline cache miss for titles: %r" % (missing,))
        return {t: (idx[t][1] if t in idx else None) for t in titles}
    out = {}
    q = resp.get("query", {})
    redirect = {r["from"]: r["to"] for r in q.get("redirects", [])}
    norm = {r["from"]: r["to"] for r in q.get("normalized", [])}
    pages = {p["title"]: p for p in q.get("pages", {}).values()}
    for t in titles:
        tt = norm.get(t, t)
        tt = redirect.get(tt, tt)
        p = pages.get(tt)
        if not p or "missing" in p or not p.get("revisions"):
            # missing page, or content left out because the response hit the
            # API's size limit (the caller can ask for that page on its own)
            out[t] = None
        else:
            out[t] = p["revisions"][0]["slots"]["main"]["*"]
    return out


def prefix(pfx, limit=500, refresh=False):
    resp = query({"action": "query", "list": "allpages", "apprefix": pfx,
                  "apnamespace": "0", "aplimit": str(limit)}, refresh=refresh)
    return [p["title"] for p in resp.get("query", {}).get("allpages", [])]
