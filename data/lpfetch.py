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
UA = ("FaceOff-CS2-insights/0.1 (non-commercial fan analytics backtest; "
      "python-urllib; low volume, cached)")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "liquipedia")
MIN_GAP = 2.5
_last = [0.0]


def _key(params):
    s = urllib.parse.urlencode(sorted(params.items()))
    h = s if API.endswith("/counterstrike/api.php") else API + "?" + s
    return hashlib.sha1(h.encode()).hexdigest()[:16], s


def query(params, offline=False):
    params = dict(params)
    params.setdefault("format", "json")
    assert params.get("action") != "parse", "parse is rate limited to 1/30s; not used"
    os.makedirs(CACHE, exist_ok=True)
    k, qs = _key(params)
    path = os.path.join(CACHE, k + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["response"]
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"url": API + "?" + qs,
                   "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "response": resp}, f, ensure_ascii=False)
    return resp


def wikitext(titles, offline=False):
    """Return {title: wikitext or None} for up to 50 titles (one request).
    offline=True never touches the network; a cache miss raises."""
    resp = query({"action": "query", "prop": "revisions", "rvprop": "content",
                  "rvslots": "main", "redirects": "1", "titles": "|".join(titles)},
                 offline=offline)
    if resp is None:
        raise RuntimeError("offline cache miss for titles: %r" % (titles,))
    out = {}
    q = resp.get("query", {})
    redirect = {r["from"]: r["to"] for r in q.get("redirects", [])}
    norm = {r["from"]: r["to"] for r in q.get("normalized", [])}
    pages = {p["title"]: p for p in q.get("pages", {}).values()}
    for t in titles:
        tt = norm.get(t, t)
        tt = redirect.get(tt, tt)
        p = pages.get(tt)
        if not p or "missing" in p:
            out[t] = None
        else:
            out[t] = p["revisions"][0]["slots"]["main"]["*"]
    return out


def prefix(pfx, limit=500):
    resp = query({"action": "query", "list": "allpages", "apprefix": pfx,
                  "apnamespace": "0", "aplimit": str(limit)})
    return [p["title"] for p in resp.get("query", {}).get("allpages", [])]
