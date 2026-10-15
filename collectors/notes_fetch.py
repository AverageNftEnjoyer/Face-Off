#!/usr/bin/env python3
"""notes_fetch.py -- web facts for the matchup notes (fact schema v1,
docs/notes-contracts.md section 1) -> data/notes_facts.json.

Facts are display context only: nothing written here is read by the
predictor, the backtest or any weight fit.

Scope: the teams with a fixture in the next 7 days (viewer/data-bundle.json,
`fixtures[].day` in [as_of, as_of + 7]); `--teams A,B` overrides it. Per scope
team there are two kinds of work unit, and every unit costs exactly ONE LLM
call (OpenAI Responses API, POST /v1/responses, key from OPENAI_API_KEY, model
from FACEOFF_NOTES_MODEL, default DEFAULT_MODEL):

  news  one call per team with the `web_search` tool: stand-ins and roster
        rumours of the last 30 days only, strict JSON (json_schema structured
        output), every item with an https source_url. Never prose.
  form  one call per rostered player (coaches skipped). Preferred path
        "scrape": a plain HTTP fetch of the player's HLTV stats page, stripped
        to text and handed to the LLM as an EXTRACTOR only (no tools). The
        HLTV URL is learnt from an earlier fact's source_url (we have no HLTV
        ids of our own). When the page cannot be read (robots.txt disallows
        it, HTTP 403 / Cloudflare challenge, network error) the same unit falls
        back to "web_search" for that player. As of 2026-10-14 BOTH apply to
        HLTV: robots.txt disallows the dated (`?startDate=`) stats pages a
        30-day rating needs, and plain HTTP gets a 403 Cloudflare challenge, so
        in practice every form unit runs on web_search; the manifest says so
        under `scrape`.

LLM output is mapped onto exact bundle team names (a team the bundle does not
know is rejected), stamped with this run's one `observed_at`, built with
notes_facts.make_fact and checked with notes_facts.validate_fact. A rejected
fact is logged in the manifest with its reasons and never written. Accepted
facts are appended with notes_facts.merge (history kept, 400-day prune); the
file is rewritten only when its content changes.

Polite crawling (scrape path; terms in data/SOURCES.md):
  * robots.txt is read once per host per run (cached) and obeyed. Python's
    urllib.robotparser does prefix matching only and so misses HLTV's
    wildcard rules (`/stats*?*startDate*`); a URL is fetched only when
    robotparser AND a wildcard-aware check of the same file both allow it.
  * descriptive User-Agent with a contact URL (UA), gzip accepted.
  * >= MIN_GAP (3 s) between requests to one host, across all worker threads.
  * every 200 response is cached under data/raw/notes_cache/ (gitignored),
    keyed by URL, for PAGE_TTL_HOURS; a host that answers 403 / a challenge is
    not asked again in the same run.
  * retries with backoff on 429 / 5xx / network errors only.

Cost bounds:
  * `--max-calls` (default 60) caps LLM calls per run. Units are planned up
    front in priority order (news for every team, then form by nearest
    fixture) and the overflow is deferred to the next run, never dropped.
  * cache first: a unit is skipped when the store already has a fact for it,
    or the last run checked it (manifest `checked`, so an empty news search
    is not repeated every run), less than CACHE_HOURS ago.
  * `--concurrency` (default 3) bounds the worker threads.
  * the pricing constants below are ESTIMATES; the manifest reports tokens
    and web_search calls so the bill can be checked against them.

Exit policy:
  0  run finished (facts may be empty: no fixtures, all cached, nothing found).
  1  more than 50% of attempted units failed (LLM / network error, output
     that is not JSON), counted once at least FAIL_MIN (3) failed, so one
     stubborn unit retried alone does not turn every run red; the good facts
     and the manifest are still written. A failed unit is not marked checked
     and is retried next run.
  1  the bundle is missing or unreadable, an unknown --teams name, or the
     existing store is not JSON (never overwritten).
  1  OPENAI_API_KEY unset in strict mode (--strict, or CI / GITHUB_ACTIONS
     set): a silently dead fetcher is the failure we must not have. The
     manifest records the error. Locally without --strict: a clear message,
     nothing written, exit 0.
  2  the manifest cannot be written.

`--dry-run` uses no network at all: LLM responses and pages come from the
recorded fixtures in tests/fixtures/notes_fetch/ (`--fixtures DIR`). Point
`--out` / `--manifest` at a scratch path so fixture facts never reach data/.

Usage:
  python collectors/notes_fetch.py                       # live, scope from the bundle
  python collectors/notes_fetch.py --teams FURIA,MOUZ --max-calls 10
  python collectors/notes_fetch.py --teams FURIA --refresh   # ignore the 12 h cache-first rule
  python collectors/notes_fetch.py --dry-run --teams FURIA,PARIVISION \\
      --out /tmp/f.json --manifest /tmp/m.json
Stdlib only.
"""
import argparse
import concurrent.futures
import difflib
import gzip
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (puts every code folder on sys.path)

import notes_facts as nf  # noqa: E402

# ---- pricing: ESTIMATES, verify against https://openai.com/api/pricing -----
# (USD; gpt-5-mini list prices as last checked by hand. Cached-input discounts
# are ignored, so the estimate errs high. Web-search result tokens are billed
# as input tokens and are already inside usage.input_tokens.)
PRICE_INPUT_PER_M = 0.25            # per 1M input tokens
PRICE_OUTPUT_PER_M = 2.00           # per 1M output tokens (reasoning included)
PRICE_WEB_SEARCH_PER_CALL = 0.01    # per web_search tool call ($10 / 1k calls)

DEFAULT_MODEL = "gpt-5-mini"
API_URL = "https://api.openai.com/v1/responses"
UA = ("FaceOff-CS2-insights/0.2 (non-commercial fan analytics; "
      "https://github.com/AverageNftEnjoyer/Face-Off; python-urllib; low volume, cached)")

SCOPE_DAYS = 7          # teams with a fixture in [as_of, as_of + 7]
CACHE_HOURS = 12        # a unit fetched / checked this recently is skipped
PAGE_TTL_HOURS = 12     # on-disk page cache lifetime
NEWS_DAYS = 30          # news window asked for (and enforced on the answer)
DEFAULT_MAX_CALLS = 60
DEFAULT_CONCURRENCY = 3
FAIL_RATIO = 0.5        # more than this share of attempted units failing -> exit 1 ...
FAIL_MIN = 3            # ... once at least this many failed (one stubborn unit retried alone is not an outage)
CHECKED_KEEP_HOURS = 48
MIN_GAP = 3.0           # seconds between requests to one host
HTTP_BACKOFF = [5, 15]  # scrape retries (429 / 5xx / network only)
LLM_BACKOFF = [5, 20, 60]
MAX_RETRY_AFTER = 120
MAX_PAGE_CHARS = 24000  # page text handed to the extractor

BUNDLE = os.path.join(paths.VIEWER, "data-bundle.json")
OUT = os.path.join(paths.DATA, "notes_facts.json")
MANIFEST = os.path.join(paths.DATA, "notes_manifest.json")
LOG_NAME = "notes_manifest_log.jsonl"
CACHE_DIR = os.path.join(paths.RAW, "notes_cache")
FIXTURES = os.path.join(paths.ROOT, "tests", "fixtures", "notes_fetch")

HLTV_PLAYER = re.compile(r"^https://www\.hltv\.org/stats/players/(\d+)/([^/?#]+)")


class FetchError(Exception):
    """A unit failed (counted against the 50% rule)."""


# ---- structured-output schemas ---------------------------------------------
def _nullable(t, enum=None):
    s = {"type": [t, "null"]}
    if enum:
        s["enum"] = list(enum) + [None]
    return s


FORM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "found": {"type": "boolean"},
        "team": {"type": "string"},
        "player": {"type": "string"},
        "rating_last30": _nullable("number"),
        "rating_baseline": _nullable("number"),
        "baseline": _nullable("string", nf.BASELINES),
        "maps_last30": _nullable("integer"),
        "rating_version": _nullable("string"),
        "source_url": _nullable("string"),
    },
}
FORM_SCHEMA["required"] = list(FORM_SCHEMA["properties"])

NEWS_ITEM = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "type": {"type": "string", "enum": list(nf.NEWS_TYPES)},
        "team": {"type": "string"},
        "player": {"type": "string"},
        "date": {"type": "string"},
        "source_url": {"type": "string"},
        "replaced": _nullable("string"),
        "until": _nullable("string"),
        "event": _nullable("string"),
        "move": _nullable("string", ("in", "out")),
        "other_team": _nullable("string"),
    },
}
NEWS_ITEM["required"] = list(NEWS_ITEM["properties"])
NEWS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
               "properties": {"items": {"type": "array", "items": NEWS_ITEM}}}

RULES = ("You are a data extractor for a Counter-Strike 2 statistics site. Answer ONLY with JSON "
         "matching the schema; never prose. Report only what a page you actually read states; "
         "never estimate, infer or invent a number, a date or a URL.")


def news_request(team, players, day):
    lo = (day - timedelta(days=NEWS_DAYS)).isoformat()
    return (f"Today is {day.isoformat()}. CS2 team: {team!r}; current players: {', '.join(players) or 'unknown'}.\n"
            f"Search the web for news dated {lo} or later, and report ONLY:\n"
            f"1. stand_in: a player standing in for {team!r} (player = the stand-in, replaced = the regular "
            f"if named, date = first day of the stand-in or the announcement, until = last day if stated, "
            f"event = the event if stated).\n"
            f"2. roster_rumor: a player reported or rumoured to be joining (move=in) or leaving (move=out) "
            f"{team!r}; other_team = the other organisation if named.\n"
            f"team = {team!r} for every item. date = YYYY-MM-DD as the source dates it. source_url = the "
            f"https URL of the article. Leave fields that do not apply null. Nothing found: items = [].")


def form_search_request(team, player, day):
    return (f"Today is {day.isoformat()}. Find the HLTV statistics of CS2 player {player!r} ({team}).\n"
            f"Report: rating_last30 = HLTV Rating over the last 30 days; maps_last30 = maps played in "
            f"that window; rating_baseline = the same rating over the last 12 months (baseline='last12m') "
            f"or the career figure (baseline='career'); rating_version as labelled (e.g. '2.1', '3.0'). "
            f"source_url = the https URL of the stats page the numbers are on (prefer "
            f"https://www.hltv.org/stats/players/<id>/<nick>). team = {team!r}, player = {player!r}. "
            f"If any required number is not on a page you read, set found=false and the numbers null.")


def form_extract_request(team, player, url, text):
    return (f"Extract the statistics of CS2 player {player!r} ({team}) from this page text "
            f"(fetched from {url}). rating_last30 / maps_last30 = the last-30-days figures; "
            f"rating_baseline with baseline='last12m' or 'career' as the page shows. team = {team!r}, "
            f"player = {player!r}, source_url = null. If the page does not show a required number, "
            f"found=false.\n\n--- page text ---\n{text}")


def payload(model, prompt, schema, name, search):
    p = {"model": model, "instructions": RULES, "input": prompt, "store": False,
         "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}}
    if search:
        p["tools"] = [{"type": "web_search", "search_context_size": "low"}]
    if model.startswith(("gpt-5", "o")):
        p["reasoning"] = {"effort": "low"}
    return p


# ---- response parsing ------------------------------------------------------
def parse_response(resp):
    """Responses-API object -> (parsed JSON, usage dict). FetchError when the
    run did not complete, refused, or answered with something that is not JSON."""
    usage = resp.get("usage") or {}
    u = {"input_tokens": int(usage.get("input_tokens") or 0),
         "output_tokens": int(usage.get("output_tokens") or 0),
         "web_search_calls": sum(1 for o in resp.get("output") or [] if o.get("type") == "web_search_call")}
    if resp.get("error"):
        raise FetchError(f"api error: {resp['error']}", u)
    if resp.get("status") not in (None, "completed"):
        why = (resp.get("incomplete_details") or {}).get("reason")
        raise FetchError(f"response {resp.get('status')}" + (f" ({why})" if why else ""), u)
    text, refusal = "", None
    for o in resp.get("output") or []:
        if o.get("type") != "message":
            continue
        for c in o.get("content") or []:
            if c.get("type") == "output_text":
                text += c.get("text") or ""
            elif c.get("type") == "refusal":
                refusal = c.get("refusal") or "refused"
    if refusal and not text:
        raise FetchError(f"model refused: {refusal[:120]}", u)
    try:
        return json.loads(text), u
    except ValueError:
        raise FetchError(f"output is not JSON (prose?): {text.strip()[:80]!r}", u)


_KEYLIKE = re.compile(r"sk-[A-Za-z0-9*_.\-]{4,}")


def redact(text, key=None):
    """Scrub the API key, and anything shaped like an OpenAI key (OpenAI's own
    error messages quote a masked one), from text bound for the manifest, the
    log line, stdout or a GitHub issue: the manifest is committed in public."""
    text = str(text)
    if key:
        text = text.replace(key, "[redacted]")
    return _KEYLIKE.sub("sk-[redacted]", text)


# ---- LLM transports --------------------------------------------------------
class LiveLLM:
    def __init__(self, key, timeout=180):
        self.key, self.timeout = key, timeout

    def __call__(self, unit_key, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(API_URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "User-Agent": UA})
        for backoff in LLM_BACKOFF + [None]:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
                except Exception:
                    pass
                if e.code not in (429, 500, 502, 503, 504) or backoff is None:
                    raise FetchError(redact(f"OpenAI HTTP {e.code}: {detail[:160]}", self.key))
                try:
                    pause = min(MAX_RETRY_AFTER, max(backoff, int(float(e.headers.get("Retry-After", "0")))))
                except ValueError:
                    pause = backoff
                time.sleep(pause)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if backoff is None:
                    raise FetchError(redact(f"OpenAI unreachable: {e}", self.key))
                time.sleep(backoff)
        raise FetchError("OpenAI: retries exhausted")


class RecordedLLM:
    """Dry run: answers from fixtures `llm[unit_key]`; '@D-4@' in a recorded
    text is the run's as_of day minus 4 days (so fixtures never go stale)."""

    def __init__(self, records, day):
        self.records, self.day = records, day

    def __call__(self, unit_key, body):
        rec = self.records.get(unit_key)
        if rec is None:
            raise FetchError(f"dry run: no recorded response for {unit_key}")
        s = re.sub(r"@D([+-]\d+)@", lambda m: (self.day + timedelta(days=int(m.group(1)))).isoformat(),
                   json.dumps(rec))
        return json.loads(s)


# ---- polite HTTP (scrape path) ---------------------------------------------
class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "div", "tr", "li", "br", "h1", "h2", "h3", "h4", "table", "section", "span"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, d):
        if not self.skip:
            self.out.append(d)


def html_to_text(html):
    p = _Text()
    p.feed(html)
    lines = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in "".join(p.out).split("\n"))
    return "\n".join(ln for ln in lines if ln)[:MAX_PAGE_CHARS]


def _robots_rules(text, agent="faceoff-cs2-insights"):
    """Disallow/Allow patterns of the group for `agent` (else '*'), in the
    wildcard ('*', '$') syntax robotparser ignores."""
    groups, cur, in_rules = {}, [], False
    for raw in text.splitlines():
        ln = raw.split("#", 1)[0].strip()
        if ":" not in ln:
            continue
        k, v = (x.strip() for x in ln.split(":", 1))
        k = k.lower()
        if k == "user-agent":
            if in_rules:
                cur, in_rules = [], False
            cur.append(v.lower())
            groups.setdefault(v.lower(), [])
        elif k in ("allow", "disallow"):
            in_rules = True
            for a in cur:
                groups[a].append((k, v))
    return groups.get(agent) or groups.get("*") or []


def _rule_match(pattern, path):
    if not pattern:
        return False
    rx = re.escape(pattern).replace(r"\*", ".*")
    if rx.endswith(r"\$"):
        rx = rx[:-2] + "$"
    return re.match(rx, path) is not None


def robots_allows(text, url):
    """robotparser AND a wildcard-aware longest-match check must both allow."""
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(text.splitlines())
    if not rp.can_fetch(UA, url):
        return False
    u = urllib.parse.urlsplit(url)
    path = (u.path or "/") + (("?" + u.query) if u.query else "")
    best = None
    for kind, pat in _robots_rules(text):
        if _rule_match(pat, path) and (best is None or len(pat) > len(best[1])
                                       or (len(pat) == len(best[1]) and kind == "allow")):
            best = (kind, pat)
    return best is None or best[0] == "allow"


class Http:
    """Polite cached GET. `recorded` (dry run): {url: {"status", "body"}} and
    no network at all."""

    def __init__(self, cache_dir=CACHE_DIR, recorded=None, now=None):
        self.cache_dir, self.recorded = cache_dir, recorded
        self.now = now or datetime.now(timezone.utc)
        self.requests = 0
        self._lock = threading.Lock()
        self._host_locks, self._last, self._robots, self._blocked = {}, {}, {}, {}

    def _host_lock(self, host):
        with self._lock:
            return self._host_locks.setdefault(host, threading.Lock())

    def _cache_path(self, url):
        return os.path.join(self.cache_dir, hashlib.sha1(url.encode()).hexdigest()[:16] + ".json")

    def _cached(self, url):
        if self.recorded is not None:
            return None
        p = self._cache_path(url)
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            if self.now - nf.parse_ts(rec["fetched"]) < timedelta(hours=PAGE_TTL_HOURS):
                return rec["body"]
        except (OSError, ValueError, KeyError):
            pass
        return None

    def _store(self, url, body):
        if self.recorded is not None:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self._cache_path(url), "w", encoding="utf-8") as f:
            json.dump({"url": url, "fetched": nf.fmt_ts(self.now), "body": body}, f, ensure_ascii=False)

    def _send(self, url):
        """-> (status, body). One request, after the per-host gap."""
        if self.recorded is not None:
            rec = self.recorded.get(url)
            if rec is None:
                return None, "dry run: no recorded page"
            with self._lock:
                self.requests += 1
            return rec.get("status", 200), rec.get("body", "")
        host = urllib.parse.urlsplit(url).netloc
        with self._host_lock(host):
            wait = MIN_GAP - (time.time() - self._last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            try:
                with self._lock:
                    self.requests += 1
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    return r.status, raw.decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", "replace")
                except Exception:
                    body = ""
                return e.code, body
            finally:
                self._last[host] = time.time()

    def get(self, url):
        """-> page body, or raises FetchError(reason)."""
        body = self._cached(url)
        if body is not None:
            return body
        host = urllib.parse.urlsplit(url).netloc
        if host in self._blocked:
            raise FetchError(f"{host} refused earlier this run ({self._blocked[host]})")
        for backoff in HTTP_BACKOFF + [None]:
            try:
                status, body = self._send(url)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                status, body = None, f"network error: {e}"
            if status == 200:
                if "Just a moment..." in body[:2000] or "cf-chl" in body[:5000]:
                    self._blocked[host] = "Cloudflare challenge"
                    raise FetchError("Cloudflare challenge (HTTP 200)")
                self._store(url, body)
                return body
            if status in (401, 403) or (body and "Just a moment..." in body[:2000]):
                why = f"HTTP {status}" + (" Cloudflare challenge" if "Just a moment" in (body or "") else "")
                self._blocked[host] = why
                raise FetchError(why)
            if status not in (None, 429, 500, 502, 503, 504) or backoff is None or self.recorded is not None:
                raise FetchError(f"HTTP {status}" if status else str(body))
            time.sleep(backoff)
        raise FetchError("retries exhausted")

    def allowed(self, url):
        """robots.txt check (fetched once per host per run). An unreadable
        robots.txt (403, error) counts as 'disallow'."""
        u = urllib.parse.urlsplit(url)
        host = u.netloc
        with self._host_lock("robots:" + host):
            if host not in self._robots:
                try:
                    self._robots[host] = self.get(f"{u.scheme}://{host}/robots.txt")
                except FetchError as e:
                    self._robots[host] = e
        r = self._robots[host]
        if isinstance(r, FetchError):
            return False, f"robots.txt unreadable ({r})"
        if not robots_allows(r, url):
            return False, f"robots.txt disallows {u.path}{'?' + u.query if u.query else ''}"
        return True, None


# ---- bundle, scope, names --------------------------------------------------
def _norm(s):
    s = re.sub(r"[^a-z0-9]", "", str(s).lower())
    for pre in ("team",):
        if s.startswith(pre) and len(s) > len(pre) + 1:
            s = s[len(pre):]
    for suf in ("esports", "esport", "gaming", "clan", "team"):
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)]
    return s


class TeamNames:
    """LLM / HLTV team names -> exact bundle names (None when unknown or
    ambiguous)."""

    def __init__(self, teams, aliases=None):
        self.teams = set(teams)
        idx = {}
        for t in teams:
            for n in [t] + list((aliases or {}).get(t) or []):
                for k in {n.lower(), _norm(n)}:
                    idx.setdefault(k, set()).add(t)
        self.idx = {k: v.pop() for k, v in idx.items() if len(v) == 1}

    def __call__(self, name):
        if not isinstance(name, str) or not name.strip():
            return None
        if name in self.teams:
            return name
        return self.idx.get(name.strip().lower()) or self.idx.get(_norm(name))


def load_aliases():
    try:
        with open(os.path.join(paths.DATA, "team_aliases.json"), encoding="utf-8") as f:
            return json.load(f).get("aliases") or {}
    except (OSError, ValueError):
        return {}


def scope(bundle, as_of, only=None):
    """[(team, nearest fixture day, [players])] sorted by (day, team)."""
    teams = bundle["data"]["teams"]
    hi = as_of + timedelta(days=SCOPE_DAYS)
    near = {}
    for fx in bundle.get("fixtures") or []:
        try:
            d = date.fromisoformat(fx["day"])
        except (KeyError, TypeError, ValueError):
            continue
        if as_of <= d <= hi:
            for t in (fx.get("team_a"), fx.get("team_b")):
                if t in teams:
                    near[t] = min(near.get(t, d), d)
    if only is not None:
        near = {t: near.get(t, as_of) for t in only}
    out = []
    for t, d in sorted(near.items(), key=lambda kv: (kv[1], kv[0])):
        roster = [r.get("id") for r in teams[t].get("roster") or [] if not r.get("coach") and r.get("id")]
        out.append((t, d, roster))
    return out


# ---- the run ---------------------------------------------------------------
def _domain_label(url):
    host = urllib.parse.urlsplit(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return {"hltv.org": "HLTV", "liquipedia.net": "Liquipedia"}.get(host, host)


def _newest(facts):
    """{key: newest observed_at} for form (team, player) and news (team)."""
    form, news = {}, {}
    for f in facts[0]:
        k = (f["team"], f["player"])
        form[k] = max(form.get(k, ""), f["observed_at"])
    for f in facts[1]:
        news[f["team"]] = max(news.get(f["team"], ""), f["observed_at"])
    return form, news


def _hltv_url(form_facts, team, player):
    best = None
    for f in form_facts:
        if f["team"] == team and f["player"].lower() == player.lower() and HLTV_PLAYER.match(f["source_url"]):
            if best is None or f["observed_at"] > best["observed_at"]:
                best = f
    m = HLTV_PLAYER.match(best["source_url"]) if best else None
    return f"https://www.hltv.org/stats/players/{m.group(1)}/{m.group(2)}" if m else None


def form_facts_from(out, unit, ctx, method, source_url):
    """LLM form answer -> ([fact], [rejection]) (empty, empty when found=false)."""
    if not isinstance(out, dict):
        return [], [{"reasons": ["answer is not an object"]}]
    if not out.get("found"):
        return [], []
    reasons = []
    team = ctx["names"](out.get("team"))
    if team is None:
        reasons.append(f"unknown team {out.get('team')!r}")
    elif team != unit["team"]:
        reasons.append(f"team {team!r} is not the requested {unit['team']!r}")
    if str(out.get("player") or "").lower() != unit["player"].lower():
        reasons.append(f"player {out.get('player')!r} is not the requested {unit['player']!r}")
    url = source_url or out.get("source_url")
    if not url:
        reasons.append("missing source_url")
    if reasons:
        return [], [{"reasons": reasons, "raw": out}]
    f = nf.make_fact(team=team, player=unit["player"], observed_at=ctx["observed_at"], source_url=url,
                     source=_domain_label(url), method=method,
                     rating_last30=out.get("rating_last30"), rating_baseline=out.get("rating_baseline"),
                     baseline=out.get("baseline"), maps_last30=out.get("maps_last30"),
                     rating_version=out.get("rating_version"))
    errs = nf.validate_fact(f, ctx["teams"])
    return ([], [{"reasons": errs, "raw": out}]) if errs else ([f], [])


def news_facts_from(out, unit, ctx):
    items = out.get("items") if isinstance(out, dict) else None
    if not isinstance(items, list):
        return [], [{"reasons": ["answer has no items list"]}]
    facts, rej = [], []
    lo = ctx["as_of"] - timedelta(days=NEWS_DAYS)
    hi = ctx["as_of"] + timedelta(days=NEWS_DAYS)
    for it in items:
        if not isinstance(it, dict):
            rej.append({"reasons": ["item is not an object"]})
            continue
        reasons = []
        team = ctx["names"](it.get("team"))
        if team is None:
            reasons.append(f"unknown team {it.get('team')!r}")
        if not it.get("source_url"):
            reasons.append("missing source_url")
        try:
            d = date.fromisoformat(str(it.get("date")))
            if not lo <= d <= (hi if it.get("type") == "stand_in" else ctx["as_of"]):
                reasons.append(f"date {d} outside the last {NEWS_DAYS} days")
            if it.get("until") and date.fromisoformat(str(it["until"])) < d:
                reasons.append("until before date")
        except ValueError:
            reasons.append(f"date {it.get('date')!r} not YYYY-MM-DD")
        if reasons:
            rej.append({"reasons": reasons, "raw": it})
            continue
        other = it.get("other_team")
        f = nf.make_fact(type=it.get("type"), team=team, player=it.get("player"),
                         observed_at=ctx["observed_at"], source_url=it["source_url"],
                         source=_domain_label(it["source_url"]), method="web_search", date=it.get("date"),
                         replaced=it.get("replaced"), until=it.get("until"), event=it.get("event"),
                         move=it.get("move"), other_team=(ctx["names"](other) or other) if other else None)
        errs = nf.validate_fact(f, ctx["teams"])
        (rej.append({"reasons": errs, "raw": it}) if errs else facts.append(f))
    return facts, rej


def run_unit(unit, ctx):
    """One unit -> result dict. Raises nothing: failures are recorded."""
    res = {"unit": unit, "facts": [], "rejected": [], "error": None, "usage": None, "scrape": None,
           "method": None, "called": False}
    try:
        if unit["kind"] == "news":
            res["method"] = "web_search"
            body = payload(ctx["model"], news_request(unit["team"], unit["players"], ctx["as_of"]),
                           NEWS_SCHEMA, "roster_news", True)
            out = _call(ctx, unit["key"], body, res)
            res["facts"], res["rejected"] = news_facts_from(out, unit, ctx)
            return res
        text, page = None, None
        if unit.get("hltv"):
            lo = (ctx["as_of"] - timedelta(days=30)).isoformat()
            page = f"{unit['hltv']}?startDate={lo}&endDate={ctx['as_of'].isoformat()}"
            ok, why = ctx["http"].allowed(page)
            try:
                if not ok:
                    raise FetchError(why)
                text = html_to_text(ctx["http"].get(page))
                res["scrape"] = "ok"
            except FetchError as e:
                res["scrape"] = str(e)
        if text:
            res["method"] = "scrape"
            body = payload(ctx["model"], form_extract_request(unit["team"], unit["player"], page, text),
                           FORM_SCHEMA, "player_form", False)
            out = _call(ctx, unit["key"].replace("form|", "form_scrape|", 1), body, res)
            res["facts"], res["rejected"] = form_facts_from(out, unit, ctx, "scrape", page)
        else:
            res["method"] = "web_search"
            body = payload(ctx["model"], form_search_request(unit["team"], unit["player"], ctx["as_of"]),
                           FORM_SCHEMA, "player_form", True)
            out = _call(ctx, unit["key"], body, res)
            res["facts"], res["rejected"] = form_facts_from(out, unit, ctx, "web_search", None)
    except FetchError as e:
        res["error"] = str(e)
    except Exception as e:      # per-unit isolation: one failure never stops the others
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def _call(ctx, key, body, res):
    res["called"] = True
    resp = ctx["llm"](key, body)
    try:
        out, usage = parse_response(resp)
    except FetchError as e:
        res["usage"] = e.args[1] if len(e.args) > 1 else None
        raise FetchError(e.args[0])
    res["usage"] = usage
    return out


def estimate_cost(input_tokens, output_tokens, web_search_calls):
    return round(input_tokens / 1e6 * PRICE_INPUT_PER_M + output_tokens / 1e6 * PRICE_OUTPUT_PER_M
                 + web_search_calls * PRICE_WEB_SEARCH_PER_CALL, 4)


def dumps(obj):
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def _write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--teams", help="comma-separated bundle team names (overrides the 7-day scope)")
    ap.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS, help="LLM calls per run (default %(default)s)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="worker threads (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="no network: recorded fixtures only")
    ap.add_argument("--fixtures", default=FIXTURES, help="dry-run fixture folder")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--log", help=f"run history (default: {LOG_NAME} beside --manifest)")
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--as-of", help="scope start day YYYY-MM-DD (default: the run's UTC date)")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache-first rule (re-fetch every unit)")
    ap.add_argument("--strict", action="store_true", help="fail on a missing API key (implied by CI)")
    return ap.parse_args(argv)


def main(argv=None, now=None, llm=None, http=None, env=None):
    a = parse_args(argv)
    env = os.environ if env is None else env
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    observed_at = nf.fmt_ts(now)
    as_of = date.fromisoformat(a.as_of) if a.as_of else now.date()
    strict = a.strict or bool(env.get("CI") or env.get("GITHUB_ACTIONS"))
    model = env.get("FACEOFF_NOTES_MODEL") or DEFAULT_MODEL
    log_path = a.log or os.path.join(os.path.dirname(os.path.abspath(a.manifest)), LOG_NAME)
    mode = "dry-run" if a.dry_run else "live"

    try:
        with open(a.bundle, encoding="utf-8") as f:
            bundle = json.load(f)
        teams = bundle["data"]["teams"]
    except (OSError, ValueError, KeyError) as e:
        print(f"notes_fetch: cannot read the bundle {a.bundle} ({e}). Build it first: "
              f"python viewer/build_bundle.py viewer/data-bundle.json", file=sys.stderr)
        return 1
    names = TeamNames(teams, load_aliases())

    only = None
    if a.teams:
        only, bad = [], []
        for t in (x.strip() for x in a.teams.split(",") if x.strip()):
            (only.append(names(t)) if names(t) else bad.append(t))
        if bad:
            for t in bad:
                print(f"notes_fetch: no team named {t!r} in the bundle; close: "
                      f"{difflib.get_close_matches(t, list(teams), 5)}", file=sys.stderr)
            return 1
    sc = scope(bundle, as_of, only)

    manifest = {"run_at": observed_at, "mode": mode, "model": model, "as_of": as_of.isoformat(),
                "scope_teams": [t for t, _, _ in sc], "error": None, "warnings": [],
                "pricing_estimate": {"note": "ESTIMATES: verify against https://openai.com/api/pricing",
                                     "input_per_m_usd": PRICE_INPUT_PER_M, "output_per_m_usd": PRICE_OUTPUT_PER_M,
                                     "web_search_per_call_usd": PRICE_WEB_SEARCH_PER_CALL}}

    def finish(code):
        manifest["exit_code"] = code
        try:
            _write(a.manifest, redact(dumps(manifest), env.get("OPENAI_API_KEY")))
            summary = {k: manifest.get(k) for k in (
                "run_at", "mode", "model", "as_of", "error", "exit_code", "llm_calls", "http_requests",
                "input_tokens", "output_tokens", "web_search_calls", "est_cost_usd", "cache_hits",
                "cache_hit_rate", "units_attempted", "units_failed", "units_deferred", "facts_written",
                "store_changed")}
            summary["scope_teams"] = len(manifest["scope_teams"])
            summary["teams_attempted"] = len(manifest.get("teams_attempted") or [])
            summary["players_attempted"] = manifest.get("players_attempted")
            summary["facts_rejected"] = len(manifest.get("facts_rejected") or [])
            summary["fail_reasons"] = sorted({x["reason"] for x in manifest.get("fetches_failed") or []})[:10]
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(redact(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                               env.get("OPENAI_API_KEY")) + "\n")
        except OSError as e:
            print(f"notes_fetch: cannot write the manifest {a.manifest}: {e}", file=sys.stderr)
            return 2
        return code

    key = env.get("OPENAI_API_KEY")
    if not a.dry_run and llm is None and not key:
        msg = "OPENAI_API_KEY is not set: no facts can be fetched"
        if strict:
            manifest["error"] = msg
            print(f"notes_fetch: ERROR {msg} (strict / CI mode)", file=sys.stderr)
            return finish(1)
        print(f"notes_fetch: {msg}. Set it, or use --dry-run. Nothing written.", file=sys.stderr)
        return 0

    try:
        store = _read_json(a.out)
    except ValueError as e:
        manifest["error"] = f"existing store {a.out} is not JSON ({e}); not overwritten"
        print(f"notes_fetch: ERROR {manifest['error']}", file=sys.stderr)
        return finish(1)
    form_old, news_old, dropped = nf.clean(store)          # structural check only: history of any team is kept
    if dropped:
        manifest["warnings"].append(f"{len(dropped)} malformed stored facts dropped")
    newest_form, newest_news = _newest((form_old, news_old))

    try:
        prev = _read_json(a.manifest) or {}
    except (OSError, ValueError):
        prev = {}
    keep_lo = nf.fmt_ts(now - timedelta(hours=CHECKED_KEEP_HOURS))
    checked = {k: v for k, v in (prev.get("checked") or {}).items() if isinstance(v, str) and keep_lo <= v <= observed_at}
    fresh_lo = nf.fmt_ts(now - timedelta(hours=CACHE_HOURS))

    def fresh(key, newest):
        if a.refresh:
            return False
        return max(newest or "", checked.get(key, "")) > fresh_lo

    if a.dry_run:
        try:
            fx = _read_json(os.path.join(a.fixtures, "responses.json")) or {}
        except ValueError as e:
            manifest["error"] = f"fixtures unreadable: {e}"
            return finish(1)
        llm = llm or RecordedLLM(fx.get("llm") or {}, as_of)
        http = http or Http(recorded=fx.get("http") or {}, now=now)
    else:
        llm = llm or LiveLLM(key)
        http = http or Http(now=now)

    # plan: news for every team first (stand-ins matter most), then form by nearest fixture
    units, hits = [], 0
    for t, d, players in sc:
        k = f"news|{t}"
        if fresh(k, newest_news.get(t)):
            hits += 1
        else:
            units.append({"kind": "news", "team": t, "day": d.isoformat(), "players": players, "key": k})
    for t, d, players in sc:
        for p in players:
            k = f"form|{t}|{p}"
            if fresh(k, newest_form.get((t, p))):
                hits += 1
            else:
                units.append({"kind": "form", "team": t, "player": p, "day": d.isoformat(), "key": k,
                              "hltv": _hltv_url(form_old, t, p)})
    cap = max(0, a.max_calls)
    todo, deferred = units[:cap], units[cap:]
    if deferred:
        manifest["warnings"].append(f"--max-calls {cap}: {len(deferred)} units deferred to the next run")

    ctx = {"model": model, "as_of": as_of, "observed_at": observed_at, "teams": teams, "names": names,
           "llm": llm, "http": http}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        results = list(ex.map(lambda u: run_unit(u, ctx), todo))   # map keeps the plan order

    new_facts, rejected, failed = [], [], []
    tok_in = tok_out = searches = calls = 0
    scrape = {"tried": 0, "ok": 0, "fell_back": 0, "reasons": {}}
    for r in results:
        u = r["unit"]
        tag = {"kind": u["kind"], "team": u["team"], **({"player": u["player"]} if u["kind"] == "form" else {})}
        calls += r["called"]
        if r["usage"]:
            tok_in += r["usage"]["input_tokens"]
            tok_out += r["usage"]["output_tokens"]
            searches += r["usage"]["web_search_calls"]
        if r["scrape"] is not None:
            scrape["tried"] += 1
            if r["scrape"] == "ok":
                scrape["ok"] += 1
            else:
                scrape["fell_back"] += 1
                scrape["reasons"][r["scrape"]] = scrape["reasons"].get(r["scrape"], 0) + 1
        if r["error"]:
            failed.append({**tag, "reason": r["error"]})
        else:
            checked[u["key"]] = observed_at
        new_facts += r["facts"]
        for x in r["rejected"]:
            rejected.append({**tag, "method": r["method"], "reasons": x["reasons"],
                             **({"raw": x["raw"]} if "raw" in x else {})})

    old_ids = {f["id"] for f in form_old + news_old}
    added = [f for f in new_facts if f["id"] not in old_ids]
    merged = nf.merge({"form": form_old, "news": news_old, "updated_at": (store or {}).get("updated_at")},
                      new_facts, now)
    pruned = len(form_old) + len(news_old) + len(added) - len(merged["form"]) - len(merged["news"])
    text = dumps(merged)
    old_text = None
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            old_text = f.read()
    store_changed = False
    if (added or pruned or dropped) and text != old_text:
        _write(a.out, text)
        store_changed = True

    attempted = len(todo)
    n_failed = len(failed)
    denom = hits + len(units)
    manifest.update({
        "teams_attempted": sorted({u["team"] for u in todo}),
        "players_attempted": sum(1 for u in todo if u["kind"] == "form"),
        "units_planned": len(units), "units_attempted": attempted, "units_failed": n_failed,
        "units_deferred": len(deferred), "deferred": [u["key"] for u in deferred],
        "facts_written": len(added), "facts_written_ids": sorted(f["id"] for f in added),
        "facts_pruned": pruned, "facts_rejected": rejected, "fetches_failed": failed,
        "cache_hits": hits, "cache_hit_rate": round(hits / denom, 3) if denom else None,
        "scrape": scrape, "http_requests": http.requests, "llm_calls": calls,
        "input_tokens": tok_in, "output_tokens": tok_out, "web_search_calls": searches,
        "est_cost_usd": estimate_cost(tok_in, tok_out, searches),
        "store": a.out if a.out != OUT else "data/notes_facts.json", "store_changed": store_changed,
        "checked": dict(sorted(checked.items())),
    })
    if scrape["tried"] and not scrape["ok"]:
        manifest["warnings"].append("scrape path unavailable this run (see scrape.reasons); form used web_search")
    code = 0
    if attempted and n_failed >= FAIL_MIN and n_failed / attempted > FAIL_RATIO:
        manifest["error"] = f"{n_failed} of {attempted} units failed (> {int(FAIL_RATIO * 100)}%)"
        print(f"notes_fetch: ERROR {manifest['error']}", file=sys.stderr)
        code = 1
    rc = finish(code)
    print(f"notes_fetch [{mode}]: scope {len(sc)} teams, {attempted} units ({n_failed} failed, "
          f"{len(deferred)} deferred), {hits} cache hits, {len(added)} new facts, {len(rejected)} rejected, "
          f"{calls} LLM calls, {tok_in}+{tok_out} tokens, {searches} searches, ~${manifest['est_cost_usd']}"
          f"{'' if store_changed else ' (store unchanged)'}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
