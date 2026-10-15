#!/usr/bin/env python3
"""notes_fetch.py -- web facts for the matchup notes (fact schema v1,
docs/notes-contracts.md section 1) -> data/notes_facts.json.

Facts are display context only: nothing written here is read by the
predictor, the backtest or any weight fit.

ONE OpenAI call per team. Scope: the teams with a fixture in the next
SCOPE_DAYS (3) days (viewer/data-bundle.json, `fixtures[].day` in [as_of,
as_of + 3]); `--teams A,B` overrides it. Per team a single Responses API call
(POST /v1/responses, key from OPENAI_API_KEY, model from FACEOFF_NOTES_MODEL,
default DEFAULT_MODEL) with the `web_search` tool returns, in one strict
json_schema output, BOTH
  players  the HLTV form of every current roster player (rating over the last
           30 days, maps in that window, a 12-month or career baseline, the
           stats page URL), and
  news     stand-ins and roster rumours of the last 30 days, each with an
           https source_url.
Never prose. (We do not fetch HLTV ourselves: its robots.txt disallows the
dated stats pages and plain HTTP gets a Cloudflare challenge.)

The answer is mapped onto exact bundle names (a team the bundle does not
know, or a player not on the team's roster, is rejected), stamped with this
run's one `observed_at`, built with notes_facts.make_fact and checked with
notes_facts.validate_fact. A rejected fact is logged in the manifest with its
reasons and never written. Accepted facts are appended with notes_facts.merge
(history kept, 400-day prune); the file is rewritten only when it changes.

Cost bounds (all four apply on every run):
  * one call per team, at most once per TEAM_TTL_HOURS (24) per team: the
    manifest's `checked` ledger (and the store's newest fact) says when a team
    was last fetched, so an empty answer is not re-asked every run.
  * `--max-calls` (default 10) per run; teams playing soonest go first, the
    rest are `deferred`, never dropped.
  * a HARD MONTHLY BUDGET, env FACEOFF_NOTES_BUDGET_USD (default 1.00):
    before each call, this calendar month's est_cost_usd from
    notes_manifest_log.jsonl (lines of the same mode: dry runs never spend the
    live budget) + this run's spend so far + calls in flight + the worst case
    of the next call (worst_call_usd) must stay within it, else the call is
    not made and the team is deferred with "monthly budget reached". That is
    not a failure (exit 0); the manifest records `budget`.
  * per call: web_search with search_context_size "low", max_tool_calls
    MAX_TOOL_CALLS (2), max_output_tokens MAX_OUTPUT_TOKENS, reasoning effort
    low. If the API rejects one of those optional parameters (HTTP 400 naming
    it) the call is retried once without it (a 400 is not billed).

Expected spend, from the PRICE_* constants below (ESTIMATES: verify against
https://openai.com/api/pricing) and the fixtures' usage numbers (about 11k
input tokens incl. search results, 2.2k output tokens incl. reasoning, 2
searches):
  per call   0.0006 (input) + 0.0009 (output) + 0.02 (2 searches) ~= $0.021
             (one search: ~$0.011). The per-search fee is ~93% of the bill.
  per day    5-8 teams in a 3-day scope, once a day: ~$0.11-0.17
  per month  ~$3.2-5.2 uncapped; the default $1.00 budget therefore binds
             after roughly 6-9 days of the month, after which calls stop
             (exit 0, "monthly budget reached") until the 1st. Fewer calls
             (a longer TTL, e.g. --ttl-hours 72: each team about once per
             3-day window, ~$1.5/month) or a larger budget are the levers.

Exit policy:
  0  run finished (facts may be empty: no fixtures, all cached, nothing
     found, budget reached).
  1  more than 50% of attempted teams failed (LLM / network error, output
     that is not JSON) and at least min(FAIL_MIN, attempted) failed; the good
     facts and the manifest are still written. A failed team is not marked
     checked and is retried next run.
  1  the bundle is missing or unreadable, an unknown --teams name, a
     FACEOFF_NOTES_BUDGET_USD that is not a number, or the existing store is
     not JSON (never overwritten).
  1  OPENAI_API_KEY unset in strict mode (--strict, or CI / GITHUB_ACTIONS
     set): a silently dead fetcher is the failure we must not have. Locally
     without --strict: a clear message, nothing written, exit 0.
  2  the manifest cannot be written.

`--dry-run` uses no network at all: LLM responses come from the recorded
fixtures in tests/fixtures/notes_fetch/ (`--fixtures DIR`). Point `--out` /
`--manifest` at a scratch path so fixture facts never reach data/.

Usage:
  python collectors/notes_fetch.py                       # live, scope from the bundle
  python collectors/notes_fetch.py --teams FURIA,MOUZ --max-calls 2
  python collectors/notes_fetch.py --teams FURIA --refresh   # ignore the 24 h rule
  python collectors/notes_fetch.py --dry-run --teams FURIA,PARIVISION \\
      --out /tmp/f.json --manifest /tmp/m.json
Stdlib only.
"""
import argparse
import concurrent.futures
import difflib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (puts every code folder on sys.path)

import notes_facts as nf  # noqa: E402

# ---- pricing: ESTIMATES, verify against https://openai.com/api/pricing -----
# (USD, gpt-5-nano list prices as last known by hand. Cached-input discounts
# are ignored, so the estimate errs high. Web-search result tokens are billed
# as input tokens and are already inside usage.input_tokens; the per-call
# search fee is separate.)
PRICE_INPUT_PER_M = 0.05            # per 1M input tokens
PRICE_OUTPUT_PER_M = 0.40           # per 1M output tokens (reasoning included)
PRICE_WEB_SEARCH_PER_CALL = 0.01    # per web_search tool call ($10 / 1k calls)

DEFAULT_MODEL = "gpt-5-nano"
API_URL = "https://api.openai.com/v1/responses"
UA = ("FaceOff-CS2-insights/0.3 (non-commercial fan analytics; "
      "https://github.com/AverageNftEnjoyer/Face-Off; python-urllib; low volume)")

SCOPE_DAYS = 3          # teams with a fixture in [as_of, as_of + 3]
TEAM_TTL_HOURS = 24     # a team fetched / checked this recently is skipped
NEWS_DAYS = 30          # news window asked for (and enforced on the answer)
DEFAULT_MAX_CALLS = 10
DEFAULT_CONCURRENCY = 2
DEFAULT_BUDGET_USD = 1.00
MAX_TOOL_CALLS = 2      # web searches per call
MAX_OUTPUT_TOKENS = 6000
WORST_INPUT_TOKENS = 40000   # budget guard's upper bound on one call's input (prompt + search results)
FAIL_RATIO = 0.5        # more than this share of attempted teams failing -> exit 1 ...
FAIL_MIN = 3            # ... once at least min(FAIL_MIN, attempted) failed
CHECKED_KEEP_HOURS = 24 * 8
LLM_BACKOFF = [5, 20, 60]
MAX_RETRY_AFTER = 120
OPTIONAL_PARAMS = ("search_context_size", "max_tool_calls", "reasoning")
BUDGET_REASON = "monthly budget reached"

BUNDLE = os.path.join(paths.VIEWER, "data-bundle.json")
OUT = os.path.join(paths.DATA, "notes_facts.json")
MANIFEST = os.path.join(paths.DATA, "notes_manifest.json")
LOG_NAME = "notes_manifest_log.jsonl"
FIXTURES = os.path.join(paths.ROOT, "tests", "fixtures", "notes_fetch")


class FetchError(Exception):
    """A team's call failed (counted against the 50% rule)."""


# ---- structured-output schema (one per team) -------------------------------
def _nullable(t, enum=None):
    s = {"type": [t, "null"]}
    if enum:
        s["enum"] = list(enum) + [None]
    return s


def _strict(props):
    return {"type": "object", "additionalProperties": False, "properties": props, "required": list(props)}


PLAYER_ITEM = _strict({
    "player": {"type": "string"},
    "found": {"type": "boolean"},
    "rating_last30": _nullable("number"),
    "rating_baseline": _nullable("number"),
    "baseline": _nullable("string", nf.BASELINES),
    "maps_last30": _nullable("integer"),
    "rating_version": _nullable("string"),
    "source_url": _nullable("string"),
})
NEWS_ITEM = _strict({
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
})
TEAM_SCHEMA = _strict({"players": {"type": "array", "items": PLAYER_ITEM},
                       "news": {"type": "array", "items": NEWS_ITEM}})

RULES = ("You are a data extractor for a Counter-Strike 2 statistics site. Answer ONLY with JSON "
         "matching the schema; never prose. Report only what a page you actually read states; "
         "never estimate, infer or invent a number, a date or a URL.")


def team_request(team, players, day):
    lo = (day - timedelta(days=NEWS_DAYS)).isoformat()
    return (f"Today is {day.isoformat()}. CS2 team: {team!r}; current players: {', '.join(players) or 'unknown'}.\n"
            f"Use at most {MAX_TOOL_CALLS} web searches (HLTV first: the team's page and its players' stats).\n"
            f"players: one entry per current player listed above, player = that exact nickname. "
            f"rating_last30 = HLTV Rating over the last 30 days; maps_last30 = maps played in that window; "
            f"rating_baseline = the same rating over the last 12 months (baseline='last12m') or the career "
            f"figure (baseline='career'); rating_version as labelled (e.g. '2.1', '3.0'); source_url = the "
            f"https URL of the stats page the numbers are on (prefer https://www.hltv.org/stats/players/<id>/<nick>). "
            f"If a required number is not on a page you read: found=false and the numbers null.\n"
            f"news: ONLY items dated {lo} or later:\n"
            f"1. stand_in: a player standing in for {team!r} (player = the stand-in, replaced = the regular if "
            f"named, date = first day of the stand-in or the announcement, until = last day if stated, event = "
            f"the event if stated).\n"
            f"2. roster_rumor: a player reported or rumoured to be joining (move=in) or leaving (move=out) "
            f"{team!r}; other_team = the other organisation if named.\n"
            f"team = {team!r} for every news item. date = YYYY-MM-DD as the source dates it. source_url = the "
            f"https URL of the article. Fields that do not apply: null. Nothing found: news = [].")


def payload(model, prompt, schema=TEAM_SCHEMA, name="team_facts"):
    p = {"model": model, "instructions": RULES, "input": prompt, "store": False,
         "max_output_tokens": MAX_OUTPUT_TOKENS, "max_tool_calls": MAX_TOOL_CALLS,
         "tools": [{"type": "web_search", "search_context_size": "low"}],
         "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}}
    if model.startswith(("gpt-5", "o")):
        p["reasoning"] = {"effort": "low"}
    return p


def _drop_param(body, name):
    """A copy of the request without the optional parameter `name`."""
    body = json.loads(json.dumps(body))
    if name == "search_context_size":
        for t in body.get("tools") or []:
            t.pop("search_context_size", None)
    else:
        body.pop(name, None)
    return body


# ---- response parsing ------------------------------------------------------
def parse_response(resp):
    """Responses-API object -> (parsed JSON, usage dict). FetchError(msg,
    usage) when the run did not complete, refused, or answered with something
    that is not JSON."""
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
        dropped = set()
        backoffs = iter(LLM_BACKOFF + [None])
        while True:
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(API_URL, data=data, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
                except Exception:
                    pass
                if e.code == 400:
                    bad = next((p for p in OPTIONAL_PARAMS if p in detail and p not in dropped), None)
                    if bad:      # the model does not take this optional knob: ask again without it
                        dropped.add(bad)
                        body = _drop_param(body, bad)
                        continue
                backoff = next(backoffs)
                if e.code not in (429, 500, 502, 503, 504) or backoff is None:
                    raise FetchError(redact(f"OpenAI HTTP {e.code}: {detail[:160]}", self.key))
                try:
                    pause = min(MAX_RETRY_AFTER, max(backoff, int(float(e.headers.get("Retry-After", "0")))))
                except ValueError:
                    pause = backoff
                time.sleep(pause)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                backoff = next(backoffs)
                if backoff is None:
                    raise FetchError(redact(f"OpenAI unreachable: {e}", self.key))
                time.sleep(backoff)


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


# ---- bundle, scope, names --------------------------------------------------
def _norm(s):
    s = re.sub(r"[^a-z0-9]", "", str(s).lower())
    if s.startswith("team") and len(s) > 5:
        s = s[4:]
    for suf in ("esports", "esport", "gaming", "clan", "team"):
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)]
    return s


class TeamNames:
    """LLM team names -> exact bundle names (None when unknown or ambiguous)."""

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


def scope(bundle, as_of, days=SCOPE_DAYS, only=None):
    """[(team, nearest fixture day, [players])] sorted by (day, team)."""
    teams = bundle["data"]["teams"]
    hi = as_of + timedelta(days=days)
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


# ---- answer -> facts -------------------------------------------------------
def _domain_label(url):
    host = urllib.parse.urlsplit(url).netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return {"hltv.org": "HLTV", "liquipedia.net": "Liquipedia"}.get(host, host)


def form_facts_from(players, unit, ctx):
    """players[] of one answer -> (facts, rejections)."""
    if not isinstance(players, list):
        return [], [{"kind": "form", "reasons": ["answer has no players list"]}]
    roster = {p.lower(): p for p in unit["players"]}
    facts, rej, seen = [], [], set()
    for it in players:
        if not isinstance(it, dict):
            rej.append({"kind": "form", "reasons": ["player entry is not an object"]})
            continue
        if not it.get("found"):
            continue
        nick = roster.get(str(it.get("player") or "").lower())
        reasons = []
        if nick is None:
            reasons.append(f"player {it.get('player')!r} is not on the {unit['team']} roster")
        elif nick in seen:
            reasons.append(f"player {nick!r} listed twice")
        if not it.get("source_url"):
            reasons.append("missing source_url")
        if reasons:
            rej.append({"kind": "form", "player": it.get("player"), "reasons": reasons, "raw": it})
            continue
        seen.add(nick)
        f = nf.make_fact(team=unit["team"], player=nick, observed_at=ctx["observed_at"],
                         source_url=it["source_url"], source=_domain_label(it["source_url"]), method="web_search",
                         rating_last30=it.get("rating_last30"), rating_baseline=it.get("rating_baseline"),
                         baseline=it.get("baseline"), maps_last30=it.get("maps_last30"),
                         rating_version=it.get("rating_version"))
        errs = nf.validate_fact(f, ctx["teams"])
        if errs:
            rej.append({"kind": "form", "player": nick, "reasons": errs, "raw": it})
        else:
            facts.append(f)
    return facts, rej


def news_facts_from(items, unit, ctx):
    """news[] of one answer -> (facts, rejections)."""
    if not isinstance(items, list):
        return [], [{"kind": "news", "reasons": ["answer has no news list"]}]
    facts, rej = [], []
    lo = ctx["as_of"] - timedelta(days=NEWS_DAYS)
    hi = ctx["as_of"] + timedelta(days=NEWS_DAYS)
    for it in items:
        if not isinstance(it, dict):
            rej.append({"kind": "news", "reasons": ["item is not an object"]})
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
            reasons.append(f"date {it.get('date')!r} / until {it.get('until')!r} not a real YYYY-MM-DD")
        if reasons:
            rej.append({"kind": "news", "player": it.get("player"), "reasons": reasons, "raw": it})
            continue
        other = it.get("other_team")
        f = nf.make_fact(type=it.get("type"), team=team, player=it.get("player"),
                         observed_at=ctx["observed_at"], source_url=it["source_url"],
                         source=_domain_label(it["source_url"]), method="web_search", date=it.get("date"),
                         replaced=it.get("replaced"), until=it.get("until"), event=it.get("event"),
                         move=it.get("move"), other_team=(ctx["names"](other) or other) if other else None)
        errs = nf.validate_fact(f, ctx["teams"])
        if errs:
            rej.append({"kind": "news", "player": it.get("player"), "reasons": errs, "raw": it})
        else:
            facts.append(f)
    return facts, rej


# ---- budget ----------------------------------------------------------------
def estimate_cost(input_tokens, output_tokens, web_search_calls):
    return round(input_tokens / 1e6 * PRICE_INPUT_PER_M + output_tokens / 1e6 * PRICE_OUTPUT_PER_M
                 + web_search_calls * PRICE_WEB_SEARCH_PER_CALL, 4)


def worst_call_usd():
    """Upper bound of one call: the guard reserves this before calling."""
    return estimate_cost(WORST_INPUT_TOKENS, MAX_OUTPUT_TOKENS, MAX_TOOL_CALLS)


def month_spent(log_path, month, mode):
    """Sum of est_cost_usd over the log lines of calendar month `month`
    ('YYYY-MM', UTC) written by runs of the same `mode`."""
    total = 0.0
    try:
        with open(log_path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    if str(r.get("run_at", "")).startswith(month) and r.get("mode") == mode:
                        total += float(r.get("est_cost_usd") or 0)
                except (ValueError, TypeError, AttributeError):
                    continue
    except OSError:
        pass
    return round(total, 4)


class Budget:
    """Thread-safe monthly guard: reserve() before a call, settle() after."""

    def __init__(self, limit, spent_before):
        self.limit, self.before = limit, spent_before
        self.run_spent, self.inflight = 0.0, 0.0
        self.stopped = False
        self._lock = threading.Lock()

    def reserve(self):
        w = worst_call_usd()
        with self._lock:
            if self.before + self.run_spent + self.inflight + w > self.limit + 1e-9:
                self.stopped = True
                return None
            self.inflight += w
            return w

    def settle(self, reserved, actual):
        with self._lock:
            self.inflight -= reserved
            self.run_spent += actual

    def report(self):
        spent = round(self.before + self.run_spent, 4)
        return {"limit_usd": self.limit, "spent_month_usd": spent, "spent_before_run_usd": self.before,
                "remaining_usd": round(max(0.0, self.limit - spent), 4), "worst_call_usd": worst_call_usd(),
                "stopped": self.stopped}


# ---- the run ---------------------------------------------------------------
def run_unit(unit, ctx):
    """One team -> result dict. Raises nothing: failures are recorded."""
    res = {"unit": unit, "facts": [], "rejected": [], "error": None, "usage": None, "called": False,
           "deferred": None}
    reserved = ctx["budget"].reserve()
    if reserved is None:
        res["deferred"] = BUDGET_REASON
        return res
    try:
        res["called"] = True
        resp = ctx["llm"](unit["key"], payload(ctx["model"], team_request(unit["team"], unit["players"], ctx["as_of"])))
        try:
            out, res["usage"] = parse_response(resp)
        except FetchError as e:
            res["usage"] = e.args[1] if len(e.args) > 1 else None
            raise FetchError(e.args[0])
        if not isinstance(out, dict):
            raise FetchError("answer is not a JSON object")
        ff, fr = form_facts_from(out.get("players"), unit, ctx)
        nfacts, nr = news_facts_from(out.get("news"), unit, ctx)
        res["facts"], res["rejected"] = ff + nfacts, fr + nr
    except FetchError as e:
        res["error"] = str(e)
    except Exception as e:      # per-team isolation: one failure never stops the others
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        u = res["usage"] or {}
        ctx["budget"].settle(reserved, estimate_cost(u.get("input_tokens", 0), u.get("output_tokens", 0),
                                                     u.get("web_search_calls", 0)))
    return res


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
    ap.add_argument("--teams", help="comma-separated bundle team names (overrides the scope)")
    ap.add_argument("--scope-days", type=int, default=SCOPE_DAYS, help="teams with a fixture in the next N days (default %(default)s)")
    ap.add_argument("--ttl-hours", type=float, default=TEAM_TTL_HOURS, help="re-fetch a team at most this often (default %(default)s)")
    ap.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS, help="LLM calls per run (default %(default)s)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="worker threads (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="no network: recorded fixtures only")
    ap.add_argument("--fixtures", default=FIXTURES, help="dry-run fixture folder")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--log", help=f"run history, also the budget's ledger (default: {LOG_NAME} beside --manifest)")
    ap.add_argument("--bundle", default=BUNDLE)
    ap.add_argument("--as-of", help="scope start day YYYY-MM-DD (default: the run's UTC date)")
    ap.add_argument("--refresh", action="store_true", help="ignore the TTL (re-fetch every team; the budget still applies)")
    ap.add_argument("--strict", action="store_true", help="fail on a missing API key (implied by CI)")
    return ap.parse_args(argv)


def main(argv=None, now=None, llm=None, env=None):
    a = parse_args(argv)
    env = os.environ if env is None else env
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    observed_at = nf.fmt_ts(now)
    as_of = date.fromisoformat(a.as_of) if a.as_of else now.date()
    strict = a.strict or bool(env.get("CI") or env.get("GITHUB_ACTIONS"))
    model = env.get("FACEOFF_NOTES_MODEL") or DEFAULT_MODEL
    log_path = a.log or os.path.join(os.path.dirname(os.path.abspath(a.manifest)), LOG_NAME)
    mode = "dry-run" if a.dry_run else "live"
    key = env.get("OPENAI_API_KEY")

    try:
        budget_limit = float(env.get("FACEOFF_NOTES_BUDGET_USD") or DEFAULT_BUDGET_USD)
    except ValueError:
        print(f"notes_fetch: FACEOFF_NOTES_BUDGET_USD={env.get('FACEOFF_NOTES_BUDGET_USD')!r} is not a number",
              file=sys.stderr)
        return 1

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
    sc = scope(bundle, as_of, a.scope_days, only)

    budget = Budget(budget_limit, month_spent(log_path, observed_at[:7], mode))
    manifest = {"run_at": observed_at, "mode": mode, "model": model, "as_of": as_of.isoformat(),
                "scope_days": a.scope_days, "ttl_hours": a.ttl_hours,
                "scope_teams": [t for t, _, _ in sc], "error": None, "warnings": [],
                "pricing_estimate": {"note": "ESTIMATES: verify against https://openai.com/api/pricing",
                                     "input_per_m_usd": PRICE_INPUT_PER_M, "output_per_m_usd": PRICE_OUTPUT_PER_M,
                                     "web_search_per_call_usd": PRICE_WEB_SEARCH_PER_CALL}}

    def finish(code):
        manifest["exit_code"] = code
        manifest.setdefault("budget", budget.report())
        try:
            _write(a.manifest, redact(dumps(manifest), key))
            summary = {k: manifest.get(k) for k in (
                "run_at", "mode", "model", "as_of", "error", "exit_code", "llm_calls",
                "input_tokens", "output_tokens", "web_search_calls", "est_cost_usd", "cache_hits",
                "cache_hit_rate", "units_attempted", "units_failed", "units_deferred", "facts_written",
                "store_changed")}
            summary["scope_teams"] = len(manifest["scope_teams"])
            summary["teams_attempted"] = len(manifest.get("teams_attempted") or [])
            summary["facts_rejected"] = len(manifest.get("facts_rejected") or [])
            summary["fail_reasons"] = sorted({x["reason"] for x in manifest.get("fetches_failed") or []})[:10]
            summary["budget_stopped"] = manifest["budget"]["stopped"]
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(redact(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                               key) + "\n")
        except OSError as e:
            print(f"notes_fetch: cannot write the manifest {a.manifest}: {e}", file=sys.stderr)
            return 2
        return code

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
    newest = {}
    for f in form_old + news_old:
        newest[f["team"]] = max(newest.get(f["team"], ""), f["observed_at"])

    try:
        prev = _read_json(a.manifest) or {}
    except (OSError, ValueError):
        prev = {}
    keep_lo = nf.fmt_ts(now - timedelta(hours=max(CHECKED_KEEP_HOURS, a.ttl_hours)))
    checked = {k: v for k, v in (prev.get("checked") or {}).items()
               if k.startswith("team|") and isinstance(v, str) and keep_lo <= v <= observed_at}
    fresh_lo = nf.fmt_ts(now - timedelta(hours=a.ttl_hours))

    if a.dry_run and llm is None:
        try:
            fx = _read_json(os.path.join(a.fixtures, "responses.json")) or {}
        except ValueError as e:
            manifest["error"] = f"fixtures unreadable: {e}"
            return finish(1)
        llm = RecordedLLM(fx.get("llm") or {}, as_of)
    llm = llm or LiveLLM(key)

    # plan: one unit per team, soonest fixture first
    units, hits = [], 0
    for t, d, players in sc:
        k = f"team|{t}"
        if not a.refresh and max(newest.get(t, ""), checked.get(k, "")) > fresh_lo:
            hits += 1
        else:
            units.append({"team": t, "day": d.isoformat(), "players": players, "key": k})
    cap = max(0, a.max_calls)
    todo = units[:cap]
    deferred = [{"team": u["team"], "reason": f"--max-calls {cap}"} for u in units[cap:]]

    ctx = {"model": model, "as_of": as_of, "observed_at": observed_at, "teams": teams, "names": names,
           "llm": llm, "budget": budget}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        results = list(ex.map(lambda u: run_unit(u, ctx), todo))   # map keeps the plan order

    new_facts, rejected, failed = [], [], []
    tok_in = tok_out = searches = calls = 0
    for r in results:
        u = r["unit"]
        if r["deferred"]:
            deferred.append({"team": u["team"], "reason": r["deferred"]})
            continue
        calls += r["called"]
        if r["usage"]:
            tok_in += r["usage"]["input_tokens"]
            tok_out += r["usage"]["output_tokens"]
            searches += r["usage"]["web_search_calls"]
        if r["error"]:
            failed.append({"team": u["team"], "reason": redact(r["error"], key)})
        else:
            checked[u["key"]] = observed_at
        new_facts += r["facts"]
        for x in r["rejected"]:
            rejected.append({"team": u["team"], **x})
    deferred.sort(key=lambda x: [u["team"] for u in units].index(x["team"]))

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

    attempted = calls
    n_failed = len(failed)
    denom = hits + len(units)
    manifest.update({
        "teams_attempted": sorted(r["unit"]["team"] for r in results if r["called"]),
        "units_planned": len(units), "units_attempted": attempted, "units_failed": n_failed,
        "units_deferred": len(deferred), "deferred": deferred,
        "facts_written": len(added), "facts_written_ids": sorted(f["id"] for f in added),
        "facts_pruned": pruned, "facts_rejected": rejected, "fetches_failed": failed,
        "cache_hits": hits, "cache_hit_rate": round(hits / denom, 3) if denom else None,
        "llm_calls": calls, "input_tokens": tok_in, "output_tokens": tok_out, "web_search_calls": searches,
        "est_cost_usd": estimate_cost(tok_in, tok_out, searches),
        "budget": budget.report(),
        "store": a.out if a.out != OUT else "data/notes_facts.json", "store_changed": store_changed,
        "checked": dict(sorted(checked.items())),
    })
    if budget.stopped:
        n = sum(1 for d in deferred if d["reason"] == BUDGET_REASON)
        b = manifest["budget"]
        line = (f"monthly budget reached: ${b['spent_month_usd']:.4f} of ${b['limit_usd']:.2f} spent this month "
                f"(next call could cost up to ${b['worst_call_usd']:.4f}); {n} teams deferred to next month "
                f"or a larger FACEOFF_NOTES_BUDGET_USD")
        manifest["warnings"].append(line)
        print(f"notes_fetch: {line}", file=sys.stderr)
    code = 0
    if attempted and n_failed >= min(FAIL_MIN, attempted) and n_failed / attempted > FAIL_RATIO:
        manifest["error"] = f"{n_failed} of {attempted} teams failed (> {int(FAIL_RATIO * 100)}%)"
        print(f"notes_fetch: ERROR {manifest['error']}", file=sys.stderr)
        code = 1
    rc = finish(code)
    print(f"notes_fetch [{mode}]: scope {len(sc)} teams ({a.scope_days} days), {attempted} calls "
          f"({n_failed} failed, {len(deferred)} deferred), {hits} cache hits, {len(added)} new facts, "
          f"{len(rejected)} rejected, {tok_in}+{tok_out} tokens, {searches} searches, "
          f"~${manifest['est_cost_usd']} (month ${manifest['budget']['spent_month_usd']} of "
          f"${budget_limit:.2f}){'' if store_changed else ' (store unchanged)'}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
