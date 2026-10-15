#!/usr/bin/env python3
"""
notes_facts.py -- the web-fact schema of the matchup notes (fact schema v1,
docs/notes-contracts.md section 1): validation, ids and point-in-time
selection. Stdlib only, no I/O: the fetcher (collectors/notes_fetch.py)
validates on write, the rule engine (notes.py) validates again on read, and a
fact that fails is dropped, never repaired.

Facts are display context only; nothing here is read by the predictor.
"""
import hashlib
import json
import re
from datetime import date, datetime, timedelta, timezone

FACT_SCHEMA = 1

FORM_TTL_DAYS = 14        # a form observation counts this long after it was read
RUMOR_TTL_DAYS = 90       # a roster rumor is shown this long after its date
STAND_IN_LEAD_DAYS = 3    # a stand-in shows this many days before its date
STAND_IN_DEFAULT_DAYS = 21  # a stand-in with no `until` applies this long
PRUNE_DAYS = 400          # observations older than this are dropped from the file
RATING_RANGE = (0.3, 2.5)
MAPS_RANGE = (0, 80)

COMMON = ("id", "team", "player", "observed_at", "source_url", "source", "method")
FORM_REQ = ("rating_last30", "rating_baseline", "baseline", "maps_last30")
FORM_OPT = ("rating_version",)
NEWS_REQ = ("type", "date")
NEWS_OPT = ("replaced", "until", "event", "move", "other_team")
NEWS_TYPES = ("stand_in", "roster_rumor")
METHODS = ("scrape", "web_search")
BASELINES = ("last12m", "career")

_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_ts(s):
    """'YYYY-MM-DDTHH:MM:SSZ' -> aware UTC datetime."""
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_ts(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cutoff_of(asof):
    """asof 'YYYY-MM-DD' (start of that day, UTC) or a timestamp -> aware
    datetime. Raises ValueError on anything else."""
    if isinstance(asof, datetime):
        return asof if asof.tzinfo else asof.replace(tzinfo=timezone.utc)
    if isinstance(asof, date):
        return datetime(asof.year, asof.month, asof.day, tzinfo=timezone.utc)
    s = str(asof)
    if _DAY.match(s):
        d = date.fromisoformat(s)
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if _TS.match(s):
        return parse_ts(s)
    raise ValueError(f"not a date or UTC timestamp: {asof!r}")


def _identity(f):
    if "rating_last30" in f:
        keys = ("team", "player", "rating_last30", "rating_baseline", "baseline", "maps_last30", "rating_version")
    else:
        keys = ("type", "team", "player", "date") + NEWS_OPT
    return {k: f.get(k) for k in keys} | {"observed_at": f.get("observed_at"), "source_url": f.get("source_url")}


def fact_id(f):
    raw = json.dumps(_identity(f), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def kind_of(f):
    return "news" if "type" in f else "form"


def validate_fact(f, teams=None):
    """-> list of problems (empty = valid). `teams`: the bundle's team names;
    a fact naming another team is rejected."""
    if not isinstance(f, dict):
        return ["not an object"]
    errs = []
    kind = kind_of(f)
    req = COMMON + (NEWS_REQ if kind == "news" else FORM_REQ)
    allowed = set(req) | set(NEWS_OPT if kind == "news" else FORM_OPT)
    for k in req:
        if f.get(k) in (None, ""):
            errs.append(f"missing {k}")
    for k in sorted(set(f) - allowed):
        errs.append(f"unknown field {k}")
    if errs:
        return errs
    for k in ("team", "player", "source", "source_url"):
        if not isinstance(f[k], str) or not f[k].strip():
            errs.append(f"{k} not a string")
    try:
        parse_ts(str(f["observed_at"]))
    except ValueError:
        errs.append("observed_at not a valid YYYY-MM-DDTHH:MM:SSZ")
    if not str(f["source_url"]).startswith("https://"):
        errs.append("source_url not https")
    if f["method"] not in METHODS:
        errs.append(f"method not in {METHODS}")
    if teams is not None and f["team"] not in teams:
        errs.append(f"team {f['team']!r} not in the bundle")
    if kind == "form":
        for k in ("rating_last30", "rating_baseline"):
            v = f[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not RATING_RANGE[0] <= v <= RATING_RANGE[1]:
                errs.append(f"{k} {v!r} outside {RATING_RANGE}")
        n = f["maps_last30"]
        if isinstance(n, bool) or not isinstance(n, int) or not MAPS_RANGE[0] <= n <= MAPS_RANGE[1]:
            errs.append(f"maps_last30 {n!r} not an int in {MAPS_RANGE}")
        if f["baseline"] not in BASELINES:
            errs.append(f"baseline not in {BASELINES}")
    else:
        if f["type"] not in NEWS_TYPES:
            errs.append(f"type not in {NEWS_TYPES}")
        for k in ("date", "until"):
            if f.get(k) is not None:
                try:
                    if not _DAY.match(str(f[k])):
                        raise ValueError
                    date.fromisoformat(str(f[k]))
                except ValueError:
                    errs.append(f"{k} not a valid YYYY-MM-DD")
        if f.get("move") is not None and f["move"] not in ("in", "out"):
            errs.append("move not in/out")
        if f["type"] != "stand_in" and any(f.get(k) for k in ("replaced", "until", "event")):
            errs.append("stand-in fields on a non-stand-in")
        if f["type"] != "roster_rumor" and any(f.get(k) for k in ("move", "other_team")):
            errs.append("rumor fields on a non-rumor")
    if not errs and f["id"] != fact_id(f):
        errs.append("id does not match the fact")
    return errs


def make_fact(**fields):
    """A fact dict with its id filled in (fields as in the schema)."""
    f = {k: v for k, v in fields.items() if v is not None}
    f["id"] = fact_id(f)
    return f


# ---- point-in-time selection (the engine's view) ---------------------------
def live_form(facts, cutoff):
    """{(team, player): newest form fact observed < cutoff and within
    FORM_TTL_DAYS of it}."""
    c = cutoff_of(cutoff)
    out = {}
    for f in facts:
        t = parse_ts(f["observed_at"])
        if not (t < c and c - t <= timedelta(days=FORM_TTL_DAYS)):
            continue
        k = (f["team"], f["player"])
        if k not in out or (f["observed_at"], f["id"]) > (out[k]["observed_at"], out[k]["id"]):
            out[k] = f
    return out


def live_news(facts, cutoff):
    """News facts observed < cutoff and active on cutoff's day; one per
    (type, team, player, date), the newest observation."""
    c = cutoff_of(cutoff)
    day = c.date()
    out = {}
    for f in facts:
        if not parse_ts(f["observed_at"]) < c:
            continue
        d = date.fromisoformat(f["date"])
        if f["type"] == "stand_in":
            end = date.fromisoformat(f["until"]) if f.get("until") else d + timedelta(days=STAND_IN_DEFAULT_DAYS)
            if not (d - timedelta(days=STAND_IN_LEAD_DAYS) <= day <= end):
                continue
        elif not (0 <= (day - d).days < RUMOR_TTL_DAYS):
            continue
        k = (f["type"], f["team"], f["player"], f["date"])
        if k not in out or (f["observed_at"], f["id"]) > (out[k]["observed_at"], out[k]["id"]):
            out[k] = f
    return [out[k] for k in sorted(out)]


def clean(store, teams=None):
    """(valid form facts, valid news facts, [(id, problems)] rejected) from a
    parsed notes_facts.json; anything malformed is dropped."""
    form, news, bad = [], [], []
    for key, out in (("form", form), ("news", news)):
        for f in (store or {}).get(key) or []:
            errs = validate_fact(f, teams)
            if not errs and kind_of(f) != key:
                errs = [f"{kind_of(f)} fact in {key}"]
            (bad.append(((f or {}).get("id") if isinstance(f, dict) else None, errs)) if errs else out.append(f))
    return form, news, bad


def merge(store, new_facts, now):
    """Append valid new facts (no duplicate ids), prune observations older
    than PRUNE_DAYS before `now`; returns the new store (sorted, stable)."""
    store = {"fact_schema": FACT_SCHEMA, "updated_at": store.get("updated_at") if store else None,
             "form": list((store or {}).get("form") or []), "news": list((store or {}).get("news") or [])}
    seen = {f["id"] for k in ("form", "news") for f in store[k]}
    for f in new_facts:
        if f["id"] not in seen:
            store[kind_of(f)].append(f)
            seen.add(f["id"])
    lo = cutoff_of(now) - timedelta(days=PRUNE_DAYS)
    for k in ("form", "news"):
        store[k] = sorted((f for f in store[k] if parse_ts(f["observed_at"]) >= lo),
                          key=lambda f: (f["team"], f["player"], f["observed_at"], f["id"]))
    store["updated_at"] = fmt_ts(cutoff_of(now))
    return store
