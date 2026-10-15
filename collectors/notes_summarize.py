#!/usr/bin/env python3
"""
notes_summarize.py -- AI-written "why the model picks this team" summaries
for scheduled matches, in plain English, from the model's own numbers.
Writes data/notes_summaries.json; the page shows them in the Notes tab.

WHAT THE AI SEES, AND ALL IT MAY SAY
  For each upcoming match (a tracked event match with a pre-match call, dated
  as_of - 1 .. as_of + SCOPE_DAYS) one prompt with: the two teams, the event,
  the model's win chance and 90% band, the predicted scoreline, every counted
  factor with its point contribution and the evidence the site shows beside
  it ("What moves the number"), the engine's warnings, and the matchup notes
  + fact sheet (notes.get_notes), labelled as context NOT counted in the win
  chance. The model only rewrites these numbers as prose:
  * GROUNDING CHECK: every number in the answer must appear in the input
    (percentages may be rounded to whole numbers); small counts 0-10 are
    allowed. Anything else rejects the summary (the page then shows its
    rule-based summary instead). The favourite's name must appear.
  * No web search, no tools: nothing outside our data can get in.

COST (estimates; verify against OpenAI's pricing page)
  gpt-5-nano, ~2.5k input + ~0.6k output tokens per match: about $0.0004 a
  summary; 5-15 scheduled matches a day = well under $0.01 a day. A summary is
  rewritten only when its input changes (input_hash), at most once per
  REWRITE_HOURS. Hard cap FACEOFF_SUMMARY_BUDGET_USD (default 5.00 a month),
  summed from data/notes_summary_log.jsonl like the fact fetcher's budget.

KEY
  OPENAI_API_KEY (the GitHub Actions secret), sent only to api.openai.com
  through notes_fetch.LiveLLM, which scrubs it from every error text.

USAGE (repo root):
  python collectors/notes_summarize.py [--dry-run] [--max-calls N] [--out PATH]
Stdlib only.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import notes as NT  # noqa: E402
import notes_facts as NF  # noqa: E402
import notes_fetch as NFX  # noqa: E402

SCHEMA = 1
OUT = os.path.join(paths.DATA, "notes_summaries.json")
LOG = os.path.join(paths.DATA, "notes_summary_log.jsonl")
BUNDLE = os.path.join(paths.VIEWER, "data-bundle.json")
SCOPE_DAYS = 3
REWRITE_HOURS = 20
KEEP_DAYS = 120
DEFAULT_MAX_CALLS = 30
DEFAULT_BUDGET_USD = 5.00
MAX_OUTPUT_TOKENS = 1500
WORST_INPUT_TOKENS = 8000
FACTORS = ["base_strength", "form_30d", "form_last5", "head_to_head", "map_veto", "roster", "map_depth"]
LABELS = {"base_strength": "Team strength (Elo rating gap)", "form_30d": "Last 30 days (results vs what the rating expects)",
          "form_last5": "Last 5 series", "head_to_head": "Head-to-head (past year)", "roster": "Roster (stand-ins, missing IGL)"}
ROSTER_TXT = ["", "stand-in", "missing IGL", "stand-in for the IGL"]

SYSTEM = ("You explain a Counter-Strike 2 match prediction to a fan in plain, calm English. Use ONLY the facts and "
          "numbers in the input; never add players, results, history or opinions that are not there, and do not "
          "give betting advice. Write a short headline and a summary of 3 to 5 sentences (60-120 words): who the "
          "model favours and its win chance with the 90% range; the two or three counted factors that matter most, "
          "with their point values and the evidence behind them; any warning about uncertainty; then one sentence of "
          "context from the notes, saying plainly that it is not counted in the win chance (skip it if there are no "
          "notes). Write percentages as given (you may round to whole numbers). Answer only with the JSON object.")
OUT_SCHEMA = NFX._strict({"headline": {"type": "string"}, "summary": {"type": "string"}})


def _pct(x):
    return f"{x * 100:.1f}%"


def factor_rows(p, a, b):
    """The counted factors, as the page's 'What moves the number' shows them:
    [{factor, points, toward, evidence}] (map veto / map depth are shown on
    the site but not counted, so they are left out)."""
    fx, out = p.get("fx") or [], []
    for name, pp in zip(FACTORS, p.get("f") or []):
        if name not in LABELS or abs(pp) < 0.05:
            continue
        ev = ""
        if fx and name == "base_strength":
            ev = f"{a if fx[0] > 0 else b} +{abs(fx[0])} Elo" if fx[0] else "even Elo"
        elif fx and name == "form_30d":
            ev = f"{a} {fx[1]}-{fx[2]}, {b} {fx[4]}-{fx[5]} in 30 days; the rating expected {fx[3] / 10:.1f} vs {fx[6] / 10:.1f} wins"
        elif fx and name == "form_last5":
            ev = f"{a} {fx[7]}-{fx[8]}, {b} {fx[9]}-{fx[10]} in the last 5 series"
        elif fx and name == "head_to_head":
            ev = f"{a} {fx[11]}-{fx[12]} {b} in the past year"
        elif fx and name == "roster":
            ra, rb = fx[16] // 4, fx[16] % 4
            ev = "; ".join(x for x in (ra and f"{a}: {ROSTER_TXT[ra]}", rb and f"{b}: {ROSTER_TXT[rb]}") if x)
        out.append({"factor": LABELS[name], "points": round(abs(pp), 1), "toward": a if pp > 0 else b, "evidence": ev})
    return sorted(out, key=lambda r: -r["points"])


def match_input(m, ev, notes_out):
    """The whole prompt input for one match (also what input_hash covers)."""
    a, b, p = m["t1"], m["t2"], m["pred"]
    fav, pf = (a, p["p"]) if p["p"] >= 0.5 else (b, 1 - p["p"])
    lo, hi = p["b"] if fav == a else [1 - p["b"][1], 1 - p["b"][0]]
    s = p.get("s") or []
    lines = dict(zip(["2-0", "2-1", "1-2", "0-2"], s))
    modal = max(lines, key=lines.get) if lines else None
    return {
        "match": {"team_a": a, "team_b": b, "event": ev["name"], "stage": m.get("stage"), "day": m.get("day"),
                  "best_of": m.get("bo")},
        "prediction": {"favourite": fav, "underdog": b if fav == a else a, "win_chance": _pct(pf),
                       "range_90": [_pct(lo), _pct(hi)], "most_likely_score": f"{modal} for {a}" if modal else None,
                       "reliability": {"H": "clear read", "M": "fairly clear read", "L": "close or uncertain read"}.get(p.get("r"), p.get("r")),
                       "counted_factors": factor_rows(p, a, b),
                       "warnings": [w.split(":")[0] + ": " + w.split(":", 1)[1].split(".")[0].strip() if ":" in w else w
                                    for w in p.get("w") or []]},
        "context_not_counted": [{"about": n["team_scope"], "note": n["headline"], "detail": n["evidence"]}
                                for n in (notes_out.get("notes") or [])] if notes_out.get("ok") else [],
    }


def input_hash(inp):
    return hashlib.sha1(json.dumps(inp, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")


def allowed_numbers(inp):
    """Every number in the input, plus whole-number roundings of percentages
    and decimals, plus small counts."""
    txt = json.dumps(inp, ensure_ascii=False)
    ok = set(str(i) for i in range(11))
    for n in _NUM.findall(txt):
        n = n.replace(",", ".")
        ok.add(n)
        try:
            f = float(n)
            ok.add(str(round(f)))
            ok.add(f"{f:.1f}")
            ok.add(f"{f:g}")
        except ValueError:
            pass
    return ok


def check(out, inp):
    """-> list of problems (empty = the summary may be shown)."""
    if not isinstance(out, dict):
        return ["answer is not an object"]
    errs = []
    head, body = str(out.get("headline") or "").strip(), str(out.get("summary") or "").strip()
    if not head or not body:
        errs.append("empty headline or summary")
    if len(body) > 1200 or len(head) > 160:
        errs.append("too long")
    ok = allowed_numbers(inp)
    bad = sorted({n for n in _NUM.findall(head + " " + body) if n.replace(",", ".") not in ok})
    if bad:
        errs.append(f"numbers not in the input: {bad[:6]}")
    fav = inp["prediction"]["favourite"]
    if fav.lower() not in (head + " " + body).lower() and fav.split()[-1].lower() not in (head + " " + body).lower():
        errs.append("favourite not named")
    return errs


def scheduled(bundle, as_of):
    lo, hi = (as_of - timedelta(days=1)).isoformat(), (as_of + timedelta(days=SCOPE_DAYS)).isoformat()
    for e in bundle["data"]["events"]:
        for m in e["matches"]:
            if (not m.get("finished") and m.get("pred") and m.get("t1") and m.get("t2")
                    and lo <= (m.get("day") or "") <= hi):
                yield m, e


def cost(u):
    return NFX.estimate_cost(u.get("input_tokens", 0), u.get("output_tokens", 0), 0)


def main(argv=None, now=None, llm=None, env=None):
    ap = argparse.ArgumentParser(description="AI summaries of the model's calls for scheduled matches")
    ap.add_argument("--dry-run", action="store_true", help="no API calls: a fixed stand-in answer (tests)")
    ap.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--bundle", default=BUNDLE)
    a = ap.parse_args(argv)
    env = os.environ if env is None else env
    now = now or datetime.now(timezone.utc)
    run_at = NF.fmt_ts(now)
    try:
        with open(a.bundle, encoding="utf-8") as f:
            bundle = json.load(f)
    except (OSError, ValueError) as e:
        print(f"notes_summarize: cannot read the bundle {a.bundle} ({e})", file=sys.stderr)
        return 1
    key = env.get("OPENAI_API_KEY")
    strict = bool(env.get("CI") or env.get("GITHUB_ACTIONS"))
    if not key and not a.dry_run and llm is None:
        print("notes_summarize: OPENAI_API_KEY is not set: no summaries written", file=sys.stderr)
        return 1 if strict else 0
    try:
        limit = float(env.get("FACEOFF_SUMMARY_BUDGET_USD") or DEFAULT_BUDGET_USD)
    except ValueError:
        print("notes_summarize: FACEOFF_SUMMARY_BUDGET_USD is not a number", file=sys.stderr)
        return 1
    model = env.get("FACEOFF_NOTES_MODEL") or NFX.DEFAULT_MODEL
    mode = "dry-run" if a.dry_run else "live"
    spent = NFX.month_spent(a.log, run_at[:7], mode)
    worst = NFX.estimate_cost(WORST_INPUT_TOKENS, MAX_OUTPUT_TOKENS, 0)
    if llm is None:
        llm = (lambda k, body: {"status": "completed", "usage": {"input_tokens": 2500, "output_tokens": 600},
                                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(
                                    {"headline": "Dry run", "summary": "Dry run: no model was called."})}]}]}
               ) if a.dry_run else NFX.LiveLLM(key)
    store = NFX._read_json(a.out) or {"schema": SCHEMA, "updated_at": None, "summaries": {}}
    sums = store.get("summaries") or {}
    as_of = date.fromisoformat(bundle["data"]["as_of"])
    stats = {"run_at": run_at, "mode": mode, "model": model, "scheduled": 0, "skipped_fresh": 0, "llm_calls": 0,
             "written": 0, "rejected": [], "failed": [], "deferred_budget": 0, "input_tokens": 0, "output_tokens": 0}
    run_cost = 0.0
    for m, ev in sorted(scheduled(bundle, as_of), key=lambda x: (x[0].get("day") or "", x[0]["id"])):
        stats["scheduled"] += 1
        notes_out = NT.get_notes(m["t1"], m["t2"], "latest", bundle, match_day=m["day"], event=ev["name"], limit=12)
        inp = match_input(m, ev, notes_out)
        h = input_hash(inp)
        old = sums.get(m["id"])
        if old and (old.get("input_hash") == h or
                    now - NF.parse_ts(old["generated_at"]) < timedelta(hours=REWRITE_HOURS)):
            stats["skipped_fresh"] += 1
            continue
        if stats["llm_calls"] >= a.max_calls:
            break
        if spent + run_cost + worst > limit + 1e-9:
            stats["deferred_budget"] += 1
            continue
        body = {"model": model, "instructions": SYSTEM, "input": json.dumps(inp, ensure_ascii=False), "store": False,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "text": {"format": {"type": "json_schema", "name": "why_summary", "schema": OUT_SCHEMA, "strict": True}}}
        if model.startswith(("gpt-5", "o")):
            body["reasoning"] = {"effort": "low"}
        stats["llm_calls"] += 1
        try:
            out, u = NFX.parse_response(llm(f"sum|{m['id']}", body))
        except NFX.FetchError as e:
            u = e.args[1] if len(e.args) > 1 and isinstance(e.args[1], dict) else {}
            run_cost += cost(u)
            stats["failed"].append({"match": m["id"], "reason": NFX.redact(e.args[0], key)})
            continue
        run_cost += cost(u)
        stats["input_tokens"] += u.get("input_tokens", 0)
        stats["output_tokens"] += u.get("output_tokens", 0)
        errs = [] if a.dry_run else check(out, inp)
        if errs:
            stats["rejected"].append({"match": m["id"], "reasons": errs})
            continue
        p = inp["prediction"]
        sums[m["id"]] = {"headline": out["headline"].strip(), "summary": out["summary"].strip(),
                         "team_a": m["t1"], "team_b": m["t2"], "day": m["day"], "favourite": p["favourite"],
                         "win_prob_a": round(m["pred"]["p"], 4), "model": model, "generated_at": run_at,
                         "input_hash": h}
        stats["written"] += 1
    keep = (as_of - timedelta(days=KEEP_DAYS)).isoformat()
    store = {"schema": SCHEMA, "updated_at": run_at if stats["written"] else store.get("updated_at"),
             "summaries": {k: v for k, v in sorted(sums.items()) if (v.get("day") or "") >= keep}}
    if stats["written"] or len(store["summaries"]) != len(sums):
        NFX._write(a.out, NFX.dumps(store))
    stats["est_cost_usd"] = round(run_cost, 4)
    stats["budget"] = {"limit_usd": limit, "spent_month_usd": round(spent + run_cost, 4)}
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    with open(a.log, "a", encoding="utf-8", newline="\n") as f:
        f.write(NFX.redact(json.dumps(stats, sort_keys=True, ensure_ascii=False, separators=(",", ":")), key) + "\n")
    print(f"notes_summarize [{mode}]: {stats['scheduled']} scheduled, {stats['skipped_fresh']} unchanged, "
          f"{stats['llm_calls']} calls, {stats['written']} written, {len(stats['rejected'])} rejected, "
          f"{len(stats['failed'])} failed, {stats['deferred_budget']} over budget; ~${stats['est_cost_usd']:.4f} "
          f"(month ${stats['budget']['spent_month_usd']:.4f} of ${limit:.2f})")
    called = stats["llm_calls"]
    return 1 if strict and called >= 3 and len(stats["failed"]) * 2 > called else 0


if __name__ == "__main__":
    sys.exit(main())
