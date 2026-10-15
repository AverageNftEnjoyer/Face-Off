#!/usr/bin/env python3
"""
notes_summarize.py -- AI-written match writing from the model's own data:
"why the model picks this team" for scheduled matches (plain words, no
figures) and post-match recaps for matches finished in the last RECAP_DAYS
(lede on the cause, map by map, the model check, what's next; figures only
from the match data, banned clichés rejected, missing player stats stated).
Writes data/notes_summaries.json; the page shows both in the Notes tab.

WHAT THE AI SEES, AND ALL IT MAY SAY
  For each upcoming match (a tracked event match with a pre-match call, dated
  as_of - 1 .. as_of + SCOPE_DAYS) one prompt with: the two teams, the event,
  the model's win chance and 90% band, the predicted scoreline, every counted
  factor with its point contribution and the evidence the site shows beside
  it ("What moves the number"), the engine's warnings, and the matchup notes
  + fact sheet (notes.get_notes), labelled as context NOT counted in the win
  chance. The model only rewrites these numbers as prose:
  * PLAIN WORDS ONLY: the summary may contain no figures at all (no
    percentages, scores, records, days); a digit left after removing team,
    player and event names rejects it, and the page shows its rule-based
    summary instead. The favourite's name must appear. It covers the counted
    factors first, then the wider picture (rest, schedule, streaks, roster
    stability, the veto maps, past meetings, player form), labelled as not
    counted in the win chance.
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

VOICE = ("Voice: a sharp human analyst for Faceoff, a CS2 analytics site. Dry, direct, confident; mix short punchy "
         "sentences with longer analytical ones. Take a position on what decides it; do not hedge everything. Hype is "
         "earned, never default. Interpret, do not list. Never write: 'what a series', 'absolute cinema', 'banger', "
         "'both teams battled hard', 'anyone's game', 'left it all on the server'. ")
SYSTEM = (VOICE + "You explain a Counter-Strike 2 match prediction to a fan. Open with the single most telling reason, not "
          "the matchup. Use ONLY the facts in "
          "the input; never add players, results, history or opinions that are not there, and do not give betting "
          "advice. Write NO digits or figures at all: no percentages, scores, records, ratings, days or counts. Say "
          "it in words instead (slight favourite, clear favourite, coin flip, a wide range of outcomes, played "
          "yesterday, a new roster, the upper hand in past meetings). Write a short headline and a summary of 4 to 6 "
          "sentences: first who the model favours and how confident it is; then the counted factors in "
          "prediction.counted_factors, strongest first, saying which team each one helps; then the wider picture "
          "from context_not_counted (rest, schedule, streaks, roster stability, the simulated veto maps, past "
          "meetings, player form, stand-ins, rumours), saying plainly that this part is not counted in the win "
          "chance. Answer only with the JSON object.")
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
        "context_not_counted": {
            "notes": [{"about": n["team_scope"], "note": n["headline"], "detail": n["evidence"]}
                      for n in (notes_out.get("notes") or [])] if notes_out.get("ok") else [],
            "fact_sheet": notes_out.get("sheet") if notes_out.get("ok") else None,
            "simulated_veto_maps": [{"map": mp, "favours": a if pa > 0.5 else b, "how_much": _how(abs(pa - 0.5))}
                                    for mp, pa in zip(p.get("vm") or [], p.get("vp") or [])],
        },
    }


def _how(d):
    return "barely" if d < 0.05 else "somewhat" if d < 0.12 else "clearly"


def input_hash(inp):
    return hashlib.sha1(json.dumps(inp, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def check(out, inp):
    """-> list of problems (empty = the summary may be shown). The text must be
    figure-free: after removing the team and event names (which may contain
    digits, e.g. 9z, 3DMAX, Season 24), no digit may remain. The favourite must
    be named."""
    if not isinstance(out, dict):
        return ["answer is not an object"]
    errs = []
    head, body = str(out.get("headline") or "").strip(), str(out.get("summary") or "").strip()
    if not head or not body:
        errs.append("empty headline or summary")
    if len(body) > 1400 or len(head) > 160:
        errs.append("too long")
    text = head + " " + body
    names = [inp["match"]["team_a"], inp["match"]["team_b"], inp["match"].get("event") or ""]
    sheet = (inp.get("context_not_counted") or {}).get("fact_sheet") or {}
    for side in ("a", "b"):
        sh = sheet.get(side) or {}
        names += [(sh.get("last") or {}).get("event") or "", (sh.get("last") or {}).get("opp") or ""]
        names += (sh.get("roster") or {}).get("five") or []
        names += [p.get("player", "") for p in sh.get("players") or []]
    names += [m.get("event") or "" for m in (sheet.get("h2h") or {}).get("recent") or []]
    bare = text
    for n in sorted({n for n in names if n}, key=len, reverse=True):
        bare = re.sub(re.escape(n), " ", bare, flags=re.I)
    digits = sorted(set(re.findall(r"\d+", bare)))
    if digits:
        errs.append(f"figures in the text: {digits[:6]}")
    fav = inp["prediction"]["favourite"]
    if fav.lower() not in text.lower() and fav.split()[-1].lower() not in text.lower():
        errs.append("favourite not named")
    return errs


# ---- post-match recaps ------------------------------------------------------
RECAP_DAYS = 3          # recap matches finished in the last few days (as_of - RECAP_DAYS .. as_of)
RECAP_SYSTEM = (VOICE + "You write the post-match recap. The reader knows the score; they come for the why. "
                "Structure, in this order: (1) a lede of one or two sentences with the single most interesting causal "
                "fact, not the scoreline; (2) how it happened, map by map, only decisive details (round score, which "
                "half or side carried it, overtime, whether the maps played were the ones the simulated veto "
                "expected); (3) standouts: per-player statistics are NOT in the data, so say so in one short clause "
                "rather than inventing names or ratings; (4) the model check: what the model predicted and what "
                "actually decided it; if the call was wrong, name the exact mechanism from the input (which "
                "assumption broke, which map diverged from the expected veto); (5) one forward-looking line about "
                "what to watch next. Every claim needs evidence from the input. Numbers are allowed ONLY if they appear "
                "in the input. 120 to 220 words. Answer only with the JSON object.")
BANNED = ("what a series", "absolute cinema", "banger", "battled hard", "anyone's game", "anyones game",
          "left it all on the server")


def recent_finished(bundle, as_of):
    lo = (as_of - timedelta(days=RECAP_DAYS)).isoformat()
    for e in bundle["data"]["events"]:
        for m in e["matches"]:
            if (m.get("finished") and m.get("pred") and m.get("t1") and m.get("t2") and m.get("maps")
                    and m.get("w1") != m.get("w2") and lo <= (m.get("day") or "") < as_of.isoformat()):
                yield m, e


def recap_input(m, ev, notes_out):
    """Everything the recap may use: the result, each map (score, halves,
    overtime, start side), the model's pre-match call and its reasons, the
    veto it simulated, and the pre-match context. Gaps are stated."""
    a, b, p = m["t1"], m["t2"], m["pred"]
    pre = match_input(m, ev, notes_out)
    winner = a if m["w1"] > m["w2"] else b
    expected = list(p.get("vm") or [])
    maps = []
    for i, mp in enumerate(m["maps"], 1):
        s1, s2 = mp.get("s1"), mp.get("s2")
        sd1, sd2 = mp.get("side1") or {}, mp.get("side2") or {}
        chance = None
        if mp.get("map") in expected:
            pa = (p.get("vp") or [])[expected.index(mp["map"])]
            chance = _pct(pa) + f" for {a}"
        maps.append({"order": i, "map": mp.get("map"), "winner": a if mp.get("w") == 1 else b,
                     "score": f"{s1}-{s2} ({a}-{b})" if s1 is not None else None,
                     "halves": {a: {"ct": sd1.get("ct"), "t": sd1.get("t")}, b: {"ct": sd2.get("ct"), "t": sd2.get("t")}}
                     if sd1 or sd2 else None,
                     "overtime": bool(mp.get("ot")), f"{a}_started_on": mp.get("first1"),
                     "in_simulated_veto": mp.get("map") in expected, "model_map_chance": chance})
    fav = pre["prediction"]["favourite"]
    return {"match": pre["match"] | {"winner": winner, "series_score": f"{m['w1']}-{m['w2']} ({a}-{b})"},
            "model_call": {"favourite": fav, "win_chance": pre["prediction"]["win_chance"],
                           "range_90": pre["prediction"]["range_90"],
                           "predicted_score": pre["prediction"]["most_likely_score"],
                           "result": "correct" if fav == winner else "upset",
                           "counted_factors": pre["prediction"]["counted_factors"],
                           "warnings": pre["prediction"]["warnings"],
                           "simulated_veto_maps": expected},
            "maps": maps, "pre_match_context": pre["context_not_counted"],
            "data_gaps": ["per-player statistics (ratings, ADR) are not in the data",
                          "who picked and who banned each map is not in the data"]}


_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")


def allowed_numbers(inp):
    """Every number in the input (with whole-number roundings) plus 0-10."""
    ok = {str(i) for i in range(11)}
    for n in _NUM.findall(json.dumps(inp, ensure_ascii=False)):
        n = n.replace(",", ".")
        ok.add(n)
        try:
            f = float(n)
            ok.update({str(round(f)), f"{f:.1f}", f"{f:g}"})
        except ValueError:
            pass
    return ok


def check_recap(out, inp):
    if not isinstance(out, dict):
        return ["answer is not an object"]
    errs = []
    head, body = str(out.get("headline") or "").strip(), str(out.get("summary") or "").strip()
    text = head + " " + body
    if not head or not body:
        errs.append("empty headline or recap")
    words = len(body.split())
    if not 80 <= words <= 300:
        errs.append(f"length {words} words")
    low = text.lower()
    hit = [b for b in BANNED if b in low]
    if hit:
        errs.append(f"banned phrases: {hit}")
    bad = sorted({n for n in _NUM.findall(text) if n.replace(",", ".") not in allowed_numbers(inp)})
    if bad:
        errs.append(f"numbers not in the input: {bad[:6]}")
    w = inp["match"]["winner"]
    if w.lower() not in low and w.split()[-1].lower() not in low:
        errs.append("winner not named")
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
    store = NFX._read_json(a.out) or {"schema": SCHEMA, "updated_at": None, "summaries": {}, "recaps": {}}
    books = {"why": store.get("summaries") or {}, "recap": store.get("recaps") or {}}
    as_of = date.fromisoformat(bundle["data"]["as_of"])
    stats = {"run_at": run_at, "mode": mode, "model": model, "scheduled": 0, "recaps_due": 0, "skipped_fresh": 0,
             "llm_calls": 0, "written": 0, "rejected": [], "failed": [], "deferred_budget": 0,
             "input_tokens": 0, "output_tokens": 0}
    jobs = []
    for m, ev in sorted(scheduled(bundle, as_of), key=lambda x: (x[0].get("day") or "", x[0]["id"])):
        stats["scheduled"] += 1
        n = NT.get_notes(m["t1"], m["t2"], "latest", bundle, match_day=m["day"], event=ev["name"], limit=12)
        jobs.append(("why", m, match_input(m, ev, n), SYSTEM, check))
    for m, ev in sorted(recent_finished(bundle, as_of), key=lambda x: (x[0].get("day") or "", x[0]["id"])):
        stats["recaps_due"] += 1
        # the context as it stood before the match (its own day as the cutoff)
        n = NT.get_notes(m["t1"], m["t2"], m["day"], bundle, match_day=m["day"], event=ev["name"], limit=12)
        jobs.append(("recap", m, recap_input(m, ev, n), RECAP_SYSTEM, check_recap))
    run_cost = 0.0
    for kind, m, inp, system, checker in jobs:
        book = books[kind]
        h = input_hash(inp)
        old = book.get(m["id"])
        if old and (old.get("input_hash") == h or (kind == "why" and
                    now - NF.parse_ts(old["generated_at"]) < timedelta(hours=REWRITE_HOURS))):
            stats["skipped_fresh"] += 1
            continue
        if stats["llm_calls"] >= a.max_calls:
            break
        if spent + run_cost + worst > limit + 1e-9:
            stats["deferred_budget"] += 1
            continue
        body = {"model": model, "instructions": system, "input": json.dumps(inp, ensure_ascii=False), "store": False,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "text": {"format": {"type": "json_schema", "name": f"{kind}_summary", "schema": OUT_SCHEMA, "strict": True}}}
        if model.startswith(("gpt-5", "o")):
            body["reasoning"] = {"effort": "low"}
        stats["llm_calls"] += 1
        try:
            out, u = NFX.parse_response(llm(f"{kind}|{m['id']}", body))
        except NFX.FetchError as e:
            u = e.args[1] if len(e.args) > 1 and isinstance(e.args[1], dict) else {}
            run_cost += cost(u)
            stats["failed"].append({"kind": kind, "match": m["id"], "reason": NFX.redact(e.args[0], key)})
            continue
        run_cost += cost(u)
        stats["input_tokens"] += u.get("input_tokens", 0)
        stats["output_tokens"] += u.get("output_tokens", 0)
        errs = [] if a.dry_run else checker(out, inp)
        if errs:
            stats["rejected"].append({"kind": kind, "match": m["id"], "reasons": errs})
            continue
        rec = {"headline": out["headline"].strip(), "summary": out["summary"].strip(), "team_a": m["t1"],
               "team_b": m["t2"], "day": m["day"], "model": model, "generated_at": run_at, "input_hash": h}
        if kind == "why":
            rec |= {"favourite": inp["prediction"]["favourite"], "win_prob_a": round(m["pred"]["p"], 4)}
        else:
            rec |= {"winner": inp["match"]["winner"], "result": inp["model_call"]["result"]}
        book[m["id"]] = rec
        stats["written"] += 1
    keep = (as_of - timedelta(days=KEEP_DAYS)).isoformat()
    kept = {k: {i: v for i, v in sorted(b.items()) if (v.get("day") or "") >= keep} for k, b in books.items()}
    new = {"schema": SCHEMA, "updated_at": run_at if stats["written"] else store.get("updated_at"),
           "summaries": kept["why"], "recaps": kept["recap"]}
    if stats["written"] or any(len(kept[k]) != len(books[k]) for k in books) or "recaps" not in store:
        NFX._write(a.out, NFX.dumps(new))
    stats["est_cost_usd"] = round(run_cost, 4)
    stats["budget"] = {"limit_usd": limit, "spent_month_usd": round(spent + run_cost, 4)}
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    with open(a.log, "a", encoding="utf-8", newline="\n") as f:
        f.write(NFX.redact(json.dumps(stats, sort_keys=True, ensure_ascii=False, separators=(",", ":")), key) + "\n")
    print(f"notes_summarize [{mode}]: {stats['scheduled']} scheduled, {stats['recaps_due']} recaps due, "
          f"{stats['skipped_fresh']} unchanged, "
          f"{stats['llm_calls']} calls, {stats['written']} written, {len(stats['rejected'])} rejected, "
          f"{len(stats['failed'])} failed, {stats['deferred_budget']} over budget; ~${stats['est_cost_usd']:.4f} "
          f"(month ${stats['budget']['spent_month_usd']:.4f} of ${limit:.2f})")
    called = stats["llm_calls"]
    return 1 if strict and called >= 3 and len(stats["failed"]) * 2 > called else 0


if __name__ == "__main__":
    sys.exit(main())
