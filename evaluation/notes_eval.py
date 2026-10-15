#!/usr/bin/env python3
"""
notes_eval.py -- how often each matchup-notes rule fires on the last 12 months
of S/A series, and how that rate moves with its threshold. Writes
data/notes_eval_report.json and prints a table.

INFORMATIONAL, NOT A SIGNAL TEST
  Notes are display context (notes.py); none is a model input. The report
  shows fire rates so thresholds stay readable (a rule that fires on most
  series says nothing; one that never fires is cut). The win rate beside each
  rule is descriptive only: it is not held out, not corrected for strength
  and must never be read as a reason to give a note predictive weight (the
  tenure lesson: that needs its own pre-registered held-out test).

Each series dated D is evaluated as of D (cutoff = start of D, the notes'
own point-in-time rule), from data files only (no bundle needed; series carry
no stage here). Web-fact rules (STAND_IN, HOT_PLAYER, COLD_PLAYER,
ROSTER_RUMOR) are live-forward: they have no history to fire on and are
reported as such, not cut.

USAGE (repo root):  python evaluation/notes_eval.py [--days 365]
Stdlib only, deterministic.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import backtest as bt  # noqa: E402
import notes as NT  # noqa: E402
import notes_data as ND  # noqa: E402

OUT = os.path.join(paths.DATA, "notes_eval_report.json")
WEB_RULES = ("STAND_IN", "HOT_PLAYER", "COLD_PLAYER", "ROSTER_RUMOR")
SWEEPS = {
    "short_rest_days": [2, 3, 4, 5, 7],
    "heavy_min_series": [6, 7, 8, 9, 10],
    "cold_rest_days": [14, 21, 28, 35, 45],
    "tenure_stable_days": [90, 120, 180, 270, 365],
    "tenure_new_min": [1, 2, 3],
    "new_addition_days": [21, 30, 45, 60, 90],
    "streak_min": [3, 4, 5, 6],
    "revenge_max_days": [60, 90, 180, 365],
}


def side_metrics(inp, team, opp, day, event):
    """Raw quantities behind the bundle-derived rules for one side of a series."""
    rows = inp.prior(team, day)
    m = {"rest": None, "same_event": None, "n14": 0, "streak": 0, "streak_won": None,
         "together": None, "new": 0, "newest_days": None, "revenge": None}
    if rows:
        last = rows[-1]
        m["rest"] = (day - last[0]).days
        m["same_event"] = last[1] == event
        m["n14"] = sum(1 for r in rows if (day - r[0]).days <= NT.CONFIG["heavy_window_days"])
        won = last[4]
        n = 0
        for r in reversed(rows):
            if r[4] != won:
                break
            n += 1
        m["streak"], m["streak_won"] = n, won
    s = inp.tenure_hist.state(team, day)
    if s["known"] and s["five"]:
        starts = [v[0] for v in s["starts"].values() if v[1] and v[0]]
        m["new"] = sum(1 for st in starts if (day - bt._d(st)).days <= NT.CONFIG["tenure_new_days"])
        m["newest_days"] = min(((day - bt._d(st)).days for st in starts), default=None)
        since, _, _ = NT.five_run(inp.tenure_hist, team, day, set(s["five"]))
        m["together"] = (day - since).days if since else None
    x, y = sorted((team, opp))
    h = [r for r in inp.h2h.get((x, y), []) if r[0] < day]
    if h and h[-1][3] != team:
        m["revenge"] = (day - h[-1][0]).days
    return m


def fires(m, cfg):
    """Rule -> bool for one side, at thresholds `cfg` (mirrors notes.py)."""
    r = m["rest"]
    return {
        "SHORT_REST": r is not None and 0 <= r <= cfg["short_rest_days"] and not m["same_event"],
        "HEAVY_SCHEDULE": m["n14"] >= cfg["heavy_min_series"],
        "COLD_REST": r is not None and r >= cfg["cold_rest_days"],
        "TENURE_STABLE": m["together"] is not None and m["together"] >= cfg["tenure_stable_days"] and m["new"] == 0,
        "TENURE_NEW": m["new"] >= cfg["tenure_new_min"],
        "NEW_ADDITION": 0 < m["new"] < cfg["tenure_new_min"] and m["newest_days"] <= cfg["new_addition_days"],
        "STREAK": m["streak"] >= cfg["streak_min"],
        "REVENGE": m["revenge"] is not None and m["revenge"] <= cfg["revenge_max_days"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="matchup-notes rule hit rates (informational)")
    ap.add_argument("--days", type=int, default=365, help="window: series in the last N days")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)

    matches, lineups, squads, overrides, facts = ND.load_inputs()
    as_of = max(bt._d(m["date"]) for m in matches) + timedelta(days=1)
    sec = ND.section(as_of, matches, [], [], {}, [], lineups, squads, overrides, facts)
    bundle = {"notes": sec}
    inp = NT.inputs_of(bundle)
    lo = as_of - timedelta(days=a.days)
    window = sorted((m for m in matches if bt._d(m["date"]) >= lo), key=lambda m: (m["date"], m["team_a"]))

    sides, shown, shown_n, wins = [], Counter(), Counter(), defaultdict(lambda: [0, 0])
    per_series, mismatch = [], []
    for m in window:
        d = bt._d(m["date"])
        for t, o in ((m["team_a"], m["team_b"]), (m["team_b"], m["team_a"])):
            sm = side_metrics(inp, t, o, d, m["event"])
            sides.append(sm)
            # the mirror must agree with the engine's own pre-cap candidates
            got = {n["rule"] for n in NT._team_candidates(inp, t, m["date"], d, m["event"])
                   + NT.h2h_notes(inp, t, o, d, d, d) if n["team_scope"] == t} - set(WEB_RULES)
            want = {r for r, hit in fires(sm, NT.CONFIG).items() if hit}
            if got != want:
                mismatch.append(f"{m['date']} {t} vs {o}: engine {sorted(got)} != mirror {sorted(want)}")
            for rule, hit in fires(sm, NT.CONFIG).items():
                if hit:
                    w = wins[rule]
                    w[0] += 1
                    w[1] += m["winner"] == t
        out = NT.get_notes(m["team_a"], m["team_b"], m["date"], bundle, match_day=m["date"], event=m["event"])
        rs = [n["rule"] for n in out["notes"]]
        per_series.append(len(rs))
        shown.update(set(rs))
        shown_n.update(rs)

    n_sides, n_series = len(sides), len(window)
    base = {r: sum(fires(s, NT.CONFIG)[r] for s in sides) for r in fires(sides[0], NT.CONFIG)} if sides else {}
    rules = {}
    for r in NT.RULES:
        if r in WEB_RULES:
            rules[r] = {"status": "live-forward (web facts begin at deployment; no history to fire on)",
                        "fired_sides": None, "shown_series": shown.get(r, 0)}
            continue
        k = base.get(r, 0)
        rules[r] = {"fired_sides": k, "fire_rate_sides": round(k / n_sides, 4) if n_sides else None,
                    "shown_series": shown.get(r, 0),
                    "shown_rate_series": round(shown.get(r, 0) / n_series, 4) if n_series else None,
                    "status": "fires" if k else "NEVER FIRES: cut",
                    "descriptive_win_rate": {"n": wins[r][0], "wins": wins[r][1],
                                             "rate": round(wins[r][1] / wins[r][0], 4) if wins[r][0] else None}}
    sweeps = {}
    rule_of = {"short_rest_days": "SHORT_REST", "heavy_min_series": "HEAVY_SCHEDULE", "cold_rest_days": "COLD_REST",
               "tenure_stable_days": "TENURE_STABLE", "tenure_new_min": "TENURE_NEW", "new_addition_days": "NEW_ADDITION", "streak_min": "STREAK",
               "revenge_max_days": "REVENGE"}
    for key, vals in SWEEPS.items():
        row = {}
        for v in vals:
            cfg = dict(NT.CONFIG, **{key: v})
            k = sum(fires(s, cfg)[rule_of[key]] for s in sides)
            row[str(v)] = round(k / n_sides, 4) if n_sides else None
        sweeps[key] = {"rule": rule_of[key], "chosen": NT.CONFIG[key], "fire_rate_sides": row}
    hist = Counter(per_series)
    report = {
        "note": "Informational: fire rates of display-only notes. Not a signal test; the win rates are "
                "descriptive (not held out, not strength-corrected) and no note is a model input.",
        "window": {"from": lo.isoformat(), "to": (as_of - timedelta(days=1)).isoformat(),
                   "series": n_series, "sides": n_sides},
        "config": dict(NT.CONFIG),
        "rules": rules,
        "notes_per_series": {str(k): hist[k] for k in sorted(hist)},
        "mean_notes_per_series": round(sum(per_series) / n_series, 3) if n_series else None,
        "shown_notes_by_rule": dict(sorted(shown_n.items())),
        "sweeps": sweeps,
        "mirror_mismatches": len(mismatch),
        "facts": {"form": len(inp.form), "news": len(inp.news), "updated_at": inp.facts_updated_at},
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print(f"{n_series} series ({report['window']['from']} .. {report['window']['to']}), {n_sides} sides; "
          f"mean {report['mean_notes_per_series']} notes shown per series")
    print(f"{'rule':16} {'fires (sides)':>14} {'shown (series)':>15}  status")
    for r in NT.RULES:
        x = rules[r]
        fr = "-" if x["fired_sides"] is None else f"{x['fire_rate_sides']:.1%}"
        sr = "-" if x.get("shown_rate_series") is None else f"{x['shown_rate_series']:.1%}"
        print(f"{r:16} {fr:>14} {sr:>15}  {x['status']}")
    for key, s in sweeps.items():
        print(f"  {key:20} chosen {s['chosen']:>4}: " + "  ".join(f"{v}->{p:.1%}" for v, p in s["fire_rate_sides"].items()))
    print(f"engine vs mirror: {len(mismatch)} mismatched sides")
    for x in mismatch[:10]:
        print("  " + x)
    print(f"wrote {a.out}")
    return report


if __name__ == "__main__":
    main()
