#!/usr/bin/env python3
"""
lower_tier_check.py -- should B-tier (and lower) results feed the ratings?

WHY: B-tier events are not shown on the site and get no predictions, but
S/A teams play them between S/A events (FaZe Clan at DraculaN Season 6,
Fnatic, Eternal Fire, ...). Their results are invisible to the ratings. This
test asks whether using them as rating DATA, never as events, makes the S/A
calls better. Data: data/matches_lower.json (python data/collect_liquipedia.py
--offline --lower; 2026 B-tier main events from the raw Liquipedia cache, the
pages are in data/raw/lp_titles_lower.txt). Lower-tier coverage starts in
January 2026; before that every variant is identical to the current engine.

PRE-REGISTERED (2026-10-08, written before the first run).

  Engine: predictor.py CONFIG as shipped (weights fitted on data up to
  2025-10-08, temperature 1), nothing refitted. Only the history changes.
  Day order: features for every S/A series on day D are built from series
  (S/A and lower) dated < D; then day D is folded in (S/A first, then lower).

  Variants (w = Elo step multiplier for a lower-tier series, on top of the
  usual K x best-of weight):
    base         lower-tier series not ingested (the current engine)
    elo_w        w in {0, 0.25, 0.5, 1.0}: lower-tier series move Elo only;
                 30-day form, last-5, volatility, maps, h2h stay S/A-only
    linked_w     w in {0.25, 0.5, 1.0}: as elo_w, but only lower-tier series
                 where at least one team has >= 1 prior S/A series (option b)
    form_w       w in {0.25, 0.5, 1.0}: as elo_w, and the series also count
                 in form30 / n30 / opp_rating30 / form5 (maps, h2h,
                 volatility stay S/A-only)
  The prior-series counts that gate the rows (and the newcomer offset) stay
  S/A-only in every variant, so every variant scores the same rows.

  Rows. PRIMARY: S/A-tier BO3 series dated 2026-02-01 .. end of data where
  both teams have >= 5 prior S/A series (one month of lower-tier history
  first). SECONDARY: the backtest.py test split (last 40% of the same eval
  rows by date), and NEWCOMER rows (S/A BO3 in the primary window where a
  team has < 5 prior S/A series; the newcomer offset was tuned without
  lower-tier data, so this view is informative only).

  Metrics: log loss (primary), Brier, accuracy, ECE10 favourite view and
  the walkforward.py favourite buckets; paired-bootstrap 95% CI (2000
  resamples, seed 12345) of each variant minus base. Elo-only probabilities
  of each variant are shown too (model-free view).

  Decision rule: wire a variant in only if, on the PRIMARY rows,
    L1 its log loss CI vs base excludes 0 (below 0);
    L2 it is not worse than base on either half of the primary window;
    L3 its favourite-view ECE10 is <= base + 0.005 and every bucket with
       n >= 30 passes the walkforward.py rule.
  Ten variants are compared, so a single CI just below 0 among them is weak
  evidence; the report says so. If several pass, the smallest w wins.

  Leakage: the script re-runs the walkforward scramble/truncate idea on the
  lower-tier list (outcomes of lower-tier series dated >= D inverted, or
  series dated > D removed) for the variant with w = 1 and form on, and
  requires day-D features to be unchanged.

USAGE (from D:/Face-Off):  python scripts/lower_tier_check.py [--quick]
Writes data/lower_tier_report.json. Stdlib only, deterministic.
"""
import copy
import json
import os
import sys
from datetime import timedelta
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402
import walkforward as wf  # noqa: E402

PRIMARY_START = "2026-02-01"
LOWER_PATH = os.path.join(ROOT, "data", "matches_lower.json")
OUT = os.path.join(ROOT, "data", "lower_tier_report.json")
VARIANTS = ([("base", None, "none")] + [(f"elo_{w}", w, "elo") for w in (0.0, 0.25, 0.5, 1.0)]
            + [(f"linked_{w}", w, "linked") for w in (0.25, 0.5, 1.0)]
            + [(f"form_{w}", w, "form") for w in (0.25, 0.5, 1.0)])


def load_lower(path=LOWER_PATH):
    with open(path, encoding="utf-8") as f:
        ms = json.load(f)
    for i, m in enumerate(ms):
        m["id"] = ("L", i)
    return sorted(ms, key=lambda m: (m["date"], m["id"][1]))


class LowerHistory(bt.History):
    """bt.History plus lower-tier series: Elo moves by K x w; with form=True
    they also count in form30 / n30 / opp_rating30 / form5. games[] (prior
    S/A series counts, maps, h2h, volatility) is never touched."""

    def __init__(self, w, form=False):
        super().__init__()
        self.w, self.form = w, form
        self.lower_games = {}

    def add_lower(self, m):
        a, b = m["team_a"], m["team_b"]
        self.clock = bt._d(m["date"])
        ea, eb = self.elo[a], self.elo[b]
        exp_a = bt.elo_expected(ea, eb)
        a_won = m["winner"] == a
        if self.form:
            for t, oe, won in ((a, eb, a_won), (b, ea, not a_won)):
                self.lower_games.setdefault(t, []).append({"date": self.clock, "won": won, "opp_elo": oe})
        na, nb = bt.elo_update(ea, eb, a_won, m.get("best_of", 3), k=bt.ELO_K * self.w)
        self.elo[a], self.elo[b] = na, nb

    def team_feats(self, team, dt):
        out = super().team_feats(team, dt)
        if not self.form or team not in self.lower_games:
            return out
        g = sorted(self.games[team] + self.lower_games[team], key=lambda r: r["date"])
        w30 = [r for r in g if dt - timedelta(days=30) <= r["date"] < dt]
        if w30:
            out["form30"] = sum(r["won"] for r in w30) / len(w30)
            out["n30"] = len(w30)
            out["opp_rating30"] = sum(bt.elo_to_rating(r["opp_elo"]) for r in w30) / len(w30)
        last5 = g[-5:]
        if last5:
            out["form5"] = sum(r["won"] for r in last5) / len(last5)
        return out


def build(top, lower, w=None, mode="none"):
    """Rows for the S/A series only, like bt.build_dataset, with lower-tier
    series folded in after each day's S/A series."""
    h: Any = bt.History() if w is None else LowerHistory(w, form=(mode == "form"))
    by_day = {}
    for m in sorted(top, key=lambda m: (m["date"], m.get("id", 0))):
        by_day.setdefault(m["date"], ([], []))[0].append(m)
    if w is not None:
        for m in lower:
            by_day.setdefault(m["date"], ([], []))[1].append(m)
    out = []
    for day in sorted(by_day):
        sa, lo = by_day[day]
        for m in sa:
            inp, meta = h.features(m)
            out.append({"match": m, "input": inp, "meta": meta})
        for m in sa:
            h.add(m)
        for m in lo:
            if mode == "linked" and not (h.n_prior(m["team_a"]) or h.n_prior(m["team_b"])):
                continue
            h.add_lower(m)
    return out, h


def predict(rows):
    return [pr.predict_match(dict(r["input"]))["p_a_exact"] for r in rows]


def summary(ps, ys):
    m = wf.metrics(ps, ys)
    return {k: m[k] for k in ("n", "accuracy", "accuracy_wilson95", "brier", "log_loss", "ece10_fav", "fav_buckets")}


def leakage(top, lower, days):
    """Scramble / truncate the LOWER-tier list from day D on (w=1, form on):
    day-D features must not change."""
    problems = []
    clean = {r["match"]["id"]: (r["input"], r["meta"]) for r in build(top, lower, 1.0, "form")[0]}
    for day in days:
        scr = copy.deepcopy(lower)
        for m in scr:
            if m["date"] >= day:
                m["winner"] = m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
        for variant in (build(top, scr, 1.0, "form")[0],
                        build([m for m in top if m["date"] <= day], [m for m in lower if m["date"] <= day], 1.0, "form")[0]):
            for r in variant:
                if r["match"]["date"] == day and (r["input"], r["meta"]) != clean[r["match"]["id"]]:
                    problems.append(f"{day} {r['match']['team_a']} vs {r['match']['team_b']}")
    # teeth: day D's own lower-tier results folded in BEFORE the features must change something
    teeth = False
    for day in days:
        early = [dict(m, date=(bt._d(m["date"]) - timedelta(days=1)).isoformat()) if m["date"] == day else m
                 for m in lower]
        moved = {r["match"]["id"]: (r["input"], r["meta"]) for r in build(top, early, 1.0, "form")[0]
                 if r["match"]["date"] == day}
        if any(moved[k] != clean[k] for k in moved):
            teeth = True
            break
    return problems, teeth


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    top = bt.load_matches()
    lower = load_lower()
    print(f"S/A series {len(top)}, lower-tier series {len(lower)} ({lower[0]['date']} .. {lower[-1]['date']})")
    base_rows, _ = build(top, lower)
    ev, _ = bt.select_eval(base_rows, bt.MIN_HISTORY)
    _, test = bt.chrono_split(ev, 0.6)
    ids = {"primary": [r["match"]["id"] for r in ev if r["match"]["date"] >= PRIMARY_START],
           "backtest_test": [r["match"]["id"] for r in test]}
    ids["newcomer"] = [r["match"]["id"] for r in base_rows if r["match"].get("best_of") == 3
                       and r["match"]["date"] >= PRIMARY_START
                       and min(r["meta"]["hist_a"], r["meta"]["hist_b"]) < bt.MIN_HISTORY]
    half = len(ids["primary"]) // 2
    while 0 < half < len(ids["primary"]) and \
            [r for r in ev if r["match"]["id"] == ids["primary"][half]][0]["match"]["date"] == \
            [r for r in ev if r["match"]["id"] == ids["primary"][half - 1]][0]["match"]["date"]:
        half += 1
    ids["primary_h1"], ids["primary_h2"] = ids["primary"][:half], ids["primary"][half:]
    rep = {"rows": {k: len(v) for k, v in ids.items()}, "primary_start": PRIMARY_START,
           "lower_series": len(lower), "variants": {}}
    preds, elo_p, ys = {}, {}, None
    ratings = {}
    for name, w, mode in VARIANTS:
        rows, h = build(top, lower, w, mode)
        byid = {r["match"]["id"]: r for r in rows}
        preds[name], elo_p[name] = {}, {}
        for view, lst in ids.items():
            rr = [byid[i] for i in lst]
            preds[name][view] = predict(rr)
            elo_p[name][view] = [r["meta"]["p_elo"] for r in rr]
        if ys is None:
            ys = {view: [wf.label(byid[i]) for i in lst] for view, lst in ids.items()}
        ratings[name] = {t: round(bt.elo_to_rating(h.elo[t]), 4) for t in
                         ("FaZe Clan", "Team Vitality", "Fnatic", "Eternal Fire", "Natus Vincere")}
    assert ys is not None
    for name, _, _ in VARIANTS:
        v = {}
        for view in ids:
            ps, y = preds[name][view], ys[view]
            v[view] = summary(ps, y)
            v[view]["elo_only_log_loss"] = bt.log_loss(elo_p[name][view], y)
            if name != "base":
                v[view]["vs_base"] = wf.paired(ps, preds["base"][view], y)
                v[view]["elo_only_vs_base"] = wf.paired(elo_p[name][view], elo_p["base"][view], y)
        if name != "base":
            pr_ = v["primary"]
            base_ece = wf.fav_ece(preds["base"]["primary"], ys["primary"])
            v["decision"] = {
                "L1_ci_below_0": pr_["vs_base"]["log_loss"]["ci95"][1] < 0,
                "L2_both_halves_not_worse": all(v[h]["log_loss"] <= rep_base_ll(preds, ys, h)
                                                for h in ("primary_h1", "primary_h2")),
                "L3_calibration": (pr_["ece10_fav"] <= base_ece + 0.005
                                   and all(b.get("pass", True) for b in pr_["fav_buckets"])),
            }
            v["decision"]["ALL"] = all(v["decision"].values())
        rep["variants"][name] = v
    rep["end_of_data_ratings"] = ratings
    days = sorted({m["date"] for m in lower if m["date"] >= "2026-01-15"})
    days = days[::max(1, len(days) // (6 if "--quick" in argv else 20))]
    problems, teeth = leakage(top, lower, days)
    rep["leakage"] = {"days_tested": len(days), "problems": problems[:20], "n_problems": len(problems),
                      "teeth": teeth, "pass": not problems and teeth}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(render(rep))
    print("wrote data/lower_tier_report.json")


def rep_base_ll(preds, ys, view):
    return bt.log_loss(preds["base"][view], ys[view])


def render(rep):
    L = [f"rows: {rep['rows']}", ""]
    for view in ("primary", "primary_h1", "primary_h2", "backtest_test", "newcomer"):
        L.append(f"[{view}] n={rep['rows'][view]}")
        L.append("  variant       acc    Brier   logloss  ECEfav   d logloss vs base [95% CI]      elo-only LL")
        for name, v in rep["variants"].items():
            b = v[view]
            d = b.get("vs_base", {}).get("log_loss")
            ds = f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else " " * 27
            L.append(f"  {name:12s} {b['accuracy']:.3f}  {b['brier']:.4f}  {b['log_loss']:.4f}  {b['ece10_fav']:.4f}"
                     f"   {ds}   {b['elo_only_log_loss']:.4f}")
        L.append("")
    L.append("Favourite buckets, primary rows (stated -> actual, n, pass):")
    for name in ("base", "elo_0.25", "elo_0.5", "elo_1.0", "form_0.5"):
        b = rep["variants"][name]["primary"]["fav_buckets"]
        L.append(f"  {name:10s} " + "  ".join(f"{x['bucket']} {x['stated']:.3f}->{x['actual']:.3f} n={x['n']}"
                                               f"{'' if x.get('pass', True) else ' FAIL'}" for x in b if x["n"]))
    L.append("")
    L.append("Decision (L1 CI<0, L2 both halves, L3 calibration):")
    for name, v in rep["variants"].items():
        if "decision" in v:
            L.append(f"  {name:12s} " + " ".join(f"{k}={'y' if x else 'n'}" for k, x in v["decision"].items()))
    L.append("")
    L.append("End-of-data proxy ratings: " + json.dumps(rep["end_of_data_ratings"]["base"]))
    for name in ("elo_0.25", "elo_0.5", "elo_1.0"):
        L.append(f"  {name}: " + json.dumps(rep["end_of_data_ratings"][name]))
    lk = rep["leakage"]
    L.append(f"Leakage (lower-tier scramble/truncate, w=1 + form): {lk['n_problems']} problems on "
             f"{lk['days_tested']} days; teeth {'yes' if lk['teeth'] else 'NO'} -> {'PASS' if lk['pass'] else 'FAIL'}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
