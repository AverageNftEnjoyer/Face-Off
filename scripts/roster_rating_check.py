#!/usr/bin/env python3
"""
roster_rating_check.py -- does a ROSTER-AWARE rating (and/or lower-tier
results as Elo data) make the S/A calls better?

WHY: Elo carries a team's results forward no matter who played them. When
an organisation replaces most of its five, the results earned by the old
roster say little about the new one. Per-match lineup features (stand-in
flag, continuity) did not add held-out signal as separate engine factors
(lineup_eval.py); this tests the other route: change the RATING itself.

DATA
  data/matches.json         S/A series (the engine's history).
  data/matches_lower.json   lower-tier series 2025-01 .. 2026-10 (B-tier
                            main events + stages, closed qualifiers of S/A
                            events, S/A events missing from matches.json;
                            python data/collect_liquipedia.py --lower).
  data/event_lineups.json   per-team, per-event lineups (python
                            data/event_lineups.py): the event's participant
                            section, else team-page squad dates.

PRE-REGISTERED (2026-10-08, written before the first run; not changed after)

  Engine: predictor.py CONFIG as shipped (weights fitted on data up to
  2025-10-08, temperature 1); nothing refitted. Only the history changes.
  Day order (as lower_tier_check.py): features of every S/A series on day D
  come from series dated < D, plus the team's OWN lineup at that series (a
  pre-match fact: the event's announced participant list); then day D is
  folded in (S/A first, then lower-tier).

  Continuity. Per team, the lineups of its last W = 10 folded series with a
  known lineup. At a series with known lineup L, if the team has >= 3 such
  lineups, c = 5 * sum_{p in L} share(p) / max(5, |L|), share(p) = fraction
  of those lineups that include p. c = 5: the same five as the rating's
  recent history; c = 2: about two of five carried over.

  Variants
    base         the current engine (S/A history only, no roster logic).
    lower025     lower-tier series move Elo with K x 0.25 (x best-of weight),
                 nothing else (the single pre-registered lower-tier
                 candidate; lower_tier_check.py's best on 2026-only data).
    R1_reset     at a series where c <= 2.0, BEFORE its features are built,
                 the team's Elo is regressed halfway to 1500:
                 elo <- 1500 + 0.5 (elo - 1500); its lineup window is then
                 reset to the new lineup (so the same roster does not
                 trigger again).
    R2_graded    regression share r = 0.2 x max(0, 4 - c) (c=3 -> 0.2,
                 c=2 -> 0.4, c=0 -> 0.8) whenever r > 0; window reset as R1.
    R3_reset_k   R1 plus K x 1.5 for that team's next 8 series.
    R1_lower025  R1 and lower025 together (lower-tier series also carry
                 lineups and can trigger the reset).
  In R1-R3 the lineup window is fed by the series the variant folds in
  (S/A only; plus lower-tier in R1_lower025). Prior-series counts that gate
  the rows, form, maps, h2h, volatility stay S/A-only in every variant, so
  every variant scores the same rows.

  Rows (S/A BO3, both teams >= 5 prior S/A series, the walkforward.py rule)
    PRIMARY      dated after 2025-10-08 (clean of the shipped weights'
                 fit; nine months of lower-tier history first).
    halves       PRIMARY split in two by date (never inside a day).
    ROSTER       PRIMARY rows where at least one team "recently changed
                 roster": a series of that team in the 90 days before D, or
                 D itself, had c <= 3.0 (>= 2 new players), measured on all
                 its S/A + lower-tier series with a plain sliding window
                 (same for every variant).
    EARLY        2024-07-01 .. 2025-10-08 (in-sample for the engine's
                 weights; informative only).

  Metrics: log loss (primary), Brier, accuracy, favourite-view ECE10 and
  buckets (walkforward.py); paired-bootstrap 95% CI (2000 resamples, seed
  12345) of each variant minus base; Elo-only probabilities too.

  Decision rule. A variant ships only if, on PRIMARY:
    D1 its log loss CI vs base is entirely below 0;
    D2 its log loss is <= base's on both halves;
    D3 favourite-view ECE10 <= base + 0.005, and no bucket with n >= 30
       that passes the walkforward.py rule for base fails for it (base
       itself already misses the 70-80% bucket, so an absolute rule would
       reject everything).
  Five candidates are compared: one CI just below 0 is weak evidence and
  the report says so. If several pass, the lowest PRIMARY log loss wins.
  The ROSTER rows are reported, not part of the rule.

  Leakage: (a) lower-tier outcomes dated >= D inverted, or series dated > D
  removed; (b) lineups of every series dated > D replaced with made-up
  players: day-D features of R1_lower025 must not change. Teeth: a variant
  that reads each team's NEXT series lineup must change day-D features.

USAGE (from D:/Face-Off):  python scripts/roster_rating_check.py [--quick]
Writes data/roster_rating_report.json. Stdlib only, deterministic.
"""
import copy
import json
import os
import sys
from collections import defaultdict, deque
from datetime import timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backtest as bt  # noqa: E402
import predictor as pr  # noqa: E402
import walkforward as wf  # noqa: E402
import event_lineups as EL  # noqa: E402

PRIMARY_START = "2025-10-09"
EARLY = ("2024-07-01", "2025-10-08")
LOWER_PATH = os.path.join(ROOT, "data", "matches_lower.json")
OUT = os.path.join(ROOT, "data", "roster_rating_report.json")
W_HIST = 10
MIN_REF = 3
ROSTER_ROW_C = 3.0
ROSTER_ROW_DAYS = 90

VARIANTS = [
    ("base", {}),
    ("lower025", {"lower": 0.25}),
    ("R1_reset", {"roster": "reset"}),
    ("R2_graded", {"roster": "graded"}),
    ("R3_reset_k", {"roster": "reset", "kmult": 1.5, "kn": 8}),
    ("R1_lower025", {"roster": "reset", "lower": 0.25}),
]


def load_lower(path=LOWER_PATH):
    with open(path, encoding="utf-8") as f:
        ms = json.load(f)
    for i, m in enumerate(ms):
        m["id"] = ("L", i)
    return sorted(ms, key=lambda m: (m["date"], m["id"][1]))


def continuity(window, L):
    if L is None or len(window) < MIN_REF:
        return None
    share = defaultdict(float)
    for lu in window:
        for p in lu:
            share[p] += 1.0 / len(window)
    return 5.0 * sum(share[p] for p in L) / max(5, len(L))


def regression_share(cfg, c):
    if c is None:
        return 0.0
    if cfg.get("roster") == "reset":
        return 0.5 if c <= 2.0 else 0.0
    if cfg.get("roster") == "graded":
        return min(1.0, 0.2 * max(0.0, 4.0 - c))
    return 0.0


class RosterHistory(bt.History):
    """bt.History with optional lower-tier Elo ingestion (K x w, Elo only) and
    optional roster-aware Elo regression. `lineup_of(m, team)` returns the
    team's lineup at series m (a pre-match fact) or None."""

    def __init__(self, cfg, lineup_of):
        super().__init__()
        self.cfg = cfg
        self.lineup_of = lineup_of
        self.window = defaultdict(lambda: deque(maxlen=W_HIST))
        self.boost = defaultdict(int)
        self.log = []                    # (date, team, c, elo_before, elo_after, tier)
        self._lu = {}

    def _check(self, m, team, tier):
        L = self.lineup_of(m, team)
        self._lu[(id(m), team)] = L
        if not self.cfg.get("roster") or L is None:
            return
        c = continuity(self.window[team], L)
        r = regression_share(self.cfg, c)
        if r > 0:
            before = self.elo[team]
            self.elo[team] = bt.ELO_INIT + (1 - r) * (before - bt.ELO_INIT)
            self.window[team].clear()
            self.window[team].append(tuple(L))
            if self.cfg.get("kn"):
                self.boost[team] = self.cfg["kn"]
            self.log.append((m["date"], team, round(c, 2), round(before, 1), round(self.elo[team], 1), tier))

    def _after(self, m, team, before, after_val):
        if self.boost[team] > 0:
            self.elo[team] = before + (after_val - before) * self.cfg.get("kmult", 1.0)
            self.boost[team] -= 1
        L = self._lu.pop((id(m), team), None)
        if self.cfg.get("roster") and L is not None:
            self.window[team].append(tuple(L))

    def features(self, m):
        dt = m["date"]
        self.clock = bt._d(dt)
        for team in (m["team_a"], m["team_b"]):
            if (id(m), team) not in self._lu:
                self._check(m, team, m.get("tier"))
        return super().features(m)

    def add(self, m):
        a, b = m["team_a"], m["team_b"]
        for t in (a, b):
            if (id(m), t) not in self._lu:
                self._check(m, t, m.get("tier"))
        ea, eb = self.elo[a], self.elo[b]
        super().add(m)
        self._after(m, a, ea, self.elo[a])
        self._after(m, b, eb, self.elo[b])

    def add_lower(self, m):
        a, b = m["team_a"], m["team_b"]
        self.clock = bt._d(m["date"])
        for t in (a, b):
            self._check(m, t, "lower")
        ea, eb = self.elo[a], self.elo[b]
        na, nb = bt.elo_update(ea, eb, m["winner"] == a, m.get("best_of", 3), k=bt.ELO_K * self.cfg["lower"])
        self.elo[a], self.elo[b] = na, nb
        self._after(m, a, ea, na)
        self._after(m, b, eb, nb)


def make_lineup_of(store, override=None):
    def lineup_of(m, team):
        if override is not None:
            return override(m, team)
        return EL.lineup(store, team, EL.page_of(m), m["date"])
    return lineup_of


def build(top, lower, cfg, lineup_of):
    h = RosterHistory(cfg, lineup_of)
    by_day = {}
    for m in sorted(top, key=lambda m: (m["date"], m.get("id", 0))):
        by_day.setdefault(m["date"], ([], []))[0].append(m)
    if cfg.get("lower"):
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
            h.add_lower(m)
    return out, h


def roster_change_days(top, lower, lineup_of):
    """team -> sorted dates of series where its sliding-window continuity
    (all S/A + lower series, no reset) was <= ROSTER_ROW_C."""
    win = defaultdict(lambda: deque(maxlen=W_HIST))
    out = defaultdict(list)
    for m in sorted(top + lower, key=lambda m: (m["date"], 0 if isinstance(m["id"], int) else 1)):
        for t in (m["team_a"], m["team_b"]):
            L = lineup_of(m, t)
            c = continuity(win[t], L)
            if c is not None and c <= ROSTER_ROW_C:
                out[t].append(m["date"])
            if L is not None:
                win[t].append(tuple(L))
    return out


def predict(rows):
    return [pr.predict_match(dict(r["input"]))["p_a_exact"] for r in rows]


def summary(ps, ys):
    m = wf.metrics(ps, ys)
    return {k: m[k] for k in ("n", "accuracy", "accuracy_wilson95", "brier", "log_loss", "ece10_fav", "fav_buckets")}


def split_halves(rows):
    half = len(rows) // 2
    while 0 < half < len(rows) and rows[half]["match"]["date"] == rows[half - 1]["match"]["date"]:
        half += 1
    return rows[:half], rows[half:]


def leakage(top, lower, store, days):
    cfg = dict(VARIANTS)["R1_lower025"]
    lo_of = make_lineup_of(store)
    clean = {r["match"]["id"]: (r["input"], r["meta"]) for r in build(top, lower, cfg, lo_of)[0]}
    problems = []
    for day in days:
        scr = copy.deepcopy(lower)
        for m in scr:
            if m["date"] >= day:
                m["winner"] = m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
        fake = make_lineup_of(store, lambda m, t, day=day: ([f"fake{i}_{t}" for i in range(5)]
                                                            if m["date"] > day else
                                                            EL.lineup(store, t, EL.page_of(m), m["date"])))
        for rows in (build(top, scr, cfg, lo_of)[0],
                     build([m for m in top if m["date"] <= day], [m for m in lower if m["date"] <= day], cfg, lo_of)[0],
                     build(top, lower, cfg, fake)[0]):
            for r in rows:
                if r["match"]["date"] == day and (r["input"], r["meta"]) != clean[r["match"]["id"]]:
                    problems.append(f"{day} {r['match']['team_a']} vs {r['match']['team_b']}")
    # teeth: each team's NEXT series lineup used instead of its own
    seq = defaultdict(list)
    for m in sorted(top + lower, key=lambda m: (m["date"], str(m["id"]))):
        for t in (m["team_a"], m["team_b"]):
            seq[t].append(m)
    nxt = {}
    for t, ms in seq.items():
        for i, m in enumerate(ms):
            nxt[(id(m), t)] = ms[i + 1] if i + 1 < len(ms) else m
    peek = make_lineup_of(store, lambda m, t: EL.lineup(store, t, EL.page_of(nxt[(id(m), t)]),
                                                         nxt[(id(m), t)]["date"]))
    dset = set(days)
    teeth = any((r["input"], r["meta"]) != clean[r["match"]["id"]]
                for r in build(top, lower, cfg, peek)[0] if r["match"]["date"] in dset)
    return problems, teeth


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    top = bt.load_matches()
    lower = load_lower()
    store = EL.load()
    lineup_of = make_lineup_of(store)
    print(f"S/A series {len(top)}, lower-tier series {len(lower)} ({lower[0]['date']} .. {lower[-1]['date']})")
    base_rows, _ = build(top, lower, {}, lineup_of)
    ev, _ = bt.select_eval(base_rows, bt.MIN_HISTORY)
    prim = [r for r in ev if r["match"]["date"] >= PRIMARY_START]
    h1, h2 = split_halves(prim)
    early = [r for r in ev if EARLY[0] <= r["match"]["date"] <= EARLY[1]]
    chg = roster_change_days(top, lower, lineup_of)

    def recent(team, day):
        lo = (bt._d(day) - timedelta(days=ROSTER_ROW_DAYS)).isoformat()
        return any(lo <= d <= day for d in chg.get(team, []))

    roster_rows = [r for r in prim if recent(r["match"]["team_a"], r["match"]["date"])
                   or recent(r["match"]["team_b"], r["match"]["date"])]
    ids = {"primary": [r["match"]["id"] for r in prim], "primary_h1": [r["match"]["id"] for r in h1],
           "primary_h2": [r["match"]["id"] for r in h2], "roster": [r["match"]["id"] for r in roster_rows],
           "early": [r["match"]["id"] for r in early]}
    rep = {"rows": {k: len(v) for k, v in ids.items()}, "primary_start": PRIMARY_START,
           "lower_series": len(lower), "variants": {}, "resets": {}}
    preds, elo_p, ys = {}, {}, None
    for name, cfg in VARIANTS:
        rows, h = build(top, lower, cfg, lineup_of)
        byid = {r["match"]["id"]: r for r in rows}
        preds[name] = {v: predict([byid[i] for i in lst]) for v, lst in ids.items()}
        elo_p[name] = {v: [byid[i]["meta"]["p_elo"] for i in lst] for v, lst in ids.items()}
        if ys is None:
            ys = {v: [wf.label(byid[i]) for i in lst] for v, lst in ids.items()}
        if h.log:
            rep["resets"][name] = {"n": len(h.log), "n_sa": sum(1 for x in h.log if x[5] != "lower"),
                                   "examples": [x for x in h.log if x[0] >= "2025-01-01"][:400]}
        print(f"  built {name}: resets {len(h.log)}")
    for name, _ in VARIANTS:
        v = {}
        for view in ids:
            ps, y = preds[name][view], ys[view]
            v[view] = summary(ps, y)
            v[view]["elo_only_log_loss"] = bt.log_loss(elo_p[name][view], y)
            if name != "base":
                v[view]["vs_base"] = wf.paired(ps, preds["base"][view], y)
                v[view]["elo_only_vs_base"] = wf.paired(elo_p[name][view], elo_p["base"][view], y)
        if name != "base":
            pb = summary(preds["base"]["primary"], ys["primary"])
            base_pass = {b["bucket"]: b.get("pass", True) for b in pb["fav_buckets"] if b["n"] >= 30}
            pr_ = v["primary"]
            v["decision"] = {
                "D1_ci_below_0": pr_["vs_base"]["log_loss"]["ci95"][1] < 0,
                "D2_both_halves_not_worse": all(v[hh]["log_loss"] <= bt.log_loss(preds["base"][hh], ys[hh])
                                                for hh in ("primary_h1", "primary_h2")),
                "D3_calibration": (pr_["ece10_fav"] <= pb["ece10_fav"] + 0.005
                                   and all(b.get("pass", True) for b in pr_["fav_buckets"]
                                           if b["n"] >= 30 and base_pass.get(b["bucket"], False))),
            }
            v["decision"]["ALL"] = all(v["decision"].values())
        rep["variants"][name] = v
    passing = [n for n, v in rep["variants"].items() if v.get("decision", {}).get("ALL")]
    rep["winner"] = min(passing, key=lambda n: rep["variants"][n]["primary"]["log_loss"]) if passing else None
    days = sorted({m["date"] for m in top if m["date"] >= "2025-03-01"})
    days = days[::max(1, len(days) // (4 if "--quick" in argv else 12))]
    problems, teeth = leakage(top, lower, store, days)
    rep["leakage"] = {"days_tested": len(days), "n_problems": len(problems), "problems": problems[:20],
                      "teeth": teeth, "pass": not problems and teeth}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(render(rep))
    print("wrote data/roster_rating_report.json")


def render(rep):
    L = [f"rows: {rep['rows']}", ""]
    for view in ("primary", "primary_h1", "primary_h2", "roster", "early"):
        L.append(f"[{view}] n={rep['rows'][view]}")
        L.append("  variant        acc    Brier   logloss  ECEfav   d logloss vs base [95% CI]      elo-only LL")
        for name, v in rep["variants"].items():
            b = v[view]
            d = b.get("vs_base", {}).get("log_loss")
            ds = f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else " " * 27
            L.append(f"  {name:13s} {b['accuracy']:.3f}  {b['brier']:.4f}  {b['log_loss']:.4f}  {b['ece10_fav']:.4f}"
                     f"   {ds}   {b['elo_only_log_loss']:.4f}")
        L.append("")
    L.append("Favourite buckets, primary rows (stated -> actual, n):")
    for name, v in rep["variants"].items():
        L.append(f"  {name:12s} " + "  ".join(f"{x['bucket']} {x['stated']:.3f}->{x['actual']:.3f} n={x['n']}"
                                               f"{'' if x.get('pass', True) else ' FAIL'}"
                                               for x in v["primary"]["fav_buckets"] if x["n"]))
    L.append("")
    L.append("Decision (D1 CI<0, D2 both halves, D3 calibration):")
    for name, v in rep["variants"].items():
        if "decision" in v:
            L.append(f"  {name:13s} " + " ".join(f"{k}={'y' if x else 'n'}" for k, x in v["decision"].items()))
    L.append(f"winner: {rep['winner']}")
    for name, r in rep["resets"].items():
        L.append(f"resets {name}: {r['n']} ({r['n_sa']} at S/A series)")
    lk = rep["leakage"]
    L.append(f"Leakage (R1_lower025; lower scramble/truncate + future lineups faked): {lk['n_problems']} problems "
             f"on {lk['days_tested']} days; teeth {'yes' if lk['teeth'] else 'NO'} -> {'PASS' if lk['pass'] else 'FAIL'}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
