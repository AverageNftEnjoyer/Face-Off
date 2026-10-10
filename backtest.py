#!/usr/bin/env python3
"""
backtest.py -- point-in-time backtest of the CS2 match-insights engine.
=======================================================================
Fan analytics only: win probabilities and calibration. No betting logic.

DATA: data/matches.json (real results scraped politely from Liquipedia's
MediaWiki API; see data/SOURCES.md and data/collect_liquipedia.py).

PIPELINE
  1. Load all finished series (BO1/BO3/BO5) sorted by date.
  2. Walk the calendar day by day. For every series on day D, build a
     predict_match() input from state that contains ONLY series dated < D
     (same-day results are never visible). Then fold day D into the state.
  3. Evaluate only BO3 series where both teams have >= MIN_HISTORY prior
     series. Chronological split: first 60% train, last 40% test (the split
     never cuts through a calendar day).
  4. Fit CONFIG["temperature"] of the current engine on TRAIN only
     (golden-section search on mean log-loss of sigmoid(T * total_logodds)).
  5. Score on TEST: current engine (fitted T), old engine (legacy/predictor_v1,
     untouched), and baselines (coin flip, Elo-only probability, higher-Elo pick).

FEATURES (all point-in-time, see FEATURE DOC in data/backtest_report.md)
  rating        proxy: series-level Elo (init 1500, K=32, BO-weighted K) mapped
                linearly: rating = 1.0 + (elo - 1500) / ELO_PER_RATING (2000).
                ~150 Elo between a top-5 and a mid team => ~0.075 rating gap.
                NOT a real HLTV rating. Teams entering the data more than 90
                days after its start begin at 1500 - NEWCOMER_OFFSET (150).
  form30/n30    series win rate / count in the 30 days before D (absent -> key omitted)
  opp_rating30  mean proxy rating of opponents faced in those 30 days
  form5         series win rate over the last 5 series (any date < D)
  h2h           series wins vs each other in the 365 days before D
  maps          per-map win rate and count over the 90 days before D
  map_pool      active pool on D: maps with >= 3 plays (all teams) in the 60
                days before D, the 7 most recently played
  permaban      None (source has no veto data)
  roster        {} (source has no stand-in data)
  stakes        "none"
  volatility    |actual wins - Elo-expected wins| / n over the last 10 series
                (any date < D), x2.5, clamped to [0, 1]; 0.3 if < 3 series

Usage:  python backtest.py [--matches data/matches.json] [--min-history 5]
                           [--train-frac 0.6] [--seed 12345] [--quiet]
Stdlib only; all randomness seeded.
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ELO_INIT = 1500.0
ELO_K = 32.0
ELO_K_BY_BO = {1: 0.75, 3: 1.0, 5: 1.25}   # BO1 results are noisier -> smaller step
# Elo step by tournament tier (match field "tier", S or A, from the event infobox). A
# tier missing from this table, or a match with no tier, counts in full.
ELO_K_BY_TIER = {"S": 1.0, "A": 1.0}
ELO_PER_RATING = 2000.0                     # rating = 1 + (elo-1500)/2000
# Newcomers: a team whose first S/A series comes more than NEWCOMER_AFTER_DAYS
# after the first series in the data starts at ELO_INIT - NEWCOMER_OFFSET
# (S/A newcomers mostly arrive from lower tiers; every team present at the
# start of the data still starts at ELO_INIT, so the offset is relative to the
# established field). Its first PROVISIONAL_N series move its Elo with
# K x PROVISIONAL_K_MULT (its opponent moves by the normal step).
# Evidence (scripts/newcomer_check.py, pre-registered, walk-forward over
# 91-day blocks, offset picked per fold on earlier newcomer rows): on 158
# held-out BO3 series where a team had < 5 prior S/A series, log loss 0.6837
# -> 0.6400 (paired 95% CI of the difference [-0.074, -0.016]), Brier 0.2451
# -> 0.2246, accuracy 55.7% -> 60.8%; graded rows (both teams >= 5 series,
# n=1260) log loss 0.6173 -> 0.6170 (CI [-0.005, +0.004]). Before it,
# established teams beat newcomers far more often than Elo said (65% when
# the established side was rated below 1500; Elo said 44%). 150 is the value
# the walk-forward picked in 7 of 8 folds (the all-rows best, 200, costs the
# graded rows more in-sample: 0.6305 vs 0.6279). The provisional-K variant
# (x2 for the first 10 series) hurt graded rows and is off.
NEWCOMER_OFFSET = 150.0
NEWCOMER_AFTER_DAYS = 90
# "start": the offset is the team's starting Elo (it then feeds every Elo
# update). "feature": the Elo table is untouched; only the rating fed to the
# engine for a late-entering team is lowered, by the offset x (MIN_HISTORY -
# prior series) / MIN_HISTORY, fading to 0 at MIN_HISTORY series. Series
# where both teams have >= MIN_HISTORY prior series are then unchanged.
NEWCOMER_MODE = "start"
NEWCOMER_FEATURE_OFFSET = 0.0
PROVISIONAL_N = 0
PROVISIONAL_K_MULT = 1.0
MIN_HISTORY = 5
VOL_WINDOW = 10
VOL_SCALE = 2.5
VOL_DEFAULT = 0.3
POOL_WINDOW_DAYS = 60
POOL_MIN_PLAYS = 3
POOL_SIZE = 7
# Map inputs to the engine. "residual": per-map results relative to what Elo
# expected in those maps (0.5 + mean(won - expected)), so a team's strength is
# not counted again through its map rates and one strong map does not make
# every other map look weak after the engine's pool-relative step. "raw":
# plain 90-day win rates (the original behaviour; the veto factor was
# anti-predictive with these, see fit_weights.py).
MAP_RATES = "residual"
# Window of the veto evidence (exclusion test in veto.py): the same 90 days as
# the per-map records the veto already uses.
VETO_EV_DAYS = 90
EPS = 1e-12


# ============================================================================
# basic math / metrics
# ============================================================================
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p, eps=1e-9):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def log_loss(ps, ys, eps=1e-15):
    s = 0.0
    for p, y in zip(ps, ys):
        p = min(1 - eps, max(eps, p))
        s -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return s / len(ps)


def accuracy(ps, ys):
    """Pick A when p>0.5, B when p<0.5; an exact 0.5 earns half credit."""
    s = 0.0
    for p, y in zip(ps, ys):
        if p == 0.5:
            s += 0.5
        elif (p > 0.5) == (y == 1):
            s += 1.0
    return s / len(ps)


def reliability_table(ps, ys, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(ps, ys):
        i = min(n_bins - 1, int(p * n_bins))
        bins[i].append((p, y))
    rows = []
    for i, b in enumerate(bins):
        if not b:
            rows.append({"bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}", "n": 0,
                         "mean_pred": None, "actual": None})
            continue
        rows.append({"bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}", "n": len(b),
                     "mean_pred": sum(p for p, _ in b) / len(b),
                     "actual": sum(y for _, y in b) / len(b)})
    return rows


def ece(ps, ys, n_bins=10):
    n = len(ps)
    return sum(r["n"] / n * abs(r["mean_pred"] - r["actual"])
               for r in reliability_table(ps, ys, n_bins) if r["n"])


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def bootstrap_ci(fn, ps, ys, n_boot=2000, seed=12345, alpha=0.05):
    rng = random.Random(seed)
    n = len(ps)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(fn([ps[i] for i in idx], [ys[i] for i in idx]))
    vals.sort()
    return (vals[int(alpha / 2 * n_boot)], vals[int((1 - alpha / 2) * n_boot) - 1])


def paired_bootstrap_diff(fn, ps1, ps2, ys, n_boot=2000, seed=12345):
    """95% CI of fn(ps1) - fn(ps2) on the same resamples."""
    rng = random.Random(seed)
    n = len(ys)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        y = [ys[i] for i in idx]
        vals.append(fn([ps1[i] for i in idx], y) - fn([ps2[i] for i in idx], y))
    vals.sort()
    return (vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1])


def fit_temperature(logits, ys, lo=0.01, hi=5.0, iters=100):
    """Golden-section search for T minimizing log-loss of sigmoid(T * logit)."""
    def f(t):
        return log_loss([sigmoid(t * x) for x in logits], ys)
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
        if b - a < 1e-6:
            break
    return (a + b) / 2


# ============================================================================
# data + point-in-time features
# ============================================================================
def load_matches(path=None):
    path = path or os.path.join(DATA, "matches.json")
    with open(path, encoding="utf-8") as f:
        ms = json.load(f)
    for i, m in enumerate(ms):
        m.setdefault("id", i)
    return sorted(ms, key=lambda m: (m["date"], m["id"]))


# ------------------------------------------------------------------ rosters
_ROSTER = None


def roster_events():
    """data/roster_events.json (python data/rosters.py): team -> tournament page
    -> {"standin", "missing_igl"}. Empty if the file is absent."""
    global _ROSTER
    if _ROSTER is None:
        path = os.path.join(DATA, "roster_events.json")
        _ROSTER = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    return _ROSTER


def event_title(m):
    """Tournament page of a match: 'source' is 'liquipedia:<page>[ (hltv match N)]';
    the viewer passes 'event_title' directly."""
    if m.get("event_title"):
        return m["event_title"]
    src = m.get("source") or ""
    if not src.startswith("liquipedia:"):
        return None
    return src[len("liquipedia:"):].split(" (hltv")[0]


def roster_flags(team, title):
    """Stand-in / missing-IGL flags for `team` at tournament page `title`. A
    stand-in listed for a parent page (".../Cologne") covers its stage pages
    (".../Cologne/Stage 1") and vice versa."""
    if not title:
        return {}
    for tour, ev in roster_events().get(team, {}).items():
        if title == tour or title.startswith(tour + "/") or tour.startswith(title + "/"):
            return {"standin": bool(ev.get("standin")), "missing_igl": bool(ev.get("missing_igl"))}
    return {}


def map_prob_from_series(p, best_of=3):
    """Per-map win probability q implied by a series win probability p,
    assuming independent maps: BO1 q = p; BO3 p = q^2 (3 - 2q); BO5 p =
    q^3 (10 - 15q + 6q^2). Solved by bisection."""
    p = min(1 - 1e-9, max(1e-9, p))
    if best_of == 1:
        return p
    f = (lambda q: q * q * (3 - 2 * q)) if best_of != 5 else (lambda q: q ** 3 * (10 - 15 * q + 6 * q * q))
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < p else (lo, mid)
    return (lo + hi) / 2


def elo_expected(ea, eb):
    return 1.0 / (1.0 + 10 ** (-(ea - eb) / 400.0))


def elo_update(ea, eb, a_won, best_of=3, k=ELO_K):
    """Return new (ea, eb) after one series. Zero-sum."""
    kk = k * ELO_K_BY_BO.get(best_of, 1.0)
    e = elo_expected(ea, eb)
    d = kk * ((1.0 if a_won else 0.0) - e)
    return ea + d, eb - d


def elo_to_rating(elo):
    return 1.0 + (elo - ELO_INIT) / ELO_PER_RATING


def _d(s):
    return date.fromisoformat(s)


def actual_scoreline(m):
    """'2-0','2-1','1-2','0-2' from team_a's view, or None if maps missing."""
    if not m.get("maps"):
        return None
    a = sum(1 for x in m["maps"] if x["winner"] == m["team_a"])
    b = sum(1 for x in m["maps"] if x["winner"] == m["team_b"])
    if max(a, b) != 2 or a + b > 3:
        return None
    return f"{a}-{b}"


class _EloTable(dict):
    """Elo by team; an unseen team reads its starting value (not stored)."""

    def __init__(self, hist):
        super().__init__()
        self.hist = hist

    def __missing__(self, team):
        return self.hist.start_elo()


class HabitAccumulator:
    """Veto phase 3: online team ban / pick habits (design ADR-8).

    When a BO3 series dated D is folded in, its posterior over the vetoes
    consistent with the played map order is computed from veto inputs of D
    (every one of them filters on dates < D) by `posterior_fn(inp, played)`
    -> {"a": {kind: {map: [c, o]}}, "b": ...} (expected choice counts c and
    opportunity counts o). Evidence for (team, D) sums records dated < D
    only: the team's own within VETO_EV_DAYS, the field's over all series
    < D. `leaky=True` is the leakage canary (records dated <= D count)."""

    KINDS = ("ban1", "pick", "ban2")

    def __init__(self, posterior_fn, leaky=False, core_fn=None):
        self.post = posterior_fn
        self.leaky = leaky
        self.core_fn = core_fn           # (team, dt) -> (core five or None, stale)
        self.recs = defaultdict(list)    # team -> [(date, counts)]
        self.recs_all = []               # (date, five or None, counts): roster-core level (phase 4)
        self.snap_dates = []             # field totals after all records dated <= snap_dates[i]
        self.snaps = []
        self.total = {k: {} for k in self.KINDS}

    def on_add(self, h, m, inp):
        if m.get("best_of") != 3 or len(m.get("maps") or []) < 2:
            return
        played = [x["map"] for x in m["maps"]]
        pool = inp.get("map_pool") or []
        if any(p not in pool for p in played) or len(set(played)) != len(played):
            return
        counts = self.post(inp, played)
        if counts is None:
            return
        dt = _d(m["date"])
        if self.snap_dates and dt < self.snap_dates[-1]:
            raise ValueError("HabitAccumulator needs series folded in date order")
        for side, team in (("a", m["team_a"]), ("b", m["team_b"])):
            self.recs[team].append((dt, counts[side]))
            self.recs_all.append((dt, h.lineup_of(m, side), counts[side]))
            for k in self.KINDS:
                tk = self.total[k]
                for mp, (c, o) in (counts[side].get(k) or {}).items():
                    cur = tk.setdefault(mp, [0.0, 0.0])
                    cur[0] += c
                    cur[1] += o
        snap = {k: {mp: list(v) for mp, v in self.total[k].items()} for k in self.KINDS}
        if self.snap_dates and self.snap_dates[-1] == dt:
            self.snaps[-1] = snap
        else:
            self.snap_dates.append(dt)
            self.snaps.append(snap)

    def _ok(self, d, dt):
        return d <= dt if self.leaky else d < dt

    @staticmethod
    def _sum(recs, pool, weight=None):
        """Sum of (c, o) counts over `recs`; `weight(date)` scales each record
        (veto phase 7: recency decay / roster discount). None = equal weights."""
        out = {k: {mp: [0.0, 0.0] for mp in sorted(pool)} for k in HabitAccumulator.KINDS}
        for d, c in recs:
            w = 1.0 if weight is None else weight(d)
            for k in HabitAccumulator.KINDS:
                ck = c.get(k) or {}
                for mp in out[k]:
                    v = ck.get(mp)
                    if v:
                        if weight is None:
                            out[k][mp][0] += v[0]
                            out[k][mp][1] += v[1]
                        else:
                            out[k][mp][0] += w * v[0]
                            out[k][mp][1] += w * v[1]
        return out

    def evidence(self, team, dt, pool, weight=None, horizon=None):
        """`weight(record_date)` and `horizon` (days) are the phase-7 options:
        both None reproduces the equal-weight VETO_EV_DAYS window exactly."""
        lo = dt - timedelta(days=VETO_EV_DAYS if horizon is None else horizon)
        own = [r for r in self.recs.get(team, []) if lo <= r[0] and self._ok(r[0], dt)]
        snap = None
        for d, sn in zip(reversed(self.snap_dates), reversed(self.snaps)):
            if self._ok(d, dt):
                snap = sn
                break
        field = {k: {mp: list((snap or {}).get(k, {}).get(mp, [0.0, 0.0])) for mp in sorted(pool)}
                 for k in self.KINDS}
        out = {"habit": self._sum(own, pool, weight), "field": field, "habit_n": len(own)}
        if self.core_fn is not None:
            core, stale = self.core_fn(team, dt)
            if core:
                rs = [(d, c) for d, five, c in self.recs_all
                      if five and lo <= d and self._ok(d, dt) and len(core & five) >= 3]
                out["core"] = self._sum(rs, pool)
                out["core_n"] = len(rs)
            out["core_stale"] = bool(stale) or not core
        return out


class History:
    """State built ONLY from series already folded in (all dated < current day)."""

    def __init__(self):
        self.elo = _EloTable(self)
        self.first_day = None            # date of the first series folded in
        self.clock = None                # date of the series being built / folded
        self.games = defaultdict(list)   # team -> [rec]
        self.map_last = {}               # map -> last date played
        self.map_dates = defaultdict(list)
        # veto evidence (veto.py): every folded series with the pool of its own
        # date and the maps played; per-date caches of the pool and of the
        # field's play rates. A cached date stays valid while only series dated
        # >= it are added (see _invalidate).
        self.vseries = []                # (date, pool frozenset, played frozenset, bo)
        self.vteam = defaultdict(list)   # team -> indexes into vseries
        self._pool_cache = {}
        self._rate_cache = {}
        self.habits = None               # optional HabitAccumulator (veto phase 3)
        self.lineups = None              # optional {"date|a|b": {"a": five, "b": five}} (veto phase 4)
        self.lineup_hist = defaultdict(list)
        self.ev_cfg = None               # veto phase-7 evidence options; None = predictor.CONFIG (see _cfg)

    def start_elo(self):
        """Starting Elo of a team not seen yet, at self.clock."""
        if (NEWCOMER_OFFSET and NEWCOMER_MODE == "start" and self.first_day is not None and self.clock is not None
                and (self.clock - self.first_day).days > NEWCOMER_AFTER_DAYS):
            return ELO_INIT - NEWCOMER_OFFSET
        return ELO_INIT

    def is_late_entrant(self, team):
        """True if the team's first series (or, unseen, the current clock) is
        more than NEWCOMER_AFTER_DAYS after the first series in the data."""
        if self.first_day is None:
            return False
        g = self.games[team]
        d = g[0]["date"] if g else self.clock
        return d is not None and (d - self.first_day).days > NEWCOMER_AFTER_DAYS

    def feature_elo(self, team):
        """Elo fed to the engine: the table value, minus the fading newcomer
        offset in "feature" mode (see NEWCOMER_MODE)."""
        e = self.elo[team]
        if NEWCOMER_MODE == "feature" and NEWCOMER_FEATURE_OFFSET:
            n = len(self.games[team])
            if n < MIN_HISTORY and self.is_late_entrant(team):
                e -= NEWCOMER_FEATURE_OFFSET * (MIN_HISTORY - n) / MIN_HISTORY
        return e

    def add(self, m):
        a, b = m["team_a"], m["team_b"]
        if self.habits is not None:
            # veto inputs of D filter on dates < D, so same-day series already
            # folded in cannot reach them
            inp, _ = self.features(m)
            self.habits.on_add(self, m, inp)
        self.clock = _d(m["date"])
        if self.first_day is None:
            self.first_day = self.clock
        ea, eb = self.elo[a], self.elo[b]
        exp_a = elo_expected(ea, eb)
        a_won = m["winner"] == a
        dt = _d(m["date"])
        for t, o, won, exp, oe in ((a, b, a_won, exp_a, eb), (b, a, not a_won, 1 - exp_a, ea)):
            self.games[t].append({"date": dt, "opp": o, "won": won, "exp": exp,
                                  "opp_elo": oe,
                                  "maps": [(x["map"], x["winner"] == t,
                                            map_prob_from_series(exp, m.get("best_of", 3)))
                                           for x in m.get("maps", [])]})
        k = ELO_K * ELO_K_BY_TIER.get(m.get("tier"), 1.0)
        na, nb = elo_update(ea, eb, a_won, m.get("best_of", 3), k=k)
        if PROVISIONAL_N and PROVISIONAL_K_MULT != 1.0:
            # games[t] already holds this series, so len - 1 = series before it
            if len(self.games[a]) - 1 < PROVISIONAL_N:
                na = ea + (na - ea) * PROVISIONAL_K_MULT
            if len(self.games[b]) - 1 < PROVISIONAL_N:
                nb = eb + (nb - eb) * PROVISIONAL_K_MULT
        self.elo[a], self.elo[b] = na, nb
        if self.lineups is not None:
            for side, t in (("a", a), ("b", b)):
                five = self.lineup_of(m, side)
                if five:
                    self.lineup_hist[t].append((dt, five))
        pool_dt = self.pool_at(dt)       # pool of this series' own date (series < dt)
        self._invalidate(dt)
        for x in m.get("maps", []):
            self.map_dates[x["map"]].append(dt)
        self.vseries.append((dt, pool_dt, frozenset(x["map"] for x in m.get("maps", [])),
                             m.get("best_of", 3)))
        for t in (a, b):
            self.vteam[t].append(len(self.vseries) - 1)

    # ------------------------------------------------------------ veto evidence
    def lineup_of(self, m, side):
        if not self.lineups:
            return None
        rec = self.lineups.get(f"{m['date']}|{m['team_a']}|{m['team_b']}") or {}
        return frozenset(rec[side]) if rec.get(side) else None

    def core_five(self, team, dt, core_n=8):
        """(core five, stale) from the team's lineups dated < dt, with the
        lineup_features rule (most appearances in its last core_n lineups,
        ties to the most recent). stale: its last series is newer than its
        last lineup (failure mode lineup_stale_X)."""
        prior = [(d, f) for d, f in self.lineup_hist.get(team, []) if d < dt]
        if not prior:
            return None, True
        recent = prior[-core_n:]
        count, seen = defaultdict(int), {}
        for i, (_, five) in enumerate(recent):
            for p in five:
                count[p] += 1
                seen[p] = i
        core = frozenset(sorted(count, key=lambda p: (-count[p], -seen[p], p))[:5])
        last_series = max((r["date"] for r in self.games.get(team, []) if r["date"] < dt), default=None)
        return core, last_series is not None and last_series > prior[-1][0]

    def _invalidate(self, dt):
        """A series dated dt changes pools / rates only for dates > dt."""
        for c in (self._pool_cache, self._rate_cache):
            if c and max(c) > dt:
                for k in [k for k in c if k > dt]:
                    del c[k]

    def pool_at(self, dt):
        """map_pool(dt), cached."""
        p = self._pool_cache.get(dt)
        if p is None:
            p = frozenset(self.map_pool(dt))
            self._pool_cache[dt] = p
        return p

    def field_rates(self, dt):
        """{map: share of series in the VETO_EV_DAYS days before dt, with that
        map in their own date's pool, in which it was played}. Series < dt."""
        r = self._rate_cache.get(dt)
        if r is None:
            lo = dt - timedelta(days=VETO_EV_DAYS)
            n, k = defaultdict(int), defaultdict(int)
            for d, pool, played, _ in reversed(self.vseries):
                if d >= dt:
                    continue
                if d < lo:
                    break
                for mp in pool:
                    n[mp] += 1
                    k[mp] += mp in played
            r = {mp: k[mp] / n[mp] for mp in n}
            self._rate_cache[dt] = r
        return r

    # ------------------------------------------------ phase 7: weighted evidence
    def _cfg(self):
        """Evidence options: self.ev_cfg, else the CONFIG of the imported
        `predictor` (read at call time; absent -> every option off)."""
        if self.ev_cfg is not None:
            return self.ev_cfg
        p = sys.modules.get("predictor")
        return getattr(p, "CONFIG", {}) if p is not None else {}

    def roster_events(self, team, dt, floor, recover):
        """(events, stale) for the roster-change discount. events = [(change
        date, factor)]: a lineup that differs from the team's previous one in
        n players (first seen in the series dated `change date`) multiplies the
        weight of every evidence record dated BEFORE that date by
        1 - (1 - floor) * min(1, n / 2) * max(0, 1 - (k - 1) / recover), k = the
        team's series played with the new lineup so far (1 right after the
        change, full weight again from k = recover + 1). Lineups dated < dt
        only. stale: the team's last series is newer than its last lineup (or it
        has no lineup): no discount, same rule as core_five."""
        hist = [(d, f) for d, f in self.lineup_hist.get(team, []) if d < dt]
        if not hist:
            return [], True
        last_series = next((r["date"] for r in reversed(self.games.get(team, [])) if r["date"] < dt), None)
        if last_series is not None and last_series > hist[-1][0]:
            return [], True
        n = len(hist)
        out = []
        for i in range(max(1, n - recover), n):
            changed = 5 - len(hist[i - 1][1] & hist[i][1])
            if changed <= 0:
                continue
            k = n - i
            f = 1.0 - (1.0 - floor) * min(1.0, changed / 2.0) * max(0.0, 1.0 - (k - 1) / float(recover))
            out.append((hist[i][0], f))
        return out, False

    def record_weights(self, team, dt, cfg):
        """(weight(record_date) or None, horizon days, roster_stale). None when
        no phase-7 option is on (legacy equal weights over VETO_EV_DAYS)."""
        hl = cfg.get("veto_decay_halflife")
        roster = bool(cfg.get("veto_roster_discount"))
        if not hl and not roster:
            return None, VETO_EV_DAYS, False
        events, stale = ([], False)
        if roster:
            events, stale = self.roster_events(team, dt, float(cfg.get("veto_roster_floor", 0.4)),
                                               int(cfg.get("veto_roster_recover", 8)))

        def weight(d):
            x = 0.5 ** ((dt - d).days / float(hl)) if hl else 1.0
            for c, f in events:
                if d < c:
                    x *= f
            return x
        horizon = int(cfg.get("veto_decay_horizon", 180)) if hl else VETO_EV_DAYS
        return weight, horizon, stale

    def veto_evidence(self, team, dt, pool):
        """Point-in-time evidence for the veto's exclusion test (veto.py):
        over the team's series in the VETO_EV_DAYS days before dt, per pool
        map: [series where it was played, series with it in their pool,
        sum of log(1 - field play rate at that series' date)].

        Phase 7 (CONFIG veto_decay_halflife / veto_roster_discount /
        veto_soft_avoid, all off by default): adds ev["wmaps"], the same
        counts with each series weighted by recency decay and the roster-change
        discount, over the decay horizon: per map [weighted plays, weighted
        series with the map in the pool, weighted sum of log(1 - rate),
        weighted expected plays (sum of rate), raw plays]. The legacy "maps"
        stay as they were. Series dated < dt only."""
        lo = dt - timedelta(days=VETO_EV_DAYS)
        js = [self.vseries[j] for j in self.vteam.get(team, []) if lo <= self.vseries[j][0] < dt]
        out = {}
        for mp in sorted(pool):
            plays = n_pool = 0
            lp = 0.0
            for d, jpool, played, _ in js:
                plays += mp in played
                if mp in jpool:
                    n_pool += 1
                    rate = min(self.field_rates(d).get(mp, 0.0), 1.0 - 1e-9)
                    lp += math.log(1.0 - rate)
            out[mp] = [plays, n_pool, lp]
        ev = {"n": len(js), "maps": out}
        cfg = self._cfg()
        weight, horizon, stale = self.record_weights(team, dt, cfg)
        if weight is not None or cfg.get("veto_soft_avoid"):
            wlo = dt - timedelta(days=horizon)
            jw = [self.vseries[j] for j in self.vteam.get(team, []) if wlo <= self.vseries[j][0] < dt]
            ws = [1.0 if weight is None else weight(d) for d, _, _, _ in jw]
            wm = {}
            for mp in sorted(pool):
                pw = nw = lw = ew = 0.0
                raw = 0
                for (d, jpool, played, _), w in zip(jw, ws):
                    if mp in played:
                        pw += w
                        raw += 1
                    if mp in jpool:
                        nw += w
                        rate = min(self.field_rates(d).get(mp, 0.0), 1.0 - 1e-9)
                        lw += w * math.log(1.0 - rate)
                        ew += w * rate
                wm[mp] = [pw, nw, lw, ew, raw]
            ev["wmaps"] = wm
            if cfg.get("veto_roster_discount"):
                ev["roster_stale"] = stale
        if self.habits is not None:
            ev.update(self.habits.evidence(team, dt, pool, weight, horizon if weight is not None else None))
        return ev

    def n_prior(self, team):
        return len(self.games[team])

    def map_pool(self, dt):
        lo = dt - timedelta(days=POOL_WINDOW_DAYS)
        cand = []
        for mp, ds in self.map_dates.items():
            recent = [d for d in ds if lo <= d < dt]
            if len(recent) >= POOL_MIN_PLAYS:
                cand.append((max(recent), len(recent), mp))
        cand.sort(reverse=True)
        return sorted(mp for _, _, mp in cand[:POOL_SIZE])

    def team_feats(self, team, dt):
        g = self.games[team]
        out = {}
        w30 = [r for r in g if dt - timedelta(days=30) <= r["date"] < dt]
        if w30:
            out["form30"] = sum(r["won"] for r in w30) / len(w30)
            out["n30"] = len(w30)
            out["opp_rating30"] = sum(elo_to_rating(r["opp_elo"]) for r in w30) / len(w30)
        last5 = g[-5:]
        if last5:
            out["form5"] = sum(r["won"] for r in last5) / len(last5)
        maps = defaultdict(lambda: [0, 0, 0.0])
        for r in g:
            if dt - timedelta(days=90) <= r["date"] < dt:
                for mp, won, q in r["maps"]:
                    maps[mp][0] += won
                    maps[mp][1] += 1
                    maps[mp][2] += won - q
        out["maps_raw"] = {mp: [round(w / n, 4), n] for mp, (w, n, _) in sorted(maps.items())}
        out["maps_resid"] = {mp: [round(min(1.0, max(0.0, 0.5 + e / n)), 4), n]
                             for mp, (_, n, e) in sorted(maps.items())}
        out["maps"] = out["maps_resid"] if MAP_RATES == "residual" else out["maps_raw"]
        lastv = g[-VOL_WINDOW:]
        if len(lastv) >= 3:
            dev = abs(sum(r["won"] for r in lastv) - sum(r["exp"] for r in lastv)) / len(lastv)
            out["volatility"] = round(min(1.0, max(0.0, VOL_SCALE * dev)), 4)
        else:
            out["volatility"] = VOL_DEFAULT
        return out

    def h2h(self, a, b, dt):
        aw = bw = 0
        for r in self.games[a]:
            if r["opp"] == b and dt - timedelta(days=365) <= r["date"] < dt:
                if r["won"]:
                    aw += 1
                else:
                    bw += 1
        return {"a_wins": aw, "b_wins": bw, "meetings": aw + bw}

    def features(self, m):
        a, b = m["team_a"], m["team_b"]
        dt = _d(m["date"])
        self.clock = dt
        fa, fb = self.team_feats(a, dt), self.team_feats(b, dt)
        ea, eb = self.feature_elo(a), self.feature_elo(b)
        pool = self.map_pool(dt)
        inp = {"team_a": a, "team_b": b,
               "rating_a": round(elo_to_rating(ea), 5),
               "rating_b": round(elo_to_rating(eb), 5),
               "h2h": self.h2h(a, b, dt),
               "maps_a": fa["maps"], "maps_b": fb["maps"],
               "maps_raw_a": fa["maps_raw"], "maps_raw_b": fb["maps_raw"],
               "map_pool": pool,
               "veto_ev_a": self.veto_evidence(a, dt, pool),
               "veto_ev_b": self.veto_evidence(b, dt, pool),
               "permaban_a": None, "permaban_b": None,
               "roster_a": roster_flags(a, event_title(m)),
               "roster_b": roster_flags(b, event_title(m)),
               "stakes_a": "none", "stakes_b": "none",
               "volatility_a": fa["volatility"], "volatility_b": fb["volatility"]}
        for side, f in (("a", fa), ("b", fb)):
            for k in ("form30", "n30", "opp_rating30", "form5"):
                if k in f:
                    inp[f"{k}_{side}"] = round(f[k], 5) if isinstance(f[k], float) else f[k]
        meta = {"elo_a": ea, "elo_b": eb,
                "p_elo": elo_expected(ea, eb),
                "hist_a": self.n_prior(a), "hist_b": self.n_prior(b)}
        return inp, meta


def build_dataset(matches):
    """Return a list (same order as `matches` sorted by date) of
    {"match", "input", "meta"} with features from strictly earlier days."""
    ms = sorted(matches, key=lambda m: (m["date"], m.get("id", 0)))
    h = History()
    out = []
    i = 0
    while i < len(ms):
        j = i
        while j < len(ms) and ms[j]["date"] == ms[i]["date"]:
            j += 1
        day = ms[i:j]
        for m in day:                    # 1) features from state of days < D
            inp, meta = h.features(m)
            out.append({"match": m, "input": inp, "meta": meta})
        for m in day:                    # 2) then fold day D in
            h.add(m)
        i = j
    return out


def select_eval(rows, min_history=MIN_HISTORY):
    ev, excluded = [], 0
    for r in rows:
        if r["match"].get("best_of") != 3:
            continue
        if min(r["meta"]["hist_a"], r["meta"]["hist_b"]) < min_history:
            excluded += 1
            continue
        ev.append(r)
    return ev, excluded


def chrono_split(rows, frac=0.6):
    """First ~frac by date -> train; never splits a calendar day."""
    if not rows:
        return [], []
    k = int(len(rows) * frac)
    k = max(1, min(len(rows) - 1, k))
    while 0 < k < len(rows) and rows[k]["match"]["date"] == rows[k - 1]["match"]["date"]:
        k += 1
    return rows[:k], rows[k:]


# ============================================================================
# engines
# ============================================================================
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_engines():
    new = old = None
    errs = {}
    try:
        new = load_module("predictor_current", os.path.join(HERE, "predictor.py"))
    except Exception as e:  # engine may be mid-rewrite
        errs["new"] = repr(e)
    try:
        old = load_module("predictor_v1", os.path.join(HERE, "legacy", "predictor_v1.py"))
    except Exception as e:
        errs["old"] = repr(e)
    return new, old, errs


def get_total_logodds(mod, inp):
    """Pre-temperature log-odds from the engine, or None if not exposed."""
    if hasattr(mod, "total_logodds"):
        return float(mod.total_logodds(dict(inp)))
    r = mod.predict_match(dict(inp))
    if "total_logodds" in r:
        return float(r["total_logodds"])
    return None


def p_exact(r):
    return float(r.get("p_a_exact", r["p_a"]))


def modal_scoreline(r):
    sp = r.get("series_probs_exact") or r.get("series_probs")
    if not sp:
        return None
    opts = {"2-0": sp["p_2_0"], "2-1": sp["p_2_1"], "1-2": sp["p_1_2"], "0-2": sp["p_0_2"]}
    return max(opts, key=lambda k: opts[k])


# ============================================================================
# evaluation
# ============================================================================
def metric_block(ps, ys, seed):
    n = len(ys)
    acc = accuracy(ps, ys)
    k = round(acc * n)
    return {"n": n, "accuracy": acc, "accuracy_ci95_wilson": wilson(k, n),
            "brier": brier(ps, ys), "brier_ci95_bootstrap": bootstrap_ci(brier, ps, ys, seed=seed),
            "log_loss": log_loss(ps, ys),
            "log_loss_ci95_bootstrap": bootstrap_ci(log_loss, ps, ys, seed=seed),
            "ece10": ece(ps, ys), "mean_pred": sum(ps) / n, "base_rate": sum(ys) / n}


def tier_table(results, ys):
    out = {}
    for tier in ("HIGH", "MEDIUM", "LOW"):
        idx = [i for i, r in enumerate(results) if r.get("reliability") == tier]
        if not idx:
            out[tier] = {"n": 0}
            continue
        ps = [p_exact(results[i]) for i in idx]
        yy = [ys[i] for i in idx]
        out[tier] = {"n": len(idx), "accuracy": accuracy(ps, yy), "brier": brier(ps, yy),
                     "mean_width_pp": sum(results[i].get("interval_width_pp", 0) for i in idx) / len(idx),
                     "mean_conf": sum(max(p, 1 - p) for p in ps) / len(ps)}
    return out


def width_quartiles(results, ys):
    idx = sorted(range(len(results)), key=lambda i: results[i].get("interval_width_pp", 0))
    q = []
    for j in range(4):
        part = idx[j * len(idx) // 4:(j + 1) * len(idx) // 4]
        if not part:
            continue
        ps = [p_exact(results[i]) for i in part]
        yy = [ys[i] for i in part]
        q.append({"quartile": j + 1, "n": len(part),
                  "width_pp_range": [results[part[0]].get("interval_width_pp"),
                                     results[part[-1]].get("interval_width_pp")],
                  "brier": brier(ps, yy), "accuracy": accuracy(ps, yy)})
    return q


def overconfidence(ps, ys, hi=0.9):
    idx = [i for i, p in enumerate(ps) if p >= hi or p <= 1 - hi]
    hits = sum(1 for i in idx if (ps[i] >= 0.5) == (ys[i] == 1))
    return {"n_extreme": len(idx), "share": len(idx) / len(ps) if ps else 0.0,
            "hit_rate": hits / len(idx) if idx else None,
            "mean_conf": sum(max(ps[i], 1 - ps[i]) for i in idx) / len(idx) if idx else None}


def scoreline_acc(preds, actual):
    pairs = [(p, a) for p, a in zip(preds, actual) if p and a]
    if not pairs:
        return None, 0
    return sum(1 for p, a in pairs if p == a) / len(pairs), len(pairs)


def run(matches_path=None, min_history=MIN_HISTORY, train_frac=0.6, seed=12345, quiet=False):
    random.seed(seed)
    matches = load_matches(matches_path)
    rows = build_dataset(matches)
    ev, excluded = select_eval(rows, min_history)
    train, test = chrono_split(ev, train_frac)
    new, old, errs = load_engines()
    rep = {"data": {
        "matches_total": len(matches),
        "date_range": [matches[0]["date"], matches[-1]["date"]] if matches else None,
        "best_of_counts": {str(k): sum(1 for m in matches if m.get("best_of") == k) for k in (1, 3, 5)},
        "bo3_total": sum(1 for m in matches if m.get("best_of") == 3),
        "bo3_excluded_min_history": excluded, "min_history": min_history,
        "eval_series": len(ev), "train": len(train), "test": len(test),
        "train_range": [train[0]["match"]["date"], train[-1]["match"]["date"]] if train else None,
        "test_range": [test[0]["match"]["date"], test[-1]["match"]["date"]] if test else None,
        "teams": len({m["team_a"] for m in matches} | {m["team_b"] for m in matches}),
    }, "engine_errors": errs, "config": {
        "elo_init": ELO_INIT, "elo_k": ELO_K, "elo_k_by_bo": ELO_K_BY_BO,
        "elo_per_rating": ELO_PER_RATING, "vol_window": VOL_WINDOW, "vol_scale": VOL_SCALE,
        "pool_window_days": POOL_WINDOW_DAYS, "pool_min_plays": POOL_MIN_PLAYS,
        "train_frac": train_frac, "seed": seed}}
    y_tr = [1 if r["match"]["winner"] == r["match"]["team_a"] else 0 for r in train]
    y_te = [1 if r["match"]["winner"] == r["match"]["team_a"] else 0 for r in test]
    act_te = [actual_scoreline(r["match"]) for r in test]
    # rating-gap sanity (proxy scale)
    gaps = sorted(abs(r["input"]["rating_a"] - r["input"]["rating_b"]) for r in ev)
    rep["data"]["rating_gap_quantiles"] = {q: gaps[int(q * (len(gaps) - 1))] for q in (0.25, 0.5, 0.75, 0.9)} if gaps else {}
    models = {}
    # baselines
    models["coin_flip"] = {"p": [0.5] * len(test)}
    models["elo_only"] = {"p": [r["meta"]["p_elo"] for r in test]}
    pick = [1.0 if r["meta"]["p_elo"] > 0.5 else (0.0 if r["meta"]["p_elo"] < 0.5 else 0.5) for r in test]
    models["higher_elo_pick"] = {"p": pick, "accuracy_only": True}
    # old engine (as-is)
    if old is not None:
        res = [old.predict_match(dict(r["input"])) for r in test]
        models["old_engine_v1"] = {"p": [float(x["p_a"]) for x in res], "results": res}
        res_tr = [old.predict_match(dict(r["input"])) for r in train]
        t_old = fit_temperature([logit(float(x["p_a"])) for x in res_tr], y_tr)
        models["old_engine_v1_posthoc_T"] = {
            "p": [sigmoid(t_old * logit(float(x["p_a"]))) for x in res],
            "note": f"diagnostic only: post-hoc temperature {t_old:.3f} on logit(p_a), fitted on train"}
    # new engine
    fitted_T = None
    if new is not None:
        try:
            cfg = getattr(new, "CONFIG", {})
            L_tr = [get_total_logodds(new, r["input"]) for r in train]
            has_T = "temperature" in cfg
            if has_T and all(x is not None for x in L_tr):
                fitted_T = fit_temperature(L_tr, y_tr)
                mode = "engine CONFIG['temperature'] fitted on total_logodds (train)"
            else:
                base_tr = [logit(p_exact(new.predict_match(dict(r["input"])))) for r in train]
                fitted_T = fit_temperature(base_tr, y_tr)
                mode = "post-hoc temperature on logit(p_a) (engine exposes no temperature)"
            res_untuned = []
            if has_T:
                prev = cfg["temperature"]
                cfg["temperature"] = 1.0
                res_untuned = [new.predict_match(dict(r["input"])) for r in test]
                cfg["temperature"] = fitted_T
                res = [new.predict_match(dict(r["input"])) for r in test]
                res_tr_fit = [new.predict_match(dict(r["input"])) for r in train]
                cfg["temperature"] = prev
                p_new = [p_exact(x) for x in res]
            else:
                res = [new.predict_match(dict(r["input"])) for r in test]
                res_tr_fit = []
                p_new = [sigmoid(fitted_T * logit(p_exact(x))) for x in res]
            models["new_engine"] = {"p": p_new, "results": res}
            if res_untuned:
                models["new_engine_T1"] = {"p": [p_exact(x) for x in res_untuned], "results": res_untuned,
                                           "note": "current engine with temperature=1.0 (unfitted)"}
            rep["temperature"] = {"fitted": fitted_T, "mode": mode,
                                  "train_log_loss_T1": log_loss([sigmoid(x) for x in L_tr], y_tr) if L_tr[0] is not None else None,
                                  "train_log_loss_fitted": log_loss([p_exact(x) for x in res_tr_fit], y_tr) if res_tr_fit else None}
        except Exception as e:
            errs["new_run"] = repr(e)
    # metrics
    out = {}
    for name, mdl in models.items():
        ps = mdl["p"]
        if mdl.get("accuracy_only"):
            acc = accuracy(ps, y_te)
            out[name] = {"n": len(ps), "accuracy": acc,
                         "accuracy_ci95_wilson": wilson(round(acc * len(ps)), len(ps))}
            continue
        blk = metric_block(ps, y_te, seed)
        blk["reliability_table"] = reliability_table(ps, y_te)
        blk["overconfidence_90"] = overconfidence(ps, y_te)
        if "note" in mdl:
            blk["note"] = mdl["note"]
        if "results" in mdl:
            res = mdl["results"]
            blk["scoreline_modal_acc"], blk["scoreline_n"] = scoreline_acc([modal_scoreline(x) for x in res], act_te)
            blk["reliability_tiers"] = tier_table(res, y_te)
            blk["band_width_quartiles"] = width_quartiles(res, y_te)
            blk["mean_interval_width_pp"] = sum(x.get("interval_width_pp", 0) for x in res) / len(res)
        out[name] = blk
    rep["test_metrics"] = out
    # paired comparisons (Brier difference, negative = first is better)
    comps = {}
    for a, b in (("new_engine", "elo_only"), ("new_engine", "old_engine_v1"), ("old_engine_v1", "elo_only"),
                 ("new_engine", "coin_flip")):
        if a in models and b in models:
            comps[f"{a} - {b}"] = {
                "brier_diff": brier(models[a]["p"], y_te) - brier(models[b]["p"], y_te),
                "brier_diff_ci95": paired_bootstrap_diff(brier, models[a]["p"], models[b]["p"], y_te, seed=seed),
                "log_loss_diff": log_loss(models[a]["p"], y_te) - log_loss(models[b]["p"], y_te)}
    rep["paired_comparisons"] = comps
    # old engine overconfidence on the whole eval set (train+test), as requested
    if old is not None:
        allp = [float(old.predict_match(dict(r["input"]))["p_a"]) for r in ev]
        ally = y_tr + y_te
        rep["old_engine_overconfidence_all_eval"] = overconfidence(allp, ally)
    # scoreline base rates
    sc = [a for a in act_te if a]
    rep["test_scoreline_distribution"] = {s: sc.count(s) / len(sc) for s in ("2-0", "2-1", "1-2", "0-2")} if sc else {}
    write_reports(rep)
    if not quiet:
        print(render_md(rep))
    return rep


# ============================================================================
# reporting
# ============================================================================
def _f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def render_md(rep):
    D = rep["data"]
    L = ["# CS2 engine backtest report", "",
         "Fan analytics only. Real Liquipedia results (data/SOURCES.md). "
         "Ratings are a point-in-time **Elo proxy**, not HLTV ratings.", "",
         "## Data", "",
         f"- series: {D['matches_total']} ({D['best_of_counts']}) from {D['date_range'][0]} to {D['date_range'][1]}, {D['teams']} teams",
         f"- BO3 series: {D['bo3_total']}; excluded (either team < {D['min_history']} prior series): {D['bo3_excluded_min_history']}",
         f"- evaluated: {D['eval_series']} -> train {D['train']} ({D['train_range'][0]}..{D['train_range'][1]}), "
         f"test {D['test']} ({D['test_range'][0]}..{D['test_range'][1]})",
         f"- |rating gap| quantiles (proxy scale): " + ", ".join(f"q{int(q*100)}={v:.3f}" for q, v in D["rating_gap_quantiles"].items()),
         ""]
    if rep.get("engine_errors"):
        L += [f"**Engine errors:** {rep['engine_errors']}", ""]
    if rep.get("temperature"):
        T = rep["temperature"]
        L += ["## Temperature (fitted on train only)", "",
              f"- T = **{T['fitted']:.3f}** ({T['mode']})",
              f"- train log-loss: T=1 {_f(T['train_log_loss_T1'], 4)} -> fitted {_f(T['train_log_loss_fitted'], 4)}", ""]
    L += ["## Test-set metrics", "",
          "| model | n | accuracy [95% Wilson] | Brier [95% boot] | log-loss [95% boot] | ECE10 | scoreline acc |",
          "|---|---|---|---|---|---|---|"]
    for name, b in rep["test_metrics"].items():
        if "brier" not in b:
            L.append(f"| {name} | {b['n']} | {b['accuracy']:.3f} [{b['accuracy_ci95_wilson'][0]:.3f}, {b['accuracy_ci95_wilson'][1]:.3f}] | n/a | n/a | n/a | - |")
            continue
        L.append(f"| {name} | {b['n']} | {b['accuracy']:.3f} [{b['accuracy_ci95_wilson'][0]:.3f}, {b['accuracy_ci95_wilson'][1]:.3f}] | "
                 f"{b['brier']:.4f} [{b['brier_ci95_bootstrap'][0]:.4f}, {b['brier_ci95_bootstrap'][1]:.4f}] | "
                 f"{b['log_loss']:.4f} [{b['log_loss_ci95_bootstrap'][0]:.4f}, {b['log_loss_ci95_bootstrap'][1]:.4f}] | "
                 f"{b['ece10']:.4f} | {_f(b.get('scoreline_modal_acc'))} |")
    L += ["", "Notes: coin flip accuracy is credited 0.5 per series by definition. "
          "higher_elo_pick is a hard pick (accuracy only). old_engine_v1_posthoc_T is a diagnostic, "
          "not the shipped old engine. Scoreline acc = modal BO3 scoreline from series_probs vs actual "
          f"(test base rates: {', '.join(f'{k} {v:.2f}' for k, v in rep['test_scoreline_distribution'].items())}).", ""]
    if rep.get("paired_comparisons"):
        L += ["## Paired Brier differences (test, 95% paired bootstrap; negative = first model better)", ""]
        for k, v in rep["paired_comparisons"].items():
            L.append(f"- {k}: {v['brier_diff']:+.4f} [{v['brier_diff_ci95'][0]:+.4f}, {v['brier_diff_ci95'][1]:+.4f}]; log-loss diff {v['log_loss_diff']:+.4f}")
        L.append("")
    for name in ("new_engine", "old_engine_v1", "elo_only"):
        b = rep["test_metrics"].get(name)
        if not b or "reliability_table" not in b:
            continue
        L += [f"## Reliability table: {name} (P(team_a) bins)", "", "| bin | n | mean pred | actual |", "|---|---|---|---|"]
        for r in b["reliability_table"]:
            L.append(f"| {r['bin']} | {r['n']} | {_f(r['mean_pred'])} | {_f(r['actual'])} |")
        L.append("")
    for name in ("new_engine", "old_engine_v1"):
        b = rep["test_metrics"].get(name)
        if not b or "reliability_tiers" not in b:
            continue
        L += [f"## Reliability tiers / bands: {name}", "", "| tier | n | accuracy | Brier | mean band width pp | mean conf |", "|---|---|---|---|---|---|"]
        for t, v in b["reliability_tiers"].items():
            if v["n"]:
                L.append(f"| {t} | {v['n']} | {v['accuracy']:.3f} | {v['brier']:.4f} | {v['mean_width_pp']:.1f} | {v['mean_conf']:.3f} |")
            else:
                L.append(f"| {t} | 0 | - | - | - | - |")
        L += ["", "Band-width quartiles (does a wider band mean a harder-to-call match?):", "",
              "| quartile | n | width pp range | Brier | accuracy |", "|---|---|---|---|---|"]
        for q in b["band_width_quartiles"]:
            L.append(f"| Q{q['quartile']} | {q['n']} | {q['width_pp_range'][0]}-{q['width_pp_range'][1]} | {q['brier']:.4f} | {q['accuracy']:.3f} |")
        L.append("")
    L += ["## Overconfidence check (outputs >= 90% or <= 10%)", ""]
    for name in ("old_engine_v1", "new_engine", "elo_only"):
        b = rep["test_metrics"].get(name)
        if b and "overconfidence_90" in b:
            o = b["overconfidence_90"]
            L.append(f"- {name} (test): {o['n_extreme']} series ({o['share']:.1%}), hit-rate {_f(o['hit_rate'])}, mean stated conf {_f(o['mean_conf'])}")
    if rep.get("old_engine_overconfidence_all_eval"):
        o = rep["old_engine_overconfidence_all_eval"]
        L.append(f"- old_engine_v1 (all {rep['data']['eval_series']} eval series): {o['n_extreme']} ({o['share']:.1%}), hit-rate {_f(o['hit_rate'])}, mean stated conf {_f(o['mean_conf'])}")
    L += ["", "## Feature derivation (point-in-time, only series dated strictly before the match day)", "",
          f"- rating: series Elo (init {ELO_INIT:.0f}, K={ELO_K:.0f} x {ELO_K_BY_BO} by best-of), rating = 1 + (elo-1500)/{ELO_PER_RATING:.0f}. PROXY, not HLTV.",
          "- form30/n30/opp_rating30: series win rate, count and mean opponent proxy rating in the 30 days before; keys omitted if no series.",
          "- form5: win rate over the last <=5 series; omitted if none.",
          "- h2h: series wins vs each other in the prior 365 days.",
          "- maps_a/b: per-map [win rate, maps played] over the prior 90 days.",
          f"- map_pool: maps with >= {POOL_MIN_PLAYS} plays (all teams) in the prior {POOL_WINDOW_DAYS} days, the {POOL_SIZE} most recently played (old engine ignores it: hardcoded pool).",
          f"- volatility: |wins - Elo-expected wins| / n over last {VOL_WINDOW} series x {VOL_SCALE}, clamped [0,1]; {VOL_DEFAULT} if < 3 series.",
          "- permaban: None (no veto data); roster: stand-in and missing-IGL flags from Liquipedia team-page stand-in tables (data/rosters.py); stakes: none.", ""]
    return "\n".join(L)


def write_reports(rep):
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "backtest_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    with open(os.path.join(DATA, "backtest_report.md"), "w", encoding="utf-8") as f:
        f.write(render_md(rep) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[1])
    ap.add_argument("--matches", default=None)
    ap.add_argument("--min-history", type=int, default=MIN_HISTORY)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(a.matches, a.min_history, a.train_frac, a.seed, a.quiet)


if __name__ == "__main__":
    main()
