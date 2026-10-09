#!/usr/bin/env python3
"""
ban_patterns.py -- research: how constant are the map-veto habits of the top teams?
===================================================================================
Fan analytics only. Research and recommendations; it never touches the engine
(veto.py, predictor.py, backtest.py are only IMPORTED, read-only, by sections
q4 and q5) and writes data/ban_patterns_report.json.

What the data can and cannot say
  data/matches*.json hold the maps that were PLAYED, in order, with the series
  winner. Bans, the picker of each map and the starter of the veto are never
  recorded. Everything about bans / who picked is therefore a posterior
  inference (section q1: an EM fit of a sequential-choice model over the
  standard BO3 order  ban, ban, pick, pick, ban, ban, decider  with a hidden
  starter), and every such number is labelled "soft" in the report. Quantities
  read straight off the played maps ("played share", the decider of 2-1
  series) are labelled "observed".

Sections (run all, or --sections q1,q2,...):
  q1  per-team entropy / first-ban repeat rate / map rates with CIs / drift
  q2  opponent dependence (strength split, pick-pair independence, matchup term)
  q3  roster changes (data/lineups.json) vs habit stability
  q4  does pick-side map strength add to the engine's win probability
      (train <= 2025-11-05, test >= 2025-11-07, point-in-time features only)
  q5  where the engine's veto predictions miss, by team, and why

USAGE (repo root):
  python scripts/ban_patterns.py                 all sections (~ a few minutes)
  python scripts/ban_patterns.py --fast          fewer EM iterations / bootstraps
  python scripts/ban_patterns.py --sections q2,q3
Stdlib only. Deterministic (seeded). No team or map name is hard-coded.
"""
import argparse
import bisect
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "ban_patterns_report.json")

SEED = 20261009
TRAIN_END = "2025-11-05"          # same split as the backtest / veto harness
TEST_START = "2025-11-07"
POOL_WINDOW_DAYS, POOL_MIN_PLAYS, POOL_SIZE = 60, 3, 7   # the engine's pool rule, re-implemented
TOP_N = 30
RECENT_DAYS = 365                  # window for the per-team description (q1)
KAPPA = 4.0                        # shrinkage pseudo-count toward the field
KINDS = ("b1", "pk", "b2")         # first ban, pick, second ban
MIN_TEAM_SERIES = 12

FAST = False


# ============================================================================
# small statistics helpers
# ============================================================================
def wilson(k, n, z=1.959964):
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def entropy(ps):
    return -sum(p * math.log(p) for p in ps if p > 0)


def norm_vec(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def js_div(p, q):
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(x, y):
        return sum(a * math.log(a / b) for a, b in zip(x, y) if a > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def cluster_boot_mean(vals, days, n_boot=1000, seed=SEED):
    """Mean of vals with a 95% percentile CI, resampling whole days."""
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    by = defaultdict(list)
    for v, d in zip(vals, days):
        by[d].append(v)
    keys = sorted(by)
    sums = [(sum(by[k]), len(by[k])) for k in keys]
    rng = random.Random(seed)
    m = mean(vals)
    bs = []
    for _ in range(n_boot):
        s = c = 0
        for _ in keys:
            a, b = sums[rng.randrange(len(sums))]
            s += a
            c += b
        bs.append(s / c)
    return {"mean": m, "lo": quantile(bs, 0.025), "hi": quantile(bs, 0.975), "n": len(vals)}


def cluster_boot_diff(v1, v2, days, n_boot=1000, seed=SEED):
    """Paired difference mean(v1 - v2) with day-cluster bootstrap CI."""
    return cluster_boot_mean([a - b for a, b in zip(v1, v2)], days, n_boot, seed)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(x) < 4:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def holm(pvals):
    """Holm-adjusted p-values (same order)."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (n - r) * pvals[i]))
        adj[i] = run
    return adj


def solve(A, b):
    """Gaussian elimination with partial pivoting (small systems)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            M[c][c] = 1e-12
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][k] * x[k] for k in range(i + 1, n))) / M[i][i]
    return x


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logistic_fit(X, y, l2=1.0, offset=None, iters=30, penalize_first=False):
    """Newton / IRLS logistic regression. X rows include their own constant
    column if wanted; `offset` is a fixed per-row logit term. L2 applies
    to all coefficients except column 0 unless penalize_first."""
    n, d = len(X), len(X[0])
    beta = [0.0] * d
    off = offset or [0.0] * n
    for _ in range(iters):
        g = [0.0] * d
        H = [[0.0] * d for _ in range(d)]
        for xi, yi, oi in zip(X, y, off):
            p = sigmoid(sum(b * v for b, v in zip(beta, xi)) + oi)
            w = p * (1 - p)
            r = yi - p
            for a in range(d):
                g[a] += r * xi[a]
                for c in range(a, d):
                    H[a][c] += w * xi[a] * xi[c]
        for a in range(d):
            for c in range(a):
                H[a][c] = H[c][a]
            if a > 0 or penalize_first:
                g[a] -= l2 * beta[a]
                H[a][a] += l2
        step = solve(H, g)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-7:
            break
    return beta


def logloss(ps, ys):
    e = 1e-12
    return -mean(y * math.log(max(p, e)) + (1 - y) * math.log(max(1 - p, e)) for p, y in zip(ps, ys))


# ============================================================================
# data
# ============================================================================
def norm_map(m):
    return re.sub(r"\s+", "", m)


def d_ord(s):
    return date.fromisoformat(s).toordinal()


def load_series():
    """All series of matches.json (src 'main') and matches_lower.json ('lower'),
    sorted by date; `ord` = date ordinal; maps are the played maps in order."""
    out = []
    for src, fn in (("main", "matches.json"), ("lower", "matches_lower.json")):
        with open(os.path.join(DATA, fn), encoding="utf-8") as f:
            ms = json.load(f)
        for i, m in enumerate(ms):
            out.append({
                "date": m["date"], "ord": d_ord(m["date"]), "a": m["team_a"], "b": m["team_b"],
                "winner": m["winner"], "bo": m["best_of"], "tier": m.get("tier"), "src": src,
                "maps": [(norm_map(x["map"]), x["winner"]) for x in m["maps"] if x.get("map")],
                "idx": i,
            })
    out.sort(key=lambda s: (s["ord"], s["src"], s["idx"]))
    for k, s in enumerate(out):
        s["k"] = k
    return out


class PoolClock:
    """The engine's pool rule (>= 3 plays in the prior 60 days, the 7 most
    recently played), recomputed from all series. Point in time: dates < D."""

    def __init__(self, series):
        self.dates = defaultdict(list)
        for s in series:
            for mp, _ in s["maps"]:
                self.dates[mp].append(s["ord"])
        for v in self.dates.values():
            v.sort()
        self.cache = {}

    def pool(self, o):
        p = self.cache.get(o)
        if p is None:
            cand = []
            for mp, ds in self.dates.items():
                i = bisect.bisect_left(ds, o - POOL_WINDOW_DAYS)
                j = bisect.bisect_left(ds, o)
                if j - i >= POOL_MIN_PLAYS:
                    cand.append((ds[j - 1], j - i, mp))
            cand.sort(reverse=True)
            p = tuple(sorted(mp for _, _, mp in cand[:POOL_SIZE]))
            self.cache[o] = p
        return p


def valid_bo3(series, clock):
    """BO3 series with >= 2 maps, distinct maps, all inside that date's pool.
    Adds s['pool']. Returns (list, counts of drops)."""
    out, drops = [], Counter()
    for s in series:
        if s["bo"] != 3 or len(s["maps"]) < 2:
            continue
        pool = clock.pool(s["ord"])
        names = [mp for mp, _ in s["maps"]]
        if len(set(names)) != len(names):
            drops["repeat_map"] += 1
            continue
        if any(mp not in pool for mp in names):
            drops["outside_pool"] += 1
            continue
        s["pool"] = pool
        out.append(s)
    return out, dict(drops)


def vrs_top(names_in_data, n=TOP_N):
    """Top-n of the global VRS ranking mapped to data team names by a
    normalised-name match (ties: the team with more recent BO3s)."""
    with open(os.path.join(DATA, "vrs.json"), encoding="utf-8") as f:
        v = json.load(f)

    def norm(s):
        s = s.lower()
        s = re.sub(r"\b(team|esports|e-sports|gaming|club|esport|clan)\b", "", s)
        return re.sub(r"[^a-z0-9]", "", s)
    idx = defaultdict(list)
    for t, c in names_in_data.items():
        idx[norm(t)].append((c, t))
    out, unmapped = [], []
    for row in v["standings"]["global"]:
        cands = sorted(idx.get(norm(row["name"]), []), reverse=True)
        if cands:
            out.append({"rank": row["rank"], "vrs_name": row["name"], "team": cands[0][1], "points": row["points"]})
        else:
            unmapped.append(row["name"])
        if len(out) >= n:
            break
    return out, unmapped, v["date"]


# ============================================================================
# q1 engine: sequential-choice model of the BO3 veto with a hidden starter
# ============================================================================
# order (S = the team that starts): S ban, O ban, S pick, O pick, S ban, O ban, decider.
# Each team has three weight vectors over the pool (first ban, pick, second ban);
# a choice is made with probability weight / (sum of weights of the maps still
# open). The played order fixes both picks (map 1 = S's, map 2 = O's) and, in a
# 2-1 series, the decider; the four bans are summed over exactly (<= 120 orders).
def _wvec(W, G, team, kind, pool):
    w = W.get(team)
    wk = w[kind] if w else None
    g = G[kind]
    if wk is None:
        return [g.get(mp, 1.0 / len(pool)) for mp in pool]
    return [wk.get(mp, g.get(mp, 1.0 / len(pool))) for mp in pool]


def series_posterior(s, W, G, pi_a, want=True):
    """P(observed order), plus (if want) the posterior marginals.

    Returns (prob, marg) where marg = {'a': {kind: [mass per pool index]},
    'b': ..., 'exposure': {team: {kind: [..]}}, 'pi': P(a started | data),
    'decider': [mass per index]} (masses sum to 1 per kind, except exposure)."""
    pool = s["pool"]
    n = len(pool)
    ix = {mp: i for i, mp in enumerate(pool)}
    m1, m2 = ix[s["maps"][0][0]], ix[s["maps"][1][0]]
    m3 = ix[s["maps"][2][0]] if len(s["maps"]) >= 3 else None
    R = [i for i in range(n) if i != m1 and i != m2]
    wv = {}
    for t in ("a", "b"):
        team = s[t]
        wv[t] = {k: _wvec(W, G, team, k, pool) for k in KINDS}
    res = {}
    tot = 0.0
    for st, pri in (("a", pi_a), ("b", 1.0 - pi_a)):
        S, O = wv[st], wv["b" if st == "a" else "a"]
        sb1, ob1, sp, op, sb2, ob2 = S["b1"], O["b1"], S["pk"], O["pk"], S["b2"], O["b2"]
        t1s, t1o, tps, tpo, t2s, t2o = sum(sb1), sum(ob1), sum(sp), sum(op), sum(sb2), sum(ob2)
        seqs = []
        for a in R:
            pa = sb1[a] / t1s
            for b in R:
                if b == a:
                    continue
                pb = ob1[b] / (t1o - ob1[a])
                zs = tps - sp[a] - sp[b]
                zo = tpo - op[a] - op[b] - op[m1]
                base = pri * pa * pb * (sp[m1] / zs) * (op[m2] / zo)
                rest = [x for x in R if x != a and x != b]
                z2s = sum(sb2[x] for x in rest)
                for c in rest:
                    pc = sb2[c] / z2s
                    z2o = sum(ob2[x] for x in rest) - ob2[c]
                    for d in rest:
                        if d == c:
                            continue
                        e = rest[0] + rest[1] + rest[2] - c - d
                        if m3 is not None and e != m3:
                            continue
                        seqs.append((base * pc * (ob2[d] / z2o), a, b, c, d, e, z2s, z2o, zs, zo))
        res[st] = (seqs, S, O, (t1s, t1o))
        tot += sum(q[0] for q in seqs)
    if not want or tot <= 0:
        return tot, None
    marg = {"a": {k: [0.0] * n for k in KINDS}, "b": {k: [0.0] * n for k in KINDS},
            "exposure": {"a": {k: [0.0] * n for k in KINDS}, "b": {k: [0.0] * n for k in KINDS}},
            "decider": [0.0] * n, "pi": 0.0}
    for st in ("a", "b"):
        seqs, S, O, (t1s, t1o) = res[st]
        so = "b" if st == "a" else "a"
        sm, om = marg[st], marg[so]
        se, oe = marg["exposure"][st], marg["exposure"][so]
        mass = 0.0
        Ma, Mab, Mabc = defaultdict(float), defaultdict(float), defaultdict(float)
        for (q, a, b, c, d, e, z2s, z2o, zs, zo) in seqs:
            q /= tot
            mass += q
            sm["b1"][a] += q
            om["b1"][b] += q
            sm["b2"][c] += q
            om["b2"][d] += q
            marg["decider"][e] += q
            Ma[a] += q
            Mab[(a, b)] += q
            Mabc[(a, b, c)] += q
        marg["pi"] += mass if st == "a" else 0.0
        sm["pk"][m1] += mass
        om["pk"][m2] += mass
        # exposure: 1 / (weight mass of the open maps) at every step, summed
        # over the posterior of the prefix (the MM denominators)
        sb1, ob1, sp, op, sb2, ob2 = S["b1"], O["b1"], S["pk"], O["pk"], S["b2"], O["b2"]
        for i in range(n):
            se["b1"][i] += mass / t1s
        for a, ma in Ma.items():
            za = t1o - ob1[a]
            for i in range(n):
                if i != a:
                    oe["b1"][i] += ma / za
        for (a, b), mab in Mab.items():
            zs = sum(sp) - sp[a] - sp[b]
            zo = sum(op) - op[a] - op[b] - op[m1]
            for i in range(n):
                if i != a and i != b:
                    se["pk"][i] += mab / zs
                    if i != m1:
                        oe["pk"][i] += mab / zo
            rest = [x for x in R if x != a and x != b]
            z2s = sum(sb2[x] for x in rest)
            for x in rest:
                se["b2"][x] += mab / z2s
        for (a, b, c), mabc in Mabc.items():
            rest = [x for x in R if x != a and x != b and x != c]
            z2o = sum(ob2[x] for x in rest)
            for x in rest:
                oe["b2"][x] += mabc / z2o
    return tot, marg


def em_fit(series, kappa=KAPPA, iters=8, weights=None, kinds=KINDS, init=None, verbose=False):
    """EM / MM fit of team weight vectors (shrunk toward the field) and the
    starter prior, on `series` (valid BO3 with 7-map pools).
    weights: optional per-series weight (recency decay). Returns dict with
    W (team -> kind -> {map: w}), G (field), pi (by src), loglik."""
    pi = dict(init["pi"]) if init else {"main": 0.5, "lower": 0.5}
    maps = sorted({mp for s in series for mp in s["pool"]})
    uni = {mp: 1.0 / len(maps) for mp in maps}
    G = {k: dict(uni) for k in KINDS}
    W = {}
    if init:
        G = {k: dict(init["G"][k]) for k in KINDS}
        W = {t: {k: (dict(v) if v is not None else None) for k, v in kv.items()} for t, kv in init["W"].items()}
    ll = None
    for it in range(iters):
        # accumulators: team -> kind -> map -> [count, exposure]; field likewise
        acc = defaultdict(lambda: {k: defaultdict(lambda: [0.0, 0.0]) for k in KINDS})
        fld = {k: defaultdict(lambda: [0.0, 0.0]) for k in KINDS}
        pi_acc = defaultdict(lambda: [0.0, 0.0])
        ll = 0.0
        wsum = 0.0
        for j, s in enumerate(series):
            wgt = weights[j] if weights else 1.0
            tot, marg = series_posterior(s, W, G, pi[s["src"]])
            if marg is None:
                continue
            ll += wgt * math.log(tot)
            wsum += wgt
            pi_acc[s["src"]][0] += wgt * marg["pi"]
            pi_acc[s["src"]][1] += wgt
            pool = s["pool"]
            for t in ("a", "b"):
                team = s[t]
                for k in KINDS:
                    mk, ek = marg[t][k], marg["exposure"][t][k]
                    for i, mp in enumerate(pool):
                        cell = acc[team][k][mp]
                        cell[0] += wgt * mk[i]
                        cell[1] += wgt * ek[i]
                        f = fld[k][mp]
                        f[0] += wgt * mk[i]
                        f[1] += wgt * ek[i]
        for k in KINDS:
            G[k] = norm_dict({mp: (c + 1e-3) / (e + 1e-3 * len(maps)) for mp, (c, e) in fld[k].items()}, maps)
        newW = {}
        for team, kd in acc.items():
            newW[team] = {}
            for k in KINDS:
                if k not in kinds:
                    newW[team][k] = None
                    continue
                w = {}
                kk = kappa[k] if isinstance(kappa, dict) else kappa
                for mp in maps:
                    c, e = kd[k].get(mp, (0.0, 0.0))
                    w[mp] = (c + kk * G[k][mp]) / (e + kk)
                newW[team][k] = norm_dict(w, maps)
        W = newW
        for src, (a, b) in pi_acc.items():
            pi[src] = min(0.95, max(0.05, a / b)) if b else 0.5
        if verbose:
            print(f"    em it {it}: ll/series {ll / max(wsum, 1e-9):.4f}", flush=True)
    return {"W": W, "G": G, "pi": pi, "loglik": ll / max(wsum, 1e-9) if wsum else None, "maps": maps}


def norm_dict(d, maps):
    s = sum(d.get(mp, 0.0) for mp in maps)
    return {mp: d.get(mp, 0.0) / s for mp in maps} if s > 0 else {mp: 1.0 / len(maps) for mp in maps}


def score_series(series, fit, kinds_override=None, uniform=False):
    """Mean log P(observed order) under a frozen fit (list per series)."""
    W, G, pi = fit["W"], fit["G"], fit["pi"]
    maps = fit["maps"]
    if uniform:
        G = {k: {mp: 1.0 / len(maps) for mp in maps} for k in KINDS}
        W = {}
    out = []
    for s in series:
        tot, _ = series_posterior(s, W, G, pi[s["src"]], want=False)
        out.append(math.log(tot) if tot > 0 else float("-inf"))
    return out


def score_series_pi(series, fit, pis):
    """Like score_series, with a per-series P(team_a starts)."""
    out = []
    for s, pi in zip(series, pis):
        tot, _ = series_posterior(s, fit["W"], fit["G"], pi, want=False)
        out.append(math.log(tot) if tot > 0 else float("-inf"))
    return out


def posterior_pass(series, fit):
    """Posterior marginals of every series under a frozen fit."""
    W, G, pi = fit["W"], fit["G"], fit["pi"]
    out = []
    for s in series:
        tot, marg = series_posterior(s, W, G, pi[s["src"]])
        out.append(marg)
    return out


# ============================================================================
# habit predictors on the OBSERVED played-map sets (no latent variables)
# ============================================================================
class FieldRates:
    """Point-in-time field play rate per map over a trailing window."""

    def __init__(self, vseries):
        by_day = defaultdict(lambda: (Counter(), Counter()))
        for s in vseries:
            pl, op = by_day[s["ord"]]
            names = {mp for mp, _ in s["maps"]}
            for mp in s["pool"]:
                op[mp] += 1
                if mp in names:
                    pl[mp] += 1
        self.days = sorted(by_day)
        self.maps = sorted({mp for d in by_day.values() for mp in d[1]})
        self.cum = {mp: ([0], [0]) for mp in self.maps}
        for d in self.days:
            pl, op = by_day[d]
            for mp in self.maps:
                c = self.cum[mp]
                c[0].append(c[0][-1] + pl[mp])
                c[1].append(c[1][-1] + op[mp])

    def rate(self, mp, o, w=180):
        c = self.cum.get(mp)
        if not c:
            return 0.25
        i = bisect.bisect_left(self.days, o - w)
        j = bisect.bisect_left(self.days, o)
        n = c[1][j] - c[1][i]
        return (c[0][j] - c[0][i] + 0.25 * 3) / (n + 3) if n >= 0 else 0.25


class TeamPlays:
    """team -> chronological [(ord, played set, pool, src)] from valid BO3s."""

    def __init__(self, vseries):
        self.ev = defaultdict(list)
        for s in vseries:
            names = frozenset(mp for mp, _ in s["maps"])
            for t in (s["a"], s["b"]):
                self.ev[t].append((s["ord"], names, s["pool"], s["k"], s))
        self.ords = {t: [e[0] for e in v] for t, v in self.ev.items()}

    def before(self, team, o, w=None):
        v = self.ev.get(team, [])
        j = bisect.bisect_left(self.ords.get(team, []), o)
        i = 0 if w is None else bisect.bisect_left(self.ords[team], o - w) if team in self.ords else 0
        return v[i:j]


def habit_prob(events, o, pool, fr, mode, param, kappa=4.0):
    """P(map played) per pool map from a team's past events.
    mode 'win': hard window of `param` days (None = all); 'hl': exponential
    decay with half-life `param` days. Shrunk to the field rate."""
    plays, opp = defaultdict(float), defaultdict(float)
    for (eo, names, epool, _, _) in events:
        w = 1.0 if mode == "win" else 0.5 ** ((o - eo) / param)
        for mp in epool:
            opp[mp] += w
            if mp in names:
                plays[mp] += w
    return {mp: (plays[mp] + kappa * fr.rate(mp, o)) / (opp[mp] + kappa) for mp in pool}, sum(opp.values()) / max(len(pool), 1)


def bern_score(q, pool, names):
    return sum(math.log(q[mp]) if mp in names else math.log(1 - q[mp]) for mp in pool)


WINDOWS = [("w30", "win", 30), ("w60", "win", 60), ("w90", "win", 90), ("w180", "win", 180),
           ("w365", "win", 365), ("all", "win", None),
           ("hl30", "hl", 30), ("hl60", "hl", 60), ("hl90", "hl", 90), ("hl180", "hl", 180), ("hl365", "hl", 365)]


def drift_windows(vseries, tp, fr, top_teams, lo_ord, hi_ord, min_prior=6):
    """Held-out log score gain (vs the field-rate baseline) of window / decay
    habit predictors, on team-series dated in [lo_ord, hi_ord]. A row needs
    >= min_prior series of that team in the prior 180 days."""
    rows = defaultdict(list)
    meta = []
    for s in vseries:
        if not (lo_ord <= s["ord"] <= hi_ord):
            continue
        names = frozenset(mp for mp, _ in s["maps"])
        for t in (s["a"], s["b"]):
            ev_all = tp.before(t, s["ord"], 730)
            prior180 = [e for e in ev_all if e[0] >= s["ord"] - 180]
            if len(prior180) < min_prior:
                continue
            base = {mp: fr.rate(mp, s["ord"]) for mp in s["pool"]}
            sb = bern_score(base, s["pool"], names)
            for name, mode, p in WINDOWS:
                ev = ev_all if mode == "hl" else (ev_all if p is None else [e for e in ev_all if e[0] >= s["ord"] - p])
                if p is None:
                    ev = tp.before(t, s["ord"])
                q, _ = habit_prob(ev, s["ord"], s["pool"], fr, mode, p)
                rows[name].append(bern_score(q, s["pool"], names) - sb)
            meta.append((t, s["ord"], t in top_teams))
    return rows, meta


def summarize_drift(rows, meta, which):
    out = {}
    sel = [i for i, m in enumerate(meta) if which(m)]
    days = [meta[i][1] for i in sel]
    for name, _, _ in WINDOWS:
        v = [rows[name][i] for i in sel]
        out[name] = cluster_boot_mean(v, days, n_boot=300 if FAST else 800)
    ref = [rows["w90"][i] for i in sel]
    for name in ("hl30", "hl60", "hl90", "hl180", "w180", "all"):
        v = [rows[name][i] for i in sel]
        out[name]["paired_gain_vs_w90"] = cluster_boot_diff(v, ref, days, n_boot=300 if FAST else 800)
    return out


# ============================================================================
# Q1
# ============================================================================
class Ctx:
    """Shared, lazily built inputs."""

    def __init__(self):
        t = time.time()
        self.series = load_series()
        self.as_of = max(s["ord"] for s in self.series)
        self.clock = PoolClock(self.series)
        self.valid, self.drops = valid_bo3(self.series, self.clock)
        self.v7 = [s for s in self.valid if len(s["pool"]) == 7]
        cnt = Counter()
        for s in self.valid:
            if s["ord"] >= self.as_of - RECENT_DAYS:
                cnt[s["a"]] += 1
                cnt[s["b"]] += 1
        self.top, self.unmapped, self.vrs_date = vrs_top(cnt, TOP_N)
        self.top_teams = {r["team"] for r in self.top}
        self.fr = FieldRates(self.valid)
        self.tp = TeamPlays(self.valid)
        self.load_s = time.time() - t

    def iso(self, o):
        return date.fromordinal(o).isoformat()

    def series_pit(self):
        """All series (any best-of), valid BO3s carrying their pool."""
        return self.series


def get_recent(ctx, kappa=8.0, iters=7):
    """Cached EM fit + posterior marginals on the last RECENT_DAYS days."""
    c = getattr(ctx, "_recent", None)
    if c is None:
        recent = [s for s in ctx.v7 if s["ord"] >= ctx.as_of - RECENT_DAYS]
        fit = em_fit(recent, kappa=kappa, iters=iters)
        c = (recent, fit, posterior_pass(recent, fit))
        ctx._recent = c
    return c


def chi2_sf(x, df):
    """Survival function of chi-square (regularised upper incomplete gamma)."""
    a, x2 = df / 2.0, x / 2.0
    if x2 <= 0:
        return 1.0
    if x2 < a + 1:
        term = s_ = 1.0 / a
        n = a
        for _ in range(500):
            n += 1
            term *= x2 / n
            s_ += term
            if abs(term) < 1e-14 * abs(s_):
                break
        return max(0.0, 1.0 - s_ * math.exp(-x2 + a * math.log(x2) - math.lgamma(a)))
    tiny = 1e-300
    b = x2 + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-14:
            break
    return min(1.0, h * math.exp(-x2 + a * math.log(x2) - math.lgamma(a)))


def era_split(ctx):
    """train / test of the veto fits, 7-map pools only."""
    tr_hi = d_ord(TRAIN_END)
    te_lo = d_ord(TEST_START)
    train = [s for s in ctx.v7 if tr_hi - 365 <= s["ord"] <= tr_hi]
    test = [s for s in ctx.v7 if s["ord"] >= te_lo]
    return train, test


def series_stats(vecs, pool_idx, n_boot, rng, field_share, n_null=300):
    """Distribution statistics of a team's soft choice vectors.
    vecs: list of dict(map -> mass) per series (each sums to <= 1)."""
    maps = pool_idx
    n = len(vecs)
    c = {mp: sum(v.get(mp, 0.0) for v in vecs) for mp in maps}
    tot = sum(c.values())
    sh = [c[mp] / tot if tot else 0.0 for mp in maps]
    H = entropy(sh)
    Hn = H / math.log(len(maps))
    top = max(range(len(maps)), key=lambda i: sh[i])
    # bootstrap over series
    hb, tb = [], []
    for _ in range(n_boot):
        acc = defaultdict(float)
        for _ in range(n):
            v = vecs[rng.randrange(n)]
            for mp, x in v.items():
                acc[mp] += x
        t = sum(acc.values())
        s2 = [acc.get(mp, 0.0) / t for mp in maps]
        hb.append(entropy(s2) / math.log(len(maps)))
        tb.append(max(s2))
    # null: n hard draws iid from the field share
    cum = []
    run = 0.0
    for p in field_share:
        run += p
        cum.append(run)
    hn, tn, jn = [], [], []
    for _ in range(n_null):
        k = [0] * len(maps)
        for _ in range(n):
            u = rng.random() * run
            k[bisect.bisect_left(cum, u)] += 1
        s2 = [x / n for x in k]
        hn.append(entropy(s2) / math.log(len(maps)))
        tn.append(max(s2))
        jn.append(js_div(s2, field_share))
    jobs = js_div(sh, field_share)
    p_js = (1 + sum(1 for x in jn if x >= jobs)) / (n_null + 1)
    uni = [1.0 / len(maps)] * len(maps)
    # effective number of maps
    return {
        "n_series": n, "shares": {mp: round(sh[i], 4) for i, mp in enumerate(maps)},
        "entropy_norm": round(Hn, 4), "entropy_norm_ci": [round(quantile(hb, 0.025), 4), round(quantile(hb, 0.975), 4)],
        "eff_maps": round(math.exp(H), 3),
        "top_map": maps[top], "top_share": round(sh[top], 4),
        "top_share_wilson": [round(x, 4) for x in wilson(sh[top] * n, n)],
        "top_share_boot_ci": [round(quantile(tb, 0.025), 4), round(quantile(tb, 0.975), 4)],
        "null_entropy_norm_mean": round(mean(hn), 4), "null_entropy_norm_p05": round(quantile(hn, 0.05), 4),
        "null_top_share_p95": round(quantile(tn, 0.95), 4),
        "entropy_z_vs_field": round((Hn - mean(hn)) / (math.sqrt(sum((x - mean(hn)) ** 2 for x in hn) / len(hn)) or 1), 3),
        "js_vs_field": round(jobs, 4), "p_js_vs_field": round(p_js, 4),
        "js_vs_uniform": round(js_div(sh, uni), 4),
    }, c


def repeat_rate(vecs_in_order, rng, n_perm=200):
    """Mean over consecutive series of sum_x q_t(x) q_{t-1}(x) and its
    permutation-null (series order shuffled)."""
    def rr(vs):
        tot = 0.0
        for a, b in zip(vs, vs[1:]):
            tot += sum(a.get(mp, 0.0) * b.get(mp, 0.0) for mp in a)
        return tot / max(len(vs) - 1, 1)
    if len(vecs_in_order) < 3:
        return None
    obs = rr(vecs_in_order)
    nulls = []
    v = list(vecs_in_order)
    for _ in range(n_perm):
        rng.shuffle(v)
        nulls.append(rr(v))
    nm = mean(nulls)
    return {"repeat": round(obs, 4), "null_mean": round(nm, 4),
            "excess": round(obs - nm, 4),
            "p": round((1 + sum(1 for x in nulls if x >= obs)) / (n_perm + 1), 4)}


def classify(st, p_holm=None):
    """constant / probable / diffuse by the top-share Wilson lower bound and
    the JS test against the field; 'outlier' flag from the Holm-adjusted p."""
    lo = st["top_share_wilson"][0]
    if lo >= 0.40:
        c = "constant"
    elif st["p_js_vs_field"] < 0.05 or lo >= 0.25:
        c = "probable"
    else:
        c = "diffuse"
    return c


def null_top_share_table(field_share, rng, ns=(20, 40, 60, 100), reps=400):
    cum, run = [], 0.0
    for p in field_share:
        run += p
        cum.append(run)
    out = {}
    for n in ns:
        tops = []
        for _ in range(reps):
            k = [0] * len(field_share)
            for _ in range(n):
                k[bisect.bisect_left(cum, rng.random() * run)] += 1
            tops.append(max(k) / n)
        out[str(n)] = {"mean": round(mean(tops), 3), "p95": round(quantile(tops, 0.95), 3)}
    return out


def q1(ctx, rep):
    rng = random.Random(SEED)
    t0 = time.time()
    R = {}
    iters = 4 if FAST else 7
    # ---------------------------------------------------------------- A. is the latent model any good? (held-out)
    train, test = era_split(ctx)
    cut = sorted({s["ord"] for s in train})[int(len(train) and len({s["ord"] for s in train}) * 0.75)]
    inner_fit = [s for s in train if s["ord"] < cut]
    inner_val = [s for s in train if s["ord"] >= cut]
    print(f"  q1: train {len(train)}  test {len(test)}  (inner {len(inner_fit)}/{len(inner_val)})", flush=True)
    kap_grid = [2.0, 8.0] if FAST else [2.0, 6.0, 16.0]
    inner = {}
    for kp in kap_grid:
        f = em_fit(inner_fit, kappa=kp, iters=iters)
        inner[kp] = mean(score_series(inner_val, f))
    kstar = max(inner, key=lambda k: inner[k])
    # separate (stronger) shrinkage for the pick weights, chosen on the same inner fold
    kp_try = {"pk_x4": {"b1": kstar, "pk": 4 * kstar, "b2": kstar}}
    f_try = em_fit(inner_fit, kappa=kp_try["pk_x4"], iters=iters)
    inner_pk = mean(score_series(inner_val, f_try))
    use_split = inner_pk > inner[kstar]
    kfull = kp_try["pk_x4"] if use_split else kstar
    fits = {"full": em_fit(train, kappa=kfull, iters=iters)}
    fits["picks_only"] = em_fit(train, kappa=kstar, iters=iters, kinds=("pk",))
    fits["picks_b1"] = em_fit(train, kappa=kstar, iters=iters, kinds=("pk", "b1"))
    fits["bans_only"] = em_fit(train, kappa=kstar, iters=iters, kinds=("b1", "b2"))
    # field-only = same fit with every team weight forced to the field
    field_fit = dict(fits["full"])
    field_fit["W"] = {}
    sc = {"uniform": score_series(test, fits["full"], uniform=True),
          "field_only": score_series(test, field_fit)}
    for k, f in fits.items():
        sc[k] = score_series(test, f)
    days = [s["ord"] for s in test]
    comp = {}
    for k in ("picks_only", "picks_b1", "bans_only", "full"):
        comp[k] = {"vs_field_only": cluster_boot_diff(sc[k], sc["field_only"], days, 400 if FAST else 1000)}
    comp["field_only_vs_uniform"] = cluster_boot_diff(sc["field_only"], sc["uniform"], days, 400 if FAST else 1000)
    comp["full_vs_picks_only"] = cluster_boot_diff(sc["full"], sc["picks_only"], days, 400 if FAST else 1000)
    # staleness: gain by months after the training cut
    stale = defaultdict(lambda: ([], []))
    for s, a, b in zip(test, sc["full"], sc["field_only"]):
        m = (s["ord"] - d_ord(TEST_START)) // 60
        stale[m][0].append(a - b)
        stale[m][1].append(s["ord"])
    R["heldout_latent_model"] = {
        "train_window": [ctx.iso(min(s["ord"] for s in train)), TRAIN_END], "n_train": len(train), "n_test": len(test),
        "kappa_inner_val": {str(k): round(v, 4) for k, v in inner.items()}, "kappa": kstar,
        "kappa_pick_x4_inner_val": round(inner_pk, 4), "kappa_pick_x4_chosen": bool(use_split),
        "mean_loglik_test": {k: round(mean(v), 4) for k, v in sc.items()},
        "gain_nats_per_series": comp,
        "gain_by_60day_block_after_cut": {f"{m * 60}-{m * 60 + 59}d": {"gain_full_vs_field": round(mean(v[0]), 4), "n": len(v[0])}
                                          for m, v in sorted(stale.items())},
        "note": "log P(observed map order) summed over hidden starter and bans; gains are team-specific "
                "habits vs the field-only weights, frozen at the training cut",
        "pi_a_starts": {k: round(v, 3) for k, v in fits["full"]["pi"].items()},
    }
    print(f"  q1 A done {time.time() - t0:.0f}s kappa*={kstar}", flush=True)
    # decay: fit on the same train with exponential recency weights
    dec = {}
    for hl in ((60, 180) if FAST else (45, 90, 180)):
        w = [0.5 ** ((d_ord(TRAIN_END) - s["ord"]) / hl) for s in train]
        f = em_fit(train, kappa=kfull, iters=iters, weights=w)
        sc_hl = score_series(test, f)
        dec[f"hl{hl}"] = {"mean_loglik_test": round(mean(sc_hl), 4),
                          "gain_vs_no_decay": cluster_boot_diff(sc_hl, sc["full"], days, 400 if FAST else 1000)}
    dec["no_decay_full"] = {"mean_loglik_test": round(mean(sc["full"]), 4)}
    R["heldout_latent_decay"] = dec
    print(f"  q1 decay done {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- B. observable played-map habits: windows / decay
    te_lo, hi = d_ord(TEST_START), ctx.as_of
    rows, meta = drift_windows(ctx.valid, ctx.tp, ctx.fr, ctx.top_teams, te_lo, hi)
    R["window_decay_observed"] = {
        "target": "log P(played set) as 7 independent map indicators; gain = nats per team-series vs the field-rate baseline",
        "test_window": [TEST_START, ctx.iso(hi)],
        "top30": summarize_drift(rows, meta, lambda m: m[2]),
        "all_teams": summarize_drift(rows, meta, lambda m: True),
    }
    tr_hi = d_ord(TRAIN_END)
    rows_tr, meta_tr = drift_windows(ctx.valid, ctx.tp, ctx.fr, ctx.top_teams, tr_hi - 270, tr_hi)
    R["window_decay_observed"]["train_all_teams"] = summarize_drift(rows_tr, meta_tr, lambda m: True)
    print(f"  q1 B done {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- C. descriptive per-team distributions (last 365 days)
    recent, fit, post = get_recent(ctx, kstar, iters)
    maps_all = fit["maps"]
    team_vecs = defaultdict(lambda: {k: [] for k in KINDS + ("dec",)})
    dec_obs = defaultdict(list)
    team_meta = defaultdict(lambda: {"tiers": Counter(), "n": 0, "opp_series": defaultdict(float)})
    for s, mg in zip(recent, post):
        if mg is None:
            continue
        for t in ("a", "b"):
            tm = s[t]
            for k in KINDS:
                team_vecs[tm][k].append((s["ord"], {mp: mg[t][k][i] for i, mp in enumerate(s["pool"])}, s["pool"]))
            if len(s["maps"]) >= 3:
                dec_obs[tm].append((s["ord"], s["maps"][2][0], s["pool"]))
            team_meta[tm]["n"] += 1
            team_meta[tm]["tiers"][s["tier"]] += 1
        # decider posterior is a series-level quantity: both teams share it
        for t in ("a", "b"):
            team_vecs[s[t]]["dec"].append((s["ord"], {mp: mg["decider"][i] for i, mp in enumerate(s["pool"])}, s["pool"]))
    # field shares per kind (all teams)
    pool_last = ctx.clock.pool(ctx.as_of)
    field = {}
    for k in KINDS + ("dec",):
        c = Counter()
        for tm, d in team_vecs.items():
            for (_, v, _) in d[k]:
                for mp, x in v.items():
                    c[mp] += x
        tot = sum(c.values())
        field[k] = {mp: c[mp] / tot for mp in maps_all}
    teams_out = {}
    nboot = 60 if FAST else 150
    fpl, fop = Counter(), Counter()
    for s in recent:
        names = {mp for mp, _ in s["maps"]}
        for mp in s["pool"]:
            fop[mp] += 1
            fpl[mp] += mp in names
    frate = {mp: fpl[mp] / fop[mp] for mp in fop}
    for r in ctx.top:
        tm = r["team"]
        if team_meta[tm]["n"] < MIN_TEAM_SERIES:
            teams_out[tm] = {"vrs_rank": r["rank"], "n_series": team_meta[tm]["n"], "status": "too_few_series"}
            continue
        entry = {"vrs_rank": r["rank"], "vrs_name": r["vrs_name"], "n_series": team_meta[tm]["n"],
                 "tiers": dict(team_meta[tm]["tiers"]), "kinds": {}}
        for k in KINDS + ("dec",):
            items = sorted(team_vecs[tm][k], key=lambda x: x[0])
            vecs = [v for _, v, _ in items]
            fs = [field[k][mp] for mp in maps_all]
            st, c = series_stats(vecs, maps_all, nboot, rng, fs)
            st["repeat_rate"] = repeat_rate(vecs, rng)
            entry["kinds"][k] = st
            # per-map rate with the number of series where that map was in the pool
            opp = Counter()
            for _, v, pool in items:
                for mp in pool:
                    opp[mp] += 1
            st["per_map"] = {}
            for mp in maps_all:
                if opp[mp] == 0:
                    continue
                lo, hi_ = wilson(c[mp], opp[mp])
                st["per_map"][mp] = {"rate_when_in_pool": round(c[mp] / opp[mp], 4), "ci95": [round(lo, 4), round(hi_, 4)],
                                     "opportunities": opp[mp], "field_share": round(field[k][mp], 4)}
        # observed decider in 2-1 series (no latent variables)
        ds = sorted(dec_obs[tm])
        if ds:
            dm = Counter(m for _, m, _ in ds)
            nd = len(ds)
            top, topc = dm.most_common(1)[0]
            sh = [dm.get(mp, 0) / nd for mp in maps_all]
            entry["decider_observed_2_1"] = {
                "n": nd, "top_map": top, "top_share": round(topc / nd, 3),
                "top_share_wilson": [round(x, 3) for x in wilson(topc, nd)],
                "entropy_norm": round(entropy(sh) / math.log(len(maps_all)), 3),
                "counts": dict(dm)}
        # played-map share (observed)
        pl, op = Counter(), Counter()
        for s in recent:
            if tm in (s["a"], s["b"]):
                names = {mp for mp, _ in s["maps"]}
                for mp in s["pool"]:
                    op[mp] += 1
                    pl[mp] += mp in names
        entry["played_share_observed"] = {mp: {"rate": round(pl[mp] / op[mp], 3), "n": op[mp],
                                                "ci95": [round(x, 3) for x in wilson(pl[mp], op[mp])]}
                                          for mp in maps_all if op[mp]}
        # observed played share vs the field (chi-square on the 7 map indicators, df 6: approximate, the
        # indicators are tied by the number of maps played)
        chi = 0.0
        zs = {}
        for mp in maps_all:
            if op[mp] and 0 < frate[mp] < 1:
                z = (pl[mp] - op[mp] * frate[mp]) / math.sqrt(op[mp] * frate[mp] * (1 - frate[mp]))
                zs[mp] = round(z, 2)
                chi += z * z
        zmax = max(zs, key=lambda m: abs(zs[m])) if zs else None
        entry["played_vs_field"] = {"chi2": round(chi, 2), "p": round(chi2_sf(chi, 6), 5), "z_by_map": zs,
                                    "most_deviant_map": zmax, "z_most_deviant": zs.get(zmax)}
        # class per kind
        entry["class"] = {k: classify(entry["kinds"][k]) for k in KINDS + ("dec",)}
        teams_out[tm] = entry
    # Holm over teams for the JS test, per kind
    for k in KINDS + ("dec",):
        names = [t for t in teams_out if "kinds" in teams_out[t]]
        adj = holm([teams_out[t]["kinds"][k]["p_js_vs_field"] for t in names])
        for t, a in zip(names, adj):
            teams_out[t]["kinds"][k]["p_js_vs_field_holm"] = round(a, 4)
    names_ok = [t for t in teams_out if "kinds" in teams_out[t]]
    # one-step posterior (field weights only, no team habit fed back): a check that EM sharpening is not driving the shares
    one_fit = dict(fit)
    one_fit["W"] = {}
    post1 = posterior_pass(recent, one_fit)
    acc1 = defaultdict(lambda: {k: defaultdict(float) for k in ("b1", "pk")})
    n1 = Counter()
    for s_, mg in zip(recent, post1):
        if mg is None:
            continue
        for t_ in ("a", "b"):
            n1[s_[t_]] += 1
            for k in ("b1", "pk"):
                for i, mp in enumerate(s_["pool"]):
                    acc1[s_[t_]][k][mp] += mg[t_][k][i]
    for t in names_ok:
        for k in ("b1", "pk"):
            tm_ = teams_out[t]["kinds"][k]["top_map"]
            teams_out[t]["kinds"][k]["top_share_one_step_field_prior"] = round(acc1[t][k][tm_] / n1[t], 4)
    chis = [teams_out[t]["played_vs_field"]["chi2"] for t in names_ok]
    med = quantile(chis, 0.5)
    mad = quantile([abs(x - med) for x in chis], 0.5) * 1.4826 or 1.0
    for t in names_ok:
        teams_out[t]["played_vs_field"]["chi2_robust_z_among_top"] = round((teams_out[t]["played_vs_field"]["chi2"] - med) / mad, 2)
    adj = holm([teams_out[t]["played_vs_field"]["p"] for t in names_ok])
    for t, a in zip(names_ok, adj):
        teams_out[t]["played_vs_field"]["p_holm"] = round(a, 5)
    R["teams"] = teams_out
    R["field_share"] = {k: {mp: round(v, 4) for mp, v in field[k].items()} for k in field}
    R["null_top_share_hard_draws"] = null_top_share_table([field["pk"][mp] for mp in maps_all], rng)
    R["fit_recent"] = {"window": [ctx.iso(ctx.as_of - RECENT_DAYS), ctx.iso(ctx.as_of)], "n_series": len(recent),
                       "kappa": kstar, "pi_a_starts": {k: round(v, 3) for k, v in fit["pi"].items()},
                       "mean_loglik_in_sample": round(fit["loglik"], 4)}
    # summary across teams
    summ = {}
    ok = [t for t in teams_out if "kinds" in teams_out[t]]
    for k in KINDS + ("dec",):
        cl = Counter(teams_out[t]["class"][k] for t in ok)
        summ[k] = {"classes": dict(cl), "n_teams": len(ok),
                   "median_entropy_norm": round(quantile([teams_out[t]["kinds"][k]["entropy_norm"] for t in ok], 0.5), 3),
                   "median_top_share": round(quantile([teams_out[t]["kinds"][k]["top_share"] for t in ok], 0.5), 3),
                   "median_repeat_excess": round(quantile([teams_out[t]["kinds"][k]["repeat_rate"]["excess"]
                                                           for t in ok if teams_out[t]["kinds"][k]["repeat_rate"]], 0.5), 4),
                   "outliers_vs_field_holm05": sorted(t for t in ok if teams_out[t]["kinds"][k]["p_js_vs_field_holm"] < 0.05)}
    R["summary"] = summ
    R["summary"]["played_share_vs_field"] = {
        "teams_holm05": sum(1 for t in ok if teams_out[t]["played_vs_field"]["p_holm"] < 0.05), "n_teams": len(ok),
        "most_deviant_among_top": sorted(
            ((t, teams_out[t]["played_vs_field"]["most_deviant_map"], teams_out[t]["played_vs_field"]["z_most_deviant"],
              teams_out[t]["played_vs_field"]["chi2_robust_z_among_top"]) for t in ok), key=lambda x: -x[3])[:8]}
    cb = []
    for t in ok:
        kb = teams_out[t]["kinds"]["b1"]
        ps_ = teams_out[t]["played_share_observed"].get(kb["top_map"])
        if ps_:
            kb["top_map_observed_avoid_rate"] = [round(1 - ps_["rate"], 3), round(1 - ps_["ci95"][1], 3), round(1 - ps_["ci95"][0], 3)]
            if teams_out[t]["class"]["b1"] == "constant":
                cb.append({"team": t, "map": kb["top_map"], "first_ban_share": kb["top_share"], "wilson": kb["top_share_wilson"],
                           "observed_avoid_rate": kb["top_map_observed_avoid_rate"]})
    R["summary"]["constant_first_ban_teams"] = sorted(cb, key=lambda x: -x["first_ban_share"])
    R["summary"]["constant_first_ban_also_strongly_avoided"] = [x["team"] for x in R["summary"]["constant_first_ban_teams"]
                                                               if x["observed_avoid_rate"][1] >= 0.70]

    # ---------------------------------------------------------------- D. per-team drift (last 90d vs prior 90d, observed played share)
    R["drift_by_team"] = team_drift(ctx, rng)
    rep["q1"] = R
    print(f"  q1 done {time.time() - t0:.0f}s", flush=True)


def team_drift(ctx, rng, n_perm=400):
    """JS divergence between the played-map share of the last 90 days and the
    90 days before, against a permutation null that shuffles series between the
    two periods."""
    o = ctx.as_of + 1
    out = {}
    pools = ctx.clock.pool(o)
    ps = []
    for r in ctx.top:
        tm = r["team"]
        ev = ctx.tp.before(tm, o, 180)
        a = [e for e in ev if e[0] >= o - 90]
        b = [e for e in ev if e[0] < o - 90]
        if len(a) < 6 or len(b) < 6:
            out[tm] = {"n_recent": len(a), "n_prior": len(b), "status": "too_few"}
            continue

        def shares(evs):
            pl, op = Counter(), Counter()
            for (_, names, pool, _, _) in evs:
                for mp in pool:
                    op[mp] += 1
                    pl[mp] += mp in names
            return [(pl[mp] + 0.5) / (op[mp] + 1.0) for mp in pools], {mp: (pl[mp], op[mp]) for mp in pools}
        sa, ca = shares(a)
        sb, cb = shares(b)
        sa, sb = norm_vec(sa), norm_vec(sb)
        obs = js_div(sa, sb)
        allv = a + b
        nulls = []
        for _ in range(n_perm if not FAST else 150):
            rng.shuffle(allv)
            x, _ = shares(allv[:len(a)])
            y, _ = shares(allv[len(a):])
            nulls.append(js_div(norm_vec(x), norm_vec(y)))
        p = (1 + sum(1 for z in nulls if z >= obs)) / (len(nulls) + 1)
        d = {mp: round(sa[i] - sb[i], 3) for i, mp in enumerate(pools)}
        mv = max(d, key=lambda m: abs(d[m]))
        out[tm] = {"n_recent": len(a), "n_prior": len(b), "js": round(obs, 4), "null_mean": round(mean(nulls), 4),
                   "p": round(p, 4), "largest_shift_map": mv, "largest_shift_share_pts": round(100 * d[mv], 1)}
        ps.append((tm, p))
    adj = holm([p for _, p in ps])
    for (tm, _), a in zip(ps, adj):
        out[tm]["p_holm"] = round(a, 4)
    return out


# ============================================================================
# point-in-time pass: Elo, map residuals, soft pick attribution
# ============================================================================
def map_prob_from_series(p, bo):
    """Per-map win chance q such that a best-of-bo series is won with chance p."""
    if bo == 1:
        return p

    def win(q):
        if bo == 3:
            return q * q * (3 - 2 * q)
        if bo == 5:
            return sum(math.comb(2 + k, k) * q ** 3 * (1 - q) ** k for k in range(3))
        return q
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if win(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class PitPass:
    """Chronological pass over `series` (any best-of). For every valid BO3
    series (key 'pool' present, >= 2 maps) a snapshot of what was known BEFORE
    its date is stored in .snap[k]. Same-day series never see each other.

    Elo: plain series-level Elo (K 20, start 1500). Map residual of a played
    map = won - q, q = per-map chance implied by the pre-series Elo.
    Pick attribution: P(team_a started | m1, m2) from the teams' running pick
    shares (decayed, shrunk to the field) and a fixed prior PI_A; the map-1
    picker is the starter, the map-2 picker the other team."""
    K = 20.0
    PI_A = 0.42
    HL = 120.0
    KP = 4.0

    def __init__(self, series, win=180):
        self.win = win
        self.elo = defaultdict(lambda: 1500.0)
        self.recs = defaultdict(list)       # team -> [(ord, [(map, won, q, w)])]
        self.pk = defaultdict(lambda: (defaultdict(float), [0]))   # team -> (counts, [last ord])
        self.pk_field = defaultdict(float)
        self.snap = {}
        self.attr = {}
        self.attr_conf = []
        i = 0
        while i < len(series):
            j = i
            while j < len(series) and series[j]["ord"] == series[i]["ord"]:
                j += 1
            day = series[i:j]
            for s in day:
                if "pool" in s and len(s["maps"]) >= 2 and s["bo"] == 3:
                    self.snap[s["k"]] = self._snapshot(s)
            for s in day:
                self._fold(s)
            i = j

    # -- running pick shares (exponential decay, lazily applied)
    def _pick_share(self, team, o, pool):
        c, last = self.pk[team]
        f = 0.5 ** ((o - last[0]) / self.HL) if last[0] else 1.0
        n = sum(c.values()) * f
        gtot = sum(self.pk_field[mp] + 0.5 for mp in pool)
        return {mp: (c.get(mp, 0.0) * f + self.KP * (self.pk_field[mp] + 0.5) / gtot) / (n + self.KP) for mp in pool}

    def _bump_pick(self, team, o, mp, w):
        c, last = self.pk[team]
        if last[0]:
            f = 0.5 ** ((o - last[0]) / self.HL)
            for k in list(c):
                c[k] *= f
        last[0] = o
        c[mp] += w

    def attribution(self, s):
        pool = s["pool"]
        pa = self._pick_share(s["a"], s["ord"], pool)
        pb = self._pick_share(s["b"], s["ord"], pool)
        m1, m2 = s["maps"][0][0], s["maps"][1][0]
        la = self.PI_A * pa[m1] * pb[m2]
        lb = (1 - self.PI_A) * pb[m1] * pa[m2]
        return la / (la + lb)       # P(team_a picked map 1, i.e. started)

    def _team_stats(self, team, o):
        lo = o - self.win
        recs = self.recs[team]
        pick = [0.0, 0.0]
        opp = [0.0, 0.0]
        dec = [0.0, 0.0]
        allr = [0.0, 0.0]
        per = defaultdict(lambda: [0.0, 0])
        nser = 0
        for (ro, mps) in reversed(recs):
            if ro >= o:
                continue
            if ro < lo:
                break
            nser += 1
            for (mp, won, q, w) in mps:
                r = won - q
                allr[0] += r
                allr[1] += 1
                per[mp][0] += r
                per[mp][1] += 1
                if w is None:
                    dec[0] += r
                    dec[1] += 1
                elif w >= 0:
                    pick[0] += w * r
                    pick[1] += w
                    opp[0] += (1 - w) * r
                    opp[1] += (1 - w)
        return {"n_series": nser, "pick": pick, "opp": opp, "dec": dec, "all": allr,
                "per": {mp: tuple(v) for mp, v in per.items()}}

    def _snapshot(self, s):
        o = s["ord"]
        ea, eb = self.elo[s["a"]], self.elo[s["b"]]
        return {"elo_a": ea, "elo_b": eb, "p_elo": 1 / (1 + 10 ** ((eb - ea) / 400)),
                "a": self._team_stats(s["a"], o), "b": self._team_stats(s["b"], o)}

    def _fold(self, s):
        a, b = s["a"], s["b"]
        ea, eb = self.elo[a], self.elo[b]
        exp_a = 1 / (1 + 10 ** ((eb - ea) / 400))
        a_won = s["winner"] == a
        qa = map_prob_from_series(exp_a, s["bo"] if s["bo"] in (1, 3, 5) else 3)
        ra = None
        if "pool" in s and len(s["maps"]) >= 2 and s["bo"] == 3:
            ra = self.attribution(s)
            self.attr[s["k"]] = ra
            self.attr_conf.append(max(ra, 1 - ra))
        la, lb = [], []
        for idx, (mp, w) in enumerate(s["maps"]):
            won_a = 1.0 if w == a else 0.0
            if ra is not None and idx == 0:
                wa, wb = ra, 1 - ra
            elif ra is not None and idx == 1:
                wa, wb = 1 - ra, ra
            elif ra is not None:
                wa = wb = None
            else:
                wa = wb = -1.0
            la.append((mp, won_a, qa, wa))
            lb.append((mp, 1 - won_a, 1 - qa, wb))
        self.recs[a].append((s["ord"], la))
        self.recs[b].append((s["ord"], lb))
        if ra is not None:
            m1, m2 = s["maps"][0][0], s["maps"][1][0]
            self._bump_pick(a, s["ord"], m1, ra)
            self._bump_pick(a, s["ord"], m2, 1 - ra)
            self._bump_pick(b, s["ord"], m2, ra)
            self._bump_pick(b, s["ord"], m1, 1 - ra)
            self.pk_field[m1] += 1
            self.pk_field[m2] += 1
        k = self.K
        self.elo[a] = ea + k * (a_won - exp_a)
        self.elo[b] = eb + k * ((not a_won) - (1 - exp_a))


# ============================================================================
# Q2  opponent dependence
# ============================================================================
def quasi_indep_g2(pairs, k):
    """G-squared of independence with an empty diagonal (IPF); pairs = [(i, j)]."""
    N = [[0.0] * k for _ in range(k)]
    for i, j in pairs:
        N[i][j] += 1
    E = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    rs = [sum(r) for r in N]
    cs = [sum(N[i][j] for i in range(k)) for j in range(k)]
    for _ in range(200):
        for i in range(k):
            t = sum(E[i])
            if t:
                f = rs[i] / t
                E[i] = [x * f for x in E[i]]
        for j in range(k):
            t = sum(E[i][j] for i in range(k))
            if t:
                f = cs[j] / t
                for i in range(k):
                    E[i][j] *= f
    g = 0.0
    for i in range(k):
        for j in range(k):
            if N[i][j] > 0 and E[i][j] > 0:
                g += 2 * N[i][j] * math.log(N[i][j] / E[i][j])
    return g


def sample_order(s, W, G, pi_a, rng):
    """Draw (m1, m2) from the sequential-choice model (same order as the fit)."""
    pool = s["pool"]
    st = "a" if rng.random() < pi_a else "b"
    ot = "b" if st == "a" else "a"
    wS = {k: _wvec(W, G, s[st], k, pool) for k in KINDS}
    wO = {k: _wvec(W, G, s[ot], k, pool) for k in KINDS}
    open_ = list(range(len(pool)))

    def draw(w):
        tot = sum(w[i] for i in open_)
        u = rng.random() * tot
        acc = 0.0
        for i in open_:
            acc += w[i]
            if u <= acc:
                return i
        return open_[-1]
    for w in (wS["b1"], wO["b1"]):
        open_.remove(draw(w))
    m1 = draw(wS["pk"])
    open_.remove(m1)
    m2 = draw(wO["pk"])
    return m1, m2


def js_across_splits(items, key_split, maps, rng, n_perm):
    """items: [(set of played maps, pool, label)]; JS between rate vectors of
    label True / False vs permutation of labels."""
    def vec(sub):
        pl, op = Counter(), Counter()
        for names, pool, _ in sub:
            for mp in pool:
                op[mp] += 1
                pl[mp] += mp in names
        return norm_vec([(pl[mp] + 0.5) / (op[mp] + 1.0) for mp in maps])
    t = [x for x in items if x[2]]
    f = [x for x in items if not x[2]]
    obs = js_div(vec(t), vec(f))
    labs = [x[2] for x in items]
    cnt = 0
    nulls = []
    for _ in range(n_perm):
        rng.shuffle(labs)
        t2 = [x for x, l in zip(items, labs) if l]
        f2 = [x for x, l in zip(items, labs) if not l]
        v = js_div(vec(t2), vec(f2))
        nulls.append(v)
        cnt += v >= obs
    return obs, mean(nulls), (1 + cnt) / (n_perm + 1)


def q2(ctx, rep):
    t0 = time.time()
    rng = random.Random(SEED + 2)
    R = {}
    iters = 4 if FAST else 7
    n_perm = 150 if FAST else 400
    pit = PitPass(ctx.series_pit())
    # ---------------------------------------------------------------- A. strength split per top team (last 365d)
    lo = ctx.as_of - RECENT_DAYS
    by_team = defaultdict(list)
    for s in ctx.valid:
        if s["ord"] >= lo and s["k"] in pit.snap:
            sn = pit.snap[s["k"]]
            names = frozenset(mp for mp, _ in s["maps"])
            by_team[s["a"]].append((names, s["pool"], sn["elo_b"] - sn["elo_a"], s))
            by_team[s["b"]].append((names, s["pool"], sn["elo_a"] - sn["elo_b"], s))
    maps_all = sorted({mp for s in ctx.valid if s["ord"] >= lo for mp in s["pool"]})
    per_team, pvals = {}, []
    for r in ctx.top:
        tm = r["team"]
        its = by_team.get(tm, [])
        if len(its) < 24:
            per_team[tm] = {"n": len(its), "status": "too_few"}
            continue
        med = quantile([x[2] for x in its], 0.5)
        items = [(x[0], x[1], x[2] > med) for x in its]
        obs, nm, p = js_across_splits(items, None, maps_all, rng, n_perm)
        # sign split (opponent stronger than the team itself)
        n_up = sum(1 for x in its if x[2] > 0)
        d = {"n": len(its), "median_opp_minus_own_elo": round(med, 1), "js": round(obs, 4), "null_mean": round(nm, 4), "p": round(p, 4),
             "n_opponent_stronger": n_up}
        # which map shifts most between the halves
        def rate(sub):
            pl, op = Counter(), Counter()
            for names, pool, _ in sub:
                for mp in pool:
                    op[mp] += 1
                    pl[mp] += mp in names
            return {mp: pl[mp] / op[mp] for mp in op if op[mp] >= 5}
        hi_, lo_ = rate([x for x in items if x[2]]), rate([x for x in items if not x[2]])
        dd = {mp: hi_[mp] - lo_[mp] for mp in hi_ if mp in lo_}
        if dd:
            mv = max(dd, key=lambda m: abs(dd[m]))
            d["largest_shift_map"] = mv
            d["largest_shift_pts_vs_stronger_opp"] = round(100 * dd[mv], 1)
        def conc(sub):
            pl, op = Counter(), Counter()
            for names, pool, _ in sub:
                for mp in pool:
                    op[mp] += 1
                    pl[mp] += mp in names
            sh = norm_vec([(pl[mp] + 0.5) / (op[mp] + 1.0) for mp in maps_all])
            return entropy(sh) / math.log(len(maps_all))
        d["entropy_norm_vs_stronger"] = round(conc([x for x in items if x[2]]), 4)
        d["entropy_norm_vs_weaker"] = round(conc([x for x in items if not x[2]]), 4)
        per_team[tm] = d
        pvals.append(p)
    ok = [t for t in per_team if "p" in per_team[t]]
    adj = holm([per_team[t]["p"] for t in ok])
    for t, a in zip(ok, adj):
        per_team[t]["p_holm"] = round(a, 4)
    fisher = -2 * sum(math.log(per_team[t]["p"]) for t in ok)
    R["strength_split"] = {
        "design": "played-map rates, series vs the stronger half of the team's own opponents vs the weaker half (point-in-time Elo), "
                  "permutation test of the half label within team",
        "n_teams_tested": len(ok), "teams": per_team,
        "share_p_below_05": round(mean(1.0 if per_team[t]["p"] < 0.05 else 0.0 for t in ok), 3) if ok else None,
        "fisher_chi2": round(fisher, 2), "fisher_df": 2 * len(ok), "fisher_p": round(chi2_sf(fisher, 2 * len(ok)), 4) if ok else None,
        "mean_js_over_null": round(mean(per_team[t]["js"] / per_team[t]["null_mean"] for t in ok), 3) if ok else None,
        "holm_significant": sorted(t for t in ok if per_team[t]["p_holm"] < 0.05),
        "played_share_entropy_stronger_minus_weaker": {
            "mean": round(mean(per_team[t]["entropy_norm_vs_stronger"] - per_team[t]["entropy_norm_vs_weaker"] for t in ok), 4),
            "teams_narrower_vs_stronger": sum(1 for t in ok if per_team[t]["entropy_norm_vs_stronger"] < per_team[t]["entropy_norm_vs_weaker"]),
            "n": len(ok)}}
    print(f"  q2 A {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- B. first pick vs second pick: independence beyond habits
    train, test = era_split(ctx)
    iters2 = iters
    fit = em_fit(train, kappa=8.0, iters=iters2)
    # most common pool in the test window
    pc = Counter(s["pool"] for s in test)
    out_b = {}
    for pool, npool in pc.most_common(2):
        sub = [s for s in test if s["pool"] == pool]
        if len(sub) < 200:
            continue
        ix = {mp: i for i, mp in enumerate(pool)}
        pairs = [(ix[s["maps"][0][0]], ix[s["maps"][1][0]]) for s in sub]
        g_obs = quasi_indep_g2(pairs, len(pool))
        sims = []
        for _ in range(60 if FAST else 150):
            pr_ = []
            for s in sub:
                m1, m2 = sample_order(s, fit["W"], fit["G"], fit["pi"][s["src"]], rng)
                pr_.append((m1, m2))
            sims.append(quasi_indep_g2(pr_, len(pool)))
        out_b[",".join(pool)] = {
            "n_series": len(sub), "G2_observed": round(g_obs, 1), "G2_df": (len(pool) - 1) ** 2 - len(pool),
            "G2_chi2_p_asymptotic": round(chi2_sf(g_obs, (len(pool) - 1) ** 2 - len(pool)), 5),
            "G2_simulated_from_habit_model_mean": round(mean(sims), 1),
            "G2_simulated_p95": round(quantile(sims, 0.95), 1),
            "p_observed_vs_simulated": round((1 + sum(1 for x in sims if x >= g_obs)) / (len(sims) + 1), 4)}
    # ---- who starts the veto: does the rating gap move P(team_a starts)?
    post_tr = posterior_pass(train, fit)
    dtr = [(pit.snap[s_["k"]]["elo_a"] - pit.snap[s_["k"]]["elo_b"]) / 100.0 if s_["k"] in pit.snap else 0.0 for s_ in train]
    ytr = [m_["pi"] if m_ else 0.4 for m_ in post_tr]
    beta_pi = logistic_fit([[1.0, d_] for d_ in dtr], ytr, l2=0.01)
    dte = [(pit.snap[s_["k"]]["elo_a"] - pit.snap[s_["k"]]["elo_b"]) / 100.0 if s_["k"] in pit.snap else 0.0 for s_ in test]
    pis = [min(0.95, max(0.05, sigmoid(beta_pi[0] + beta_pi[1] * d_))) for d_ in dte]
    const = [fit["pi"][s_["src"]] for s_ in test]
    ll_c = score_series_pi(test, fit, const)
    ll_p = score_series_pi(test, fit, pis)
    qd = defaultdict(list)
    for s_, m_ in zip(train, post_tr):
        if m_ is not None and s_["k"] in pit.snap:
            d_ = (pit.snap[s_["k"]]["elo_a"] - pit.snap[s_["k"]]["elo_b"])
            qd[0 if d_ < -100 else 1 if d_ < -30 else 2 if d_ < 30 else 3 if d_ < 100 else 4].append(m_["pi"])
    R["starter_prior"] = {
        "design": "posterior P(team_a starts) from the fitted veto model vs the point-in-time Elo gap (team_a minus team_b, per 100 Elo)",
        "logit_intercept": round(beta_pi[0], 3), "logit_slope_per_100_elo": round(beta_pi[1], 3),
        "mean_posterior_by_gap_bucket(train)": {lab: [round(mean(qd[i]), 3), len(qd[i])] for i, lab in
                                                enumerate(["gap<-100", "-100..-30", "-30..30", "30..100", ">100"]) if qd[i]},
        "test_loglik_gain_vs_constant_pi": cluster_boot_diff(ll_p, ll_c, [s_["ord"] for s_ in test], 300 if FAST else 1000),
        "pi_constant_fit": {k_: round(v_, 3) for k_, v_ in fit["pi"].items()},
        "note": "the engine's point veto always lets team_a start; the fits put P(team_a starts) well below 0.5"}
    R["pick_pair_dependence"] = {
        "design": "G2 of (map 1, map 2) quasi-independence vs the same statistic on orders simulated from the habit model "
                  "(independent picks given team habits and bans), fit on training window only",
        "pools": out_b}
    print(f"  q2 B {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- C. map-played model with matchup terms
    fr = ctx.fr
    tp = ctx.tp

    def rows_for(series_list, kappa_res=6.0):
        X, y, meta = [], [], []
        for s in series_list:
            sn = pit.snap.get(s["k"])
            if sn is None:
                continue
            o = s["ord"]
            ha = habit_prob(tp.before(s["a"], o, 365), o, s["pool"], fr, "hl", 90)[0]
            hb = habit_prob(tp.before(s["b"], o, 365), o, s["pool"], fr, "hl", 90)[0]
            names = {mp for mp, _ in s["maps"]}
            dE = abs(sn["elo_a"] - sn["elo_b"]) / 100.0
            for mp in s["pool"]:
                f = min(max(fr.rate(mp, o), 0.01), 0.99)
                la = math.log(ha[mp] / (1 - ha[mp]))
                lb = math.log(hb[mp] / (1 - hb[mp]))
                ra = sn["a"]["per"].get(mp, (0.0, 0))
                rb = sn["b"]["per"].get(mp, (0.0, 0))
                ea = ra[0] / (ra[1] + kappa_res)
                eb = rb[0] / (rb[1] + kappa_res)
                edge = ea - eb
                X.append({"c": 1.0, "lf": math.log(f / (1 - f)), "mean_l": (la + lb) / 2, "dis_l": abs(la - lb),
                          "abs_edge": abs(edge), "both_pos": max(0.0, min(ea, eb)) + max(0.0, min(-ea, -eb)),
                          "mean_l_x_gap": (la + lb) / 2 * dE})
                y.append(1.0 if mp in names else 0.0)
                meta.append(s["ord"])
        return X, y, meta
    tr_rows = rows_for(train)
    te_rows = rows_for(test)
    specs = {"field_only": ["c", "lf"],
             "+team_habits": ["c", "lf", "mean_l", "dis_l"],
             "+matchup_edge": ["c", "lf", "mean_l", "dis_l", "abs_edge", "both_pos"],
             "+strength_interaction": ["c", "lf", "mean_l", "dis_l", "mean_l_x_gap"]}
    outc = {}
    losses = {}
    for name, cols in specs.items():
        Xtr = [[r[c] for c in cols] for r in tr_rows[0]]
        Xte = [[r[c] for c in cols] for r in te_rows[0]]
        beta = logistic_fit(Xtr, tr_rows[1], l2=1.0)
        ps = [sigmoid(sum(b * v for b, v in zip(beta, x))) for x in Xte]
        e = 1e-12
        losses[name] = [-(yy * math.log(max(p, e)) + (1 - yy) * math.log(max(1 - p, e))) for p, yy in zip(ps, te_rows[1])]
        outc[name] = {"coef": {c: round(b, 4) for c, b in zip(cols, beta)}, "test_logloss": round(mean(losses[name]), 5)}
    # day-cluster CI on paired differences (rows -> series blocks of 7: use the date list)
    for name in specs:
        if name != "field_only":
            base = "+team_habits" if name in ("+matchup_edge", "+strength_interaction") else "field_only"
            outc[name]["gain_vs_" + base] = cluster_boot_diff(losses[base], losses[name], te_rows[2], 300 if FAST else 800)
    R["matchup_model"] = {"design": "logistic on (series, pool map) rows, y = map was played; point-in-time habits (decay 90d), "
                                    "matchup edge = shrunk Elo-residual map result gap; fit <= %s, scored >= %s" % (TRAIN_END, TEST_START),
                          "n_train_rows": len(tr_rows[1]), "n_test_rows": len(te_rows[1]), "models": outc}
    R["attribution_confidence_mean"] = round(mean(pit.attr_conf), 3)
    rep["q2"] = R
    print(f"  q2 done {time.time() - t0:.0f}s", flush=True)


# ============================================================================
# Q3  roster changes
# ============================================================================
def load_lineups():
    with open(os.path.join(DATA, "lineups.json"), encoding="utf-8") as f:
        return json.load(f)["lineups"]


def q3(ctx, rep):
    t0 = time.time()
    rng = random.Random(SEED + 3)
    lu = load_lineups()
    main = [s for s in ctx.series if s["src"] == "main"]
    # lineup chain per team (all main series, any best-of)
    chain = defaultdict(list)         # team -> [(ord, frozenset, k)]
    key_info = {}                     # (k, side) -> dict
    for s in main:
        rec = lu.get(f"{s['date']}|{s['a']}|{s['b']}")
        if not rec:
            continue
        for side in ("a", "b"):
            five = rec.get(side)
            if five and len(five) == 5:
                chain[s[side]].append((s["ord"], frozenset(five), s["k"]))
    cover = sum(len(v) for v in chain.values())
    # change descriptors
    desc = {}
    for team, ch in chain.items():
        same_run = 0
        for i, (o, five, k) in enumerate(ch):
            if i == 0:
                desc[(k, team)] = {"change": None, "since": 0, "gap": None, "core_overlap": None}
                continue
            prev = ch[i - 1][1]
            ch_k = 5 - len(five & prev)
            same_run = same_run + 1 if ch_k == 0 else 0
            recent = ch[max(0, i - 8):i]
            cnt, seen = Counter(), {}
            for j, (_, f, _) in enumerate(recent):
                for pl in f:
                    cnt[pl] += 1
                    seen[pl] = j
            core = frozenset(sorted(cnt, key=lambda pl: (-cnt[pl], -seen[pl], pl))[:5])
            desc[(k, team)] = {"change": ch_k, "since": same_run, "gap": o - ch[i - 1][0], "core_overlap": len(five & core)}
    # main-only habit machinery (so history matches the lineup coverage)
    vmain = [s for s in ctx.valid if s["src"] == "main"]
    fr = FieldRates(vmain)
    tp = TeamPlays(vmain)
    lineup_of = {}
    for team, ch in chain.items():
        for o, five, k in ch:
            lineup_of[(k, team)] = five
    rows = []
    for s in vmain:
        names = frozenset(mp for mp, _ in s["maps"])
        for t in (s["a"], s["b"]):
            d = desc.get((s["k"], t))
            if not d or d["change"] is None:
                continue
            ev = tp.before(t, s["ord"], 365)
            prior180 = [e for e in ev if e[0] >= s["ord"] - 180]
            if len(prior180) < 4:
                continue
            base = {mp: fr.rate(mp, s["ord"]) for mp in s["pool"]}
            sb = bern_score(base, s["pool"], names)
            q_all, _ = habit_prob(ev, s["ord"], s["pool"], fr, "hl", 90)
            g_all = bern_score(q_all, s["pool"], names) - sb
            cur = lineup_of[(s["k"], t)]
            same = [e for e in ev if (lineup_of.get((e[3], t)) is not None and len(lineup_of[(e[3], t)] & cur) >= 4)]
            q_same, _ = habit_prob(same, s["ord"], s["pool"], fr, "hl", 90)
            g_same = bern_score(q_same, s["pool"], names) - sb
            q_mix = {mp: (q_all[mp] + q_same[mp]) / 2 for mp in s["pool"]}
            g_mix = bern_score(q_mix, s["pool"], names) - sb
            rows.append({"team": t, "ord": s["ord"], "change": d["change"], "since": d["since"], "core": d["core_overlap"],
                         "n_same": len(same), "g_all": g_all, "g_same": g_same, "g_mix": g_mix})
    def cat(r):
        return "0 players" if r["change"] == 0 else ("1 player" if r["change"] == 1 else "2+ players")
    out = {}
    for name in ("0 players", "1 player", "2+ players"):
        sub = [r for r in rows if cat(r) == name]
        out[name] = {"all_history": cluster_boot_mean([r["g_all"] for r in sub], [r["ord"] for r in sub], 500 if FAST else 1000)}
        sub2 = [r for r in sub if r["n_same"] >= 2]
        out[name]["same_core_history"] = cluster_boot_mean([r["g_same"] for r in sub2], [r["ord"] for r in sub2], 500 if FAST else 1000)
        out[name]["same_core_rows"] = len(sub2)
        out[name]["all_history_on_same_core_rows"] = cluster_boot_mean([r["g_all"] for r in sub2], [r["ord"] for r in sub2], 500 if FAST else 1000)
        out[name]["blend_on_same_core_rows"] = cluster_boot_mean([r["g_mix"] for r in sub2], [r["ord"] for r in sub2], 500 if FAST else 1000)
    # recovery: rows with a stable lineup, by series since the last change
    rec_out = {}
    for lab, lo_, hi_ in (("1-3", 1, 3), ("4-8", 4, 8), ("9+", 9, 10 ** 6)):
        sub = [r for r in rows if r["change"] == 0 and lo_ <= r["since"] <= hi_]
        rec_out[lab] = cluster_boot_mean([r["g_all"] for r in sub], [r["ord"] for r in sub], 500 if FAST else 1000)
    # the first series with a new lineup (since==0 and change>0) is covered by the 1 / 2+ rows above
    # ---- pre / post habit distance around change events
    maps_all = sorted({mp for s in vmain if s["ord"] >= ctx.as_of - 540 for mp in s["pool"]})

    def share_vec(evs):
        pl, op = Counter(), Counter()
        for (_, names, pool, _, _) in evs:
            for mp in pool:
                op[mp] += 1
                pl[mp] += mp in names
        return pl, op

    def jsd(pre, post):
        a, b = share_vec(pre), share_vec(post)
        ms = [mp for mp in maps_all if a[1][mp] and b[1][mp]]
        if len(ms) < 4:
            return None
        va = norm_vec([(a[0][mp] + 0.5) / (a[1][mp] + 1.0) for mp in ms])
        vb = norm_vec([(b[0][mp] + 0.5) / (b[1][mp] + 1.0) for mp in ms])
        return js_div(va, vb)
    W = 8
    ev_pairs = {"major(2+)": [], "minor(1)": [], "stable": []}
    for team, ch in chain.items():
        evs = tp.ev.get(team, [])
        if len(evs) < 2 * W:
            continue
        ks = {e[3]: i for i, e in enumerate(evs)}
        for (o, five, k) in ch:
            i = ks.get(k)
            if i is None or i < W or i + W > len(evs):
                continue
            d = desc.get((k, team))
            if not d or d["change"] is None:
                continue
            pre, post = evs[i - W:i], evs[i:i + W]
            if post[-1][0] - pre[0][0] > 365:
                continue
            # lineups inside the post window must stay mostly the new one; none changes inside the pre window
            inner = [desc.get((e[3], team)) for e in evs[i - W + 1:i + W] if (e[3], team) in desc and e[3] != k]
            inner_changes = [x["change"] for x in inner if x and x["change"] is not None and x["change"] > 0]
            j = jsd(pre, post)
            if j is None:
                continue
            if d["change"] >= 2:
                ev_pairs["major(2+)"].append(j)
            elif d["change"] == 1:
                ev_pairs["minor(1)"].append(j)
            elif not inner_changes and i % 4 == 0:      # thinned control windows, no change anywhere inside
                ev_pairs["stable"].append(j)
    dist = {}
    for kname, v in ev_pairs.items():
        if v:
            bs = []
            for _ in range(500):
                bs.append(mean(v[rng.randrange(len(v))] for _ in v))
            dist[kname] = {"n": len(v), "mean_js": round(mean(v), 4), "ci95": [round(quantile(bs, 0.025), 4), round(quantile(bs, 0.975), 4)]}
    perm = None
    if ev_pairs["major(2+)"] and ev_pairs["stable"]:
        a_, b_ = ev_pairs["major(2+)"], ev_pairs["stable"]
        obs = mean(a_) - mean(b_)
        allv = a_ + b_
        cnt = 0
        for _ in range(1000):
            rng.shuffle(allv)
            cnt += (mean(allv[:len(a_)]) - mean(allv[len(a_):])) >= obs
        perm = {"diff_major_minus_stable": round(obs, 4), "p_one_sided": round((1 + cnt) / 1001, 4)}
    # share of top-30 teams that changed lineup recently
    recent_changes = {}
    for r in ctx.top:
        ch = chain.get(r["team"], [])
        if not ch:
            continue
        last_change = None
        for i in range(len(ch) - 1, 0, -1):
            if 5 - len(ch[i][1] & ch[i - 1][1]) >= 1:
                last_change = (ch[i][0], 5 - len(ch[i][1] & ch[i - 1][1]))
                break
        n365 = sum(1 for i in range(1, len(ch)) if ch[i][0] >= ctx.as_of - 365 and 5 - len(ch[i][1] & ch[i - 1][1]) >= 1)
        big365 = sum(1 for i in range(1, len(ch)) if ch[i][0] >= ctx.as_of - 365 and 5 - len(ch[i][1] & ch[i - 1][1]) >= 2)
        recent_changes[r["team"]] = {"changes_last_365d": n365, "changes_2plus_last_365d": big365, "last_lineup_seen": ctx.iso(ch[-1][0]),
                                     "last_change_date": ctx.iso(last_change[0]) if last_change else None,
                                     "players_changed": last_change[1] if last_change else None}
    rep["q3"] = {
        "coverage": {"lineup_rows": cover, "teams": len(chain), "scored_team_series": len(rows),
                     "note": "lineups exist for matches.json series only, so this section uses main-tier history for the habit predictor"},
        "habit_skill_by_change": {"unit": "log-score nats per team-series vs the field-rate baseline; habits = 90-day-half-life played-map rates",
                                  "by_players_changed_vs_previous_series": out,
                                  "stable_lineup_by_series_since_last_change": rec_out},
        "pre_post_distance": {"window_series": W, "groups": dist, "major_vs_stable_perm": perm,
                              "note": "JS divergence of played-map shares, 8 series before vs 8 series from the first new-lineup series"},
        "rows_core_overlap_share": {
            "core_overlap_lt4": sum(1 for r in rows if r["core"] is not None and r["core"] < 4),
            "all": len(rows)},
        "top30_recent_lineup_state": recent_changes,
    }
    print(f"  q3 done {time.time() - t0:.0f}s", flush=True)


# ============================================================================
# Q4  does pick-side map strength add to the win probability?
# ============================================================================
K_SHRINK_RES = 6.0


def feat_row(sn, k_s=K_SHRINK_RES):
    """Differences (team_a - team_b) of shrunk Elo-residual map results."""
    def res(st, key):
        v = st[key]
        return v[0] / (v[1] + k_s)
    out = {}
    for name, key in (("d_all", "all"), ("d_pick", "pick"), ("d_opp", "opp"), ("d_dec", "dec")):
        out[name] = res(sn["a"], key) - res(sn["b"], key)
    out["d_gap"] = out["d_pick"] - out["d_opp"]
    out["n_min"] = min(sn["a"]["pick"][1], sn["b"]["pick"][1])
    return out


def fit_eval(train, test, cols, l2_grid=(1.0, 10.0, 100.0), inner_frac=0.25):
    """Logistic: y ~ 1 + engine logit (unpenalised) + cols (ridge, scale
    fixed on train). L2 chosen on the last 25% of training dates only."""
    names = ["c", "L"] + list(cols)
    sd = {c: (math.sqrt(mean(r[c] ** 2 for r in train)) or 1.0) for c in cols}

    def X(rows):
        return [[1.0, r["L"]] + [r[c] / sd[c] for c in cols] for r in rows]
    days = sorted({r["ord"] for r in train})
    cut = days[int(len(days) * (1 - inner_frac))]
    fit_, val_ = [r for r in train if r["ord"] < cut], [r for r in train if r["ord"] >= cut]
    best = None

    def fit(rows, l2):
        # penalise features only: intercept and engine slope get a negligible ridge
        n = len(rows)
        Xr = X(rows)
        beta = logistic_fit(Xr, [r["y"] for r in rows], l2=l2, iters=40)
        return beta
    if cols:
        for l2 in l2_grid:
            b = fit(fit_, l2)
            ll = logloss([sigmoid(sum(u * v for u, v in zip(b, x))) for x in X(val_)], [r["y"] for r in val_])
            if best is None or ll < best[0]:
                best = (ll, l2)
        l2 = best[1]
    else:
        l2 = 1.0
    beta = fit(train, l2)
    ps = [sigmoid(sum(u * v for u, v in zip(beta, x))) for x in X(test)]
    return {"coef": {n: round(b, 4) for n, b in zip(names, beta)}, "l2": l2, "p": ps}


def q4(ctx, rep):
    t0 = time.time()
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import backtest as bt        # read-only imports of the engine
    import predictor as pr
    ms = bt.load_matches()
    built = bt.build_dataset(ms)
    ev, _ = bt.select_eval(built)
    by_idx = {s["idx"]: s for s in ctx.series if s["src"] == "main"}
    base = []
    for r in ev:
        m = r["match"]
        s = by_idx.get(m["id"])
        if s is None or "pool" not in s or s["k"] is None:
            continue
        base.append({"s": s, "L": bt.get_total_logodds(pr, r["input"]), "y": 1.0 if m["winner"] == m["team_a"] else 0.0,
                     "ord": s["ord"], "date": s["date"]})
    print(f"  q4: {len(base)} engine-eligible BO3 rows ({time.time() - t0:.0f}s)", flush=True)
    variants = {"history_main_only": PitPass([s for s in ctx.series if s["src"] == "main"]),
                "history_main_plus_lower": PitPass(ctx.series)}
    out = {}
    specs = {"engine_only": [], "+overall_map_residual": ["d_all"],
             "+pick_split(pick,opp,decider)": ["d_pick", "d_opp", "d_dec"],
             "+pick_split_with_overall": ["d_all", "d_pick", "d_opp", "d_dec"],
             "+picker_edge(pick-opp)": ["d_gap"]}
    def eval_rows(rows):
        train = [r for r in rows if r["date"] <= TRAIN_END]
        test = [r for r in rows if r["date"] >= TEST_START]
        ys = [r["y"] for r in test]
        days = [r["ord"] for r in test]
        res = {}
        losses = {}
        for name, cols in specs.items():
            fe = fit_eval(train, test, cols)
            p = fe["p"]
            losses[name] = [-(y * math.log(max(q, 1e-12)) + (1 - y) * math.log(max(1 - q, 1e-12))) for q, y in zip(p, ys)]
            briers = [(q - y) ** 2 for q, y in zip(p, ys)]
            res[name] = {"coef": fe["coef"], "l2": fe["l2"], "test_logloss": round(mean(losses[name]), 5),
                         "test_brier": round(mean(briers), 5),
                         "test_accuracy": round(mean(1.0 if (q > 0.5) == (y > 0.5) else 0.0 for q, y in zip(p, ys)), 4)}
        for name in specs:
            if name != "engine_only":
                res[name]["logloss_gain_vs_engine_only"] = cluster_boot_diff(losses["engine_only"], losses[name], days, 500 if FAST else 2000)
        res["+pick_split_with_overall"]["logloss_gain_vs_overall_residual"] = cluster_boot_diff(
            losses["+overall_map_residual"], losses["+pick_split_with_overall"], days, 500 if FAST else 2000)
        res["feature_sd_test"] = {c: round(math.sqrt(mean(r[c] ** 2 for r in test)), 4) for c in ("d_all", "d_pick", "d_opp", "d_dec", "d_gap")}
        res["mean_prior_pick_weight_min_team"] = round(mean(r["n_min"] for r in test), 2)
        res["n_train"], res["n_test"] = len(train), len(test)
        return res, test
    for vname, pit in variants.items():
        rows = []
        for b in base:
            sn = pit.snap.get(b["s"]["k"])
            if sn is None:
                continue
            r = dict(b)
            r.update(feat_row(sn))
            rows.append(r)
        res, test = eval_rows(rows)
        raw_engine = [-(r["y"] * math.log(sigmoid(r["L"])) + (1 - r["y"]) * math.log(1 - sigmoid(r["L"]))) for r in test]
        res["engine_as_shipped_logloss"] = round(mean(raw_engine), 5)
        out[vname] = res
    # larger sample: every tier, own point-in-time Elo as the baseline (no engine)
    pit_all = variants["history_main_plus_lower"]
    rows = []
    for s in ctx.valid:
        sn = pit_all.snap.get(s["k"])
        if sn is None or sn["a"]["n_series"] < 4 or sn["b"]["n_series"] < 4:
            continue
        r = {"L": math.log(sn["p_elo"] / (1 - sn["p_elo"])), "y": 1.0 if s["winner"] == s["a"] else 0.0, "ord": s["ord"], "date": s["date"]}
        r.update(feat_row(sn))
        rows.append(r)
    out["all_tiers_own_elo_baseline"], _ = eval_rows(rows)
    # picker advantage at map level
    pit = variants["history_main_plus_lower"]
    mrows = []
    for s in ctx.valid:
        r = pit.attr.get(s["k"])
        sn = pit.snap.get(s["k"])
        if r is None or sn is None:
            continue
        q = map_prob_from_series(sn["p_elo"], 3)
        lq = math.log(q / (1 - q))
        for idx, (mp, w) in enumerate(s["maps"][:2]):
            sign = (2 * r - 1) if idx == 0 else (1 - 2 * r)   # +1 team_a picked it
            ra_ = sn["a"]["per"].get(mp, (0.0, 0))
            rb_ = sn["b"]["per"].get(mp, (0.0, 0))
            edge = ra_[0] / (ra_[1] + K_SHRINK_RES) - rb_[0] / (rb_[1] + K_SHRINK_RES)
            mrows.append({"y": 1.0 if w == s["a"] else 0.0, "off": lq, "z": sign, "ord": s["ord"], "src": s["src"],
                          "conf": abs(2 * r - 1), "slot": idx, "edge": edge, "zshuf": 0.0})
    shuf_rng = random.Random(SEED + 9)
    zs_all = [r["z"] for r in mrows]
    shuf_rng.shuffle(zs_all)
    for r, z in zip(mrows, zs_all):
        r["zshuf"] = z
    def pick_adv(rows, label, conf_min=0.0, zkey="z", control_edge=False):
        rows = [r for r in rows if r["conf"] >= conf_min]
        if len(rows) < 200:
            return None
        cols = [zkey] + (["edge"] if control_edge else [])
        X = [[1.0] + [r[c] for c in cols] for r in rows]
        beta = logistic_fit(X, [r["y"] for r in rows], l2=0.01, offset=[r["off"] for r in rows])
        # day-cluster bootstrap of the z coefficient
        by = defaultdict(list)
        for r in rows:
            by[r["ord"]].append(r)
        ks = sorted(by)
        rng = random.Random(SEED + 4)
        bs = []
        for _ in range(100 if FAST else 300):
            sample = []
            for _ in ks:
                sample += by[ks[rng.randrange(len(ks))]]
            b = logistic_fit([[1.0] + [r[c] for c in cols] for r in sample], [r["y"] for r in sample], l2=0.01,
                             offset=[r["off"] for r in sample], iters=12)
            bs.append(b[1])
        out_ = {"label": label, "n_maps": len(rows), "picker_edge_logit": round(beta[1], 4),
                "ci95": [round(quantile(bs, 0.025), 4), round(quantile(bs, 0.975), 4)],
                "approx_pp_at_even": round(100 * (sigmoid(beta[1]) - 0.5), 2)}
        if control_edge:
            out_["map_specific_edge_coef"] = round(beta[2], 4)
        return out_
    pa = {}
    for lab, sel in (("train_all", lambda r: r["ord"] <= d_ord(TRAIN_END)), ("test_all", lambda r: r["ord"] >= d_ord(TEST_START)),
                     ("train_main_only", lambda r: r["ord"] <= d_ord(TRAIN_END) and r["src"] == "main"),
                     ("test_main_only", lambda r: r["ord"] >= d_ord(TEST_START) and r["src"] == "main")):
        pa[lab] = pick_adv([r for r in mrows if sel(r)], lab)
    te = [r for r in mrows if r["ord"] >= d_ord(TEST_START)]
    pa["test_all_controlling_map_specific_edge"] = pick_adv(te, "test, + shrunk team map-result gap on that map", control_edge=True)
    pa["test_map1_only"] = pick_adv([r for r in te if r["slot"] == 0], "test, map 1")
    pa["test_map2_only"] = pick_adv([r for r in te if r["slot"] == 1], "test, map 2")
    ctl = []
    for sd_ in range(12 if FAST else 30):
        zz = [r["z"] for r in te]
        random.Random(SEED + 100 + sd_).shuffle(zz)
        ctl.append(logistic_fit([[1.0, z_] for z_ in zz], [r["y"] for r in te], l2=0.01, offset=[r["off"] for r in te], iters=12)[1])
    pa["test_negative_control_shuffled_picker"] = {
        "label": "picker label shuffled across test maps, %d shuffles" % len(ctl),
        "coef_mean": round(mean(ctl), 4), "coef_sd": round(math.sqrt(mean((c - mean(ctl)) ** 2 for c in ctl)), 4)}
    pa["test_all_confident_attribution(>=0.5)"] = pick_adv([r for r in mrows if r["ord"] >= d_ord(TEST_START)], "conf>=0.5", 0.5)
    rep["q4"] = {
        "design": "walk-forward free: train <= %s, test >= %s, engine logit (pre-temperature) recalibrated on train; "
                  "features are differences of shrunk (k=%g) Elo-residual map results over the prior 180 days, split by "
                  "whether the team (probably) picked the map; picks attributed softly" % (TRAIN_END, TEST_START, K_SHRINK_RES),
        "variants": out, "picker_advantage_by_map": pa,
        "attribution_confidence_mean": round(mean(pit.attr_conf), 3),
        "note": "l2 chosen on the last 25% of training dates only; test window scored once per spec"}
    print(f"  q4 done {time.time() - t0:.0f}s", flush=True)


# ============================================================================
# Q5  where the engine's veto predictions miss
# ============================================================================
def swap_input(inp):
    out = dict(inp)
    for a, b in (("maps_a", "maps_b"), ("maps_raw_a", "maps_raw_b"), ("permaban_a", "permaban_b"),
                 ("veto_ev_a", "veto_ev_b"), ("team_a", "team_b")):
        out[a], out[b] = inp.get(b), inp.get(a)
    return out


def habit_pick_set(inp, pool):
    """Trivial baseline: each team's most played pool map over the prior 90
    days (second most if both name the same one)."""
    def ranked(side):
        mm = inp.get("maps_raw_" + side) or {}
        return sorted(pool, key=lambda mp: (-(mm.get(mp) or [0, 0])[1], mp))
    ra, rb = ranked("a"), ranked("b")
    a1 = ra[0]
    b1 = next(mp for mp in rb if mp != a1)
    return {a1, b1}


def q5(ctx, rep):
    t0 = time.time()
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import backtest as bt
    import predictor as pr
    import veto as vt
    ms = bt.load_matches()
    built = bt.build_dataset(ms)
    rows = []
    for r in built:
        m = r["match"]
        if m.get("best_of") != 3 or len(m.get("maps") or []) < 2:
            continue
        inp = r["input"]
        pool = pr._pool(inp)
        played = [x["map"] for x in m["maps"]]
        if any(p_ not in pool for p_ in played) or len(set(played)) != len(played) or len(pool) < 3:
            continue
        rows.append((m, inp, pool, played, r["meta"]))
    chance_pick = 2.0 / 7.0
    recs = []
    for m, inp, pool, played, meta in rows:
        va = pr.simulate_veto(inp, 3)
        vb = pr.simulate_veto(swap_input(inp), 3)
        act = set(played[:2])
        hit_a = len(set(va["maps"][:2]) & act) / 2.0
        hit_b = len(set(vb["maps"][:2]) & act) / 2.0
        hab = len(habit_pick_set(inp, pool) & act) / 2.0
        dec_a = float(va["maps"][2] == played[2]) if len(played) >= 3 else None
        dec_b = float(vb["maps"][2] == played[2]) if len(played) >= 3 else None
        # picks on maps the picker has not played in the prior 90 days
        up = 0
        for mp_, side in ((va["maps"][0], "maps_a"), (va["maps"][1], "maps_b")):
            e = (inp.get(side) or {}).get(mp_)
            up += not (e and e[1])
        o_ = d_ord(m["date"])
        fld = len(set(sorted(pool, key=lambda mp: (-ctx.fr.rate(mp, o_), mp))[:2]) & act) / 2.0
        novel = float(any(not ((inp.get("maps_raw_a") or {}).get(mp_, [0, 0])[1] or (inp.get("maps_raw_b") or {}).get(mp_, [0, 0])[1])
                          for mp_ in played[:2]))
        recs.append({"m": m, "date": m["date"], "ord": o_, "hit_a": hit_a, "hit_b": hit_b, "hab": hab, "fld": fld, "novel": novel,
                     "dec_a": dec_a, "dec_b": dec_b, "unplayed": up / 2.0,
                     "min_hist": min(meta["hist_a"], meta["hist_b"]), "inp": inp, "pool": pool, "played": played})
    print(f"  q5: {len(recs)} rows scored ({time.time() - t0:.0f}s)", flush=True)
    test_lo = d_ord(TEST_START)

    def agg(rs):
        n = len(rs)
        if not n:
            return None
        e = mean((r["hit_a"] + r["hit_b"]) / 2 for r in rs)       # starter-agnostic
        ea = mean(r["hit_a"] for r in rs)
        h = mean(r["hab"] for r in rs)
        fb = mean(r["fld"] for r in rs)
        ds = [r for r in rs if r["dec_a"] is not None]
        return {"n": n, "engine_hit_a_starts": round(ea, 4), "engine_hit_starter_avg": round(e, 4),
                "engine_hit_best_starter": round(mean(max(r["hit_a"], r["hit_b"]) for r in rs), 4),
                "habit_baseline_hit": round(h, 4), "field_top2_baseline_hit": round(fb, 4), "chance_hit": round(chance_pick, 4),
                "played_pick_map_unseen_by_both_90d": round(mean(r["novel"] for r in rs), 4),
                "engine_decider_hit": round(mean((r["dec_a"] + r["dec_b"]) / 2 for r in ds), 4) if ds else None, "n_decider": len(ds),
                "decider_chance": round(1 / 7.0, 4),
                "unplayed_pick_share": round(mean(r["unplayed"] for r in rs), 4)}
    overall = {"all": agg(recs), "test": agg([r for r in recs if r["ord"] >= test_lo]),
               "train": agg([r for r in recs if r["ord"] <= d_ord(TRAIN_END)])}
    # paired CIs engine vs habit baseline and vs chance (overall, test)
    tst = [r for r in recs if r["ord"] >= test_lo]
    overall["test_engine_minus_habit"] = cluster_boot_diff([(r["hit_a"] + r["hit_b"]) / 2 for r in tst], [r["hab"] for r in tst],
                                                          [r["ord"] for r in tst], 500 if FAST else 2000)
    overall["test_engine_minus_field_top2"] = cluster_boot_diff([(r["hit_a"] + r["hit_b"]) / 2 for r in tst], [r["fld"] for r in tst],
                                                               [r["ord"] for r in tst], 500 if FAST else 2000)
    overall["test_engine_minus_chance"] = cluster_boot_mean([(r["hit_a"] + r["hit_b"]) / 2 - chance_pick for r in tst],
                                                           [r["ord"] for r in tst], 500 if FAST else 2000)
    # per team
    by_team = defaultdict(list)
    for r in recs:
        by_team[r["m"]["team_a"]].append(r)
        by_team[r["m"]["team_b"]].append(r)
    q1r = rep.get("q1", {}).get("teams", {})
    q1d = rep.get("q1", {}).get("drift_by_team", {})
    q3r = rep.get("q3", {}).get("top30_recent_lineup_state", {})
    rng = random.Random(SEED + 5)
    teams = {}
    for t, rs in by_team.items():
        top = t in ctx.top_teams
        if not top and len(rs) < 40:
            continue
        rs = sorted(rs, key=lambda r: r["ord"])
        d = agg(rs)
        eng = [(r["hit_a"] + r["hit_b"]) / 2 for r in rs]
        hab = [r["hab"] for r in rs]
        days = [r["ord"] for r in rs]
        d["lift_vs_chance"] = cluster_boot_mean([x - chance_pick for x in eng], days, 300 if FAST else 800)
        d["engine_minus_field_top2"] = cluster_boot_diff(eng, [r["fld"] for r in rs], days, 300 if FAST else 800)
        d["engine_minus_habit"] = cluster_boot_diff(eng, hab, days, 300 if FAST else 800)
        rr = [r for r in rs if r["ord"] >= test_lo]
        d["test_n"] = len(rr)
        d["test_engine_hit_starter_avg"] = round(mean((r["hit_a"] + r["hit_b"]) / 2 for r in rr), 4) if rr else None
        d["in_top30"] = top
        d["vrs_rank"] = next((x["rank"] for x in ctx.top if x["team"] == t), None)
        k1 = q1r.get(t, {}).get("kinds")
        if k1:
            d["pick_top_share_recent"] = k1["pk"]["top_share"]
            d["pick_entropy_norm_recent"] = k1["pk"]["entropy_norm"]
            d["b1_top_share_recent"] = k1["b1"]["top_share"]
        dr = q1d.get(t)
        if dr and "p" in dr:
            d["recent_drift_p"] = dr["p"]
        if t in q3r:
            d["changes_last_365d"] = q3r[t]["changes_last_365d"]
            d["last_lineup_change"] = q3r[t]["last_change_date"]
            d["players_changed_last"] = q3r[t]["players_changed"]
        # share of the team's picked-map predictions that were on a map the picker had no 90d record on
        teams[t] = d
    # ranking
    cand = [(t, d) for t, d in teams.items() if d["n"] >= 25]
    worst_lift = sorted(cand, key=lambda x: x[1]["engine_minus_field_top2"]["mean"])[:10]
    worst_vs_habit = sorted(cand, key=lambda x: x[1]["engine_minus_habit"]["mean"])[:10]
    sp = None
    both = [(d.get("pick_top_share_recent"), d["engine_hit_starter_avg"], d["habit_baseline_hit"]) for t, d in cand if d.get("pick_top_share_recent") is not None]
    if len(both) >= 8:
        sp = {"n_teams": len(both),
              "spearman_pick_concentration_vs_engine_hit": round(spearman([x[0] for x in both], [x[1] for x in both]), 3),
              "spearman_pick_concentration_vs_habit_baseline_hit": round(spearman([x[0] for x in both], [x[2] for x in both]), 3)}
    for lab, key in (("pick_entropy", "pick_entropy_norm_recent"), ("roster_changes_365d", "changes_last_365d")):
        pairs = [(d[key], d["engine_minus_field_top2"]["mean"]) for t, d in cand if d.get(key) is not None]
        if len(pairs) >= 8:
            if sp is None:
                sp = {}
            sp["spearman_%s_vs_engine_minus_field" % lab] = round(spearman([x[0] for x in pairs], [x[1] for x in pairs]), 3)
            sp["n_teams_" + lab] = len(pairs)
    # reasons
    def reasons(d):
        out = []
        if d["engine_minus_field_top2"]["mean"] < 0.05:
            out.append("engine_adds_little_over_field_top2(%+.2f)" % d["engine_minus_field_top2"]["mean"])
        if d["played_pick_map_unseen_by_both_90d"] > 0.12:
            out.append("plays_maps_neither_team_played_in_90d(%.2f)" % d["played_pick_map_unseen_by_both_90d"])
        if d["engine_minus_habit"]["mean"] < -0.03:
            out.append("trivial_habit_baseline_better")
        if d.get("pick_top_share_recent") is not None and d["pick_top_share_recent"] >= 0.35:
            out.append("concentrated_picks(top share %.2f)" % d["pick_top_share_recent"])
        if d.get("pick_entropy_norm_recent") is not None and d["pick_entropy_norm_recent"] >= 0.85:
            out.append("diffuse_picks(entropy %.2f)" % d["pick_entropy_norm_recent"])
        if d.get("recent_drift_p") is not None and d["recent_drift_p"] < 0.1:
            out.append("recent_habit_shift(p=%.2f)" % d["recent_drift_p"])
        if d.get("players_changed_last") and d["players_changed_last"] >= 2 and d.get("last_lineup_change") and d["last_lineup_change"] >= ctx.iso(ctx.as_of - 240):
            out.append("recent_roster_change(%d players, %s)" % (d["players_changed_last"], d["last_lineup_change"]))
        if d["unplayed_pick_share"] > 0.08:
            out.append("engine_picks_maps_team_has_not_played(%.2f)" % d["unplayed_pick_share"])
        return out
    for t, d in teams.items():
        d["flags"] = reasons(d)
    # dist-mode scoring (not shipped): log-likelihood of the played order vs a uniform legal veto, test window
    dist_gain = {}
    try:
        w = dict(pr.veto_params(3))
        wu = {k: (0.0 if k.startswith(("alpha_", "gamma_", "rho_", "eta_")) else v) for k, v in w.items()}
        wu["lookahead"] = False
        saved_alpha = pr.CONFIG["veto_excl_alpha"]
        per = defaultdict(list)
        sub = tst if not FAST else tst[:150]
        for r in sub:
            ctx_ = pr.veto_context(r["inp"], 3)
            ll_m = vt.loglik(ctx_, w, r["played"])[0]
            pr.CONFIG["veto_excl_alpha"] = 0.0
            try:
                ctx_u = pr.veto_context(r["inp"], 3)
                ll_u = vt.loglik(ctx_u, wu, r["played"])[0]
            finally:
                pr.CONFIG["veto_excl_alpha"] = saved_alpha
            g = ll_m - ll_u
            if g == float("-inf") or g != g:
                g = -10.0
            for t in (r["m"]["team_a"], r["m"]["team_b"]):
                per[t].append(g)
        for t, v in per.items():
            if len(v) >= 8 and (t in ctx.top_teams or len(v) >= 20):
                dist_gain[t] = {"n": len(v), "mean_loglik_gain_vs_uniform": round(mean(v), 3)}
        dist_overall = round(mean(g for v in per.values() for g in v), 3) if per else None
    except Exception as e:      # the opt-in distribution mode must never break the report
        dist_overall = None
        dist_gain = {"error": repr(e)[:200]}
    rep["q5"] = {
        "design": "shipped point veto (simulate_veto) vs played maps. pick-set hit = |predicted picks & played maps 1-2| / 2, "
                  "chance 2/7; starter unknown so the shipped team_a-starts figure, the average over both starters and the "
                  "best of both are given; habit baseline = each team's most played map of the prior 90 days",
        "overall": overall,
        "teams": teams,
        "ranking_worst_engine_minus_field_top2": [{"team": t, "n": d["n"], "hit": d["engine_hit_starter_avg"], "field_top2": d["field_top2_baseline_hit"],
                                                 "lift": round(d["engine_minus_field_top2"]["mean"], 3),
                                                 "lift_ci": [round(d["engine_minus_field_top2"]["lo"], 3), round(d["engine_minus_field_top2"]["hi"], 3)], "flags": d["flags"]}
                                                for t, d in worst_lift],
        "ranking_worst_engine_minus_habit": [{"team": t, "n": d["n"], "engine": d["engine_hit_starter_avg"], "habit": d["habit_baseline_hit"],
                                              "diff": round(d["engine_minus_habit"]["mean"], 3),
                                              "diff_ci": [round(d["engine_minus_habit"]["lo"], 3), round(d["engine_minus_habit"]["hi"], 3)], "flags": d["flags"]}
                                             for t, d in worst_vs_habit],
        "across_teams": sp,
        "dist_mode_loglik_gain_vs_uniform_test": {"overall_per_team_series": dist_overall, "by_team": dist_gain,
                                                  "note": "opt-in distribution mode with the stored p2 parameters; not the shipped point veto"},
    }
    for r in recs:      # drop heavy references before serialisation
        r.pop("inp", None)
        r.pop("m", None)
    print(f"  q5 done {time.time() - t0:.0f}s", flush=True)


# ============================================================================
# headline + recommendations (numbers are read back from the sections above)
# ============================================================================
def _g(d, *path, default=None):
    for p_ in path:
        if isinstance(d, dict) and p_ in d:
            d = d[p_]
        else:
            return default
    return d


def build_headline(rep):
    q1_, q2_, q3_, q4_, q5_ = (rep.get(k, {}) for k in ("q1", "q2", "q3", "q4", "q5"))
    h = {}
    sm = q1_.get("summary", {})
    h["q1_classes"] = {k: _g(sm, k, "classes") for k in ("b1", "pk", "b2", "dec")}
    h["q1_median_top_share"] = {k: _g(sm, k, "median_top_share") for k in ("b1", "pk", "b2", "dec")}
    h["q1_median_entropy_norm"] = {k: _g(sm, k, "median_entropy_norm") for k in ("b1", "pk", "b2", "dec")}
    h["q1_constant_first_ban_teams"] = sm.get("constant_first_ban_teams")
    h["q1_constant_first_ban_also_strongly_avoided"] = sm.get("constant_first_ban_also_strongly_avoided")
    h["q1_played_share_vs_field"] = sm.get("played_share_vs_field")
    h["q1_null_top_share_hard_draws"] = q1_.get("null_top_share_hard_draws")
    h["q1_heldout_gain_nats_per_series"] = {k: _g(q1_, "heldout_latent_model", "gain_nats_per_series", k, "vs_field_only", "mean")
                                            for k in ("picks_only", "picks_b1", "bans_only", "full")}
    h["q1_gain_decays_with_age"] = _g(q1_, "heldout_latent_model", "gain_by_60day_block_after_cut")
    h["q1_best_window_observed"] = {g: sorted(((k, round(v["mean"], 4)) for k, v in _g(q1_, "window_decay_observed", g, default={}).items()),
                                              key=lambda x: -x[1])[:3] for g in ("top30", "all_teams")}
    h["q1_decay_paired_gain_vs_w90"] = {k: _g(q1_, "window_decay_observed", "all_teams", k, "paired_gain_vs_w90")
                                        for k in ("hl30", "hl60", "hl90", "hl180", "w180", "all")}
    h["q1_latent_decay_gain"] = {k: _g(v, "gain_vs_no_decay", "mean") for k, v in (q1_.get("heldout_latent_decay") or {}).items() if isinstance(v, dict)}
    dr = q1_.get("drift_by_team", {})
    h["q1_teams_with_drift_p_lt_05"] = sorted(t for t, d in dr.items() if d.get("p") is not None and d["p"] < 0.05)
    h["q1_teams_with_drift_holm05"] = sorted(t for t, d in dr.items() if d.get("p_holm") is not None and d["p_holm"] < 0.05)
    ss = q2_.get("strength_split", {})
    h["q2_strength_split"] = {k: ss.get(k) for k in ("n_teams_tested", "share_p_below_05", "fisher_p", "mean_js_over_null", "holm_significant",
                                                    "played_share_entropy_stronger_minus_weaker")}
    h["q2_pick_pair_dependence_p"] = {k: v.get("p_observed_vs_simulated") for k, v in (_g(q2_, "pick_pair_dependence", "pools", default={}) or {}).items()}
    mm = _g(q2_, "matchup_model", "models", default={})
    h["q2_matchup_edge_gain_over_habits"] = _g(mm, "+matchup_edge", "gain_vs_+team_habits")
    h["q2_habit_gain_over_field"] = _g(mm, "+team_habits", "gain_vs_field_only")
    h["q2_starter_prior"] = {k: _g(q2_, "starter_prior", k) for k in ("logit_slope_per_100_elo", "test_loglik_gain_vs_constant_pi", "pi_constant_fit")}
    byc = _g(q3_, "habit_skill_by_change", "by_players_changed_vs_previous_series", default={})
    h["q3_habit_skill_nats"] = {k: [_g(v, "all_history", "mean"), _g(v, "all_history", "n")] for k, v in byc.items()}
    h["q3_recovery"] = {k: _g(v, "mean") for k, v in (_g(q3_, "habit_skill_by_change", "stable_lineup_by_series_since_last_change", default={}) or {}).items()}
    h["q3_pre_post"] = _g(q3_, "pre_post_distance", "groups")
    h["q3_major_vs_stable_perm"] = _g(q3_, "pre_post_distance", "major_vs_stable_perm")
    vv = q4_.get("variants", {})
    h["q4_series_win_logloss_gain_vs_engine"] = {
        v: {k: _g(vv, v, k, "logloss_gain_vs_engine_only") for k in ("+overall_map_residual", "+pick_split(pick,opp,decider)",
                                                                       "+pick_split_with_overall", "+picker_edge(pick-opp)")} for v in vv}
    h["q4_picker_advantage_by_map"] = {k: (v if v is None else {a: v[a] for a in v if a in ("n_maps", "picker_edge_logit", "ci95", "approx_pp_at_even",
                                                                                            "map_specific_edge_coef", "coef_mean", "coef_sd")})
                                       for k, v in (q4_.get("picker_advantage_by_map") or {}).items()}
    ov = q5_.get("overall", {})
    h["q5_overall_test"] = ov.get("test")
    h["q5_engine_minus_field_top2_test"] = ov.get("test_engine_minus_field_top2")
    h["q5_engine_minus_habit_test"] = ov.get("test_engine_minus_habit")
    h["q5_weakest_vs_field"] = q5_.get("ranking_worst_engine_minus_field_top2", [])[:5]
    h["q5_weakest_vs_habit"] = q5_.get("ranking_worst_engine_minus_habit", [])[:3]
    h["q5_across_teams"] = q5_.get("across_teams")
    rep["headline"] = h
    rep["recommendations"] = [
        {"rank": 1, "title": "Recency-decayed habit evidence instead of a flat 90-day window",
         "action": "In the preference estimator replace equal weights over the last 90 days by exponential decay with a half-life of 60-90 days "
                   "(observed played-set score peaks at 60, latent-model score at 90-180; 45 is not better); apply the same weights to the exclusion "
                   "test and to any habit counts. A flat window longer than 180 days, or all history, is clearly worse.",
         "evidence": {"observed_played_set_gain_vs_flat_90d": h["q1_decay_paired_gain_vs_w90"], "latent_model_gain_vs_no_decay": h["q1_latent_decay_gain"],
                      "gain_vs_age": h["q1_gain_decays_with_age"]}},
        {"rank": 2, "title": "Model avoidance (bans) as the primary habit signal; keep pick habits heavily shrunk",
         "action": "Held-out information is concentrated on the ban side: team-specific ban weights add about three times what pick weights add, and "
                   "pick weights only stop hurting when they are shrunk about 4x harder than ban weights (then both together equal bans alone). Use a soft 'avoid' score (played-rate shortfall "
                   "vs the field, decayed) in place of the binary zero-plays exclusion, with per-kind shrinkage chosen on a training-only inner fold.",
         "evidence": {"gain_nats_per_series": h["q1_heldout_gain_nats_per_series"],
                      "pick_x4_shrinkage_chosen": _g(q1_, "heldout_latent_model", "kappa_pick_x4_chosen")}},
        {"rank": 3, "title": "Collect the actual veto sequences",
         "action": "Every ban / picker / starter statement here is a posterior from played-map order. Storing the veto log per series (who "
                   "banned, picked, started) removes the identification problem, makes first-ban consistency directly measurable and "
                   "lets the starter rule below be validated instead of inferred.",
         "evidence": {"first_ban_identifiability": "ban shares are soft; classes are conditional on the sequential-choice model"}},
        {"rank": 4, "title": "Stop treating team_a as the starter; use a rating-gap rule or the mixture",
         "action": "The fits put P(team_a starts) near 0.4 and falling with team_a's rating advantage. The point veto always lets team_a start. "
                   "Use the veto mixture over starters with P(a starts)=sigmoid(c + b * gap), or at least a documented starter rule. "
                   "The held-out gain from the gap term is small and borderline (CI touches 0); the larger point is the level, which matters most "
                   "for attributing bans when learning habits.",
         "evidence": h["q2_starter_prior"]},
        {"rank": 5, "title": "Add a picker term to the per-map win model (scoreline / decider), not to the series win chance",
         "action": "Teams win the maps they (probably) picked more often than the rating gap implies: about +0.25 logit (+6 points at an even map), "
                   "stable across train and test and still +0.18 after controlling for the team's own record on that exact map. It nets out of "
                   "the series win chance when both teams pick once, so use it for per-map and scoreline outputs only.",
         "evidence": h["q4_picker_advantage_by_map"]},
        {"rank": 6, "title": "Discount habit evidence after roster changes instead of filtering by core",
         "action": "Habit skill falls from about 0.26 nats per team-series with a stable lineup to about 0.15 after a one-player change and to about 0 "
                   "after 2+ changes (small n), and recovers over 4-8 series. Multiply the evidence weight by a factor that is low right after a "
                   "change and returns to 1 over roughly 8 series. Core-filtered counts did not beat plain decayed counts.",
         "evidence": {"skill": h["q3_habit_skill_nats"], "recovery": h["q3_recovery"], "pre_post": h["q3_pre_post"]}},
        {"rank": 7, "title": "Keep preferences opponent-independent; do not add pick-split map strength to the win probability",
         "action": "Matchup terms add about 1.7 percent of what team habits add; the strength split is weakly significant in aggregate but "
                   "not directional and not significant per team after Holm; the second pick does not respond to the first pick beyond habits "
                   "(borderline). Pick-split map-strength features move series log loss by less than the CI half-width (about 0.002).",
         "evidence": {"strength_split": h["q2_strength_split"], "pair_dependence_p": h["q2_pick_pair_dependence_p"],
                      "matchup_gain": h["q2_matchup_edge_gain_over_habits"], "habit_gain": h["q2_habit_gain_over_field"],
                      "series_win_gain": h["q4_series_win_logloss_gain_vs_engine"]}},
        {"rank": 8, "title": "Spot-check the few weak teams; no top-30 team is a systematic outlier",
         "action": "The shipped point veto beats a top-two-field-maps baseline by about 0.13 in pick-set hit and a most-played-map baseline by about 0.07. "
                   "Weakest relative to the field: see ranking; most are explained by recent roster change or a recent habit shift, which "
                   "recommendations 1 and 6 target. Team-specific decider modelling is not worth it (deciders are mostly diffuse).",
         "evidence": {"weakest": h["q5_weakest_vs_field"], "across_teams": h["q5_across_teams"]}},
    ]
    rep["caveats"] = [
        "Bans, pickers and the veto starter are not in the data: all of them are posterior inferences from the order of the played maps.",
        "Soft shares come from a self-consistent EM fit; they can be sharper than the truth. Held-out log-likelihood gains (section q1 A) are the "
        "check that the habits are real; the one-step field-prior shares in the team table bound the sharpening.",
        "Pick attribution in q4 is a point-in-time posterior with a fixed starter prior; its mean confidence is reported. Errors attenuate the picker edge.",
        "Elo here is a plain series-level Elo (K 20) used for strength splits and map residuals; the engine's own logit is used as the q4 baseline.",
        "Test window is scored once per specification; no choice (shrinkage, windows, kappa) was made on it.",
    ]


# ============================================================================
# main
# ============================================================================
SECTIONS = {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5}


def main(argv=None):
    global FAST
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="q1,q2,q3,q4,q5")
    ap.add_argument("--fast", action="store_true", help="fewer EM iterations / bootstraps (smoke test)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    FAST = a.fast
    t0 = time.time()
    ctx = Ctx()
    rep = {}
    if os.path.exists(a.out) and a.sections != "q1,q2,q3,q4,q5":
        with open(a.out, encoding="utf-8") as f:
            rep = json.load(f)
    rep["meta"] = {
        "script": "scripts/ban_patterns.py", "as_of": ctx.iso(ctx.as_of), "vrs_date": ctx.vrs_date,
        "split": {"train_end": TRAIN_END, "test_start": TEST_START}, "seed": SEED, "fast": FAST,
        "series_total": len(ctx.series), "bo3_valid": len(ctx.valid), "bo3_valid_7map_pool": len(ctx.v7),
        "dropped": ctx.drops, "top_teams": ctx.top, "vrs_unmapped_in_top": ctx.unmapped[:10],
        "data_note": "played maps only; bans, pickers and the veto starter are latent (posterior, 'soft')",
    }
    for name in a.sections.split(","):
        name = name.strip()
        if name in SECTIONS:
            print(f"[{name}]", flush=True)
            SECTIONS[name](ctx, rep)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1, sort_keys=False)
    build_headline(rep)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, sort_keys=False)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
