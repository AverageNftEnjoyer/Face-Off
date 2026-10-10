#!/usr/bin/env python3
"""
veto.py -- the veto subsystem v2: a calibrated distribution over map vetoes.
===========================================================================
Fan analytics only. Replaces the single simulated veto (predictor.simulate_veto,
which stays as the point engine for CONFIG["veto_mode"] = "point") with an
exact probability distribution over every legal veto.

Components (design: veto_architecture v2, section A):
  FormatSpec            format_steps(): the explicit step list (role, kind)
  PreferenceEstimator   team_prefs(): comfort, exclusion set, habits -- a
                        function of (team, D) only, never of the opponent (I8)
  Policy                quantal response (softmax over a linear utility) at
                        one state; myopic (phase 1) or with lookahead values V
                        from backward induction (phase 2+)
  VetoOrchestrator      Tree: every state of the draft, memoised; the closed-
                        loop policy is read from it
  VetoDistribution      distribution(): outcomes mixed over the hidden
                        starter, marginals, entropy, modal sequence
  ScorelineAdapter      mixture_scoreline(): the BO3 scoreline as a mixture
                        over outcomes with one shared logit shift; the ONLY
                        place that sees p_a, read-only (I6)
  Likelihood            loglik(): P(observed map order), summing over every
                        consistent veto and both starters (used by the fit
                        and the harness)

No team name and no map name appears as a literal in this file (I0). Every
number that is a modelling choice is passed in from predictor.CONFIG.
Deterministic: maps are iterated by index in sorted pool order, no randomness.
Stdlib only.
"""
import math
from typing import Any

VERSION = "veto-v2"
ROLES = ("S", "O")
KINDS = ("ban1", "pick", "ban2")
LOGIT_EPS = 1e-12
TAIL_P = 1e-6          # outcomes below this are folded into tail_mass in the output
CONSERVE_TOL = 1e-9    # I5 tolerance (the design's 1e-12 is below float noise for 400 terms)
EXACT_MAX_POOL = 8     # exact enumeration up to this pool size; larger pools raise (predictor
                       # then falls back to the point veto with warning pool_too_large)


class VetoError(ValueError):
    pass


# ============================================================================
# numeric helpers
# ============================================================================
def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p):
    if p < LOGIT_EPS:
        p = LOGIT_EPS
    elif p > 1.0 - LOGIT_EPS:
        p = 1.0 - LOGIT_EPS
    return math.log(p / (1.0 - p))


def _softmax(us):
    mx = max(us)
    es = [math.exp(u - mx) for u in us]
    s = sum(es)
    return [e / s for e in es]


def series_win(ps):
    """P(A wins a majority of len(ps) maps), maps independent."""
    need = (len(ps) + 1) // 2
    dist = [1.0]
    for p in ps:
        nxt = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            nxt[k] += v * (1.0 - p)
            nxt[k + 1] += v * p
        dist = nxt
    return sum(dist[need:])


def _win3(p1, p2, p3):
    return p1 * p2 + (p1 * (1.0 - p2) + (1.0 - p1) * p2) * p3


# ============================================================================
# FormatSpec
# ============================================================================
def format_steps(best_of, n_pool, veto_order, veto_format=None):
    """Explicit step list [(role, kind)], role in {S, O} (S = the team that
    starts the veto), kind in {ban1, pick, ban2}; the decider is implicit
    (the one map left). Default: today's VETO_ORDER rules (predictor): bans
    are skipped when the pool is too small to keep room for the remaining
    picks plus a decider, and extra alternating bans (S first) are added when
    it is larger than 7. A ban is ban1 while no pick has been made and fewer
    than two bans happened, else ban2.

    veto_format (optional input): a list of tokens "S_ban" / "O_ban" /
    "S_pick" / "O_pick" for an event with another order. It must have
    n_pool - 1 steps of which best_of - 1 are picks (I3)."""
    if best_of not in veto_order:
        raise VetoError(f"best_of must be one of {sorted(veto_order)}, got {best_of!r}")
    if n_pool < best_of:
        raise VetoError(f"map pool needs at least {best_of} maps for a BO{best_of} veto, got {n_pool}")
    n_picks = best_of - 1
    steps = []
    if veto_format:
        picks = bans = 0
        for tok in veto_format:
            role, _, act = str(tok).partition("_")
            if role not in ROLES or act not in ("ban", "pick"):
                raise VetoError(f"bad veto_format token {tok!r}")
            if act == "pick":
                steps.append((role, "pick"))
                picks += 1
            else:
                steps.append((role, "ban1" if picks == 0 and bans < 2 else "ban2"))
                bans += 1
        if len(steps) != n_pool - 1 or picks != n_picks:
            raise VetoError(f"veto_format has {len(steps)} steps / {picks} picks; "
                            f"need {n_pool - 1} / {n_picks}")
        return tuple(steps)
    rem, picks, bans = n_pool, 0, 0
    for act in veto_order[best_of]:
        team, kind = act.split("_")
        role = "S" if team == "A" else "O"
        if kind == "pick":
            steps.append((role, "pick"))
            picks += 1
            rem -= 1
        elif rem - (n_picks - picks) > 1:
            steps.append((role, "ban1" if picks == 0 and bans < 2 else "ban2"))
            bans += 1
            rem -= 1
    turn = "S"
    while rem > 1:
        steps.append((turn, "ban2"))
        rem -= 1
        turn = "O" if turn == "S" else "S"
    if sum(1 for _, k in steps if k == "pick") != n_picks:
        raise VetoError("format has the wrong number of picks")
    return tuple(steps)


# ============================================================================
# PreferenceEstimator -- (team, D) only
# ============================================================================
EXCL_PLAY_EPS = 0.25   # weighted plays below this count as "no plays" (phase 7); one equal-weight play is 1.0


def exclusion_set(evidence, pool, alpha_excl, weighted=False):
    """E_X: maps with zero plays in the window whose zero count would be
    unlikely (prob <= alpha_excl) if the team played them at the field's
    point-in-time rate. evidence = {"maps": {map: [plays, n_pool_series,
    log_p0]}} from backtest.History (series dated < D only).

    weighted=True (phase 7) reads evidence["wmaps"] instead: every series is
    weighted by recency decay and the roster-change discount, "zero plays" is
    weighted plays < EXCL_PLAY_EPS (so an old or discounted play no longer
    protects a map) and the zero-count probability is exp(weighted sum of
    log(1 - rate)). With all weights 1 this is the baseline rule."""
    if not isinstance(evidence, dict) or not alpha_excl or alpha_excl <= 0:
        return set()
    if weighted and evidence.get("wmaps") is not None:
        out = set()
        la = math.log(alpha_excl)
        for mp in pool:
            e = evidence["wmaps"].get(mp)
            if not isinstance(e, (list, tuple)) or len(e) < 5:
                continue
            if e[0] < EXCL_PLAY_EPS and e[1] > 0 and e[2] <= la:
                out.add(mp)
        return out
    ev = evidence.get("maps") or {}
    out = set()
    la = math.log(alpha_excl)
    for mp in pool:
        e = ev.get(mp)
        if not isinstance(e, (list, tuple)) or len(e) < 3:
            continue
        plays, n_pool, log_p0 = e[0], e[1], e[2]
        if plays == 0 and n_pool > 0 and log_p0 <= la:
            out.add(mp)
    return out


def shrink_k(cfg):
    """{kind: pseudo-count} for habit shares and avoid scores. Default: the
    phase-3 veto_habit_k for every kind. With veto_habit_k_ban set (phase 7),
    ban kinds use it and the pick kind uses veto_pick_shrink_mult times as
    much (pick habits carry about a quarter of the held-out information)."""
    kh = float(cfg.get("veto_habit_k", 4.0))
    kb = cfg.get("veto_habit_k_ban")
    if kb is None:
        return {k: kh for k in KINDS}
    kb = float(kb)
    return {"ban1": kb, "ban2": kb, "pick": kb * float(cfg.get("veto_pick_shrink_mult", 4.0))}


def avoid_scores(evidence, pool, cfg):
    """Phase 7 soft avoid score: per kind and map, log((E + k) / (P + k)) with
    P the (decayed, roster-discounted) plays of the team and E the plays the
    field's play rate would have given it over the same series; k as in
    shrink_k. Positive = played less than the field (avoided), negative =
    played more; 0 with no evidence. Reads evidence["wmaps"]."""
    out = {k: {mp: 0.0 for mp in pool} for k in KINDS}
    wm = evidence.get("wmaps") if isinstance(evidence, dict) else None
    if not wm:
        return out
    ks = shrink_k(cfg)
    for k in KINDS:
        for mp in pool:
            e = wm.get(mp)
            if isinstance(e, (list, tuple)) and len(e) >= 5:
                out[k][mp] = math.log((e[3] + ks[k]) / (e[0] + ks[k]))
    return out


def habit_scores(evidence, pool, cfg):
    """Phase 3: h_k,X(m) = log p_k,X(m) - log g_k(m), the team's shrunk choice
    share at kind k over the field's. evidence["habit"] = {kind: {map: [c, o]}}
    soft counts (posterior expectation over hidden vetoes, series < D) and
    evidence["field"] = {kind: {map: [c, o]}} for the field prior g. Roster
    core (phase 4): evidence["core"] = {kind: {map: [c, o]}} is the middle
    level, skipped when evidence["core_stale"] is true. Zero evidence -> 0."""
    out = {k: {mp: 0.0 for mp in pool} for k in KINDS}
    if not isinstance(evidence, dict):
        return out
    hab = evidence.get("habit") or {}
    field = evidence.get("field") or {}
    core = evidence.get("core") if not evidence.get("core_stale") else None
    khs = shrink_k(cfg)
    kr = float(cfg.get("veto_roster_k", 4.0))
    use_core = bool(cfg.get("veto_use_roster", False)) and isinstance(core, dict)
    for k in KINDS:
        kh = khs[k]
        fk = field.get(k) or {}
        hk = hab.get(k) or {}
        ck = (core or {}).get(k) or {} if use_core else {}
        legal = len(pool)
        for mp in pool:
            fc, fo = (fk.get(mp) or [0.0, 0.0])[:2]
            g = (fc + 1.0) / (fo + legal) if fo > 0 else 1.0 / legal
            prior = g
            if use_core:
                cc, co = (ck.get(mp) or [0.0, 0.0])[:2]
                prior = (cc + kr * g) / (co + kr)
            c, o = (hk.get(mp) or [0.0, 0.0])[:2]
            p = (c + kh * prior) / (o + kh)
            out[k][mp] = math.log(max(p, 1e-12)) - math.log(g)
    return out


def team_prefs(comfort, evidence, pool, cfg, use_habits=False, plain=False):
    """PreferenceEstimator for one team: {"kappa", "excl", "habit", "avoid",
    "cold"}. Depends only on the team's own inputs (I8). Phase 7: with
    evidence["wmaps"] present the exclusion test uses the weighted counts; with
    cfg veto_soft_avoid the binary exclusion is replaced by the soft avoid
    scores ("avoid", None otherwise). plain=True ignores all of it (the
    posterior that labels past vetoes for the habit counts is always plain)."""
    has_w = (not plain) and isinstance(evidence, dict) and evidence.get("wmaps") is not None
    soft = has_w and bool(cfg.get("veto_soft_avoid"))
    excl = set() if soft else exclusion_set(evidence, pool, cfg.get("veto_excl_alpha", 0.0), weighted=has_w)
    avoid = avoid_scores(evidence, pool, cfg) if soft else None
    hab = habit_scores(evidence, pool, cfg) if use_habits else {k: {mp: 0.0 for mp in pool} for k in KINDS}
    n_series = evidence.get("n", 0) if isinstance(evidence, dict) else 0
    cold = (not any(v > 0 for v in comfort.values())) and not n_series
    return {"kappa": dict(comfort), "excl": excl, "habit": hab, "avoid": avoid, "cold": cold}


# ============================================================================
# Context: everything the tree needs, by map index (sorted pool order)
# ============================================================================
class Context:
    """One series, both teams. ell[i] = team_a's map logit vs team_b on map i
    (MapModel, unshifted: map_scale * relative edge). Team-specific arrays are
    keyed by "a" / "b"."""

    def __init__(self, pool, best_of, steps, ell, prefs, permaban=None, warnings=None):
        self.pool = list(pool)
        self.n = len(self.pool)
        self.idx = {mp: i for i, mp in enumerate(self.pool)}
        self.best_of = best_of
        self.steps = tuple(steps)
        self.ell = {"a": [ell.get(mp, 0.0) for mp in self.pool]}
        self.ell["b"] = [-x for x in self.ell["a"]]
        self.kap = {t: [prefs[t]["kappa"].get(mp, 0.0) for mp in self.pool] for t in "ab"}
        self.excl = {t: [mp in prefs[t]["excl"] for mp in self.pool] for t in "ab"}
        self.hab = {t: {k: [prefs[t]["habit"][k].get(mp, 0.0) for mp in self.pool] for k in KINDS}
                    for t in "ab"}
        self.has_avoid = any(prefs[t].get("avoid") for t in "ab")
        self.avoid = {t: {k: [(prefs[t].get("avoid") or {}).get(k, {}).get(mp, 0.0) for mp in self.pool]
                          for k in KINDS} for t in "ab"} if self.has_avoid else None
        self.cold = {t: bool(prefs[t]["cold"]) for t in "ab"}
        self.masked = {t: sorted(prefs[t]["excl"]) for t in "ab"}
        pb = permaban or {}
        self.permaban = {t: self.idx.get(pb.get(t)) for t in "ab"}
        self.warnings = list(warnings or [])
        self.n_picks = best_of - 1
        first = {}
        for t, (role, kind) in enumerate(self.steps):
            if kind != "pick" and role not in first:
                first[role] = t
        self.first_ban = first
        self.full = (1 << self.n) - 1


def _other(t):
    return "b" if t == "a" else "a"


# ============================================================================
# Policy + VetoOrchestrator
# ============================================================================
class Tree:
    """The draft for one starter hypothesis sigma ("a" or "b" starts).

    State (t, R, picks): step index, bitmask of maps left, picks so far (map
    indices). Bans matter only through R (deliberate Markov choice). node()
    returns (V_S, choices) where choices = [(map index, prob, child state)]
    and V_S = P(starter wins the series) when both teams follow the model
    (computed only with lookahead). Every state's choice probabilities are
    computed from that exact state, so the policy is closed-loop (ADR-1)."""

    def __init__(self, ctx, w, sigma):
        self.ctx, self.w, self.sigma = ctx, w, sigma
        self.team = {"S": sigma, "O": _other(sigma)}
        self.look = bool(w.get("lookahead"))
        temp = float(w.get("temp", 1.0)) or 1.0
        self.inv_t = 1.0 / temp
        self.memo = {}
        self.mask_exhausted = set()
        n = ctx.n
        # constant utility parts per (team, kind), by map index
        self.const = {}
        for t in "ab":
            o = _other(t)
            for k in KINDS:
                a = float(w.get(f"alpha_{k}", 0.0))
                g = float(w.get(f"gamma_{k}", 0.0))
                r = float(w.get(f"rho_{k}", 0.0))
                e = float(w.get(f"eta_{k}", 0.0))
                ell, kap, kapo, hab = ctx.ell[t], ctx.kap[t], ctx.kap[o], ctx.hab[t][k]
                if k == "pick":
                    if self.look:
                        c = [g * kap[i] + e * hab[i] for i in range(n)]
                    else:
                        c = [a * ell[i] + g * kap[i] + e * hab[i] for i in range(n)]
                else:
                    if self.look:
                        c = [-g * kap[i] + e * hab[i] for i in range(n)]
                    else:
                        c = [-a * ell[i] - g * kap[i] + r * kapo[i] + e * hab[i] for i in range(n)]
                z = float(w.get(f"zeta_{k}", 0.0))
                if z and ctx.has_avoid:      # phase 7: soft avoid (bans want it, picks shun it)
                    av = ctx.avoid[t][k]
                    c = [c[i] + (-z if k == "pick" else z) * av[i] for i in range(n)]
                self.const[(t, k)] = c
        # leaf value: P(S wins | q = sigmoid(ell_S)) on the played maps
        q = [_sigmoid(x) for x in ctx.ell[sigma]]
        self.q = q

    def leaf_value(self, maps):
        ps = [self.q[i] for i in maps]
        if len(ps) == 3:
            return _win3(*ps)
        if len(ps) == 1:
            return ps[0]
        return series_win(ps)

    def node(self, t, R, picks) -> Any:
        # leaf: (value, None); inner node: (value, choices, forced)
        key = (t, R, picks)
        got = self.memo.get(key)
        if got is not None:
            return got
        ctx = self.ctx
        if t == len(ctx.steps):
            d = R.bit_length() - 1
            if R != (1 << d):
                raise VetoError("decider state with more than one map left (I1)")
            v = self.leaf_value(picks + (d,)) if self.look else 0.0
            res = (v, None)
            self.memo[key] = res
            return res
        role, kind = ctx.steps[t]
        team = self.team[role]
        legal = [i for i in range(ctx.n) if R >> i & 1]
        if kind == "pick":
            ex = ctx.excl[team]
            allowed = [i for i in legal if not ex[i]]
            if not allowed:          # failure mode mask_exhausted_X: lift for this step only
                allowed = legal
                self.mask_exhausted.add(team)
        else:
            allowed = legal
        pb = ctx.permaban[team]
        forced = kind != "pick" and ctx.first_ban.get(role) == t and pb is not None and (R >> pb & 1)
        kids = []
        for i in allowed:
            if kind == "pick":
                kids.append((i, (t + 1, R & ~(1 << i), picks + (i,))))
            else:
                kids.append((i, (t + 1, R & ~(1 << i), picks)))
        if forced:
            kids = [k for k in kids if k[0] == pb]
            probs = [1.0]
            vs = [self.node(*kids[0][1])[0]] if self.look else None
        else:
            c = self.const[(team, kind)]
            if self.look:
                a = float(self.w.get(f"alpha_{kind}", 0.0))
                vs = [self.node(*ch)[0] for _, ch in kids]
                if role == "S":
                    us = [(a * _logit(v) + c[i]) * self.inv_t for (i, _), v in zip(kids, vs)]
                else:
                    us = [(a * _logit(1.0 - v) + c[i]) * self.inv_t for (i, _), v in zip(kids, vs)]
            else:
                vs = None
                us = [c[i] * self.inv_t for i, _ in kids]
            probs = _softmax(us)
        v = sum(p * x for p, x in zip(probs, vs or [])) if self.look else 0.0
        res = (v, [(i, p, ch) for (i, ch), p in zip(kids, probs)], bool(forced))
        self.memo[key] = res
        return res

    def root(self):
        return (0, self.ctx.full, ())


# ============================================================================
# Likelihood of an observed map order
# ============================================================================
def _obs_prob(tree, obs):
    """P(observed played maps | sigma) summed over every consistent veto.
    obs = map indices in played order (picks in pick order, then the decider
    if it was played)."""
    ctx = tree.ctx
    npk = ctx.n_picks
    obs = tuple(obs)
    dec = obs[npk] if len(obs) > npk else None
    obs_set = 0
    for i in obs:
        obs_set |= 1 << i
    memo = {}

    def rec(t, R, picks):
        key = (t, R, picks)
        if key in memo:
            return memo[key]
        if t == len(ctx.steps):
            d = R.bit_length() - 1
            r = 1.0 if (dec is None or d == dec) else 0.0
            memo[key] = r
            return r
        kind = ctx.steps[t][1]
        _, choices, _ = tree.node(t, R, picks)
        tot = 0.0
        if kind == "pick":
            j = len(picks)
            want = obs[j] if j < len(obs) else None
            for i, p, ch in choices:
                if want is None:
                    if obs_set >> i & 1:
                        continue
                elif i != want:
                    continue
                if p > 0.0:
                    tot += p * rec(*ch)
        else:
            for i, p, ch in choices:
                if obs_set >> i & 1:
                    continue
                if p > 0.0:
                    tot += p * rec(*ch)
        memo[key] = tot
        return tot

    return rec(*tree.root())


def loglik(ctx, w, obs_maps):
    """log P(observed map order) mixing both starters with prior pi =
    P(team_a started). obs_maps: map names in played order."""
    pi = float(w.get("pi", 0.5))
    obs = [ctx.idx[mp] for mp in obs_maps]
    pa = _obs_prob(Tree(ctx, w, "a"), obs)
    pb = _obs_prob(Tree(ctx, w, "b"), obs)
    p = pi * pa + (1.0 - pi) * pb
    return math.log(p) if p > 0 else float("-inf"), pa, pb


def posterior_counts(ctx, w, obs_maps):
    """Phase 3 (HabitAccumulator): expected choice counts c and opportunity
    counts o per (team, kind, map) under the posterior over vetoes consistent
    with the observed map order (both starters, prior pi). Returns
    {team: {kind: {map: [c, o]}}} or None if the order has probability 0."""
    pi = float(w.get("pi", 0.5))
    obs = tuple(ctx.idx[mp] for mp in obs_maps)
    npk = ctx.n_picks
    dec = obs[npk] if len(obs) > npk else None
    obs_set = 0
    for i in obs:
        obs_set |= 1 << i
    trees = {s: Tree(ctx, w, s) for s in "ab"}
    betas = {}
    for s, tree in trees.items():
        memo = {}

        def rec(t, R, picks, tree=tree, memo=memo):
            key = (t, R, picks)
            if key in memo:
                return memo[key]
            if t == len(ctx.steps):
                d = R.bit_length() - 1
                r = 1.0 if (dec is None or d == dec) else 0.0
                memo[key] = r
                return r
            kind = ctx.steps[t][1]
            _, choices, _ = tree.node(t, R, picks)
            tot = 0.0
            for i, p, ch in choices:
                if kind == "pick":
                    j = len(picks)
                    want = obs[j] if j < len(obs) else None
                    if (want is None and obs_set >> i & 1) or (want is not None and i != want):
                        continue
                elif obs_set >> i & 1:
                    continue
                if p > 0.0:
                    tot += p * rec(*ch)
            memo[key] = tot
            return tot

        rec(*tree.root())
        betas[s] = memo
    z = {s: betas[s].get(trees[s].root(), 0.0) for s in "ab"}
    total = pi * z["a"] + (1.0 - pi) * z["b"]
    if total <= 0:
        return None
    out = {t: {k: {mp: [0.0, 0.0] for mp in ctx.pool} for k in KINDS} for t in "ab"}
    for s, tree in trees.items():
        prior = pi if s == "a" else 1.0 - pi
        if z[s] <= 0:
            continue
        beta = betas[s]
        layer = {tree.root(): prior / total}
        for t in range(len(ctx.steps)):
            role, kind = ctx.steps[t]
            team = tree.team[role]
            nxt = {}
            for st in sorted(layer):
                a = layer[st]
                _, choices, _ = tree.node(*st)
                reach = a * beta.get(st, 0.0)       # posterior P(reach st)
                if reach <= 0:
                    continue
                cell = out[team][kind]
                for i, p, ch in choices:
                    cell[ctx.pool[i]][1] += reach     # m was available at this choice
                    b = beta.get(ch, 0.0)
                    if p <= 0 or b <= 0:
                        continue
                    cell[ctx.pool[i]][0] += a * p * b
                    nxt[ch] = nxt.get(ch, 0.0) + a * p
            layer = nxt
    return out


# ============================================================================
# VetoDistribution
# ============================================================================
def _forward(tree, prefix=None):
    """Reach probabilities layer by layer; returns (leaves, marginals,
    prefix_mass). leaves: {(picks..., decider): p}. marginals by (team,
    what, map index). prefix: [(team or None, action, map index)] observed
    steps; the mass that survives them is P(prefix | sigma)."""
    ctx = tree.ctx
    layer = {tree.root(): 1.0}
    marg = {}
    first_done = set()
    for t in range(len(ctx.steps)):
        role, kind = ctx.steps[t]
        team = tree.team[role]
        con = prefix[t] if prefix is not None and t < len(prefix) else None
        if con is not None and con[0] is not None and con[0] != team:
            return {}, {}, 0.0
        nxt = {}
        is_first = kind != "pick" and ctx.first_ban.get(role) == t
        for st in sorted(layer):
            a = layer[st]
            if a <= 0.0:
                continue
            _, choices, _ = tree.node(*st)
            for i, p, ch in choices:
                if con is not None and (i != con[2] or (con[1] == "pick") != (kind == "pick")):
                    continue
                q = a * p
                if q <= 0.0:
                    continue
                if kind == "pick":
                    key = (team, "picked", i)
                else:
                    key = (team, "banned", i)
                marg[key] = marg.get(key, 0.0) + q
                if is_first:
                    k2 = (team, "first_ban", i)
                    marg[k2] = marg.get(k2, 0.0) + q
                nxt[ch] = nxt.get(ch, 0.0) + q
        layer = nxt
        first_done.add(role)
    leaves = {}
    mass = 0.0
    for (t, R, picks), a in layer.items():
        d = R.bit_length() - 1
        leaves[picks + (d,)] = leaves.get(picks + (d,), 0.0) + a
        mass += a
    return leaves, marg, mass


def _viterbi(tree, target):
    """Most probable full sequence ending at the leaf `target` (picks +
    decider). Ties: higher probability first, then sorted map names (map
    indices follow sorted names)."""
    ctx = tree.ctx
    best: dict = {tree.root(): (1.0, ())}
    for t in range(len(ctx.steps)):
        nxt: dict = {}
        for st in sorted(best):
            p0, path = best[st]
            _, choices, forced = tree.node(*st)
            for i, p, ch in choices:
                if p <= 0.0:
                    continue
                cand = (p0 * p, path + ((t, i, forced),))
                cur = nxt.get(ch)
                if cur is None or cand[0] > cur[0] or (cand[0] == cur[0] and
                                                       [x[1] for x in cand[1]] < [x[1] for x in cur[1]]):
                    nxt[ch] = cand
        best = nxt
    for (t, R, picks), (p, path) in sorted(best.items()):
        d = R.bit_length() - 1
        if picks + (d,) == tuple(target):
            return p, path
    return 0.0, ()


def distribution(ctx, w, prefix=None, pi=None):
    """VetoDistribution for one series. Returns (dist, internals) where dist
    is the add-only output field veto_dist and internals carries the full
    outcome list (for the scoreline) and the modal sequence."""
    if ctx.n > EXACT_MAX_POOL:
        raise VetoError(f"pool of {ctx.n} maps is above the exact-enumeration cap {EXACT_MAX_POOL}")
    pi = float(w.get("pi", 0.5)) if pi is None else float(pi)
    pref = None
    if prefix:
        pref = []
        for s in prefix:
            mp = s.get("map") if isinstance(s, dict) else None
            if mp not in ctx.idx:
                raise VetoError(f"veto_prefix map {mp!r} not in the pool")
            pref.append((s.get("team"), s.get("action", "ban"), ctx.idx[mp]))
    trees = {s: Tree(ctx, w, s) for s in "ab"}
    fw = {s: _forward(trees[s], pref) for s in "ab"}
    if pref is None:
        for s in "ab":
            tot = fw[s][2]
            if abs(tot - 1.0) > CONSERVE_TOL or not math.isfinite(tot):
                raise VetoError(f"outcome probabilities sum to {tot!r} for starter {s} (I5)")
        post = {"a": pi, "b": 1.0 - pi}
    else:
        za, zb = pi * fw["a"][2], (1.0 - pi) * fw["b"][2]
        if za + zb <= 0:
            raise VetoError("veto_prefix has probability 0 under both starters")
        post = {"a": za / (za + zb), "b": zb / (za + zb)}
    # outcomes with pickers
    outcomes = {}
    npk = ctx.n_picks
    for s in "ab":
        leaves, _, mass = fw[s]
        if mass <= 0:
            continue
        for maps, p in leaves.items():
            pickers = tuple(trees[s].team[ctx.steps[t][0]] for t in range(len(ctx.steps))
                            if ctx.steps[t][1] == "pick")
            key = (maps, pickers)
            outcomes[key] = outcomes.get(key, 0.0) + post[s] * p / mass
    tot = sum(outcomes.values())
    if abs(tot - 1.0) > CONSERVE_TOL or not math.isfinite(tot):
        raise VetoError(f"mixed outcome probabilities sum to {tot!r} (I5)")
    ranked = sorted(outcomes.items(), key=lambda kv: (-kv[1], [ctx.pool[i] for i in kv[0][0]], kv[0][1]))
    # marginals
    names = ("played", "picked_a", "picked_b", "decider", "banned_a", "banned_b",
             "first_ban_a", "first_ban_b")
    marg = {mp: {k: 0.0 for k in names} for mp in ctx.pool}
    for s in "ab":
        leaves, mg, mass = fw[s]
        if mass <= 0:
            continue
        f = post[s] / mass
        for (team, what, i), q in mg.items():
            marg[ctx.pool[i]][f"{what}_{team}"] += f * q
    for (maps, pickers), p in outcomes.items():
        marg[ctx.pool[maps[-1]]]["decider"] += p
        for i in maps:
            marg[ctx.pool[i]]["played"] += p
    played_sum = sum(v["played"] for v in marg.values())
    if abs(played_sum - ctx.best_of) > 1e-6:
        raise VetoError(f"played marginals sum to {played_sum!r}, not {ctx.best_of} (I5)")
    # modal outcome and its most probable sequence
    (m_maps, m_pickers), m_p = ranked[0]
    s_modal = m_pickers[0] if m_pickers else ("a" if post["a"] >= post["b"] else "b")
    if m_pickers:
        s_modal = next(s for s in "ab" if trees[s].team[ctx.steps[
            next(t for t in range(len(ctx.steps)) if ctx.steps[t][1] == "pick")][0]] == m_pickers[0])
    _, path = _viterbi(trees[s_modal], m_maps)
    log = []
    for t, i, forced in path:
        role, kind = ctx.steps[t]
        side = trees[s_modal].team[role].upper()
        verb = "picks" if kind == "pick" else "bans"
        log.append(f"{side} {verb} {ctx.pool[i]}" + (" (permaban)" if forced else ""))
    log.append(f"decider: {ctx.pool[m_maps[-1]]}")
    ent = -sum(p * math.log2(p) for p in outcomes.values() if p > 0)
    acc, hpd = 0.0, 0
    for _, p in ranked:
        if acc >= 0.8:
            break
        acc += p
        hpd += 1
    warns = list(ctx.warnings)
    for t in "ab":
        for s in "ab":
            if t in trees[s].mask_exhausted and f"mask_exhausted_{t}" not in warns:
                warns.append(f"mask_exhausted_{t}")
    shown = [{"maps": [ctx.pool[i] for i in maps], "pickers": list(pk), "p": p}
             for (maps, pk), p in ranked if p >= TAIL_P]
    tail = sum(p for _, p in ranked if p < TAIL_P)
    dist = {
        "version": w.get("version", VERSION),
        "best_of": ctx.best_of,
        "starter_a": post["a"],
        "outcomes": shown,
        "tail_mass": tail,
        "marginals": marg,
        "entropy_bits": ent,
        "hpd80": hpd,
        "masked": {t: list(ctx.masked[t]) for t in "ab"},
        "cold_start": dict(ctx.cold),
        "warnings": warns,
    }
    internals = {"outcomes": ranked, "modal_maps": [ctx.pool[i] for i in m_maps],
                 "modal_log": log, "modal_starter": s_modal, "trees": trees, "post": post}
    return dist, internals


# ============================================================================
# ScorelineAdapter (BO3)
# ============================================================================
def mixture_scoreline(ranked, ell_a_by_index, scale, p_a):
    """BO3 scoreline as a mixture over outcomes (ADR-4): one shift c by
    bisection (bounds +-60, at most 200 halvings, stopped when the bracket
    stops moving) so that sum_v p_v P_A(series | sigmoid(s*ell + c)) = p_a.
    Returns ((p_2_0, p_2_1, p_1_2, p_0_2), c). Never writes upstream."""
    by_set, by_pair = {}, {}
    for (maps, _), p in ranked:
        k3 = tuple(sorted(maps))
        by_set[k3] = by_set.get(k3, 0.0) + p
        k2 = (min(maps[0], maps[1]), max(maps[0], maps[1]), maps[2])
        by_pair[k2] = by_pair.get(k2, 0.0) + p
    base = [scale * x for x in ell_a_by_index]
    sets = sorted(by_set.items())

    def win(c):
        q = [_sigmoid(b + c) for b in base]
        return sum(p * _win3(q[i], q[j], q[k]) for (i, j, k), p in sets)

    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if win(mid) < p_a:
            nlo, nhi = mid, hi
        else:
            nlo, nhi = lo, mid
        if nlo == lo and nhi == hi:
            break
        lo, hi = nlo, nhi
    c = (lo + hi) / 2.0
    q = [_sigmoid(b + c) for b in base]
    s20 = s21 = s12 = s02 = 0.0
    for (i, j, k), p in sorted(by_pair.items()):
        p1, p2, p3 = q[i], q[j], q[k]
        split = p1 * (1 - p2) + (1 - p1) * p2
        s20 += p * p1 * p2
        s21 += p * split * p3
        s12 += p * split * (1 - p3)
        s02 += p * (1 - p1) * (1 - p2)
    return (s20, s21, s12, s02), c


def compact(dist, top=8):
    """The site-build form: top outcomes plus marginals (display is J3)."""
    if not dist:
        return None
    return {k: (dist[k][:top] if k == "outcomes" else dist[k]) for k in dist}
