#!/usr/bin/env python3
"""
tenure.py -- point-in-time roster tenure: how long the current five have
played together, from dated per-series lineups. Stdlib only.

WHAT IT MEASURES (design: the tenure spec, section 3)
  For team X on day D, the regular five R and, for each of its 10 pairs
  (p, q), the pair tenure s_pq = D - start of the pair's current run of
  series together (ANY team, so a core that moves organisation keeps its
  history). A run breaks when the pair goes more than GAP_DAYS without a
  series together. Team familiarity
      T = mean over resolved pairs of ln(1 + min(s, CAP_DAYS)) / ln(1 + CAP_DAYS)
  in [0, 1]: 0 = five strangers, 1 = every pair a year or more together.
  One new regular resets 4 of 10 pairs, two reset 7, a rebuild all 10.

SOURCES
  data/lineups.json        five per series side (Valve), with avail_X: the
                           day that row became public (when present; see load()).
  data/event_lineups.json  Liquipedia squad histories (join / leave dates):
                           only to resolve runs censored by the start of the
                           Valve data, and to confirm a regular.
  data/roster_overrides.json  hand-entered moves: LIVE ONLY (live=True).
                           Backtests never read them: history entered by hand
                           keeps the moves that stuck.

POINT-IN-TIME (evaluation/leakage_check.py tests the notes' use of it)
  * A lineup row is read on day D only if it is dated < D and, in a backtest
    (live=False), its avail < D: what this source had published by then.
    Live (live=True) every row in the file is public by construction.
  * Squad spells count on day D only with joindate <= D; a leave dated > D
    is still active on D.
  * An override dated F is read only by days D > F.

REGULARS, STAND-INS
  R = players of the team's latest readable five who are CONFIRMED (>=
  CONFIRM_N series for the team in the readable rows of the last CAP_DAYS,
  an active squad spell on D, or an override five), the rest filled from
  the core (most appearances in the last CORE_N readable rows, ties to the
  most recent). An unconfirmed player is a stand-in: not a regular, no clock,
  resets nobody; stand-in effects stay in predictor.py's roster factor.

NEVER GUESS: known=False (and factor 0) when the team has no readable row,
its latest readable row is older than STALE_DAYS, or fewer than 5 regulars
resolve. A pair whose run may have begun before the Valve data starts and
that neither the squad history nor the cap resolves is left out; the factor
is then multiplied by the resolved share (toward 0, never imputed).

No team or player names live in this file; thresholds are the constants below.
"""
import json
import math
import os
import re
from collections import defaultdict
from datetime import date, timedelta
from itertools import combinations

import paths

DATA = paths.DATA

# ---- pre-registered constants (tenure_eval.py; do not tune) ---------------
CAP_DAYS = 365      # tenure beyond a year adds nothing (buckets stop differing)
GAP_DAYS = 90       # a pair / player run breaks after this long without a series together
STALE_DAYS = 75     # latest readable lineup older than this -> unknown
CONFIRM_N = 3       # series with the team before an unlisted player counts as a regular
CORE_N = 8          # lineups the core-five fill looks back over (lineup_features.CORE_N)
# ---- display only (never in a number) -------------------------------------
GELL_DAYS = 90      # "new roster" window
NEW_MIN = 2         # regulars joined inside the window before the marker shows

_LOG_CAP = math.log1p(CAP_DAYS)


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _d(s):
    return s if isinstance(s, date) else date.fromisoformat(s)


def g(days):
    """Tenure in days -> [0, 1] on the log scale, capped."""
    return math.log1p(min(max(days, 0), CAP_DAYS)) / _LOG_CAP


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Tenure:
    """rows from data files; query with state(team, day) / factor(a, b, day).

    lineups   {"date|a|b": {"a": five, "b": five, "avail_a": iso, ...}}
    squads    {team: [[player, joindate, enddate or ""]]}
    overrides lineup_features.load_overrides() list (used only if live)
    live      True: every row is public (the site today); False: a backtest,
              rows are read only after their avail date.
    first_seen_censored
              True: a run that begins within GAP_DAYS of the team's FIRST
              row in the data is censored too (a team new to the S/A data
              did not just form), and resolves through the squad history or
              stays unresolved. False (the default, and what tenure_eval.py
              pre-registered) censors only at the start of the Valve data.
              The display uses True."""

    def __init__(self, lineups, squads=None, overrides=None, live=False, first_seen_censored=False):
        self.live = live
        self.first_seen_censored = first_seen_censored
        self.team_rows = defaultdict(list)     # team -> [(date, avail, frozenset five, override?)]
        for key, rec in lineups.items():
            dt, a, b = key.split("|", 2)
            for side, team in (("a", a), ("b", b)):
                five = rec.get(side)
                if five and len(set(five)) == 5:
                    av = rec.get(f"avail_{side}")
                    self.team_rows[team].append((_d(dt), _d(av) if av else None,
                                                 frozenset(_norm(p) for p in five), False))
        if live:
            for o in overrides or ():
                if o.get("kind") == "roster_change":
                    f = _d(o["from"])
                    self.team_rows[o["team"]].append((f, f, frozenset(_norm(p) for p in o["players"]), True))
        for rows in self.team_rows.values():
            rows.sort(key=lambda r: (r[0], r[3]))
        starts = [r[0] for rows in self.team_rows.values() for r in rows if not r[3]]
        self.data_start = min(starts) if starts else None
        # pair -> sorted [(date, avail)] of series the two played together (any team)
        self.pairs = defaultdict(list)
        for rows in self.team_rows.values():
            for dt, av, five, _ in rows:
                for p, q in combinations(sorted(five), 2):
                    self.pairs[(p, q)].append((dt, av))
        for v in self.pairs.values():
            v.sort(key=lambda r: (r[0], r[1] or date.max))
        self.by_player = defaultdict(list)     # player -> [(date, team)] (display resolution)
        for team, rows in self.team_rows.items():
            for dt, _, five, ov in rows:
                if not ov:
                    for p in five:
                        self.by_player[p].append((dt, team))
        self.squads = defaultdict(dict)        # team -> norm player -> [(join, end or None)]
        for team, spells in (squads or {}).items():
            for row in spells:
                try:
                    j = _d(row[1])
                    e = _d(row[2]) if len(row) > 2 and row[2] else None
                except (TypeError, ValueError, IndexError):
                    continue
                self.squads[team].setdefault(_norm(row[0]), []).append((j, e))

    # ---- point-in-time access ------------------------------------------------
    def _ok(self, dt, av, day):
        return dt < day and (self.live or (av is not None and av < day))

    def readable(self, team, day):
        day = _d(day)
        return [r for r in self.team_rows.get(team, ()) if self._ok(r[0], r[1], day)]

    def squad_join(self, team, p, day):
        """Join date of p's squad spell on `team` active on day (join <= day,
        no end <= day), else None."""
        for j, e in self.squads.get(team, {}).get(p, ()):
            if j <= day and (e is None or e > day):
                return j
        return None

    def _run_start(self, dates):
        """Start of the run ending at the last of sorted dates (gaps <= GAP_DAYS)."""
        i = len(dates) - 1
        while i > 0 and (dates[i] - dates[i - 1]).days <= GAP_DAYS:
            i -= 1
        return dates[i]

    def _censored(self, start, team=None):
        if self.data_start is not None and (start - self.data_start).days <= GAP_DAYS:
            return True
        if self.first_seen_censored and team is not None:
            rows = [r[0] for r in self.team_rows.get(team, ()) if not r[3]]
            return bool(rows) and (start - min(rows)).days <= GAP_DAYS
        return False

    # ---- the team on day D ---------------------------------------------------
    def regulars(self, team, day, rows=None):
        """(R, stand-ins) from readable rows, or (None, []) if R does not resolve."""
        day = _d(day)
        rows = self.readable(team, day) if rows is None else rows
        if not rows:
            return None, []
        last = rows[-1]
        lo = day - timedelta(days=CAP_DAYS)
        count = defaultdict(int)
        for dt, _, five, _ in rows:
            if dt >= lo:
                for p in five:
                    count[p] += 1
        if last[3]:                              # an override five is the roster
            return sorted(last[2]), []
        conf = [p for p in last[2] if count[p] >= CONFIRM_N or self.squad_join(team, p, day)]
        subs = sorted(set(last[2]) - set(conf))
        base = rows[-CORE_N:]
        seen, cnt = {}, defaultdict(int)
        for i, (_, _, five, _) in enumerate(base):
            for p in five:
                cnt[p] += 1
                seen[p] = i
        fill = [p for p in sorted(cnt, key=lambda p: (-cnt[p], -seen[p], p)) if p not in conf and p not in subs]
        R = sorted(conf + fill[:5 - len(conf)])
        return (R, subs) if len(R) == 5 else (None, subs)

    def player_start(self, team, p, day, rows):
        """Start of p's current run with `team` (readable rows), resolved
        through the squad history when the run reaches the data start.
        -> (date, resolved)."""
        dates = [dt for dt, _, five, _ in rows if p in five]
        if not dates:
            return None, False
        start = self._run_start(dates)
        if self.first_seen_censored:
            # display resolution: a team that plays S/A series rarely breaks
            # its run on the GAP_DAYS rule; the squad join date is the tenure
            # unless the player is seen with another team since joining.
            j = self.squad_join(team, p, day)
            if j is not None and j < start and not any(
                    t != team and j <= d < start for d, t in self.by_player.get(p, ())):
                return j, True
        if not self._censored(start, team):
            return start, True
        j = self.squad_join(team, p, day)
        if j is not None and j <= start:
            return j, True
        return start, (day - start).days >= CAP_DAYS

    def pair_start(self, team, p, q, day, last_row):
        """Start of the current run of p and q together. -> (date, resolved)."""
        key = (p, q) if p < q else (q, p)
        dates = [dt for dt, av in self.pairs.get(key, ()) if self._ok(dt, av, day)]
        if not dates or (last_row - dates[-1]).days > GAP_DAYS:
            return None, False                   # not a current pair in the readable data
        start = self._run_start(dates)
        if not self._censored(start, team):
            return start, True
        jp, jq = self.squad_join(team, p, day), self.squad_join(team, q, day)
        if jp is not None and jq is not None and max(jp, jq) <= start:
            return max(jp, jq), True
        return start, (day - start).days >= CAP_DAYS

    def state(self, team, day):
        """Tenure state of `team` on day D (ISO or date):
        {known, why, five, stand_ins, T, cov, days_together, new, thru, starts}."""
        day = _d(day)
        rows = self.readable(team, day)
        out = {"known": False, "why": "", "five": None, "stand_ins": [], "T": None, "cov": 0.0,
               "days_together": None, "new": [], "thru": rows[-1][0].isoformat() if rows else None,
               "starts": {}}
        if not rows:
            out["why"] = "no published lineup yet"
            return out
        if (day - rows[-1][0]).days > STALE_DAYS:
            out["why"] = f"latest published lineup is older than {STALE_DAYS} days"
            return out
        R, subs = self.regulars(team, day, rows)
        out["stand_ins"] = subs
        if R is None:
            out["why"] = "the regular five does not resolve"
            return out
        out["five"] = R
        starts, raw = {}, {}
        for p in R:
            s, ok = self.player_start(team, p, day, rows)
            raw[p] = (s, ok)
            starts[p] = (s.isoformat(), ok) if s else (None, False)
        vals = []
        for p, q in combinations(R, 2):
            s, ok = self.pair_start(team, p, q, day, rows[-1][0])
            if self.first_seen_censored and raw[p][1] and raw[q][1]:
                s, ok = max(raw[p][0], raw[q][0]), True
            if ok:
                vals.append(g((day - s).days))
        out["starts"] = starts
        if not vals:
            out["why"] = "no pair tenure resolves"
            return out
        out.update(known=True, T=sum(vals) / len(vals), cov=len(vals) / 10.0)
        if all(ok for _, ok in starts.values()):
            newest = max(_d(s) for s, _ in starts.values())
            out["days_together"] = (day - newest).days
        out["new"] = sorted(p for p, (s, ok) in starts.items()
                            if s and ok and (day - _d(s)).days <= GELL_DAYS)
        return out

    def factor(self, a, b, day, L=None):
        """(x, z, info): x = c (T_a - T_b), z = c L ((1-T_a) + (1-T_b)) with
        c = min(cov_a, cov_b); both 0 unless both teams are known."""
        sa, sb = self.state(a, day), self.state(b, day)
        if not (sa["known"] and sb["known"]):
            return 0.0, 0.0, (sa, sb)
        c = min(sa["cov"], sb["cov"])
        x = c * (sa["T"] - sb["T"])
        z = c * L * ((1 - sa["T"]) + (1 - sb["T"])) if L is not None else 0.0
        return x, z, (sa, sb)


def load(live=False, lineups=None, squads=None, overrides=None, first_seen_censored=False):
    """Tenure over the repo's data files. live=True also reads the overrides."""
    if lineups is None:
        lineups = _load_json(os.path.join(DATA, "lineups.json"), {}).get("lineups", {})
    if squads is None:
        squads = _load_json(os.path.join(DATA, "event_lineups.json"), {}).get("squads", {})
    if overrides is None and live:
        import lineup_features as LF
        overrides = LF.load_overrides()
    return Tenure(lineups, squads, overrides, live=live, first_seen_censored=first_seen_censored)
