#!/usr/bin/env python3
"""
roster_flags.py -- DISPLAY-ONLY "treat with caution" roster flags.

A team in a prediction gets a flag when it is fielding a stand-in or has
changed its lineup recently. The flag is shown beside the prediction; it is
NOT a model input and moves no probability (predictor.py already reads the
per-event stand-in flags through backtest.roster_flags; nothing here changes
that). Making a flag a model input would need its own validation gate.

Sources (read only; roster moves are entered by hand, never written here):
  data/roster_events.json    stand-ins per team per TOURNAMENT PAGE, undated
                             (backtest.roster_events()).
  data/lineups.json          five-man lineup per series, dated; lags the
                             results (Valve publishes it in monthly snapshots).
  data/roster_overrides.json manual moves with a `from` date
                             (lineup_features.load_overrides / team_history).

POINT-IN-TIME. A flag for a series on day D uses only what was knowable
before D:
  * lineups / overrides: rows dated strictly before D (lineup_features rule;
    an override dated F is first read by D > F). Series counts use results
    dated before D.
  * tournament stand-in lists carry no date, so they are used
      - for a series not yet played at build time: the list as it stands now
        (it was published before the series is played), and
      - for a finished series: only when a dated lineup of that team in a
        series at the same event (dated >= event start and < D) already
        contains the stand-in. Otherwise a mid-event stand-in could be
        flagged on a match he never played (leakage_check.py "NOT COVERED").

No team or player names, matches or dates live in this file; the thresholds
are the named constants below.
"""
from collections import defaultdict
from datetime import date, timedelta

import backtest as bt
import lineup_features as LF

# ---- thresholds (the one place they live) --------------------------------
# A lineup counts as "settled" after this many series with it: by then it
# fills the whole core window, so the core five is the new lineup itself.
RECENT_SERIES = LF.CORE_N
# The reference lineup is the core five over the team's CORE_N lineups before
# the latest RECENT_SERIES (lineup_features.CORE_N, the stand-in feature's rule).
CORE_N = LF.CORE_N
# A change older than this many days is not "recent" however few series the
# team has played since (lineup_features.CONT_DAYS, the continuity window).
MAX_DAYS = LF.CONT_DAYS
# New players before a lineup change is flagged: 2, where the backtest roster
# discount reaches its full size (min(1, n / 2)). One new player was in 27% of
# 2025-26 series with no worse calls (Brier .218 vs .217); stand-ins come from
# the tournament lists and stand_in overrides, which are flagged at any size.
MIN_NEW = 2


def _title_match(title, tour):
    """backtest.roster_flags' page rule: a parent page covers its stage pages."""
    return bool(title) and (title == tour or title.startswith(tour + "/") or tour.startswith(title + "/"))


def event_standins(team, title, events=None):
    """[(stand-in, replaced, missing_igl)] listed for `team` at tournament page
    `title` (all matching pages). Undated: see the module docstring."""
    events = bt.roster_events() if events is None else events
    out = []
    for tour, ev in (events.get(team) or {}).items():
        if not _title_match(title, tour) or not ev.get("standin"):
            continue
        for row in ev.get("players") or []:
            sp = str(row[0]).strip() if row else ""
            rep = str(row[1]).strip() if len(row) > 1 else ""
            if sp and all((x[0], x[1]) != (sp, rep) for x in out):
                out.append((sp, rep, bool(ev.get("missing_igl"))))
    return out


def _core(rows):
    """Core five of (date, five) rows: lineup_features' rule (most appearances,
    ties to the most recent; temporary override rows left out when possible)."""
    base = [r for r in rows if not getattr(r[1], "temporary", False)] or rows
    count, seen = defaultdict(int), {}
    for i, (_, five) in enumerate(base):
        for p in five:
            count[p] += 1
            seen[p] = i
    return set(sorted(count, key=lambda p: (-count[p], -seen[p], p))[:5])


def lineup_change(rows, day, played=None):
    """Recent lineup change from one team's dated lineup rows, using only rows
    dated < day. -> None or {new, series, since, thru, temporary}.
      new    players of the latest five outside the core five of the CORE_N
             lineups before the latest RECENT_SERIES (sorted); at least
             MIN_NEW of them unless the latest five is a stand_in override
      since  date of the series where the earliest of them joined
      series k = the team's series dated >= since and < day; from `played`
             (its series dates) so series whose lineup is not published yet
             still count, else from lineup rows. Flag only while k <=
             RECENT_SERIES and since is within MAX_DAYS of day.
      thru   date of the newest lineup used (lineups lag the results)."""
    prior = [(d, f) for d, f in rows if d < day]
    if len(prior) < 2:
        return None
    base = prior[:-RECENT_SERIES][-CORE_N:]
    if not base:                 # too little history to tell a change from a debut
        return None
    core = _core(base)
    last = prior[-1][1]
    new = sorted(p for p in last if p not in core)
    temporary = bool(getattr(last, "temporary", False))
    manual = isinstance(last, LF.Five)    # override rows are Five; Valve lineup rows are plain tuples
    if not new or (len(new) < MIN_NEW and not temporary):
        return None
    start = len(prior) - 1       # first series of the new players' current run
    for p in new:
        i = len(prior) - 1
        while i > 0 and p in prior[i - 1][1]:
            i -= 1
        start = min(start, i)
    since = prior[start][0]
    k = len(prior) - start if played is None else sum(1 for d in played if since <= d < day)
    lo = (date.fromisoformat(day) - timedelta(days=MAX_DAYS)).isoformat()
    if k > RECENT_SERIES or since < lo:
        return None
    return {"new": new, "series": k, "since": since, "thru": prior[-1][0], "temporary": temporary,
            "manual": manual}


class Flagger:
    """Builds flags for (team, day) against fixed inputs.
    hist     lineup_features.team_history(matches, lineups, overrides)
    events   roster_events.json content (None reads the file)
    played   team -> [series dates] (series_dates(matches))
    display  lower-cased player id -> shown name"""

    def __init__(self, hist, events=None, played=None, display=None):
        self.hist = hist
        self.events = bt.roster_events() if events is None else events
        self.played = played or {}
        self.display = display or {}

    def _played_before(self, team, name, lo, day):
        """True if `name` is in a dated lineup of `team` with lo <= date < day."""
        n = LF._norm(name)
        return any(lo <= d < day and n in f for d, f in self.hist.get(team, ()))

    def flags(self, team, day, title=None, finished=False, event_start=None):
        """Display flags for `team` in a series on `day` (ISO) at tournament page
        `title`. finished: the series is already played at build time (its
        undated stand-in list then needs a dated lineup to confirm it).
        -> list of {k, src, r, ...}: k "standin" | "change", r the reason line."""
        out, named = [], set()
        for sp, rep, igl in event_standins(team, title, self.events):
            if finished and not (event_start and self._played_before(team, sp, event_start, day)):
                continue
            named.add(LF._norm(sp))
            out.append({"k": "standin", "src": "event",
                        "r": f"Stand-in: {sp} for {rep}" if rep else f"Stand-in: {sp}",
                        **({"igl": 1} if igl else {})})
        ch = lineup_change(self.hist.get(team, ()), day, self.played.get(team))
        if ch:
            new = [self.display.get(p, p) for p in ch["new"] if p not in named]
            if new and (ch["temporary"] or len(new) >= MIN_NEW):
                n, k = len(new), ch["series"]
                if ch["temporary"]:
                    r = f"Stand-in: {', '.join(new)}"
                elif k:
                    r = f"{n} new player{'s' if n > 1 else ''} in the last {k} series: {', '.join(new)}"
                else:   # a hand-entered move with no series played since
                    r = f"{n} new player{'s' if n > 1 else ''} since {ch['since']}, no series played yet: {', '.join(new)}"
                out.append({"k": "standin" if ch["temporary"] else "change",
                            "src": "override" if ch["temporary"] or ch["manual"] else "lineups",
                            "r": r, "since": ch["since"], "thru": ch["thru"]})
        return out


def series_dates(matches):
    """team -> [ISO date of each of its series]."""
    out = defaultdict(list)
    for m in matches:
        out[m["team_a"]].append(m["date"])
        out[m["team_b"]].append(m["date"])
    return out


def load_flagger(matches=None, display=None):
    """Flagger over the repo's data files (overrides included)."""
    matches = bt.load_matches() if matches is None else matches
    return Flagger(LF.team_history(matches, LF.load()), played=series_dates(matches), display=display)
