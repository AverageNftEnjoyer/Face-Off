#!/usr/bin/env python3
"""
Cross-source agreement: data/matches.json (primary) against the second source
data/valve_results.json (Valve regional-standings match rows, built by
data/valve_results.py). Read-only: it never writes matches.json.

Matching: same team pair, dates at most valve_results.DATE_WINDOW_DAYS apart,
one-to-one (closest dates first). The second source only knows the series
winner, so the check is the winner.

Buckets (inside the overlap window, i.e. the dates both sources cover):
  agree           matched, same winner
  disagree        matched, different winner                 -> needs a human
  primary_only    a matches.json series the second source should have (one of
                  the teams has a Valve page covering that date) but does not
  second_only     a second-source series between two tracked teams that
                  matches.json lacks; split into
                    lower_tier  present in matches_lower.json (B-tier etc.,
                                deliberately not tracked)
                    untracked   in neither file (another event the primary
                                does not track, or a series it missed)

Freshness (measured from the Valve repository tree): when a series played on
day D can first be seen in the second source = the first snapshot after D.

USAGE (repo root):
  python scripts/cross_source_check.py [--write] [--fill-candidates]
    --write            write data/cross_source_report.json
    --fill-candidates  ALSO write data/valve_fill.candidate.json (gitignored):
                       untracked second-source series dated after the newest
                       primary series, in the matches.json record shape with
                       every unknown field null. A review list only: nothing
                       reads it and nothing merges it (see docs/pipeline.md).
Exit 0 always unless the inputs cannot be read (exit 2); the blocking check
is the cross_source gate in scripts/data_gates.py.
Stdlib only, no network.
"""
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))

import valve_results as vr  # noqa: E402

REPORT = os.path.join(ROOT, "data", "cross_source_report.json")
FILL = os.path.join(ROOT, "data", "valve_fill.candidate.json")


def _pair(m):
    return tuple(sorted((m["team_a"], m["team_b"])))


def match_series(primary, second, window=None):
    """One-to-one matching on (team pair, |date gap| <= window), closest first.
    Returns {primary index: second index}."""
    window = vr.DATE_WINDOW_DAYS if window is None else window
    by_pair = {}
    for j, s in enumerate(second):
        by_pair.setdefault(_pair(s), []).append(j)
    cands = []
    for i, p in enumerate(primary):
        for j in by_pair.get(_pair(p), []):
            gap = vr.days_apart(p["date"], second[j]["date"])
            if gap <= window:
                cands.append((gap, p["date"], i, j))
    used_p, used_s, out = set(), set(), {}
    for gap, _, i, j in sorted(cands):
        if i in used_p or j in used_s:
            continue
        used_p.add(i)
        used_s.add(j)
        out[i] = j
    return out


def window_of(window):
    return vr.DATE_WINDOW_DAYS if window is None else window


def identity_mismatches(prim, sec, window=None):
    """Pairs (primary series, second-source series) within the date window
    that share exactly one team, whose result for that team is the same, one
    candidate each way (an ambiguous day is left out)."""
    window = window_of(window)
    cands = {}
    for i, p in enumerate(prim):
        for j, s in enumerate(sec):
            if vr.days_apart(p["date"], s["date"]) > window:
                continue
            shared = {p["team_a"], p["team_b"]} & {s["team_a"], s["team_b"]}
            if len(shared) != 1:
                continue
            t = shared.pop()
            if (p["winner"] == t) != (s["winner"] == t):
                continue
            cands.setdefault(("p", i), []).append(j)
            cands.setdefault(("s", j), []).append(i)
    out = []
    for (kind, i), js in sorted(cands.items()):
        if kind != "p" or len(js) != 1 or len(cands[("s", js[0])]) != 1:
            continue
        p, s = prim[i], sec[js[0]]
        t = ({p["team_a"], p["team_b"]} & {s["team_a"], s["team_b"]}).pop()
        po = p["team_b"] if p["team_a"] == t else p["team_a"]
        so = s["team_b"] if s["team_a"] == t else s["team_a"]
        out.append({"match_id": p["match_id"], "date": p["date"], "event": p.get("event"),
                    "team": t, "primary_opponent": po, "second_opponent": so,
                    "second_opponent_valve_name": s["valve_names"][[s["team_a"], s["team_b"]].index(so)],
                    "second_date": s["date"], "winner_primary": p["winner"]})
    return out


def compare(primary, second_file, lower=None, window=None):
    """The agreement report (a dict). primary: matches.json records;
    second_file: valve_results.load() output; lower: matches_lower.json."""
    second = second_file["series"]
    spans = second_file.get("spans") or {}
    if not second or not primary:
        return {"overlap": None, "note": "nothing to compare"}
    lo = max(min(s["date"] for s in second), min(p["date"] for p in primary))
    hi = min(max(s["date"] for s in second), max(p["date"] for p in primary))
    P = [p for p in primary if lo <= p["date"] <= hi]
    S = [s for s in second if lo <= s["date"] <= hi]
    pairs = match_series(P, S, window)
    matched_s = set(pairs.values())

    agree, disagree = 0, []
    for i, j in pairs.items():
        p, s = P[i], S[j]
        if p["winner"] == s["winner"]:
            agree += 1
        else:
            disagree.append({"match_id": p["match_id"], "date": p["date"],
                             "teams": [p["team_a"], p["team_b"]], "event": p.get("event"),
                             "primary_winner": p["winner"], "second_winner": s["winner"],
                             "second_date": s["date"], "snapshot": s.get("snapshot")})
    # an event still running when the newest snapshot was cut is not in it yet
    # (Valve lists an event's series once it has finished): its series are
    # "pending", not missing
    newest_snap = max((s.get("snapshot") or "") for s in second).replace("_", "-")
    running = {p.get("event_page") or p.get("event") for p in primary
               if p["date"] >= vr.shift(newest_snap, -window_of(window))}
    eligible = [i for i, p in enumerate(P)
                if vr.covered(spans, p["team_a"], p["date"]) or vr.covered(spans, p["team_b"], p["date"])]
    pending = [i for i in eligible if i not in pairs
               and (P[i].get("event_page") or P[i].get("event")) in running]
    pend = set(pending)
    missing = [i for i in eligible if i not in pairs and i not in pend]
    never_seen = sorted({t for p in P for t in (p["team_a"], p["team_b"])} - set(spans))

    rest = [S[j] for j in range(len(S)) if j not in matched_s]
    lower_hit = set()
    if lower:
        L = [m for m in lower if lo <= m["date"] <= hi]
        lower_hit = set(match_series(rest, L, window))
    second_only = {"lower_tier": len(lower_hit), "untracked": len(rest) - len(lower_hit)}
    untracked = [s for k, s in enumerate(rest) if k not in lower_hit]

    # identity mismatches: a primary series nobody matched and an unmatched
    # second-source series on the same day(s), sharing one team whose result
    # is the same in both, but with a DIFFERENT opponent. Either one source
    # names the opponent wrongly or a name maps to the wrong team.
    ident = identity_mismatches([P[i] for i in missing], untracked, window)
    ident_ids = {x["match_id"] for x in ident}
    primary_only = [{"match_id": P[i]["match_id"], "date": P[i]["date"], "event": P[i].get("event"),
                     "teams": [P[i]["team_a"], P[i]["team_b"]], "winner": P[i]["winner"]}
                    for i in missing if P[i]["match_id"] not in ident_ids]
    by_names = {}
    for x in ident:
        k = (x["primary_opponent"], x["second_opponent"])
        by_names[k] = by_names.get(k, 0) + 1

    n = agree + len(disagree)
    return {
        "overlap": [lo, hi],
        "date_window_days": vr.DATE_WINDOW_DAYS if window is None else window,
        "primary_series_in_overlap": len(P),
        "primary_series_checkable": len(eligible),
        "matched": n,
        "coverage_of_checkable": round(len(set(pairs) & set(eligible)) / len(eligible), 4) if eligible else None,
        "agree": agree,
        "disagree": len(disagree),
        "agreement_rate": round(agree / n, 4) if n else None,
        "disagreements": disagree,
        "identity_mismatch": len(ident),
        "identity_mismatch_by_name": [{"primary": a, "second": b, "series": n}
                                      for (a, b), n in sorted(by_names.items(), key=lambda x: (-x[1], x[0]))],
        "identity_mismatches": ident,
        "pending": len(pending),
        "primary_only": len(primary_only),
        "primary_only_sample": primary_only[:40],
        "primary_teams_without_valve_page": never_seen,
        "second_series_in_overlap": len(S),
        "second_only": second_only,
        "second_only_untracked_sample": [{"date": s["date"], "teams": [s["team_a"], s["team_b"]],
                                          "winner": s["winner"]} for s in untracked[:40]],
        "_untracked": untracked,
        "_matched_second_ids": [S[j]["match_id"] for j in pairs.values()],
    }


def freshness(second, matched_ids, tree_snaps, today=None):
    """How late the second source is, MEASURED: for every matched series, the
    days from its date to the first cached snapshot that listed it
    (first_seen). Snapshots read every other month (the initial build) make
    this an upper bound for that period; incremental runs read every new one.
    Plus the publication cadence from the repository tree and today's gap."""
    lags = sorted((date.fromisoformat(r["first_seen"].replace("_", "-"))
                   - date.fromisoformat(r["date"])).days for r in second
                  if r["match_id"] in matched_ids)
    snaps = sorted(date.fromisoformat(s.replace("_", "-")) for s in tree_snaps)
    gaps = sorted((b - a).days for a, b in zip(snaps, snaps[1:]))

    def q(xs, f):
        return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None
    today = today or date.today()
    return {"snapshots_in_tree": len(snaps),
            "newest_snapshot": snaps[-1].isoformat() if snaps else None,
            "days_since_newest_snapshot": (today - snaps[-1]).days if snaps else None,
            "snapshot_gap_days": {"median": q(gaps, 0.5), "max": gaps[-1] if gaps else None},
            "lag_days_to_first_listing": {"n": len(lags), "median": q(lags, 0.5), "p90": q(lags, 0.9),
                                          "min": lags[0] if lags else None,
                                          "max": lags[-1] if lags else None}}


def tree_snapshots():
    """Every live/ details snapshot in the cached Valve tree (no network)."""
    try:
        import lineups
        lineups.OFFLINE = True
        paths = lineups.tree_paths()
    except (SystemExit, OSError, ValueError):
        return []
    return sorted({p.split("/")[3] for p in paths
                   if p.startswith("live/") and "/details/" in p and p.count("/") == 4})


def _load(name):
    p = os.path.join(ROOT, "data", name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    primary = _load("matches.json")
    second = vr.load()
    if primary is None or second is None:
        print("cross_source_check: data/matches.json or data/valve_results.json missing "
              "(run python data/valve_results.py)", file=sys.stderr)
        return 2
    rep = compare(primary, second, _load("matches_lower.json"))
    untracked = rep.pop("_untracked", [])
    rep["second_source_snapshots"] = second.get("snapshots")
    rep["freshness"] = freshness(second["series"], set(rep.pop("_matched_second_ids", [])),
                                 tree_snapshots() or second.get("snapshots") or [])
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("primary_only_sample", "second_only_untracked_sample",
                                   "identity_mismatches")}, indent=1))
    if "--write" in argv:
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
        print("wrote data/cross_source_report.json")
    if "--fill-candidates" in argv:
        newest = max(p["date"] for p in primary)
        cands = [s for s in untracked if s["date"] > newest]
        with open(FILL, "w", encoding="utf-8") as f:
            json.dump({"note": "REVIEW LIST ONLY: second-source series after the newest primary "
                               "series; winner only, no maps / event / best-of. Never merged.",
                       "series": cands}, f, indent=1, ensure_ascii=False)
        print(f"wrote data/valve_fill.candidate.json ({len(cands)} series after {newest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
