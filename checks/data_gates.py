#!/usr/bin/env python3
"""
Data-quality gates for the Faceoff pipeline. Compares a freshly rebuilt
data/matches.json against the last published one and BLOCKS the publish when
anything looks wrong. A green run means the data is safe to ship; a red run
means a human must look first.

Gates:
  schema         every record has the required fields with the right types; the
                 winner is one of the two teams; no wiki markup leaks into names
  duplicate_ids  no two records share a match_id
  sane_counts    total series must not collapse (mass deletion) or explode
                 (absurd spike) in one update
  changed_results  a series that is already published (same match_id) must not
                 change its winner or its map winners. A handful happen (Liquipedia
                 corrections); a flood means the parser or the page broke.
  team_names     team names are non-empty display names; brand-new teams are
                 REPORTED (info), not failed. The phantom-team protection lives in
                 collectors/collect_liquipedia.py, which refuses to resolve unknown
                 codes offline
  elo_bounds     per-team Elo, recomputed on old and new data. A team may move at
                 most max(--elo-bound, 40 x its new or changed series). In an
                 append-only update (every new series is dated on or after the
                 newest old one) a team with no new series must not move at all.
  cross_source   agreement with a second, independent results source
                 (data/valve_results.json, Valve's regional-standings match rows;
                 winner only). Fails when the update ADDS a winner disagreement or
                 an identity mismatch (same day, team and result, different
                 opponent) beyond valve_results.GATE_NEW_DISAGREE_MAX. Skipped
                 with a note when the second source is absent (never blocks on
                 the second source being down). Report: checks/cross_source_check.py
  vrs_freshness  the stored Valve Regional Standings (data/vrs.json) must not be
                 older than the newest Valve published at the last refresh
                 (data/vrs_status.json) nor older than Valve's measured cadence
                 allows (max recent gap + 7 days). Offline; a fetch failure
                 alone warns here and is raised by `vrs.py --check` in the workflow.
  lower_tier     data/matches_lower.json schema (same rules + a known `kind`).
                 REPORT ONLY, never blocks: lower-tier data must not stop an S/A
                 publish. The Elo bounds above stay S/A-only for the same reason.

USAGE:
  python checks/data_gates.py [--new data/matches.json] [--old PATH | --old-git]
         [--elo-bound N] [--allow-backfill] [--allow-no-baseline]
  --old-git compares against git HEAD (what the workflows use). Without --old or
  --old-git there is no baseline and the count / change / Elo gates are skipped
  (reported as such). With --old-git a missing baseline is an ERROR (exit 2)
  unless --allow-no-baseline is given: a gate that cannot compare must not pass.
  --allow-backfill is for an intentional large historical backfill: it lifts the
  append-only rule and the spike cap (env FACEOFF_ALLOW_BACKFILL=1 does the same).

  Exit 0 = all gates pass. Exit 1 = a gate failed (report on stdout as JSON).
  Exit 2 = could not run (no baseline, git missing, unreadable input).

Stdlib only. Never touches the network.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

REQUIRED = {
    "match_id": str, "date": str, "event": str, "team_a": str, "team_b": str,
    "winner": str, "maps": list, "event_page": str, "tier": str,
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_RE = re.compile(r"^[0-9a-f]{12}$")
WIKI_RE = re.compile(r"[{}\[\]|]")
# A single update adding more series than this is a spike, not a match day.
SPIKE_CAP = 1000
# The total may shrink a little (deleted pages); more than this is a collapse.
SHRINK_FRAC = 0.01
SHRINK_MIN = 10
# Published series whose winner / map winners changed. Real history has had one
# (a Liquipedia correction); more than this many (or this share) is corruption.
CHANGED_MAX = 3
CHANGED_FRAC = 0.002
ELO_BOUND: float = 150  # floor of the per-team allowance
ELO_PER_SERIES = 40  # one series moves a team at most ~40 Elo (K 32 x 1.25, BO5)
ELO_IDLE = 1.0       # a team with no new series in an append-only update


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class NoBaseline(Exception):
    """The baseline (the last published matches.json) could not be read."""


def load_old_git():
    """matches.json as committed at HEAD. Raises NoBaseline with the reason."""
    try:
        out = subprocess.run(["git", "show", "HEAD:data/matches.json"], cwd=ROOT,
                             capture_output=True)
    except FileNotFoundError as e:
        raise NoBaseline("git is not installed or not on PATH") from e
    if out.returncode != 0:
        raise NoBaseline("git show HEAD:data/matches.json failed: "
                         + out.stderr.decode("utf-8", "replace").strip()[:200])
    # bytes -> UTF-8 explicitly: text=True would use the locale code page
    # (cp1252 on Windows), which changes the derived ids of non-ASCII names
    return json.loads(out.stdout.decode("utf-8"))


def ensure_ids(matches):
    """Baselines written before stable_match_id existed get the same
    deterministic IDs derived from their fields, so old-vs-new comparison
    still works."""
    from collect_liquipedia import stable_match_id  # noqa: E402
    for m in matches:
        if "match_id" not in m:
            ext = m.get("ext_match_id") or ""
            src = m.get("source") or ""          # baselines written before event_page / ext_match_id
            if not ext and "hltv match" in src:
                ext = src.split("hltv match ")[-1].rstrip(")")
            m["match_id"] = stable_match_id(
                m["date"], m["team_a"], m["team_b"], m.get("event") or "", ext)
    return matches


def gate_schema(new):
    bad = []
    for i, m in enumerate(new):
        for k, t in REQUIRED.items():
            if k not in m:
                bad.append(f"record {i}: missing field {k!r}")
            elif not isinstance(m[k], t):
                bad.append(f"record {i}: field {k!r} is {type(m[k]).__name__}, want {t.__name__}")
        if bad and len(bad) > 10:
            break
        if isinstance(m.get("match_id"), str) and not ID_RE.match(m["match_id"]):
            bad.append(f"record {i}: match_id {m['match_id']!r} is not 12-hex")
        if isinstance(m.get("date"), str) and not DATE_RE.match(m["date"]):
            bad.append(f"record {i}: date {m['date']!r} is not YYYY-MM-DD")
        for k in ("team_a", "team_b", "event"):
            v = m.get(k)
            if isinstance(v, str) and (not v.strip() or WIKI_RE.search(v)):
                bad.append(f"record {i}: field {k!r} looks wrong: {v!r}")
        if m.get("winner") not in (m.get("team_a"), m.get("team_b")):
            bad.append(f"record {i}: winner {m.get('winner')!r} is not one of the teams")
        if isinstance(m.get("maps"), list):
            for mp in m["maps"]:
                if not isinstance(mp, dict) or mp.get("winner") not in (m.get("team_a"), m.get("team_b")):
                    bad.append(f"record {i}: map entry has bad winner: {mp!r}")
                    break
    return bad


LOWER_KINDS = {"main", "qualifier", "sa_gap"}


def gate_lower(path=None):
    """data/matches_lower.json (lower-tier series: Elo data and team-page
    results, never events). WARN ONLY: the S/A publish must never be blocked by
    lower-tier data. The same schema rules as matches.json plus a known
    `kind`; malformed records are reported here and skipped by
    backtest.read_lower, and the refresh never writes a file that fails them
    (pipeline/lower_refresh.py)."""
    path = path or os.path.join(ROOT, "data", "matches_lower.json")
    if not os.path.exists(path):
        return {"note": "no data/matches_lower.json; ratings use S/A series only", "warnings": []}
    try:
        lower = load(path)
        if not isinstance(lower, list):
            raise ValueError("not a list")
    except (OSError, ValueError) as e:
        return {"warnings": [f"unreadable: {e}"]}
    warn = gate_schema(lower)
    warn += [f"record {i}: unknown kind {m.get('kind')!r}" for i, m in enumerate(lower)
             if isinstance(m, dict) and m.get("kind") not in LOWER_KINDS][:10]
    dups = gate_duplicate_ids(lower)
    if dups:
        warn.append(f"{len(dups)} duplicate match_ids, e.g. {dups[:3]}")
    dates = [m["date"] for m in lower if isinstance(m, dict) and isinstance(m.get("date"), str)]
    return {"series": len(lower), "newest": max(dates) if dates else None,
            "warning_count": len(warn), "warnings": warn[:20]}


def gate_duplicate_ids(new):
    seen, dups = {}, []
    for m in new:
        mid = m.get("match_id")
        if mid in seen:
            dups.append(mid)
        seen[mid] = True
    return sorted(set(dups))


def gate_sane_counts(old, new, allow_backfill=False):
    if old is None:
        return {"note": "no baseline; counts unchecked"}
    old_ids = {m["match_id"] for m in old}
    new_ids = {m["match_id"] for m in new}
    added = len(new_ids - old_ids)
    removed = len(old_ids - new_ids)
    problems = []
    if added > SPIKE_CAP and not allow_backfill:
        problems.append(f"{added} new series in one update (cap {SPIKE_CAP}; "
                        "--allow-backfill lifts it for an intentional backfill)")
    shrink_allow = max(SHRINK_MIN, int(len(old) * SHRINK_FRAC))
    if removed > shrink_allow:
        problems.append(f"{removed} series vanished (allow {shrink_allow})")
    return {"added": added, "removed": removed, "problems": problems}


def _result_key(m):
    """What a published series' outcome is: winner, best-of and the map winners."""
    return (m.get("winner"), m.get("best_of"),
            tuple((x.get("map"), x.get("winner")) for x in m.get("maps") or []))


def changed_ids(old, new):
    """match_ids present in both whose outcome differs."""
    old_by = {m["match_id"]: _result_key(m) for m in old}
    return sorted(m["match_id"] for m in new
                  if m["match_id"] in old_by and _result_key(m) != old_by[m["match_id"]])


def gate_changed_results(old, new):
    if old is None:
        return {"note": "no baseline; changed results unchecked"}
    ids = changed_ids(old, new)
    allow = max(CHANGED_MAX, int(len(old) * CHANGED_FRAC))
    by_id = {m["match_id"]: m for m in new}
    sample = [{"match_id": i, "teams": [by_id[i]["team_a"], by_id[i]["team_b"]],
               "date": by_id[i]["date"]} for i in ids[:20]]
    return {"changed": len(ids), "allow": allow, "sample": sample,
            "problems": ([f"{len(ids)} published series changed their result (allow {allow})"]
                         if len(ids) > allow else [])}


def gate_team_names(old, new):
    if old is None:
        return {"new_teams": sorted({m["team_a"] for m in new} | {m["team_b"] for m in new})}
    old_teams = {m["team_a"] for m in old} | {m["team_b"] for m in old}
    new_teams = {m["team_a"] for m in new} | {m["team_b"] for m in new}
    return {"new_teams": sorted(new_teams - old_teams)}


def elo_table(matches):
    import backtest as bt
    h = bt.History()
    for m in sorted(matches, key=lambda x: x["date"]):
        h.add(m)
    return {t: h.elo[t] for t in h.games}


def gate_elo_bounds(old, new, bound: float = ELO_BOUND, allow_backfill=False):
    if old is None:
        return {"note": "no baseline; Elo unchecked"}
    try:
        eo, en = elo_table(old), elo_table(new)
    except Exception as e:  # noqa: BLE001 - a crash here is itself a gate failure signal
        return {"error": f"Elo recompute crashed: {e}"}
    old_ids = {m["match_id"] for m in old}
    touched = changed_ids(old, new)
    new_series = [m for m in new if m["match_id"] not in old_ids or m["match_id"] in set(touched)]
    n_for = {}
    for m in new_series:
        for t in (m["team_a"], m["team_b"]):
            n_for[t] = n_for.get(t, 0) + 1
    last_old = max((m["date"] for m in old), default="")
    append_only = (not allow_backfill and not touched and
                   all(m["date"] >= last_old for m in new if m["match_id"] not in old_ids))
    viol = []
    for t, v in en.items():
        d = abs(v - eo.get(t, v))
        if t not in eo:
            continue   # a brand-new team starts at the default; nothing to compare
        allow = max(bound, ELO_PER_SERIES * n_for.get(t, 0))
        if d > allow:
            viol.append({"team": t, "old": round(eo[t], 1), "new": round(v, 1),
                         "delta": round(v - eo[t], 1), "allow": allow,
                         "reason": "moved more than the series it played justify"})
        elif append_only and n_for.get(t, 0) == 0 and d >= ELO_IDLE:
            viol.append({"team": t, "old": round(eo[t], 1), "new": round(v, 1),
                         "delta": round(v - eo[t], 1), "allow": ELO_IDLE,
                         "reason": "no new series, yet its rating moved (append-only update)"})
    return {"append_only": append_only, "violations": sorted(viol, key=lambda x: -abs(x["delta"]))}


def gate_cross_source(old, new, second=None):
    """Agreement with the second source (data/valve_results.json, Valve's
    regional-standings match rows). Fails only on what THIS update introduced:
    winner disagreements or identity mismatches (same day, same team and
    result, different opponent) present on the new data but not on the
    published data, both compared with the same second-source file. Older,
    already-published discrepancies are reported, never re-failed. A missing
    or unreadable second source passes with a note: it is a cross-check, and
    its outage must never block a refresh."""
    try:
        import valve_results as vr
        import cross_source_check as xs
        if second is None:
            second = vr.load()
    except Exception as e:  # noqa: BLE001 - the cross-check is optional input
        return {"note": f"second source unreadable ({type(e).__name__}: {e}); cross-check skipped"}
    if not second or not second.get("series"):
        return {"note": "no second source (data/valve_results.json); cross-check skipped"}

    def issues(matches):
        rep = xs.compare(matches, second)
        return ({d["match_id"]: d for d in rep.get("disagreements", [])},
                {d["match_id"]: d for d in rep.get("identity_mismatches", [])}, rep)
    dn, inew, rep = issues(new)
    out = {"matched": rep.get("matched"), "agreement_rate": rep.get("agreement_rate"),
           "disagree": len(dn), "identity_mismatch": len(inew),
           "second_source_snapshots": [min(second.get("snapshots") or [""]),
                                       max(second.get("snapshots") or [""])]}
    if old is None:
        out["note"] = "no baseline: discrepancies reported, not gated"
        return out
    do, iold, _ = issues(old)
    added_d = [dn[k] for k in sorted(dn) if k not in do]
    added_i = [inew[k] for k in sorted(inew) if k not in iold]
    out["new_disagreements"] = added_d[:20]
    out["new_identity_mismatches"] = added_i[:20]
    allow = vr.GATE_NEW_DISAGREE_MAX
    problems = []
    if len(added_d) > allow:
        problems.append(f"{len(added_d)} series now disagree with the second source on the winner "
                        f"(allow {allow})")
    if len(added_i) > allow:
        problems.append(f"{len(added_i)} series now name a different opponent than the second source "
                        f"(allow {allow})")
    out["problems"] = problems
    return out


def gate_vrs_freshness(today=None, vrs_path=None, status_path=None):
    """Valve standings freshness (offline: reads data/vrs.json and the
    data/vrs_status.json the refresh wrote). Fails when the stored standings are
    older than the newest Valve published at the last check, older than Valve's
    measured cadence allows (vrs.assess), or the last fetch failed while the
    stored copy is already past that limit."""
    from datetime import date as _date
    import vrs
    def rd(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    stored = (rd(vrs_path or os.path.join(ROOT, "data", "vrs.json")) or {}).get("date")
    status = rd(status_path or os.path.join(ROOT, "data", "vrs_status.json"))
    problems = vrs.assess(stored, status, today or _date.today())
    # a fetch failure alone is reported loudly by the workflow's `vrs.py --check`
    # step and issue; it blocks the publish only when the stored copy is stale
    blocking = [p for p in problems if not p.startswith("the last Valve standings refresh failed")]
    return {"stored": stored, "published": (status or {}).get("published"),
            "expected_next": (status or {}).get("expected_next"),
            "problems": blocking, "warnings": [p for p in problems if p not in blocking]}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    new_path = argv[argv.index("--new") + 1] if "--new" in argv else os.path.join(ROOT, "data", "matches.json")
    allow_backfill = "--allow-backfill" in argv or os.environ.get("FACEOFF_ALLOW_BACKFILL") == "1"
    bound = float(argv[argv.index("--elo-bound") + 1]) if "--elo-bound" in argv else ELO_BOUND

    old_raw, baseline_note = None, None
    try:
        if "--old" in argv:
            old_raw = load(argv[argv.index("--old") + 1])
        elif "--old-git" in argv:
            old_raw = load_old_git()
        else:
            baseline_note = "no --old / --old-git given: count, change and Elo gates skipped"
    except NoBaseline as e:
        if "--allow-no-baseline" not in argv:
            print(json.dumps({"passed": False, "error": f"no baseline: {e}",
                              "hint": "pass --allow-no-baseline only for a first-ever run"}, indent=1))
            return 2
        baseline_note = f"no baseline ({e}); allowed by --allow-no-baseline"
    except (OSError, ValueError) as e:
        print(json.dumps({"passed": False, "error": f"could not read the baseline: {e}"}, indent=1))
        return 2
    old = ensure_ids(old_raw) if old_raw is not None else None

    try:
        new = ensure_ids(load(new_path))
    except (OSError, ValueError) as e:
        print(json.dumps({"passed": False, "error": f"could not read {new_path}: {e}"}, indent=1))
        return 2

    gates = {}
    schema_bad = gate_schema(new)
    gates["schema"] = {"passed": not schema_bad, "issues": schema_bad[:20],
                       "issue_count": len(schema_bad)}
    dups = gate_duplicate_ids(new)
    gates["duplicate_ids"] = {"passed": not dups, "duplicate_ids": dups[:20]}
    counts = gate_sane_counts(old, new, allow_backfill)
    gates["sane_counts"] = {"passed": not counts.get("problems"), **counts}
    changed = gate_changed_results(old, new)
    gates["changed_results"] = {"passed": not changed.get("problems"), **changed}
    names = gate_team_names(old, new)
    gates["team_names"] = {"passed": True, **names}  # informational by design
    elo = gate_elo_bounds(old, new, bound, allow_backfill)
    gates["elo_bounds"] = {"passed": not elo.get("violations") and not elo.get("error"), **elo}
    xs = gate_cross_source(old, new)
    gates["cross_source"] = {"passed": not xs.get("problems"), **xs}

    vf = gate_vrs_freshness()
    gates["vrs_freshness"] = {"passed": not vf["problems"], **vf}
    for w in vf["warnings"]:
        print(f"WARNING vrs_freshness: {w}", file=sys.stderr)

    passed = all(g["passed"] for g in gates.values())
    # lower-tier data: reported, never blocking (added after `passed` on purpose)
    gates["lower_tier"] = {"passed": True, "blocking": False, **gate_lower()}
    if gates["lower_tier"].get("warnings"):
        print(f"WARNING lower_tier: {len(gates['lower_tier']['warnings'])} issue(s) in data/matches_lower.json "
              "(not blocking)", file=sys.stderr)
    report = {"passed": passed, "new_series": len(new),
              "old_series": len(old) if old is not None else None,
              "baseline": baseline_note or "compared with the last published matches.json",
              "gates": gates}
    print(json.dumps(report, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
