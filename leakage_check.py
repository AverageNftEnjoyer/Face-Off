#!/usr/bin/env python3
"""
leakage_check.py -- prove that backtest features are strictly pre-match.

CUTOFF RULE (the contract every feature must satisfy)
  A series dated D (event-local calendar day, the only precision the source
  has) may use:
    * outcomes (series winner, map winners, map names played) of series
      dated strictly before D -- never anything from day D itself, not even an
      earlier series on the same day (start times are not in the data);
    * facts published before D's matches start: the two team names, best-of,
      the tournament page, and its stand-in / lineup flags.
  It may NOT use any outcome dated >= D.

TESTS (run on a sample of match days, or every day with --all)
  1. SCRAMBLE: invert every outcome dated >= D (series winner and each map
     winner) and rename each of those maps to a different pool map. Every
     feature of every series dated D must be bit-identical to the clean build.
  2. TRUNCATE: delete every series dated > D. Same requirement.
  3. CANARY: a deliberately leaky builder (folds day D in BEFORE computing
     day D's features) must FAIL test 1. This proves the check has teeth.
Exit code 0 = no leakage found, 1 = leakage (offending fields are printed).

  4. LINEUPS (data/lineups.json via lineup_features.py): replace every lineup
     dated >= D with five made-up players. The stand-in and continuity
     features of every series dated D must not change. CANARY: the same
     features computed with day D's own lineups visible must change on at
     least one sampled day.

  Tests 1-2 also cover the plain 90-day map rates (input maps_raw_a/_b, used
  for the map_depth row) and the newcomer Elo offset (backtest.NEWCOMER_*),
  whose only inputs are the match date and the date of the first series in
  the data. Test 5 (below) checks the offset with it switched on explicitly.

  6. VETO (veto.py): the whole veto_dist (outcomes, marginals, masks,
     warnings) of every BO1/BO3 series dated D, as canonical JSON with repr
     floats, must be bit-identical under SCRAMBLE and TRUNCATE (tests 1-2
     already cover its inputs veto_ev_a/_b, and with habits on, the
     HabitAccumulator state). CANARIES: an exclusion set computed with day
     D's own plays must change at least one mask on at least one sampled
     day; with CONFIG["veto_use_habits"] on, an accumulator that counts day D
     before predicting day D must fail the scramble check.

NOT COVERED, AND WHY IT CANNOT BE:
  data/roster_events.json is a present-day scrape of Liquipedia stand-in
  tables keyed by TOURNAMENT, with no date per stand-in. There is nothing
  dated to scramble, so no point-in-time test can be written for it: a
  stand-in who joined mid-event is flagged for the whole event, including
  matches played before he joined. Treat it as unverified. data/lineups.json
  (dated per match, test 4) is its replacement for any feature that ships.

USAGE (from D:/Face-Off):  python leakage_check.py [--days 40] [--all]
Stdlib only, deterministic.
"""
import argparse
import json
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backtest as bt  # noqa: E402


def features_by_id(rows):
    return {r["match"]["id"]: (r["input"], r["meta"]) for r in rows}


def scramble_from(matches, day):
    """Copy of `matches` with every outcome dated >= day inverted and every
    such map renamed to the next map name in sorted order."""
    names = sorted({x["map"] for m in matches for x in m.get("maps", [])})
    nxt = {mp: names[(i + 1) % len(names)] for i, mp in enumerate(names)}
    out = copy.deepcopy(matches)
    for m in out:
        if m["date"] < day:
            continue
        a, b = m["team_a"], m["team_b"]
        flip = {a: b, b: a}
        m["winner"] = flip.get(m["winner"], m["winner"])
        for x in m.get("maps", []):
            x["winner"] = flip.get(x["winner"], x["winner"])
            x["map"] = nxt[x["map"]]
    return out


def leaky_build(matches, history_factory=None):
    """CANARY ONLY: folds each day into history BEFORE computing that day's
    features, i.e. the bug the check must catch."""
    ms = sorted(matches, key=lambda m: (m["date"], m.get("id", 0)))
    h = history_factory() if history_factory else bt.History()
    out = []
    i = 0
    while i < len(ms):
        j = i
        while j < len(ms) and ms[j]["date"] == ms[i]["date"]:
            j += 1
        for m in ms[i:j]:
            h.add(m)
        for m in ms[i:j]:
            inp, meta = h.features(m)
            out.append({"match": m, "input": inp, "meta": meta})
        i = j
    return out


def diff_fields(x, y, prefix=""):
    """List of dotted paths where two nested dict/list/scalar values differ."""
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in sorted(set(x) | set(y), key=str):
            if k not in x or k not in y:
                out.append(f"{prefix}{k} (present in one build only)")
            else:
                out += diff_fields(x[k], y[k], f"{prefix}{k}.")
        return out
    return [] if x == y else [prefix.rstrip(".")]


def check_day(clean, variant_rows, day):
    """Compare features of every series dated `day`; return list of problems."""
    var = features_by_id(variant_rows)
    problems = []
    for mid, (inp, meta) in clean.items():
        if mid[0] != day:
            continue
        got = var.get(mid[1])
        if got is None:
            problems.append(f"{day} match {mid[1]}: missing from variant build")
            continue
        for f in diff_fields({"input": inp, "meta": meta}, {"input": got[0], "meta": got[1]}):
            problems.append(f"{day} match {mid[1]}: {f}")
    return problems


def check_lineups(matches, test_days):
    """Test 4: lineups dated >= D must not reach day-D lineup features."""
    import lineup_features as LF
    lineups = LF.load()
    if not lineups:
        return [], 0, 0
    clean_hist = LF.team_history(matches, lineups)
    problems, caught, tested = [], 0, 0
    for day in test_days:
        todays = [m for m in matches if m["date"] == day]
        scrambled = {k: {s: ([f"fake{i}_{s}_{k}" for i in range(5)] if v.get(s) else None) for s in ("a", "b")}
                     if k.split("|", 1)[0] >= day else v for k, v in lineups.items()}
        hist = LF.team_history(matches, scrambled)
        nxt = (bt._d(day) + bt.timedelta(days=1)).isoformat()
        any_known = False
        hit = False
        for m in todays:
            for team in (m["team_a"], m["team_b"]):
                f_clean = LF.features(clean_hist.get(team, []), day)
                any_known |= f_clean["known"]
                if f_clean != LF.features(hist.get(team, []), day):
                    problems.append(f"{day} {team}: lineup features changed when lineups >= D were scrambled")
                # canary: let day D's own (scrambled) lineups through
                if LF.features(clean_hist.get(team, []), day) != LF.features(hist.get(team, []), nxt):
                    hit = True
        tested += any_known
        caught += hit and any_known
    return problems, caught, tested


def _veto_dumps(rows, day, build=None):
    """Canonical JSON (repr floats) of veto_dist for every series dated `day`,
    keyed by match id. BO1 and BO3 series; BO5 stays on the point veto."""
    import predictor as pr
    out = {}
    for r in rows:
        m = r["match"]
        if m["date"] != day or m.get("best_of") not in (1, 3):
            continue
        try:
            dist, _, _ = pr.veto_distribution(r["input"], m["best_of"])
        except ValueError as e:
            dist = {"error": str(e)}
        out[m["id"]] = json.dumps(dist, sort_keys=True, default=repr)
    return out


def check_veto(matches, test_days, history_factory=None, n_habit_days=4):
    """Test 6. Returns (problems, days where the exclusion canary changed a
    mask, habit canary status: "off" | "caught" | "BLIND")."""
    import predictor as pr
    import veto as V
    problems = []
    if history_factory is not None:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
        import veto_harness as H
        build = lambda ms: H.build_dataset(ms, history_factory)   # noqa: E731
        days = test_days[:: max(1, len(test_days) // n_habit_days)][:n_habit_days]
    else:
        build = bt.build_dataset
        days = test_days
    clean_rows = build(matches)
    for day in days:
        clean = _veto_dumps(clean_rows, day)
        for name, variant in (("scramble", scramble_from(matches, day)),
                              ("truncate", [m for m in matches if m["date"] <= day])):
            got = _veto_dumps(build(variant), day)
            for mid, s in clean.items():
                if got.get(mid) != s:
                    problems.append(f"{day} match {mid}: veto_dist differs under {name}")
    # exclusion canary: evidence that sees day D's own plays
    caught = 0
    h = bt.History()
    ms = sorted(matches, key=lambda m: (m["date"], m["id"]))
    i = 0
    alpha = pr.CONFIG["veto_excl_alpha"]
    for day in test_days:
        while i < len(ms) and ms[i]["date"] <= day:
            h.add(ms[i])
            i += 1
        dt = bt._d(day)
        leak_dt = dt + bt.timedelta(days=1)
        pool = h.map_pool(dt)
        hit = False
        for m in ms:
            if m["date"] != day:
                continue
            for t in (m["team_a"], m["team_b"]):
                ok = V.exclusion_set(h.veto_evidence(t, dt, pool), pool, alpha)
                bad = V.exclusion_set(h.veto_evidence(t, leak_dt, pool), pool, alpha)
                hit |= ok != bad
        caught += hit
    # habit canary (phase 3): only when habits are switched on
    status = "off"
    if pr.CONFIG.get("veto_use_habits") and history_factory is not None:
        status = "BLIND"
        for day in test_days[:: max(1, len(test_days) // n_habit_days)][:n_habit_days]:
            # the bug the check must catch: day D counted (<= D) and folded in first
            a = _veto_dumps(leaky_build(matches, lambda: history_factory(leaky=True)), day)
            b = _veto_dumps(leaky_build(scramble_from(matches, day), lambda: history_factory(leaky=True)), day)
            if a != b:
                status = "caught"
                break
    return problems, caught, status


def sample_days(days, k):
    if k >= len(days):
        return days
    step = (len(days) - 1) / (k - 1)
    return sorted({days[round(i * step)] for i in range(k)})


def main(argv=None):
    ap = argparse.ArgumentParser(description="strict pre-match feature check")
    ap.add_argument("--days", type=int, default=40, help="match days to test (evenly spaced)")
    ap.add_argument("--all", action="store_true", help="test every match day (slow)")
    ap.add_argument("--roster", action="store_true", help="with --habits: phase-4 roster-core level on too")
    ap.add_argument("--habits", action="store_true",
                    help="also run test 6 with the phase-3 HabitAccumulator on (slow; 4 days)")
    a = ap.parse_args(argv)

    matches = bt.load_matches()
    rows = bt.build_dataset(matches)
    clean = {(r["match"]["date"], r["match"]["id"]): (r["input"], r["meta"]) for r in rows}
    days = sorted({m["date"] for m in matches})
    # skip the first 30 days: teams have no history yet, features are trivially empty
    usable = days[30:]
    test_days = usable if a.all else sample_days(usable, a.days)
    print(f"{len(matches)} series, {len(days)} match days; testing {len(test_days)} days "
          f"({test_days[0]} .. {test_days[-1]})")

    failures = []
    for day in test_days:
        failures += check_day(clean, bt.build_dataset(scramble_from(matches, day)), day)
        failures += check_day(clean, bt.build_dataset([m for m in matches if m["date"] <= day]), day)

    # canary: the leaky builder must be caught on at least one of the sampled days
    caught = 0
    leaky_clean = {(r["match"]["date"], r["match"]["id"]): (r["input"], r["meta"])
                   for r in leaky_build(matches)}
    for day in test_days[:10]:
        if check_day(leaky_clean, leaky_build(scramble_from(matches, day)), day):
            caught += 1
    canary_ok = caught > 0

    # 4. lineups
    lineup_fail, lineup_caught, lineup_days = check_lineups(matches, test_days)
    failures += lineup_fail
    canary_ok = canary_ok and (lineup_days == 0 or lineup_caught > 0)
    print(f"lineups: {lineup_days} days with lineup data tested, {len(lineup_fail)} leaks; "
          f"lineup canary caught on {lineup_caught} days")

    # 5. newcomer Elo offset, both modes switched on explicitly (scramble + truncate)
    saved = (bt.NEWCOMER_OFFSET, bt.NEWCOMER_MODE, bt.NEWCOMER_FEATURE_OFFSET)
    nc_fail, nc_moved = [], True
    keyed = lambda rows: {(r["match"]["date"], r["match"]["id"]): (r["input"], r["meta"]) for r in rows}
    try:
        bt.NEWCOMER_OFFSET, bt.NEWCOMER_MODE, bt.NEWCOMER_FEATURE_OFFSET = 0.0, "start", 0.0
        off = keyed(bt.build_dataset(matches))
        for setting in ((150.0, "start", 0.0), (0.0, "feature", 150.0)):
            bt.NEWCOMER_OFFSET, bt.NEWCOMER_MODE, bt.NEWCOMER_FEATURE_OFFSET = setting
            on = keyed(bt.build_dataset(matches))
            # teeth: the setting must change at least one rating vs no offset
            nc_moved &= any(on[k][0]["rating_a"] != off[k][0]["rating_a"] for k in on)
            for day in test_days[::4]:
                nc_fail += check_day(on, bt.build_dataset(scramble_from(matches, day)), day)
                nc_fail += check_day(on, bt.build_dataset([m for m in matches if m["date"] <= day]), day)
    finally:
        bt.NEWCOMER_OFFSET, bt.NEWCOMER_MODE, bt.NEWCOMER_FEATURE_OFFSET = saved
    failures += nc_fail
    print(f"newcomer offset (start 150 / feature 150): {len(nc_fail)} leaks on {len(test_days[::4])} days; "
          f"offset changes features: {'yes' if nc_moved else 'NO (check is blind)'}")
    canary_ok = canary_ok and nc_moved

    # 6. veto distribution (veto.py): the whole veto_dist of every day-D series
    #    must be bit-identical under SCRAMBLE and TRUNCATE; canaries: an
    #    exclusion set computed with day D's own plays must change a mask, and
    #    (with habits on) an accumulator that folds day D in first must fail.
    v_fail, excl_caught, habit_status = check_veto(matches, test_days)
    if a.habits:
        import predictor as pr
        sys.path.insert(0, os.path.join(HERE, "scripts"))
        import veto_fit as F
        import veto_harness as H
        rep = H.read_report()
        saved = dict(pr.CONFIG)
        try:
            ph = "p4" if a.roster else "p3"
            pr.CONFIG.update(veto_use_habits=True, veto_habit_k=rep["phases"]["p3"]["grid_value"],
                             veto_params_bo3=rep["phases"][ph]["params"])
            if a.roster:
                pr.CONFIG.update(veto_use_roster=True, veto_roster_k=rep["phases"]["p4"]["grid_value"])
            hv_fail, _, habit_status = check_veto(matches, test_days,
                                                  F.history_factory(rep["phases"]["p2"]["params"],
                                                                    roster=a.roster))
        finally:
            pr.CONFIG.clear()
            pr.CONFIG.update(saved)
        print(f"veto_dist with habits: {len(hv_fail)} differing distributions")
        v_fail += hv_fail
    failures += v_fail
    canary_ok = canary_ok and excl_caught > 0 and habit_status in ("off", "caught")
    print(f"veto_dist: {len(v_fail)} differing distributions on {len(test_days)} days; exclusion canary "
          f"changed a mask on {excl_caught} days; habit canary: {habit_status}")

    print(f"scramble + truncate: {len(failures)} differing fields")
    for f in failures[:30]:
        print("  LEAK", f)
    print(f"canary (deliberately leaky builder) caught on {caught}/{min(10, len(test_days))} days: "
          f"{'OK' if canary_ok else 'CHECK IS BLIND'}")
    ok = not failures and canary_ok
    print("RESULT:", "PASS - no outcome dated >= D reaches a day-D feature" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
