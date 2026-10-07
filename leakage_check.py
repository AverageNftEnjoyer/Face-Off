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


def leaky_build(matches):
    """CANARY ONLY: folds each day into history BEFORE computing that day's
    features, i.e. the bug the check must catch."""
    ms = sorted(matches, key=lambda m: (m["date"], m.get("id", 0)))
    h = bt.History()
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


def sample_days(days, k):
    if k >= len(days):
        return days
    step = (len(days) - 1) / (k - 1)
    return sorted({days[round(i * step)] for i in range(k)})


def main(argv=None):
    ap = argparse.ArgumentParser(description="strict pre-match feature check")
    ap.add_argument("--days", type=int, default=40, help="match days to test (evenly spaced)")
    ap.add_argument("--all", action="store_true", help="test every match day (slow)")
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
