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
  the data. Test 5 checks the offset with it switched on explicitly.

  6. VETO (veto.py): the whole veto_dist (outcomes, marginals, masks,
     warnings) of every BO1/BO3 series dated D, as canonical JSON with repr
     floats, must be bit-identical under SCRAMBLE and TRUNCATE (tests 1-2
     already cover its inputs veto_ev_a/_b). CANARY: an exclusion set
     computed with day D's own plays must change at least one mask on at
     least one sampled day.

  7. LOWER TIER (data/matches_lower.json, folded into series Elo at
     backtest.LOWER_TIER_ELO_WEIGHT; when that is 0 the check switches the
     candidate weight LOWER_TIER_CANDIDATE_WEIGHT on explicitly, like test 5):
     SCRAMBLE inverts every lower-tier outcome dated >= D (series and map
     winners), TRUNCATE removes every lower-tier series dated > D; every
     feature of every S/A series dated D must be bit-identical. CANARIES:
     the lower-tier data must change at least one feature (the check sees
     it), and a deliberately leaky builder that folds day D's lower-tier
     series in before day D's features must fail the scramble.

  8. NOTES (notes.py over notes_data.section; display only): for each D,
     the notes of every series dated D (matchup + both team views) must be
     byte-identical when outcomes dated >= D are inverted, lineups dated >= D
     are made-up fives, squads gain spells joined after D, hand-entered
     overrides are added (history never reads them) and web facts observed
     at or after D's start are added. CANARIES: facts observed just before D
     must show on every day, and the notes computed with a one-day-late
     cutoff must differ on at least one day.

NOT COVERED, AND WHY IT CANNOT BE:
  data/roster_events.json is a present-day scrape of Liquipedia stand-in
  tables keyed by TOURNAMENT, with no date per stand-in. There is nothing
  dated to scramble, so no point-in-time test can be written for it: a
  stand-in who joined mid-event is flagged for the whole event, including
  matches played before he joined. Treat it as unverified. data/lineups.json
  (dated per match, test 4) is its replacement for any feature that ships.
  The display-only roster caution flags (roster_flags.py) use the stand-in
  list for a finished series only once a lineup dated before its day shows
  the stand-in (tests/test_roster_flags.py).

USAGE (from D:/Face-Off):  python evaluation/leakage_check.py [--days 40] [--all]
Stdlib only, deterministic.
"""
import argparse
import json
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

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


def scramble_lower_from(lower, day):
    """Copy of the lower-tier list with every outcome dated >= day inverted
    (series winner and each map winner)."""
    out = copy.deepcopy(lower)
    for m in out:
        if m["date"] < day:
            continue
        flip = {m["team_a"]: m["team_b"], m["team_b"]: m["team_a"]}
        m["winner"] = flip.get(m["winner"], m["winner"])
        for x in m.get("maps") or []:
            x["winner"] = flip.get(x["winner"], x["winner"])
    return out


def leaky_lower_build(matches, lower):
    """CANARY ONLY: folds day D's lower-tier series in BEFORE computing day D's
    features (the bug test 7 must catch)."""
    h = bt.History()
    out = []
    for _, day, lo in bt.day_groups(matches, lower):
        for m in lo:
            h.add_lower(m)
        for m in day:
            inp, meta = h.features(m)
            out.append({"match": m, "input": inp, "meta": meta})
        for m in day:
            h.add(m)
    return out


def check_lower(matches, lower, test_days, weight):
    """Test 7 at the given lower-tier weight. Returns (problems, days tested,
    lower data changes features?, leaky-builder canary caught on n days)."""
    saved = bt.LOWER_TIER_ELO_WEIGHT
    keyed = lambda rows: {(r["match"]["date"], r["match"]["id"]): (r["input"], r["meta"]) for r in rows}  # noqa: E731
    problems = []
    try:
        bt.LOWER_TIER_ELO_WEIGHT = weight
        clean = keyed(bt.build_dataset(matches, lower))
        without = keyed(bt.build_dataset(matches))
        sees = any(clean[k] != without[k] for k in clean)
        first = lower[0]["date"] if lower else "9999"
        days = [d for d in test_days if d >= first]
        for day in days:
            problems += check_day(clean, bt.build_dataset(matches, scramble_lower_from(lower, day)), day)
            problems += check_day(clean, bt.build_dataset([m for m in matches if m["date"] <= day],
                                                          [m for m in lower if m["date"] <= day]), day)
        # canary days: days where a team plays an S/A series AND a lower-tier
        # series, so a builder that lets day D's lower-tier results in early
        # must be caught by the scramble (picked from the data, evenly spaced)
        sa_teams = {}
        for m in matches:
            sa_teams.setdefault(m["date"], set()).update((m["team_a"], m["team_b"]))
        both = sorted({m["date"] for m in lower
                       if {m["team_a"], m["team_b"]} & sa_teams.get(m["date"], set())})
        leaky = keyed(leaky_lower_build(matches, lower))
        caught = sum(1 for day in sample_days(both, 6) if both
                     and check_day(leaky, leaky_lower_build(matches, scramble_lower_from(lower, day)), day))
    finally:
        bt.LOWER_TIER_ELO_WEIGHT = saved
    return problems, len(days), sees, caught


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


def check_veto(matches, test_days):
    """Test 6. Returns (problems, days where the exclusion canary changed a mask)."""
    import predictor as pr
    import veto as V
    problems = []
    lower = bt.load_lower()      # the site's history (empty while the weight is 0)
    build = lambda ms: bt.build_dataset(ms, [m for m in lower if m["date"] <= ms[-1]["date"]])  # noqa: E731
    clean_rows = build(matches)
    for day in test_days:
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
    return problems, caught


def _notes_day(section, matches, day, cutoff=None):
    """Canonical JSON of the notes (both team views and the matchup) of every
    series dated `day`, computed as of `cutoff` (default: the day itself)."""
    import notes as NT
    b = {"notes": section}
    out = []
    for m in sorted((m for m in matches if m["date"] == day), key=lambda m: (m["team_a"], m["team_b"])):
        a, t = m["team_a"], m["team_b"]
        asof = cutoff or day
        out.append([a, t, NT.get_notes(a, t, asof, b, match_day=day, event=m["event"]),
                    NT.team_notes(a, asof, b, day), NT.team_notes(t, asof, b, day)])
    return NT.dumps(out)


def _canary_facts(teams, day, after):
    """Web facts for `teams` observed just before `day` (after=False: they must
    show) or at / after its start (after=True: they must never show)."""
    import notes_facts as NF
    d = bt._d(day)
    ts = [NF.fmt_ts(NF.cutoff_of(d) + bt.timedelta(seconds=s)) for s in ((0, 3600, 86400 * 3) if after else (-60,))]
    out = []
    for t in sorted(teams):
        for i, obs in enumerate(ts):
            tag = f"{'late' if after else 'early'}{i}"
            base = dict(team=t, observed_at=obs, source_url=f"https://example.org/{tag}", source="CANARY",
                        method="scrape")
            out.append(NF.make_fact(player=f"canary_{tag}", rating_last30=1.9 if after else 1.3,
                                    rating_baseline=1.0, baseline="last12m", maps_last30=20, **base))
            out.append(NF.make_fact(type="stand_in", player=f"sub_{tag}", date=day, **base))
            out.append(NF.make_fact(type="roster_rumor", player=f"rumor_{tag}", date=day, **base))
    return out


def check_notes(matches, test_days):
    """Test 8: matchup notes (notes.py over notes_data.section). For each day D,
    every series dated D must get byte-identical notes when, in the variant
    inputs, every outcome dated >= D is inverted, every lineup dated >= D is
    five made-up players, squads gain spells joined after D, hand-entered roster
    overrides are added on and before D, and web facts observed at or after
    D's start are added. Canary facts observed before D must show (the check
    sees facts at all), and the same notes computed one day late (cutoff D+1)
    on the variant must differ on at least one day (the check has teeth).
    -> (problems, days tested, early facts seen on, leaky cutoff caught on)."""
    import notes_data as ND
    _, lineups, squads, _, _ = ND.load_inputs()
    as_of = max(bt._d(m["date"]) for m in matches) + bt.timedelta(days=1)
    empty = {"fact_schema": 1, "updated_at": None, "form": [], "news": []}
    problems, seen, caught = [], 0, 0
    for day in test_days:
        teams = {t for m in matches if m["date"] == day for t in (m["team_a"], m["team_b"])}
        early = _canary_facts(teams, day, after=False)
        clean_facts = dict(empty, form=[f for f in early if "rating_last30" in f],
                           news=[f for f in early if "type" in f])
        clean = ND.section(as_of, matches, [], [], {}, [], lineups, squads, [], clean_facts)
        late = _canary_facts(teams, day, after=True)
        var_facts = dict(clean_facts, form=clean_facts["form"] + [f for f in late if "rating_last30" in f],
                         news=clean_facts["news"] + [f for f in late if "type" in f])
        var_lineups = {k: ({s: ([f"fake{i}_{s}_{k}" for i in range(5)] if v.get(s) else None) for s in ("a", "b")}
                           if k.split("|", 1)[0] >= day else v) for k, v in lineups.items()}
        # squad spells joined after D for the players each team last fielded
        nxt = (bt._d(day) + bt.timedelta(days=1)).isoformat()
        var_squads = {t: list(rows) for t, rows in squads.items()}
        for t in sorted(teams):
            rows = sorted((k, v) for k, v in lineups.items() if k < day and t in k.split("|")[1:])
            for k, v in rows[-1:]:
                five = v.get("a") if k.split("|")[1] == t else v.get("b")
                var_squads.setdefault(t, []).extend([[p, nxt, ""] for p in five or []])
        overrides = [{"team": t, "kind": "roster_change", "from": f, "players": [f"ov{i}" for i in range(5)]}
                     for t in sorted(teams) for f in ((bt._d(day) - bt.timedelta(days=5)).isoformat(), day)]
        variant = ND.section(as_of, scramble_from(matches, day), [], [], {}, [], var_lineups, var_squads,
                             overrides, var_facts)
        got_clean = _notes_day(clean, matches, day)
        if got_clean != _notes_day(variant, matches, day):
            problems.append(f"{day}: notes changed when information dated >= D was altered")
        seen += "sub_early0" in got_clean
        caught += _notes_day(variant, matches, day, (bt._d(day) + bt.timedelta(days=1)).isoformat()) != got_clean
    return problems, len(test_days), seen, caught


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
    lower = bt.load_lower()          # [] while bt.LOWER_TIER_ELO_WEIGHT is 0
    rows = bt.build_dataset(matches, lower)
    clean = {(r["match"]["date"], r["match"]["id"]): (r["input"], r["meta"]) for r in rows}
    days = sorted({m["date"] for m in matches})
    # skip the first 30 days: teams have no history yet, features are trivially empty
    usable = days[30:]
    test_days = usable if a.all else sample_days(usable, a.days)
    print(f"{len(matches)} series, {len(days)} match days; testing {len(test_days)} days "
          f"({test_days[0]} .. {test_days[-1]})")

    failures = []
    for day in test_days:
        failures += check_day(clean, bt.build_dataset(scramble_from(matches, day), lower), day)
        failures += check_day(clean, bt.build_dataset([m for m in matches if m["date"] <= day],
                                                      [m for m in lower if m["date"] <= day]), day)

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

    # 5. newcomer Elo offset switched on explicitly (scramble + truncate)
    saved = bt.NEWCOMER_OFFSET
    nc_fail = []
    keyed = lambda rows: {(r["match"]["date"], r["match"]["id"]): (r["input"], r["meta"]) for r in rows}
    try:
        bt.NEWCOMER_OFFSET = 0.0
        off = keyed(bt.build_dataset(matches))
        bt.NEWCOMER_OFFSET = 150.0
        on = keyed(bt.build_dataset(matches))
        # teeth: the offset must change at least one rating vs no offset
        nc_moved = any(on[k][0]["rating_a"] != off[k][0]["rating_a"] for k in on)
        for day in test_days[::4]:
            nc_fail += check_day(on, bt.build_dataset(scramble_from(matches, day)), day)
            nc_fail += check_day(on, bt.build_dataset([m for m in matches if m["date"] <= day]), day)
    finally:
        bt.NEWCOMER_OFFSET = saved
    failures += nc_fail
    print(f"newcomer offset (150): {len(nc_fail)} leaks on {len(test_days[::4])} days; "
          f"offset changes features: {'yes' if nc_moved else 'NO (check is blind)'}")
    canary_ok = canary_ok and nc_moved

    # 6. veto distribution (veto.py): the whole veto_dist of every day-D series
    #    must be bit-identical under SCRAMBLE and TRUNCATE; canaries: an
    #    exclusion set computed with day D's own plays must change a mask.
    v_fail, excl_caught = check_veto(matches, test_days)
    failures += v_fail
    canary_ok = canary_ok and excl_caught > 0
    print(f"veto_dist: {len(v_fail)} differing distributions on {len(test_days)} days; exclusion canary "
          f"changed a mask on {excl_caught} days")

    # 7. lower-tier series (Elo data): scramble / truncate them from day D on
    w_low = bt.LOWER_TIER_ELO_WEIGHT or bt.LOWER_TIER_CANDIDATE_WEIGHT
    low_all = bt.read_lower()
    l_fail, l_days, l_sees, l_caught = check_lower(matches, low_all, test_days, w_low)
    failures += l_fail
    canary_ok = canary_ok and l_sees and l_caught > 0
    print(f"lower tier (weight {w_low}{'' if bt.LOWER_TIER_ELO_WEIGHT else ', switched on for the check'}, "
          f"{len(low_all)} series): {len(l_fail)} leaks on {l_days} days; lower-tier data changes features: "
          f"{'yes' if l_sees else 'NO (check is blind)'}; leaky-builder canary caught on {l_caught} days")

    # 8. matchup notes (display only): results, lineups, squads, overrides and
    #    web facts dated >= D never reach a day-D note
    n_fail, n_days, n_seen, n_caught = check_notes(matches, test_days)
    failures += n_fail
    canary_ok = canary_ok and n_seen == n_days and n_caught > 0
    print(f"notes: {len(n_fail)} leaks on {n_days} days; facts observed before D shown on {n_seen}/{n_days} "
          f"days; one-day-late cutoff caught on {n_caught} days")

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
