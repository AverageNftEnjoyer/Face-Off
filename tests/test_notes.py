"""notes.py (rule engine + tool interface) and notes_facts.py on SYNTHETIC
fixtures (made-up teams "Alpha", "Bravo", ...; never written to data/), plus
a determinism check on the real bundle when one has been built."""
import json
import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import notes as NT  # noqa: E402
import notes_data as ND  # noqa: E402
import notes_facts as NF  # noqa: E402

AS_OF = date(2026, 6, 1)
EMPTY = {"fact_schema": 1, "updated_at": None, "form": [], "news": []}


def series(day, a, b, winner, ev="Cup", score="2-0"):
    maps = [{"map": "Mirage", "winner": winner}, {"map": "Nuke", "winner": winner}]
    if score == "2-1":
        loser = b if winner == a else a
        maps = [maps[0], {"map": "Nuke", "winner": loser}, {"map": "Inferno", "winner": winner}]
    return {"date": day, "event": ev, "team_a": a, "team_b": b, "winner": winner, "maps": maps}


def iso(d):
    return d.isoformat()


def world():
    """Alpha: busy (9 series in 14 days, won 6 straight). Bravo: idle 30 days.
    Charlie beat Alpha 2-1 40 days before AS_OF."""
    ms = []
    d0 = AS_OF - timedelta(days=13)
    opps = ["Delta", "Echo", "Foxtrot"]
    for i in range(9):
        w = "Alpha" if i >= 3 else opps[i % 3]
        ms.append(series(iso(d0 + timedelta(days=i)), "Alpha", opps[i % 3], w, ev="Major"))
    ms.append(series(iso(AS_OF - timedelta(days=40)), "Charlie", "Alpha", "Charlie", score="2-1"))
    ms.append(series(iso(AS_OF - timedelta(days=30)), "Bravo", "Delta", "Bravo"))
    ms.append(series(iso(AS_OF - timedelta(days=200)), "Bravo", "Charlie", "Bravo"))
    return sorted(ms, key=lambda m: m["date"])


def lineups_world():
    """Charlie fields one five for 300 days (stable); Echo changed two players 30 days ago."""
    lu = {}
    old = ["c1", "c2", "c3", "c4", "c5"]
    for k in range(0, 300, 10):
        d = AS_OF - timedelta(days=300 - k)
        lu[f"{iso(d)}|Charlie|Zulu{k}"] = {"a": old, "b": None}
    # an unrelated early row pushes the data start back so Charlie's run is not censored
    lu["2024-01-01|Yankee|Xray"] = {"a": ["y1", "y2", "y3", "y4", "y5"], "b": None}
    e_old, e_new = ["e1", "e2", "e3", "e4", "e5"], ["e1", "e2", "e3", "n1", "n2"]
    for k in range(0, 200, 10):
        d = AS_OF - timedelta(days=200 - k)
        lu[f"{iso(d)}|Echo|Zulu{k}"] = {"a": e_old if k < 170 else e_new, "b": None}
    for k in range(170, 200, 5):
        d = AS_OF - timedelta(days=200 - k) + timedelta(days=2)
        lu[f"{iso(d)}|Echo|Whiskey{k}"] = {"a": e_new, "b": None}
    return lu


def build(facts=None, matches=None, overrides=None, fixtures=None):
    ms = matches if matches is not None else world()
    # Charlie's squad history resolves its five's start (the run begins with its first row in the data)
    squads = {"Charlie": [[p, "2025-08-01", ""] for p in ("c1", "c2", "c3", "c4", "c5")]}
    sec = ND.section(AS_OF, ms, [], ["Kilo"], {}, fixtures or [], lineups_world(), squads, overrides or [],
                     facts or EMPTY)
    return {"notes": sec}


def form_fact(team, player, r30, base, maps, obs):
    return NF.make_fact(team=team, player=player, observed_at=obs, source_url=f"https://www.hltv.org/stats/players/1/{player}",
                        source="HLTV", method="scrape", rating_last30=r30, rating_baseline=base,
                        baseline="last12m", maps_last30=maps)


def rules(out):
    return [n["rule"] for n in out["notes"]]


class RuleTest(unittest.TestCase):
    def test_schedule_rules(self):
        out = NT.team_notes("Alpha", iso(AS_OF), build())
        self.assertTrue(out["ok"])
        r = rules(out)
        self.assertNotIn("SHORT_REST", r)          # last series 5 days before
        self.assertIn("HEAVY_SCHEDULE", r)
        self.assertIn("STREAK", r)
        n = {x["rule"]: x for x in out["notes"]}
        self.assertEqual(n["HEAVY_SCHEDULE"]["values"]["n14"], 9)
        self.assertEqual(n["STREAK"]["values"], {"result": "W", "n": 6, "since": iso(AS_OF - timedelta(days=10)),
                                                 "last_date": iso(AS_OF - timedelta(days=5))})

    def test_short_rest_threshold_and_same_event(self):
        b = build()
        # last Alpha series is AS_OF - 5: on AS_OF it is 5 days (> 4), on AS_OF - 1 it is 4
        self.assertNotIn("SHORT_REST", rules(NT.team_notes("Alpha", iso(AS_OF), b, iso(AS_OF + timedelta(days=0)))))
        self.assertIn("SHORT_REST", rules(NT.team_notes("Alpha", iso(AS_OF), b, iso(AS_OF - timedelta(days=1)))))
        same = NT.get_notes("Alpha", "Bravo", iso(AS_OF), b, match_day=iso(AS_OF - timedelta(days=1)), event="Major")
        self.assertNotIn("SHORT_REST", rules(same))

    def test_past_match_day_clamps_the_cutoff(self):
        # an unplayed fixture dated before the cutoff must not read later results
        b = build()
        day = iso(AS_OF - timedelta(days=12))
        late = NT.get_notes("Alpha", "Bravo", "latest", b, match_day=day)
        self.assertEqual(NT.dumps(late["notes"]), NT.dumps(NT.get_notes("Alpha", "Bravo", day, b, match_day=day)["notes"]))
        for n in late["notes"]:
            self.assertGreaterEqual(n["values"].get("days", 0), 0)

    def test_same_day_match_keeps_latest_facts(self):
        obs = "2026-06-01T06:00:00Z"
        f = dict(EMPTY, updated_at=obs, form=[form_fact("Alpha", "hot", 1.30, 1.10, 12, obs)])
        out = NT.get_notes("Alpha", "Bravo", "latest", build(f), match_day=iso(AS_OF))
        self.assertIn("HOT_PLAYER", rules(out))

    def test_cold_rest(self):
        n = [x for x in NT.team_notes("Bravo", iso(AS_OF), build())["notes"] if x["rule"] == "COLD_REST"]
        self.assertEqual(len(n), 1)
        self.assertEqual(n[0]["values"]["days"], 30)
        self.assertEqual(n[0]["age_days"], 30)

    def test_no_series_no_schedule_note(self):
        self.assertEqual([r for r in rules(NT.team_notes("Kilo", iso(AS_OF), build()))
                          if r in ("SHORT_REST", "COLD_REST", "STREAK")], [])

    def test_revenge(self):
        out = NT.get_notes("Alpha", "Charlie", iso(AS_OF), build())
        rv = [n for n in out["notes"] if n["rule"] == "REVENGE"]
        self.assertEqual(len(rv), 1)
        self.assertEqual(rv[0]["team_scope"], "Alpha")
        self.assertEqual(rv[0]["scope"], "matchup")
        self.assertEqual(rv[0]["values"]["score"], "1-2")
        self.assertEqual(rv[0]["values"]["days"], 40)
        # 200 days ago is past revenge_max_days
        self.assertNotIn("REVENGE", rules(NT.get_notes("Bravo", "Charlie", iso(AS_OF), build())))

    def test_tenure_rules(self):
        b = build()
        c = NT.team_notes("Charlie", iso(AS_OF), b)
        self.assertIn("TENURE_STABLE", rules(c))
        e = NT.team_notes("Echo", iso(AS_OF), b)
        self.assertIn("TENURE_NEW", rules(e))
        self.assertNotIn("NEW_ADDITION", rules(e))       # covered by TENURE_NEW, not repeated
        self.assertNotIn("TENURE_STABLE", rules(e))

    def test_form_rules_and_expiry(self):
        obs = "2026-05-28T06:00:00Z"
        f = dict(EMPTY, updated_at=obs, form=[form_fact("Alpha", "hot", 1.30, 1.10, 12, obs),
                                              form_fact("Alpha", "cold", 0.90, 1.05, 9, obs),
                                              form_fact("Alpha", "few", 1.50, 1.00, 3, obs),
                                              form_fact("Alpha", "flat", 1.10, 1.05, 20, obs)])
        b = build(f)
        r = rules(NT.team_notes("Alpha", iso(AS_OF), b))
        self.assertEqual(r.count("HOT_PLAYER"), 1)
        self.assertEqual(r.count("COLD_PLAYER"), 1)
        hot = [n for n in NT.team_notes("Alpha", iso(AS_OF), b)["notes"] if n["rule"] == "HOT_PLAYER"][0]
        self.assertEqual(hot["source_url"], "https://www.hltv.org/stats/players/1/hot")
        self.assertEqual(hot["values"]["delta"], 0.2)
        self.assertEqual(hot["age_days"], 4)
        # 14 days after observation: still shown; 15: expired
        self.assertIn("HOT_PLAYER", rules(NT.team_notes("Alpha", "2026-06-11T06:00:00Z", b)))
        self.assertNotIn("HOT_PLAYER", rules(NT.team_notes("Alpha", "2026-06-11T06:00:01Z", b)))
        # observed after the cutoff: invisible
        self.assertNotIn("HOT_PLAYER", rules(NT.team_notes("Alpha", "2026-05-28", b)))

    def test_news_rules(self):
        obs = "2026-05-30T06:00:00Z"
        base = dict(observed_at=obs, source="HLTV", method="web_search")
        news = [NF.make_fact(type="stand_in", team="Bravo", player="sub", replaced="reg", date="2026-06-02",
                             source_url="https://www.hltv.org/news/1/x", **base),
                NF.make_fact(type="roster_rumor", team="Bravo", player="star", date="2026-05-20", move="out",
                             other_team="Charlie", source_url="https://www.hltv.org/news/2/y", **base)]
        b = build(dict(EMPTY, updated_at=obs, news=news))
        out = NT.team_notes("Bravo", iso(AS_OF), b)
        n = {x["rule"]: x for x in out["notes"]}
        self.assertEqual(n["STAND_IN"]["headline"], "Stand-in: sub in for reg")
        self.assertEqual(n["ROSTER_RUMOR"]["kind"], "rumor")
        self.assertTrue(n["ROSTER_RUMOR"]["headline"].startswith("Rumor: "))
        self.assertEqual(out["notes"][0]["rule"], "STAND_IN")        # rank 1
        # stand-in over (default 21 days after its date) and rumor past 90 days
        late = NT.team_notes("Bravo", "2026-08-25", b)
        self.assertNotIn("STAND_IN", rules(late))
        self.assertNotIn("ROSTER_RUMOR", rules(late))

    def test_overrides_only_for_the_present(self):
        ov = [{"team": "Charlie", "kind": "roster_change", "from": iso(AS_OF - timedelta(days=3)),
               "players": ["c1", "c2", "c3", "x1", "x2"]}]
        b = build(overrides=ov)
        self.assertIn("TENURE_NEW", rules(NT.team_notes("Charlie", iso(AS_OF), b)))
        self.assertNotIn("TENURE_NEW", rules(NT.team_notes("Charlie", iso(AS_OF - timedelta(days=1)), b)))


class AssemblyTest(unittest.TestCase):
    def test_cap_rank_order_and_unique(self):
        obs = "2026-05-28T06:00:00Z"
        form = [form_fact(t, f"p{t}{i}", 1.4 + i / 100, 1.0, 10, obs) for t in ("Alpha", "Charlie") for i in range(4)]
        out = NT.get_notes("Alpha", "Charlie", iso(AS_OF), build(dict(EMPTY, updated_at=obs, form=form)))
        self.assertLessEqual(len(out["notes"]), NT.CONFIG["max_matchup_notes"])
        self.assertGreater(out["omitted"], 0)
        ranks = [n["rank"] for n in out["notes"]]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(len({n["id"] for n in out["notes"]}), len(out["notes"]))
        per = {}
        for n in out["notes"]:
            per[(n["rule"], n["team_scope"])] = per.get((n["rule"], n["team_scope"]), 0) + 1
        self.assertLessEqual(max(per.values()), NT.CONFIG["max_per_rule_team"])
        # the stronger delta of a team's hot players comes first
        hot = [n for n in out["notes"] if n["rule"] == "HOT_PLAYER" and n["team_scope"] == "Alpha"]
        self.assertEqual([n["values"]["player"] for n in hot], ["pAlpha3", "pAlpha2"][:len(hot)])

    def test_note_schema(self):
        out = NT.get_notes("Alpha", "Charlie", iso(AS_OF), build())
        keys = {"id", "rule", "scope", "team_scope", "kind", "headline", "evidence", "values", "source",
                "source_url", "observed_at", "age_days", "rank"}
        for n in out["notes"]:
            self.assertEqual(set(n), keys)
            self.assertIn(n["rule"], NT.RULES)
            self.assertIn(n["scope"], ("team", "matchup"))
            self.assertGreaterEqual(n["age_days"], 0)
        self.assertFalse(out["counted_in_win_probability"])
        self.assertEqual(out["tool_version"], NT.TOOL_VERSION)


class ErrorContractTest(unittest.TestCase):
    def test_unknown_team_lists_close_matches(self):
        out = NT.get_notes("Alpah", "Bravo", iso(AS_OF), build())
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"]["code"], "UNKNOWN_TEAM")
        self.assertEqual(out["error"]["team"], "Alpah")
        self.assertIn("Alpha", out["error"]["close_matches"])
        self.assertNotIn("notes", out)

    def test_case_insensitive_name(self):
        self.assertEqual(NT.get_notes("alpha", "BRAVO", iso(AS_OF), build())["team_a"], "Alpha")

    def test_missing_fields(self):
        b = build()
        del b["notes"]["inputs"]["lineups"]
        out = NT.get_notes("Alpha", "Bravo", iso(AS_OF), b)
        self.assertEqual(out["error"]["code"], "MISSING_FIELDS")
        self.assertEqual(out["error"]["fields"], ["notes.inputs.lineups"])
        self.assertEqual(NT.get_notes("Alpha", "Bravo", iso(AS_OF), {})["error"]["code"], "MISSING_FIELDS")

    def test_bad_date_same_team_bad_schema(self):
        b = build()
        self.assertEqual(NT.get_notes("Alpha", "Bravo", "June 1", b)["error"]["code"], "BAD_DATE")
        self.assertEqual(NT.get_notes("Alpha", "Bravo", iso(AS_OF), b, match_day="x")["error"]["code"], "BAD_DATE")
        self.assertEqual(NT.get_notes("Alpha", "alpha", iso(AS_OF), b)["error"]["code"], "SAME_TEAM")
        b["notes"]["schema"] = {"note": 99, "fact": 1, "inputs": 1}
        self.assertEqual(NT.get_notes("Alpha", "Bravo", iso(AS_OF), b)["error"]["code"], "BAD_SCHEMA")

    def test_batch_isolates_errors(self):
        out = NT.get_notes_batch([{"team_a": "Alpha", "team_b": "Bravo"}, {"team_a": "Nope", "team_b": "Bravo"},
                                  {"team_a": "Alpha", "team_b": "Charlie", "match_day": iso(AS_OF)}],
                                 iso(AS_OF), build())
        self.assertTrue(out["ok"])
        self.assertEqual([r["ok"] for r in out["results"]], [True, False, True])

    def test_latest(self):
        out = NT.get_notes("Alpha", "Bravo", "latest", build())
        self.assertTrue(out["ok"])


class ManifestTest(unittest.TestCase):
    """docs/notes-tool-manifest.json is what an LLM reads: it must match the code."""

    def setUp(self):
        path = os.path.join(paths.ROOT, "docs", "notes-tool-manifest.json")
        if not os.path.exists(path):         # docs/ is gitignored: absent in a fresh checkout
            self.skipTest("docs/notes-tool-manifest.json not present")
        with open(path, encoding="utf-8") as f:
            self.m = json.load(f)

    def test_versions_and_rules(self):
        self.assertEqual(self.m["name"], NT.TOOL_NAME)
        self.assertEqual(self.m["version"], NT.TOOL_VERSION)
        self.assertEqual(self.m["note_schema"], NT.NOTE_SCHEMA)
        self.assertEqual([r.strip() for r in self.m["rules"]["order"].split(",")], list(NT.RULES))
        self.assertEqual(self.m["schemas"]["Note"]["properties"]["rule"]["enum"], list(NT.RULES))

    def test_thresholds_stated_match_config(self):
        r, c = self.m["rules"], NT.CONFIG
        for rule, nums in (("SHORT_REST", [c["short_rest_days"]]), ("COLD_REST", [c["cold_rest_days"]]),
                           ("TENURE_NEW", [c["tenure_new_min"], c["tenure_new_days"]]),
                           ("NEW_ADDITION", [c["new_addition_days"]]), ("REVENGE", [c["revenge_max_days"]]),
                           ("STREAK", [c["streak_min"]]),
                           ("HEAVY_SCHEDULE", [c["heavy_min_series"], c["heavy_window_days"]]),
                           ("TENURE_STABLE", [c["tenure_stable_days"]]), ("HOT_PLAYER", [c["form_min_maps"]])):
            for n in nums:
                self.assertIn(str(n), r[rule], rule)

    def test_example_matches_the_schema(self):
        res = self.m["worked_example"]["result"]
        req = set(self.m["schemas"]["Note"]["required"])
        self.assertTrue(res["ok"])
        self.assertEqual(set(self.m["schemas"]["GetNotesResult"]["required"]) - set(res), set())
        for n in res["notes"] + [self.m["worked_example"]["web_fact_note_example"]]:
            self.assertEqual(set(n), req)
        out = NT.get_notes("Alpha", "Charlie", iso(AS_OF), build())
        self.assertEqual(set(self.m["schemas"]["GetNotesResult"]["required"]) - set(out), set())


class AliasTest(unittest.TestCase):
    def test_alias_resolves(self):
        ms = world()
        sec = ND.section(AS_OF, ms, [], [], {}, [], lineups_world(), {}, [], EMPTY, {"Alpha": ["alp"]})
        out = NT.get_notes("ALP", "Bravo", iso(AS_OF), {"notes": sec})
        self.assertEqual(out["team_a"], "Alpha")


class FactSchemaTest(unittest.TestCase):
    def test_validation(self):
        good = form_fact("Alpha", "p", 1.1, 1.0, 10, "2026-05-28T06:00:00Z")
        self.assertEqual(NF.validate_fact(good, {"Alpha"}), [])
        self.assertTrue(NF.validate_fact(good, {"Bravo"}))                  # unknown team
        for k, v in (("rating_last30", 3.1), ("maps_last30", 2.5), ("source_url", "http://x"),
                     ("observed_at", "2026-05-28"), ("method", "guess")):
            bad = dict(good, **{k: v})
            bad["id"] = NF.fact_id(bad)
            self.assertTrue(NF.validate_fact(bad, {"Alpha"}), k)
        self.assertTrue(NF.validate_fact(dict(good, rating_last30=1.2), {"Alpha"}))   # id no longer matches
        self.assertTrue(NF.validate_fact({k: v for k, v in good.items() if k != "source_url"}, {"Alpha"}))

    def test_merge_appends_and_is_idempotent(self):
        a = form_fact("Alpha", "p", 1.1, 1.0, 10, "2026-05-28T06:00:00Z")
        s1 = NF.merge(None, [a], "2026-05-28T06:00:00Z")
        s2 = NF.merge(s1, [a], "2026-05-28T06:00:00Z")
        self.assertEqual(s1, s2)
        b = form_fact("Alpha", "p", 1.2, 1.0, 11, "2026-05-29T06:00:00Z")
        s3 = NF.merge(s2, [b], "2026-05-29T06:00:00Z")
        self.assertEqual(len(s3["form"]), 2)                               # history kept
        self.assertEqual(NF.live_form(s3["form"], "2026-05-30")[("Alpha", "p")]["id"], b["id"])
        self.assertEqual(NF.live_form(s3["form"], "2026-05-29")[("Alpha", "p")]["id"], a["id"])

    def test_malformed_fact_dropped_not_fatal(self):
        bad = {"team": "Alpha", "player": "x"}
        b = build(dict(EMPTY, form=[bad]))
        self.assertTrue(NT.team_notes("Alpha", iso(AS_OF), b)["ok"])


class DeterminismTest(unittest.TestCase):
    def test_synthetic_twice(self):
        x = NT.dumps(NT.get_notes("Alpha", "Charlie", iso(AS_OF), build()))
        NT._CACHE.clear()
        y = NT.dumps(NT.get_notes("Alpha", "Charlie", iso(AS_OF), build()))
        self.assertEqual(x, y)

    def test_frozen_bundle_twice(self):
        path = os.path.join(paths.VIEWER, "data-bundle.json")
        if not os.path.exists(path):
            self.skipTest("no viewer/data-bundle.json (build it to run this)")
        with open(path, encoding="utf-8") as f:
            bundle = json.load(f)
        if "inputs_version" not in bundle.get("notes", {}):
            self.skipTest("bundle built before the on-demand notes snapshot existed")
        sec = bundle["notes"]
        teams = sorted(sec["inputs"]["teams"])[:12]
        days = sorted({r[0] for r in sec["inputs"]["series"]})[-200::20]

        def run():
            NT._CACHE.clear()
            calls = [{"team_a": a, "team_b": b, "match_day": d} for a in teams[:4] for b in teams[4:8]
                     for d in days]
            return NT.dumps([NT.get_notes_batch(calls, d, bundle) for d in days[:3]]
                            + [NT.team_notes(t, "latest", bundle) for t in teams])
        self.assertEqual(run(), run())
        # the version names exactly this snapshot
        self.assertEqual(ND.inputs_version(sec), sec["inputs_version"])

if __name__ == "__main__":
    unittest.main()
