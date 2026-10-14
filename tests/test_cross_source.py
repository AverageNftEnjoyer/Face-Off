"""Second results source (data/valve_results.py), the cross-source report
(scripts/cross_source_check.py) and the cross_source gate. Every fixture is
SYNTHETIC (made-up teams, dates and pages); nothing touches the network or the
real cache."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "data", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cross_source_check as xs  # noqa: E402
import data_gates as gates  # noqa: E402
import valve_results as vr  # noqa: E402
from collect_liquipedia import stable_match_id  # noqa: E402


def page(team, rows):
    """A Valve roster detail page in the repository's markdown layout.
    rows: [(local id, date, opponent, 'W'|'L')]"""
    head = (f"### Roster Details<br />\nTeam Name: {team}<br />\nRoster: a, b, c, d, e<br />\n\n"
            "| Match Played | Match ID | Date       | Opponent      | W/L | Age Weight | Event Weight |"
            " Bounty Collected | Opponent Network | LAN Wins  | H2H Adj. | Roster |\n"
            "| -: | -: | :- | :- | :- | :- | :- | :- | :- | :- | -: | :- |\n")
    body = "".join(f"| {n} | {lid} | {d} | {opp} | {wl} | 1.000 | - | - | - | - | 1.00 | a, b, c, d, e |\n"
                   for n, (lid, d, opp, wl) in enumerate(rows, 1))
    tail = ("\n| Event Date | Age Weight | Prize Winnings | Scaled Winnings |\n| :- | -: | :- | :- |\n"
            "| 2030-01-05 |      1.000 | $1,000.00    | $1,000.00     |\n")
    return head + body + tail


def path(snap, n, slug):
    return f"live/2030/details/{snap}/{n:04d}--{slug}--a-b-c-d-e.md"


def pm(date, a, b, winner, event="Cup", ext=""):
    """A primary (matches.json-shaped) series."""
    return {"match_id": stable_match_id(date, a, b, event, ext), "date": date, "event": event,
            "event_page": event, "team_a": a, "team_b": b, "winner": winner, "best_of": 3,
            "maps": [{"map": "Mirage", "winner": winner}, {"map": "Nuke", "winner": winner}],
            "tier": "S", "ext_match_id": ext or None}


NAMES = {"Team Alpha", "Bravo Esports", "Charlie Gaming", "Delta"}
INDEX = vr.name_index(NAMES, {"Bravo Esports": ["brv"]})


class TestParse(unittest.TestCase):
    def test_parse_detail_reads_rows_and_header(self):
        team, rows = vr.parse_detail(page("Alpha", [(17, "2030-01-03", "Bravo", "W"),
                                                    (40, "2030-01-02", "Charlie", "L")]))
        self.assertEqual(team, "Alpha")
        self.assertEqual(rows, [(17, "2030-01-03", "Bravo", True), (40, "2030-01-02", "Charlie", False)])

    def test_prize_table_is_not_a_match_row(self):
        _, rows = vr.parse_detail(page("Alpha", []))
        self.assertEqual(rows, [])

    def test_slug_and_snapshot(self):
        p = path("2030_02_04", 3, "team_alpha")
        self.assertEqual(vr.slug_name(p), "team alpha")
        self.assertEqual(vr.snapshot_of(p), "2030_02_04")


class TestNames(unittest.TestCase):
    def test_derived_index_drops_filler_words_and_uses_alias_codes(self):
        self.assertEqual(INDEX["alpha"], "Team Alpha")       # "Team" is filler
        self.assertEqual(INDEX["bravo"], "Bravo Esports")    # "Esports" is filler
        self.assertEqual(INDEX["brv"], "Bravo Esports")      # alias code
        self.assertEqual(INDEX["charlie"], "Charlie Gaming")

    def test_ambiguous_key_is_dropped(self):
        idx = vr.name_index({"Echo", "Team Echo"})
        self.assertNotIn("echo", idx)

    def test_alias_for_unknown_team_is_ignored(self):
        idx = vr.name_index({"Delta"}, {"Not Ours": ["delta2"]})
        self.assertNotIn("delta2", idx)


def two_sided(snap="2030_02_04"):
    """Alpha beat Bravo (local id 5) and lost to Charlie (id 6), seen from both sides."""
    return [
        (path(snap, 1, "alpha"), page("Alpha", [(5, "2030-01-10", "Bravo", "W"),
                                                (6, "2030-01-11", "Charlie", "L")])),
        (path(snap, 2, "bravo"), page("Bravo", [(5, "2030-01-10", "Alpha", "L")])),
        (path(snap, 3, "charlie"), page("Charlie", [(6, "2030-01-11", "Alpha", "W")])),
    ]


class TestBuild(unittest.TestCase):
    def test_one_record_per_series_with_null_unknowns(self):
        recs, spans, stats = vr.build_series(two_sided(), INDEX)
        self.assertEqual(len(recs), 2)
        r = next(r for r in recs if {r["team_a"], r["team_b"]} == {"Team Alpha", "Bravo Esports"})
        self.assertEqual(r["winner"], "Team Alpha")
        for k in ("event", "event_page", "best_of", "tier", "ext_match_id"):
            self.assertIsNone(r[k])
        self.assertEqual(r["maps"], [])
        self.assertEqual(r["match_id"], stable_match_id(r["date"], r["team_a"], r["team_b"], vr.SOURCE_TAG, ""))
        self.assertEqual(spans["Team Alpha"], [["2030-01-10", "2030-02-04"]])

    def test_pages_that_disagree_drop_the_series(self):
        pages = [(path("s1", 1, "alpha"), page("Alpha", [(5, "2030-01-10", "Bravo", "W")])),
                 (path("s1", 2, "bravo"), page("Bravo", [(5, "2030-01-10", "Alpha", "W")]))]
        recs, _, stats = vr.build_series(pages, INDEX)
        self.assertEqual(recs, [])
        self.assertEqual(stats["inconsistent"], 1)

    def test_newest_snapshot_wins_and_first_seen_is_kept(self):
        old = [(path("2030_02_04", 1, "alpha"), page("Alpha", [(5, "2030-01-10", "Bravo", "L")]))]
        new = [(path("2030_03_04", 1, "alpha"), page("Alpha", [(9, "2030-01-10", "Bravo", "W")]))]
        recs, _, _ = vr.build_series(old + new, INDEX)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["winner"], "Team Alpha")
        self.assertEqual(recs[0]["snapshot"], "2030_03_04")
        self.assertEqual(recs[0]["first_seen"], "2030_02_04")

    def test_same_day_rematch_gets_its_own_id(self):
        pages = [(path("s1", 1, "alpha"), page("Alpha", [(5, "2030-01-10", "Bravo", "W"),
                                                         (8, "2030-01-10", "Bravo", "L")]))]
        recs, _, _ = vr.build_series(pages, INDEX)
        self.assertEqual(len(recs), 2)
        self.assertEqual(len({r["match_id"] for r in recs}), 2)

    def test_unmapped_opponent_is_counted_not_guessed(self):
        pages = [(path("s1", 1, "alpha"), page("Alpha", [(5, "2030-01-10", "Zulu Five", "W")]))]
        recs, _, stats = vr.build_series(pages, INDEX)
        self.assertEqual(recs, [])
        self.assertIn(("Zulu Five", 1), stats["unmapped_names"])

    def test_keep_filters_to_tracked_pairs(self):
        recs, _, _ = vr.build_series(two_sided(), INDEX, keep={"Team Alpha", "Bravo Esports"})
        self.assertEqual([(r["team_a"], r["team_b"]) for r in recs], [("Bravo Esports", "Team Alpha")])

    def test_write_load_roundtrip(self):
        recs, spans, _ = vr.build_series(two_sided(), INDEX)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "v.json")
            vr.write({"snapshots": ["2030_02_04"], "spans": spans, "series": recs}, p)
            back = vr.load(p)
        self.assertEqual(back["series"], recs)
        self.assertIsNone(vr.load(os.path.join(ROOT, "no", "such.json")))


class TestInferAliases(unittest.TestCase):
    def _pages(self, n):
        """Alpha meets 'Bravo Rebrand' (no name of ours) on n days."""
        rows = [(i, f"2030-01-{10 + 3 * i:02d}", "Bravo Rebrand", "W") for i in range(n)]
        return [(path("s1", 1, "alpha"), page("Alpha", rows))]

    def _primary(self, n):
        return [pm(f"2030-01-{10 + 3 * i:02d}", "Team Alpha", "Bravo Esports", "Team Alpha") for i in range(n)]

    def test_enough_independent_days_infer_the_name(self):
        n = vr.ALIAS_MIN_VOTES
        self.assertEqual(vr.infer_aliases(self._pages(n), INDEX, self._primary(n)),
                         {"Bravo Rebrand": "Bravo Esports"})

    def test_too_few_days_infer_nothing(self):
        n = vr.ALIAS_MIN_VOTES - 1
        self.assertEqual(vr.infer_aliases(self._pages(n), INDEX, self._primary(n)), {})

    def test_inference_ignores_who_won(self):
        n = vr.ALIAS_MIN_VOTES
        prim = [dict(m, winner="Bravo Esports") for m in self._primary(n)]
        self.assertEqual(vr.infer_aliases(self._pages(n), INDEX, prim), {"Bravo Rebrand": "Bravo Esports"})

    def test_a_name_that_maps_to_another_team_is_never_remapped(self):
        n = vr.ALIAS_MIN_VOTES
        rows = [(i, f"2030-01-{10 + 3 * i:02d}", "Charlie", "W") for i in range(n)]
        pages = [(path("s1", 1, "alpha"), page("Alpha", rows))]
        self.assertEqual(vr.infer_aliases(pages, INDEX, self._primary(n)), {})


def second_file(records, snaps=("2030_02_04",)):
    spans = {}
    for r in records:
        for t in (r["team_a"], r["team_b"]):
            spans[t] = [["2029-08-01", "2030-02-04"]]
    return {"snapshots": list(snaps), "spans": spans, "series": records}


def sec(date, a, b, winner, snap="2030_02_04"):
    a, b = sorted((a, b))
    return vr.make_record(date, a, b, winner, snap, snap, (a, b))


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.second = second_file([
            sec("2030-01-10", "Team Alpha", "Bravo Esports", "Team Alpha"),
            sec("2030-01-12", "Team Alpha", "Charlie Gaming", "Charlie Gaming"),   # one day off
            sec("2030-01-20", "Delta", "Charlie Gaming", "Delta"),
            sec("2030-02-01", "Bravo Esports", "Delta", "Delta"),
        ])

    def test_buckets(self):
        primary = [pm("2030-01-10", "Team Alpha", "Bravo Esports", "Team Alpha"),
                   pm("2030-01-11", "Charlie Gaming", "Team Alpha", "Charlie Gaming"),
                   pm("2030-01-20", "Delta", "Charlie Gaming", "Charlie Gaming"),   # disagrees
                   pm("2030-02-01", "Bravo Esports", "Delta", "Delta")]
        rep = xs.compare(primary, self.second)
        self.assertEqual((rep["matched"], rep["agree"], rep["disagree"]), (4, 3, 1))
        self.assertEqual(rep["disagreements"][0]["second_winner"], "Delta")
        self.assertEqual(rep["primary_only"], 0)

    def test_outside_the_date_window_does_not_match(self):
        primary = [pm("2030-01-13", "Team Alpha", "Bravo Esports", "Team Alpha"),
                   pm("2030-02-01", "Bravo Esports", "Delta", "Delta")]
        rep = xs.compare(primary, self.second)
        self.assertEqual(rep["matched"], 1)

    def test_identity_mismatch_same_day_same_result_other_opponent(self):
        primary = [pm("2030-01-10", "Team Alpha", "Delta", "Team Alpha", event="Old Cup"),
                   pm("2030-02-01", "Bravo Esports", "Delta", "Delta")]
        rep = xs.compare(primary, self.second)
        self.assertEqual(rep["identity_mismatch"], 1)
        self.assertEqual(rep["identity_mismatch_by_name"],
                         [{"primary": "Delta", "second": "Bravo Esports", "series": 1}])

    def test_event_running_at_snapshot_is_pending_not_missing(self):
        primary = [pm("2030-01-10", "Team Alpha", "Bravo Esports", "Team Alpha"),
                   pm("2030-02-01", "Team Alpha", "Delta", "Delta", event="Live Cup"),
                   pm("2030-02-04", "Team Alpha", "Charlie Gaming", "Team Alpha", event="Live Cup")]
        rep = xs.compare(primary, self.second)
        self.assertEqual(rep["pending"], 1)
        self.assertEqual(rep["primary_only"], 0)

    def test_second_only_split_by_lower_tier(self):
        primary = [pm("2030-01-10", "Team Alpha", "Bravo Esports", "Team Alpha"),
                   pm("2030-02-01", "Bravo Esports", "Delta", "Delta")]
        lower = [pm("2030-01-20", "Delta", "Charlie Gaming", "Delta", event="B Cup")]
        rep = xs.compare(primary, self.second, lower)
        self.assertEqual(rep["second_only"], {"lower_tier": 1, "untracked": 1})


class TestGate(unittest.TestCase):
    def setUp(self):
        self.second = second_file([sec("2030-01-10", "Team Alpha", "Bravo Esports", "Team Alpha"),
                                   sec("2030-01-20", "Delta", "Charlie Gaming", "Delta")])
        self.old = [pm("2030-01-10", "Team Alpha", "Bravo Esports", "Team Alpha"),
                    pm("2030-01-20", "Delta", "Charlie Gaming", "Delta")]

    def test_unchanged_data_passes(self):
        out = gates.gate_cross_source(self.old, [dict(m) for m in self.old], self.second)
        self.assertEqual(out["problems"], [])
        self.assertEqual(out["agreement_rate"], 1.0)

    def test_flipped_winner_fails(self):
        new = [dict(m) for m in self.old]
        new[1] = pm("2030-01-20", "Delta", "Charlie Gaming", "Charlie Gaming")
        out = gates.gate_cross_source(self.old, new, self.second)
        self.assertEqual(len(out["problems"]), 1)
        self.assertEqual(out["new_disagreements"][0]["match_id"], new[1]["match_id"])

    def test_new_identity_mismatch_fails(self):
        new = [self.old[0], pm("2030-01-20", "Delta", "Team Alpha", "Delta")]
        out = gates.gate_cross_source(self.old, new, self.second)
        self.assertTrue(any("different opponent" in p for p in out["problems"]))

    def test_existing_discrepancy_is_not_re_failed(self):
        bad = [self.old[0], pm("2030-01-20", "Delta", "Charlie Gaming", "Charlie Gaming")]
        out = gates.gate_cross_source(bad, [dict(m) for m in bad], self.second)
        self.assertEqual(out["problems"], [])
        self.assertEqual(out["disagree"], 1)

    def test_missing_second_source_never_blocks(self):
        out = gates.gate_cross_source(self.old, self.old, {"series": []})
        self.assertNotIn("problems", out)
        self.assertIn("skipped", out["note"])

    def test_no_baseline_reports_only(self):
        out = gates.gate_cross_source(None, self.old, self.second)
        self.assertNotIn("problems", out)


class TestFillCandidatesNeverMerged(unittest.TestCase):
    def test_nothing_reads_the_candidate_file(self):
        """The fill-in list is a review list: no pipeline module may read it."""
        hits = []
        for sub in ("", "data", "scripts", "viewer"):
            d = os.path.join(ROOT, sub)
            for fn in os.listdir(d):
                if fn.endswith(".py") and fn != "cross_source_check.py":
                    with open(os.path.join(d, fn), encoding="utf-8") as f:
                        if "valve_fill" in f.read():
                            hits.append(os.path.join(sub, fn))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
