"""Tests for pipeline hardening: data gates, idempotent rebuild, the derived
bundle's pure functions, loud parse failure, and the supervised roster diff.

Everything is SYNTHETIC except TestIdempotentBuild, which rebuilds from the
committed Liquipedia cache twice and compares bytes. Nothing touches the
network. The full end-to-end acceptance demos (bad-input block in the
workflow, deterministic bundle rebuild) are run manually and reported, not
in this suite, because a full bundle build takes ~7 minutes.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "data", "scripts", "viewer"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import data_gates as gates  # noqa: E402
import roster_watch as rw  # noqa: E402


def good_match(i, date="2026-10-09", a="Team A", b="Team B", winner=None):
    w = winner or a
    return {"match_id": f"abc{i:09d}", "date": date, "team_a": a, "team_b": b,
            "event": "Some Event", "stage": "Final", "best_of": 3, "tier": "S",
            "winner": w, "score_a": 2, "score_b": 0,
            "maps": [{"map": "Mirage", "winner": w}, {"map": "Nuke", "winner": w}],
            "source": "liquipedia:X"}


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestGatesBadInput(unittest.TestCase):
    def _run(self, old, new):
        with tempfile.TemporaryDirectory() as d:
            po, pn = os.path.join(d, "old.json"), os.path.join(d, "new.json")
            with open(po, "w", encoding="utf-8") as f:
                json.dump(old, f)
            with open(pn, "w", encoding="utf-8") as f:
                json.dump(new, f)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = gates.main(["--old", po, "--new", pn])
            return code, json.loads(buf.getvalue())

    def test_old_git_without_baseline_is_an_error_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            pn = os.path.join(d, "new.json")
            with open(pn, "w", encoding="utf-8") as f:
                json.dump([good_match(0)], f)
            buf = io.StringIO()
            with mock.patch.object(gates, "load_old_git", side_effect=gates.NoBaseline("no HEAD")), \
                    redirect_stdout(buf):
                code = gates.main(["--old-git", "--new", pn])
            self.assertEqual(code, 2)
            self.assertIn("no baseline", json.loads(buf.getvalue())["error"])
            # an explicit opt-in lets a first-ever run through
            buf = io.StringIO()
            with mock.patch.object(gates, "load_old_git", side_effect=gates.NoBaseline("no HEAD")), \
                    redirect_stdout(buf):
                code = gates.main(["--old-git", "--allow-no-baseline", "--new", pn])
            self.assertEqual(code, 0)

    def test_no_flag_does_not_silently_read_git(self):
        # the old `elif "--old-git":` was always true and read git even with no flag
        with tempfile.TemporaryDirectory() as d:
            pn = os.path.join(d, "new.json")
            with open(pn, "w", encoding="utf-8") as f:
                json.dump([good_match(0)], f)
            with mock.patch.object(gates, "load_old_git", side_effect=AssertionError("must not be called")), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(gates.main(["--new", pn]), 0)

    def test_git_output_is_decoded_as_utf8(self):
        raw = json.dumps([{"team_a": "Grün"}], ensure_ascii=False).encode("utf-8")
        done = mock.Mock(returncode=0, stdout=raw, stderr=b"")
        with mock.patch.object(gates.subprocess, "run", return_value=done):
            self.assertEqual(gates.load_old_git()[0]["team_a"], "Grün")

    def test_clean_passes(self):
        old = [good_match(i) for i in range(30)]
        new = [good_match(i) for i in range(30)] + [good_match(30)]
        code, report = self._run(old, new)
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])

    def test_schema_violation_blocks(self):
        old = [good_match(i) for i in range(30)]
        bad = good_match(99)
        del bad["winner"]  # missing required field
        code, report = self._run(old, old + [bad])
        self.assertEqual(code, 1)
        self.assertFalse(report["gates"]["schema"]["passed"])
        self.assertIn("winner", report["gates"]["schema"]["issues"][0])

    def test_duplicate_ids_block(self):
        old = [good_match(i) for i in range(30)]
        dup = good_match(0)  # same match_id as record 0
        code, report = self._run(old, old + [dup])
        self.assertEqual(code, 1)
        self.assertFalse(report["gates"]["duplicate_ids"]["passed"])

    def test_mass_drop_blocks(self):
        # a parse break that silently drops half the matches must not ship
        old = [good_match(i) for i in range(100)]
        new = old[:50]
        code, report = self._run(old, new)
        self.assertEqual(code, 1)
        self.assertFalse(report["gates"]["sane_counts"]["passed"])
        self.assertEqual(report["gates"]["sane_counts"]["removed"], 50)

    def test_spike_blocks(self):
        old = [good_match(i) for i in range(30)]
        new = old + [good_match(1000 + i) for i in range(1200)]
        code, report = self._run(old, new)
        self.assertEqual(code, 1)
        self.assertFalse(report["gates"]["sane_counts"]["passed"])

    def test_normal_update_passes_elo(self):
        # one new series moves a team by far less than its allowance
        one = good_match(0)
        code, report = self._run([one], [one, good_match(1, winner="Team B")])
        self.assertEqual(code, 0)
        self.assertTrue(report["gates"]["elo_bounds"]["passed"])

    def test_many_new_series_scale_the_allowance(self):
        # 40 new results for a team is a busy catch-up, not corruption: the
        # allowance grows with the series the team actually played
        old = [good_match(i, winner="Team A") for i in range(40)]
        new = old + [good_match(1000 + i, winner="Team B") for i in range(40)]
        code, report = self._run(old, new)
        self.assertEqual(code, 0)
        self.assertTrue(report["gates"]["elo_bounds"]["passed"])

    def test_flipped_published_results_block(self):
        # already-published series that change their winner is the corruption
        # signal the Elo bound alone cannot see (a few flips barely move Elo)
        old = [good_match(i, winner="Team A") for i in range(100)]
        new = [dict(m) for m in old]
        for m in new[:10]:
            m["winner"] = "Team B"
            m["maps"] = [{"map": "Mirage", "winner": "Team B"}, {"map": "Nuke", "winner": "Team B"}]
        code, report = self._run(old, new)
        self.assertEqual(code, 1)
        ch = report["gates"]["changed_results"]
        self.assertFalse(ch["passed"])
        self.assertEqual(ch["changed"], 10)
        self.assertEqual(len(ch["sample"]), 10)

    def test_one_correction_is_tolerated(self):
        old = [good_match(i, winner="Team A") for i in range(100)]
        new = [dict(m) for m in old]
        new[0]["winner"] = "Team B"
        new[0]["maps"] = [{"map": "Mirage", "winner": "Team B"}, {"map": "Nuke", "winner": "Team B"}]
        code, report = self._run(old, new)
        self.assertEqual(report["gates"]["changed_results"]["changed"], 1)
        self.assertTrue(report["gates"]["changed_results"]["passed"])
        self.assertEqual(code, 0)

    def test_append_only_detection_and_idle_teams_stay_put(self):
        # a team with no new series cannot change rating in an append-only update
        old = [good_match(i, a="Team A", b="Team B") for i in range(20)] + \
              [good_match(50 + i, date="2026-10-01", a="Team C", b="Team D") for i in range(10)]
        # the new series is dated in the past -> not append-only, so an idle team CAN move
        past = good_match(900, date="2026-09-01", a="Team A", b="Team B")
        code, report = self._run(old, old + [past])
        self.assertFalse(report["gates"]["elo_bounds"]["append_only"])
        # same series dated after every old one -> append-only; C and D are idle and must not move
        future = good_match(901, date="2026-10-10", a="Team A", b="Team B")
        code, report = self._run(old, old + [future])
        self.assertTrue(report["gates"]["elo_bounds"]["append_only"])
        self.assertEqual(code, 0)


class TestIdempotentBuild(unittest.TestCase):
    def test_rebuild_twice_byte_identical(self):
        import collect_liquipedia as cl
        cl.OFFLINE = True
        titles = cl.read_titles(os.path.join(ROOT, "data", "raw", "lp_titles_selected.txt"))
        a = json.dumps(cl.build(titles)[0], indent=1, ensure_ascii=False, sort_keys=True)
        b = json.dumps(cl.build(titles)[0], indent=1, ensure_ascii=False, sort_keys=True)
        self.assertEqual(a, b)
        self.assertGreater(len(json.loads(a)), 2000)


class TestBundlePureFunctions(unittest.TestCase):
    def setUp(self):
        import build_bundle as bb
        self.bb = bb

    def test_fixture_id_deterministic(self):
        f = self.bb.fixture_id
        self.assertEqual(f("e", "2026-10-10", "A", "B"), f("e", "2026-10-10", "A", "B"))
        self.assertNotEqual(f("e", "2026-10-10", "A", "B"), f("e", "2026-10-10", "B", "A"))

    def test_decode_veto(self):
        v = self.bb.decode_veto(["A-Train", "B+Inferno", "DNuke"])
        self.assertEqual(v, [{"team": "A", "action": "ban", "map": "Train"},
                             {"team": "B", "action": "pick", "map": "Inferno"},
                             {"team": None, "action": "decider", "map": "Nuke"}])

    def test_expand_pred_contract(self):
        import build_viewer as bv
        p = {"p": 0.68, "b": [0.5, 0.81], "s": [0.34, 0.34, 0.19, 0.13],
             "vm": ["Dust2", "Mirage"], "vp": [0.7, 0.6], "v": ["A-Dust2"],
             "f": [19.4, -1.2, 0, 0, 0, 0, 0], "w": [], "r": "HIGH", "e": [1808, 1730]}
        e = self.bb.expand_pred(p)
        self.assertEqual(e["win_prob_a"], 0.68)
        self.assertEqual(e["ci_90"], [0.5, 0.81])
        self.assertEqual(set(e["scoreline_probs"]), {"2-0", "2-1", "1-2", "0-2"})
        self.assertEqual(e["modal_scoreline"], "2-0")
        self.assertEqual(e["factors"]["base_strength"], 19.4)
        self.assertEqual(list(e["factors"]), bv.FACTORS)
        self.assertEqual(e["pre_match_elo"], [1808, 1730])


class TestLoudParseFailure(unittest.TestCase):
    def test_truncated_page_yields_no_phantom_matches(self):
        import collect_liquipedia as cl
        # a structurally broken page must not produce fake results
        self.assertEqual(cl.parse_page("X", "<html><body>truncated garbage{{{"), [])
        self.assertEqual(cl.parse_page("X", ""), [])

    def test_gates_catch_parse_break(self):
        # simulate a parse break that drops matches: gates go red
        old = [good_match(i) for i in range(200)]
        new = old[:100]
        with tempfile.TemporaryDirectory() as d:
            po, pn = os.path.join(d, "old.json"), os.path.join(d, "new.json")
            _dump(old, po)
            _dump(new, pn)
            with redirect_stdout(io.StringIO()):
                code = gates.main(["--old", po, "--new", pn])
        self.assertEqual(code, 1)


class TestRosterWatch(unittest.TestCase):
    def _files(self, d, live, cand):
        pl = os.path.join(d, "roster_events.json")
        pc = os.path.join(d, "roster_events.candidate.json")
        if live is not None:
            _dump(live, pl)
        _dump(cand, pc)
        return pl, pc

    def test_identical_is_clean(self):
        ev = {"Team A": {"Event X": {"players": [["s1mple", "ZywOo"]], "missing_igl": False}}}
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(rw, "LIVE", os.path.join(d, "roster_events.json")), \
                 mock.patch.object(rw, "CAND", os.path.join(d, "roster_events.candidate.json")):
                self._files(d, ev, ev)
                self.assertEqual(rw.main(["--report", os.path.join(d, "r.md")]), 0)
                self.assertFalse(os.path.exists(os.path.join(d, "r.md")))

    def test_difference_is_loud(self):
        live = {"Team A": {"Event X": {"players": [["s1mple", "ZywOo"]], "missing_igl": False}}}
        cand = {"Team A": {"Event X": {"players": [["s1mple", "donk"]], "missing_igl": False}}}
        with tempfile.TemporaryDirectory() as d:
            rp = os.path.join(d, "r.md")
            with mock.patch.object(rw, "LIVE", os.path.join(d, "roster_events.json")), \
                 mock.patch.object(rw, "CAND", os.path.join(d, "roster_events.candidate.json")), \
                 mock.patch.object(rw, "DATA", d):
                self._files(d, live, cand)
                self.assertEqual(rw.main(["--report", rp]), 1)
                body = _read(rp)
                self.assertIn("Team A", body)
                self.assertIn("donk", body)
                self.assertIn("scripts/promote_rosters.py --yes", body)

    def test_no_candidate_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(rw, "CAND", os.path.join(d, "nope.json")):
                self.assertEqual(rw.main(["--report", os.path.join(d, "r.md")]), 2)


if __name__ == "__main__":
    unittest.main()
