"""collectors/notes_summarize.py on a SYNTHETIC bundle (made-up teams, no network):
the grounding check, rewrite-on-change caching, the monthly budget, the key policy."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401

import notes_data as ND  # noqa: E402
import notes_summarize as S  # noqa: E402

AS_OF = date(2026, 6, 1)
NOW = datetime(2026, 6, 1, 7, 0, tzinfo=timezone.utc)
EMPTY = {"fact_schema": 1, "updated_at": None, "form": [], "news": []}
PRED = {"p": 0.6512, "b": [0.52, 0.77], "s": [0.35, 0.3, 0.2, 0.15], "r": "M", "w": [],
        "f": [9.1, 3.2, -1.4, 0.0, 0.0, 0.0, 0.0], "fx": [60, 4, 1, 25, 2, 3, 28, 4, 1, 2, 3, 0, 0, 2, 1, 7, 0]}


def bundle():
    ms = [{"date": (AS_OF - timedelta(days=3)).isoformat(), "event": "Cup", "team_a": "Alpha", "team_b": "Bravo",
           "winner": "Alpha", "maps": [{"map": "Nuke", "winner": "Alpha"}, {"map": "Mirage", "winner": "Alpha"}]}]
    sec = ND.section(AS_OF, ms, [], ["Alpha", "Bravo"], {}, [], {}, {}, [], EMPTY)
    match = {"id": "cup-1", "t1": "Alpha", "t2": "Bravo", "day": (AS_OF + timedelta(days=1)).isoformat(),
             "stage": "Final", "bo": 3, "finished": False, "pred": dict(PRED)}
    return {"data": {"as_of": AS_OF.isoformat(), "events": [{"name": "Cup", "matches": [match]}]}, "notes": sec}


def answer(text):
    return lambda key, body: {"status": "completed", "usage": {"input_tokens": 2000, "output_tokens": 500},
                              "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(text)}]}]}


GOOD = {"headline": "Alpha are the favourites",
        "summary": "The model has Alpha as the favourite with moderate confidence. Alpha are clearly the stronger team "
                   "on rating and have done better than expected over the last month. Not counted in the win chance: "
                   "Alpha won the most recent meeting."}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.b = os.path.join(self.dir, "bundle.json")
        with open(self.b, "w", encoding="utf-8") as f:
            json.dump(bundle(), f)
        self.out, self.log = os.path.join(self.dir, "s.json"), os.path.join(self.dir, "log.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir)

    def run_(self, llm, env=None, now=NOW, extra=()):
        return S.main(["--bundle", self.b, "--out", self.out, "--log", self.log, *extra], now=now, llm=llm,
                      env=env or {"OPENAI_API_KEY": "sk-test"})

    def store(self):
        with open(self.out, encoding="utf-8") as f:
            return json.load(f)


class GroundingTest(Base):
    def test_grounded_summary_is_written(self):
        self.assertEqual(self.run_(answer(GOOD)), 0)
        s = self.store()["summaries"]["cup-1"]
        self.assertEqual((s["favourite"], s["headline"]), ("Alpha", GOOD["headline"]))

    def test_invented_number_rejected(self):
        bad = dict(GOOD, summary=GOOD["summary"] + " Alpha won 17 of their last 23 maps.")
        self.assertIn("figures in the text", S.check(bad, S.match_input(bundle()["data"]["events"][0]["matches"][0],
                                                                        {"name": "Cup"}, {"ok": False}))[0])
        self.run_(answer(bad))
        self.assertFalse(os.path.exists(self.out))
        with open(self.log, encoding="utf-8") as f:
            self.assertIn("figures in the text", f.read())

    def test_favourite_must_be_named(self):
        inp = S.match_input(bundle()["data"]["events"][0]["matches"][0], {"name": "Cup"}, {"ok": False})
        self.assertIn("favourite not named", S.check({"headline": "x", "summary": "A close one."}, inp))

    def test_input_carries_counted_factors_only(self):
        inp = S.match_input(bundle()["data"]["events"][0]["matches"][0], {"name": "Cup"}, {"ok": False})
        names = [r["factor"] for r in inp["prediction"]["counted_factors"]]
        self.assertEqual(names[0], S.LABELS["base_strength"])
        self.assertEqual(inp["prediction"]["win_chance"], "65.1%")


class NamesWithDigitsTest(unittest.TestCase):
    def test_digits_inside_names_are_fine(self):
        m = dict(bundle()["data"]["events"][0]["matches"][0], t1="9z Team", t2="3DMAX")
        inp = S.match_input(m, {"name": "Season 24 Finals"}, {"ok": False})
        ok = {"headline": "9z Team edge it", "summary": "The model leans to 9z Team over 3DMAX at Season 24 Finals."}
        self.assertEqual(S.check(ok, inp), [])


class CacheBudgetKeyTest(Base):
    def test_unchanged_input_is_not_rewritten(self):
        self.run_(answer(GOOD))
        calls = []

        def llm(k, body):
            calls.append(k)
            return answer(GOOD)(k, body)
        self.run_(llm, now=NOW + timedelta(days=2))
        self.assertEqual(calls, [])

    def test_budget_stops_calls(self):
        calls = []

        def llm(k, body):
            calls.append(k)
            return answer(GOOD)(k, body)
        self.assertEqual(self.run_(llm, env={"OPENAI_API_KEY": "sk-test", "FACEOFF_SUMMARY_BUDGET_USD": "0.0001"}), 0)
        self.assertEqual(calls, [])

    def test_no_key(self):
        self.assertEqual(self.run_(None, env={}), 0)
        self.assertEqual(self.run_(None, env={"CI": "true"}), 1)

    def test_key_never_logged(self):
        def boom(k, body):
            raise S.NFX.FetchError("OpenAI HTTP 401: Incorrect API key provided: sk-test-SECRET", {})
        self.run_(boom, env={"OPENAI_API_KEY": "sk-test-SECRET"})
        with open(self.log, encoding="utf-8") as f:
            self.assertNotIn("SECRET", f.read())


def finished_match():
    return {"id": "cup-0", "t1": "Alpha", "t2": "Bravo", "day": (AS_OF - timedelta(days=1)).isoformat(), "stage": "Semifinal",
            "bo": 3, "finished": True, "w1": 0, "w2": 2, "pred": dict(PRED, vm=["Nuke", "Mirage", "Inferno"],
                                                                     vp=[0.7, 0.6, 0.55]),
            "maps": [{"map": "Nuke", "s1": 1, "s2": 13, "w": 2, "ot": 0, "first1": "ct",
                      "side1": {"ct": 1, "t": 0}, "side2": {"ct": 1, "t": 12}},
                     {"map": "Ancient", "s1": 11, "s2": 13, "w": 2, "ot": 0, "first1": "t",
                      "side1": {"ct": 6, "t": 5}, "side2": {"ct": 7, "t": 6}}]}


RECAP_GOOD = {"headline": "Bravo flip the script on Nuke",
              "summary": ("Bravo turned Alpha's best map into a rout and the series never recovered. Nuke went 13-1 "
                          "to Bravo, twelve of those rounds on the T side, on a map where the model leaned Alpha. "
                          "Ancient was the divergence: it was never in the simulated veto, and Bravo closed it 13-11. "
                          "Player statistics are not in our data, so standouts go unnamed here. The model had Alpha "
                          "as the favourite on rating strength; the mechanism of the miss was the map pool, with Bravo "
                          "winning one map the veto expected and one it did not. Alpha lost a map the model rated as "
                          "theirs by a wide margin, which matters more than the scoreline suggests. Watch whether "
                          "Bravo's T side on Nuke holds up next time out, because that half decided everything here.")}


class RecapTest(unittest.TestCase):
    def inp(self):
        return S.recap_input(finished_match(), {"name": "Cup"}, {"ok": False})

    def test_input_flags_the_miss(self):
        inp = self.inp()
        self.assertEqual(inp["model_call"]["result"], "upset")
        self.assertEqual(inp["match"]["winner"], "Bravo")
        self.assertFalse(inp["maps"][1]["in_simulated_veto"])
        self.assertIn("per-player statistics (ratings, ADR) are not in the data", inp["data_gaps"])

    def test_good_recap_passes(self):
        self.assertEqual(S.check_recap(RECAP_GOOD, self.inp()), [])

    def test_banned_invented_short(self):
        inp = self.inp()
        self.assertTrue(any("banned" in e for e in S.check_recap(
            dict(RECAP_GOOD, summary=RECAP_GOOD["summary"] + " What a series."), inp)))
        self.assertTrue(any("numbers not in the input" in e for e in S.check_recap(
            dict(RECAP_GOOD, summary=RECAP_GOOD["summary"] + " Bravo's star posted a 1.61 rating."), inp)))
        self.assertTrue(any("length" in e for e in S.check_recap({"headline": "x", "summary": "Bravo won."}, inp)))


class RecapRunTest(Base):
    def setUp(self):
        super().setUp()
        b = bundle()
        b["data"]["events"][0]["matches"].append(finished_match())
        with open(self.b, "w", encoding="utf-8") as f:
            json.dump(b, f)

    def test_recap_written_once(self):
        def llm(k, body):
            return answer(RECAP_GOOD if k.startswith("recap|") else GOOD)(k, body)
        self.assertEqual(self.run_(llm), 0)
        st = self.store()
        self.assertEqual(st["recaps"]["cup-0"]["result"], "upset")
        self.assertIn("cup-1", st["summaries"])
        calls = []
        self.run_(lambda k, body: (calls.append(k), llm(k, body))[1], now=NOW + timedelta(hours=1))
        self.assertEqual(calls, [])          # unchanged inputs: nothing rewritten


if __name__ == "__main__":
    unittest.main()
