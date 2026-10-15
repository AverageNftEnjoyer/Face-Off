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


GOOD = {"headline": "Alpha favoured at 65.1%",
        "summary": "The model gives Alpha a 65.1% chance (range 52% to 77%). Team strength adds 9.1 points for Alpha, "
                   "and the last 30 days 3.2 more."}


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
        self.run_(answer(bad))
        self.assertFalse(os.path.exists(self.out))
        with open(self.log, encoding="utf-8") as f:
            self.assertIn("numbers not in the input", f.read())

    def test_favourite_must_be_named(self):
        inp = S.match_input(bundle()["data"]["events"][0]["matches"][0], {"name": "Cup"}, {"ok": False})
        self.assertIn("favourite not named", S.check({"headline": "x", "summary": "A close one."}, inp))

    def test_input_carries_counted_factors_only(self):
        inp = S.match_input(bundle()["data"]["events"][0]["matches"][0], {"name": "Cup"}, {"ok": False})
        names = [r["factor"] for r in inp["prediction"]["counted_factors"]]
        self.assertEqual(names[0], S.LABELS["base_strength"])
        self.assertEqual(inp["prediction"]["win_chance"], "65.1%")


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


if __name__ == "__main__":
    unittest.main()
