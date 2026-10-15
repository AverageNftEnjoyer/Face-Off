"""collectors/notes_fetch.py: one call per team (form + news in one answer),
3-day scope, 24 h TTL, monthly budget guard, schema rejection, idempotent
merge, failure policy, per-team isolation, manifest and cost. No network: the
LLM is a fake, the bundle is a small synthetic one in a temp folder."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

import notes_facts as nf  # noqa: E402
import notes_fetch  # noqa: E402

NOW = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
URL = "https://fixtures.invalid/t"


def bundle(teams, day="2026-10-16"):
    """{team: [players]} -> a minimal bundle; teams play pairwise on `day`."""
    names = list(teams)
    fx = [{"day": day, "team_a": names[i], "team_b": names[i + 1] if i + 1 < len(names) else names[0]}
          for i in range(0, len(names), 2)]
    return {"fixtures": fx, "data": {
        "teams": {t: {"roster": [{"id": p, "coach": False} for p in ps] + [{"id": "coach_" + t, "coach": True}]}
                  for t, ps in teams.items()},
        "players": {}}}


def resp(obj=None, text=None, inp=1000, out=200, searches=1):
    o = [{"type": "web_search_call", "status": "completed"} for _ in range(searches)]
    o.append({"type": "message", "content": [{"type": "output_text",
                                               "text": text if text is not None else json.dumps(obj)}]})
    return {"status": "completed", "output": o, "usage": {"input_tokens": inp, "output_tokens": out}}


def player(p, r30=1.1, rb=1.0, maps=20, url=URL + "/form", found=True):
    return {"player": p, "found": found, "rating_last30": r30, "rating_baseline": rb, "baseline": "last12m",
            "maps_last30": maps, "rating_version": "3.0", "source_url": url}


def item(team, **kw):
    return dict({"type": "stand_in", "team": team, "player": "sub", "date": "2026-10-12",
                 "source_url": URL + "/news", "replaced": None, "until": None, "event": None,
                 "move": None, "other_team": None}, **kw)


def answer(players=(), news=()):
    return {"players": list(players), "news": list(news)}


class FakeLLM:
    """key -> response (dict) or exception (raised)."""

    def __init__(self, table, default=None):
        self.table, self.default, self.calls = table, default, []

    def __call__(self, key, body):
        self.calls.append((key, body))
        r = self.table.get(key, self.default)
        if isinstance(r, Exception):
            raise r
        if r is None:
            raise notes_fetch.FetchError("no answer")
        return r


def good_table(teams):
    return {f"team|{t}": resp(answer([player(p) for p in ps]), searches=2) for t, ps in teams.items()}


class Base(unittest.TestCase):
    TEAMS = {"Alpha": ["a1", "a2"], "Bravo Esports": ["b1", "b2"]}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bundle = os.path.join(self.tmp, "bundle.json")
        self.write_bundle(bundle(self.TEAMS))
        self.out = os.path.join(self.tmp, "facts.json")
        self.man = os.path.join(self.tmp, "manifest.json")
        self.log = os.path.join(self.tmp, notes_fetch.LOG_NAME)

    def write_bundle(self, b):
        with open(self.bundle, "w", encoding="utf-8") as f:
            json.dump(b, f)

    def run_fetch(self, llm, now=NOW, extra=(), env=None):
        argv = ["--bundle", self.bundle, "--out", self.out, "--manifest", self.man, "--concurrency", "2"] + list(extra)
        with contextlib.redirect_stderr(io.StringIO()):
            return notes_fetch.main(argv, now=now, llm=llm, env={} if env is None else env)

    def manifest(self):
        with open(self.man, encoding="utf-8") as f:
            return json.load(f)

    def store(self):
        with open(self.out, encoding="utf-8") as f:
            return json.load(f)

    def raw(self):
        with open(self.out, "rb") as f:
            return f.read()

    def log_lines(self):
        with open(self.log, encoding="utf-8") as f:
            return [json.loads(x) for x in f]


class OneCallPerTeam(Base):
    def test_form_and_news_from_one_response(self):
        t = good_table(self.TEAMS)
        t["team|Alpha"] = resp(answer([player("a1"), player("A2", r30=0.9)], [item("Alpha")]), searches=2)
        llm = FakeLLM(t)
        self.assertEqual(self.run_fetch(llm), 0)
        self.assertEqual(sorted(k for k, _ in llm.calls), ["team|Alpha", "team|Bravo Esports"])
        body = llm.calls[0][1]
        self.assertEqual(body["tools"], [{"type": "web_search", "search_context_size": "low"}])
        self.assertEqual(body["max_tool_calls"], notes_fetch.MAX_TOOL_CALLS)
        self.assertEqual(body["max_output_tokens"], notes_fetch.MAX_OUTPUT_TOKENS)
        self.assertEqual(body["model"], "gpt-5-nano")
        s = self.store()
        form, news, bad = nf.clean(s, set(self.TEAMS))
        self.assertEqual((len(form), len(news), bad), (4, 1, []))
        self.assertIn(("Alpha", "a2"), {(f["team"], f["player"]) for f in form})   # roster spelling kept
        self.assertTrue(all(f["observed_at"] == "2026-10-14T12:00:00Z" and f["method"] == "web_search"
                            for f in form + news))
        m = self.manifest()
        self.assertEqual((m["llm_calls"], m["facts_written"], m["exit_code"]), (2, 5, 0))
        self.assertEqual(m["teams_attempted"], ["Alpha", "Bravo Esports"])
        with open(self.out, encoding="utf-8") as f:
            self.assertEqual(f.read(), json.dumps(s, sort_keys=True, indent=1, ensure_ascii=False) + "\n")

    def test_three_day_scope(self):
        b = bundle(self.TEAMS)
        b["fixtures"] = [{"day": "2026-10-17", "team_a": "Alpha", "team_b": "Gamma"},          # day +3: in
                         {"day": "2026-10-18", "team_a": "Bravo Esports", "team_b": "Alpha"}]  # day +4: out
        self.write_bundle(b)
        llm = FakeLLM(good_table(self.TEAMS))
        self.run_fetch(llm)
        self.assertEqual(self.manifest()["scope_teams"], ["Alpha"])
        self.assertEqual([k for k, _ in llm.calls], ["team|Alpha"])
        self.run_fetch(FakeLLM(good_table(self.TEAMS)), extra=["--scope-days", "4", "--refresh"])
        self.assertEqual(self.manifest()["scope_teams"], ["Alpha", "Bravo Esports"])

    def test_cost_arithmetic_and_log(self):
        self.run_fetch(FakeLLM(good_table(self.TEAMS)))
        m = self.manifest()
        self.assertEqual((m["input_tokens"], m["output_tokens"], m["web_search_calls"]), (2000, 400, 4))
        want = 2000 / 1e6 * notes_fetch.PRICE_INPUT_PER_M + 400 / 1e6 * notes_fetch.PRICE_OUTPUT_PER_M \
            + 4 * notes_fetch.PRICE_WEB_SEARCH_PER_CALL
        self.assertAlmostEqual(m["est_cost_usd"], round(want, 4))
        self.assertIn("ESTIMATES", m["pricing_estimate"]["note"])
        self.run_fetch(FakeLLM(good_table(self.TEAMS)), now=NOW + timedelta(hours=1))
        self.assertEqual([x["llm_calls"] for x in self.log_lines()], [2, 0])
        self.assertEqual(self.log_lines()[1]["cache_hits"], 2)

    def test_team_names_map_onto_the_bundle(self):
        names = notes_fetch.TeamNames(["Team Falcons", "FURIA", "9z Team"], {"9z Team": ["9z"]})
        self.assertEqual(names("Falcons"), "Team Falcons")
        self.assertEqual(names("FURIA Esports"), "FURIA")
        self.assertEqual(names("9z"), "9z Team")
        self.assertIsNone(names("Vitality"))


class Rejections(Base):
    def test_rejection_paths(self):
        t = good_table(self.TEAMS)
        t["team|Alpha"] = resp(answer(
            [player("a1", r30=3.4), player("a2", url=None), player("zz9")],
            [item("Alpha", source_url=""), item("Nobody FC"), item("Alpha", player="ok_sub")]))
        t["team|Bravo Esports"] = resp(text="Bravo are playing really well lately.")      # prose
        self.assertEqual(self.run_fetch(FakeLLM(t)), 0)      # 1 failure of 2: not > 50%
        m = self.manifest()
        reasons = {(r["kind"], r.get("player")): r["reasons"] for r in m["facts_rejected"]}
        self.assertIn("rating_last30 3.4 outside", reasons[("form", "a1")][0])
        self.assertEqual(reasons[("form", "a2")], ["missing source_url"])
        self.assertEqual(reasons[("form", "zz9")], ["player 'zz9' is not on the Alpha roster"])
        news_rej = sorted(r["reasons"][0] for r in m["facts_rejected"] if r["kind"] == "news")
        self.assertEqual(news_rej, ["missing source_url", "unknown team 'Nobody FC'"])
        self.assertEqual(len(m["fetches_failed"]), 1)
        self.assertIn("not JSON", m["fetches_failed"][0]["reason"])
        s = self.store()
        self.assertEqual(s["form"], [])
        self.assertEqual([f["player"] for f in s["news"]], ["ok_sub"])
        self.assertNotIn("team|Bravo Esports", m["checked"])   # a failed team is retried next run
        self.assertIn("team|Alpha", m["checked"])

    def test_news_outside_the_window_and_bad_dates(self):
        t = good_table(self.TEAMS)
        t["team|Alpha"] = resp(answer([], [item("Alpha", type="roster_rumor", date="2026-08-01", move="in"),
                                            item("Alpha", date="2026-02-30")]))
        self.run_fetch(FakeLLM(t))
        r = [x["reasons"][0] for x in self.manifest()["facts_rejected"]]
        self.assertIn("outside the last 30 days", r[0])
        self.assertIn("not a real YYYY-MM-DD", r[1])

    def test_not_found_players_are_skipped(self):
        t = good_table(self.TEAMS)
        t["team|Alpha"] = resp(answer([player("a1", found=False, r30=None, rb=None, maps=None, url=None)]))
        self.run_fetch(FakeLLM(t))
        m = self.manifest()
        self.assertEqual((m["facts_rejected"], m["fetches_failed"]), ([], []))
        self.assertEqual({f["team"] for f in self.store()["form"]}, {"Bravo Esports"})


class CacheAndMerge(Base):
    def test_24h_ttl(self):
        t = good_table(self.TEAMS)
        self.run_fetch(FakeLLM(t))
        before = self.raw()
        llm = FakeLLM(t)
        self.assertEqual(self.run_fetch(llm, now=NOW + timedelta(hours=23)), 0)
        self.assertEqual(llm.calls, [])
        m = self.manifest()
        self.assertEqual((m["cache_hits"], m["cache_hit_rate"], m["store_changed"]), (2, 1.0, False))
        self.assertEqual(self.raw(), before)          # byte-identical
        llm = FakeLLM(t)
        self.run_fetch(llm, now=NOW + timedelta(hours=25))
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(self.store()["form"]), 8)                  # append-only history

    def test_empty_answer_is_not_asked_again_within_the_ttl(self):
        t = {k: resp(answer()) for k in good_table(self.TEAMS)}
        self.run_fetch(FakeLLM(t))
        self.assertFalse(os.path.exists(self.out))                      # nothing found, nothing written
        llm = FakeLLM(t)
        self.run_fetch(llm, now=NOW + timedelta(hours=12))
        self.assertEqual(llm.calls, [])

    def test_rerun_of_the_same_observations_writes_nothing(self):
        t = good_table(self.TEAMS)
        self.run_fetch(FakeLLM(t))
        before = self.raw()
        self.run_fetch(FakeLLM(t), extra=["--refresh"])      # same observations fetched again ...
        m = self.manifest()
        self.assertEqual((m["llm_calls"], m["facts_written"], m["store_changed"]), (2, 0, False))
        self.assertEqual(self.raw(), before)  # ... same ids, so merge adds nothing

    def test_dry_run_with_the_recorded_fixtures_is_idempotent(self):
        teams = {"PARIVISION": ["Jame", "xiELO", "zweih", "FL1T", "HObbit"],
                 "FURIA": ["yuurih", "KSCERATO", "FalleN", "molodoy", "YEKINDAR"], "Team Falcons": ["NiKo"]}
        self.write_bundle(bundle(teams))
        argv = ["--bundle", self.bundle, "--out", self.out, "--manifest", self.man, "--dry-run",
                "--teams", "PARIVISION,FURIA"]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(notes_fetch.main(argv, now=NOW, env={}), 0)
        m = self.manifest()
        self.assertEqual((m["mode"], m["llm_calls"], m["facts_written"], len(m["facts_rejected"]),
                          len(m["fetches_failed"])), ("dry-run", 2, 9, 4, 0))
        self.assertLess(m["est_cost_usd"] / m["llm_calls"], 0.03)        # the one-call shape: ~$0.02 a team
        rumor = [f for f in self.store()["news"] if f["type"] == "roster_rumor"][0]
        self.assertEqual(rumor["other_team"], "Team Falcons")
        before = self.raw()
        with contextlib.redirect_stderr(io.StringIO()):
            notes_fetch.main(argv, now=NOW + timedelta(hours=1), env={})
        self.assertEqual(self.raw(), before)
        self.assertEqual((self.manifest()["llm_calls"], self.manifest()["facts_written"]), (0, 0))


class Budget(Base):
    TEAMS = {"Alpha": ["a1"], "Bravo Esports": ["b1"], "Charlie": ["c1"], "Delta": ["d1"]}

    def seed_log(self, *lines):
        with open(self.log, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(json.dumps(ln) + "\n")

    def test_budget_stops_calls_without_failing(self):
        worst = notes_fetch.worst_call_usd()
        # this month's live spend leaves room for exactly two worst-case calls; other months and
        # dry runs do not count
        self.seed_log({"run_at": "2026-10-01T06:47:00Z", "mode": "live", "est_cost_usd": 1.0 - 2.5 * worst},
                      {"run_at": "2026-09-30T06:47:00Z", "mode": "live", "est_cost_usd": 50},
                      {"run_at": "2026-10-02T06:47:00Z", "mode": "dry-run", "est_cost_usd": 50})
        expensive = {k: resp(answer([player(k.split("|")[1][0].lower() + "1")]), inp=40000, out=6000, searches=2)
                     for k in good_table(self.TEAMS)}
        llm = FakeLLM(expensive)
        self.assertEqual(self.run_fetch(llm, extra=["--concurrency", "1"], env={"OPENAI_API_KEY": "x"}), 0)
        m = self.manifest()
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual([d["reason"] for d in m["deferred"]], ["monthly budget reached"] * 2)
        b = m["budget"]
        self.assertTrue(b["stopped"])
        self.assertEqual(b["limit_usd"], 1.0)
        self.assertAlmostEqual(b["spent_before_run_usd"], round(1.0 - 2.5 * worst, 4))
        self.assertAlmostEqual(b["spent_month_usd"], round(b["spent_before_run_usd"] + m["est_cost_usd"], 4))
        self.assertLessEqual(b["spent_month_usd"], 1.0)
        self.assertAlmostEqual(b["remaining_usd"], round(1.0 - b["spent_month_usd"], 4))
        self.assertIn("monthly budget reached", " ".join(m["warnings"]))
        self.assertEqual(m["fetches_failed"], [])
        self.assertNotIn("team|Charlie", m["checked"])                 # deferred, asked next run
        self.assertTrue(self.log_lines()[-1]["budget_stopped"])

    def test_spent_budget_makes_no_call_and_env_sets_the_limit(self):
        self.seed_log({"run_at": "2026-10-03T06:47:00Z", "mode": "live", "est_cost_usd": 0.99})
        llm = FakeLLM(good_table(self.TEAMS))
        self.assertEqual(self.run_fetch(llm), 0)
        self.assertEqual(llm.calls, [])
        self.assertEqual(self.manifest()["units_deferred"], 4)
        llm = FakeLLM(good_table(self.TEAMS))
        self.run_fetch(llm, env={"FACEOFF_NOTES_BUDGET_USD": "5"})
        self.assertEqual(len(llm.calls), 4)
        self.assertEqual(self.manifest()["budget"]["limit_usd"], 5.0)

    def test_bad_budget_value_exits_1(self):
        self.assertEqual(self.run_fetch(FakeLLM({}), env={"FACEOFF_NOTES_BUDGET_USD": "one dollar"}), 1)

    def test_call_cap_defers_the_rest_soonest_first(self):
        b = bundle(self.TEAMS)
        b["fixtures"] = [{"day": "2026-10-16", "team_a": "Delta", "team_b": "Charlie"},
                         {"day": "2026-10-14", "team_a": "Bravo Esports", "team_b": "Alpha"}]
        self.write_bundle(b)
        llm = FakeLLM(good_table(self.TEAMS))
        self.run_fetch(llm, extra=["--max-calls", "2"])
        m = self.manifest()
        self.assertEqual(sorted(k for k, _ in llm.calls), ["team|Alpha", "team|Bravo Esports"])
        self.assertEqual(m["deferred"], [{"team": "Charlie", "reason": "--max-calls 2"},
                                         {"team": "Delta", "reason": "--max-calls 2"}])
        self.assertEqual(notes_fetch.DEFAULT_MAX_CALLS, 10)


class FailurePolicy(Base):
    def test_more_than_half_failed_exits_non_zero(self):
        llm = FakeLLM({}, default=notes_fetch.FetchError("OpenAI HTTP 401: bad key"))
        self.assertEqual(self.run_fetch(llm), 1)
        m = self.manifest()
        self.assertEqual((m["units_failed"], m["units_attempted"], m["exit_code"]), (2, 2, 1))
        self.assertIn("2 of 2 teams failed", m["error"])
        self.assertFalse(os.path.exists(self.out))

    def test_half_failed_is_not_an_outage(self):
        t = good_table(self.TEAMS)
        t["team|Alpha"] = notes_fetch.FetchError("boom")
        self.assertEqual(self.run_fetch(FakeLLM(t)), 0)     # 1 of 2 is not > 50%

    def test_per_team_isolation(self):
        t = good_table(self.TEAMS)
        t["team|Alpha"] = RuntimeError("unexpected crash")
        self.run_fetch(FakeLLM(t))
        s = self.store()
        self.assertEqual(sorted({f["team"] for f in s["form"]}), ["Bravo Esports"])
        self.assertEqual(len(s["form"]), 2)
        self.assertIn("RuntimeError", self.manifest()["fetches_failed"][0]["reason"])

    def test_missing_key(self):
        argv = ["--bundle", self.bundle, "--out", self.out, "--manifest", self.man]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(notes_fetch.main(argv, now=NOW, env={}), 0)          # local: message only
            self.assertFalse(os.path.exists(self.man))
            self.assertEqual(notes_fetch.main(argv, now=NOW, env={"GITHUB_ACTIONS": "true"}), 1)
        self.assertIn("OPENAI_API_KEY", self.manifest()["error"])

    def test_missing_bundle_and_unknown_team(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(notes_fetch.main(["--bundle", os.path.join(self.tmp, "nope.json"), "--out", self.out,
                                               "--manifest", self.man], now=NOW, llm=FakeLLM({}), env={}), 1)
        self.assertEqual(self.run_fetch(FakeLLM({}), extra=["--teams", "Alpah"]), 1)

    def test_unwritable_manifest_exits_2(self):
        os.makedirs(self.man)                 # a folder where the manifest file should go
        self.assertEqual(self.run_fetch(FakeLLM(good_table(self.TEAMS))), 2)


class Secrets(unittest.TestCase):
    """The manifest and log are committed in public and pasted into issues:
    the API key must never reach them."""

    def test_redact(self):
        key = "sk-proj-AbC123_def-456"
        msg = f"Incorrect API key provided: sk-proj-****456. Bearer {key}"
        out = notes_fetch.redact(msg, key)
        self.assertNotIn(key, out)
        self.assertNotIn("sk-proj-****456", out)
        self.assertIn("[redacted]", out)
        self.assertEqual(notes_fetch.redact("no secrets here"), "no secrets here")

    def test_http_error_detail_is_redacted(self):
        import urllib.error
        key = "sk-test-SECRETSECRET"
        body = io.BytesIO(json.dumps({"error": {"message": f"Incorrect API key provided: {key}"}}).encode())
        err = urllib.error.HTTPError(notes_fetch.API_URL, 401, "Unauthorized", {}, body)
        llm = notes_fetch.LiveLLM(key)
        orig = notes_fetch.urllib.request.urlopen
        notes_fetch.urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(err)
        try:
            with self.assertRaises(notes_fetch.FetchError) as cm:
                llm("unit", {})
        finally:
            notes_fetch.urllib.request.urlopen = orig
        self.assertNotIn("SECRETSECRET", str(cm.exception))


class Parsing(unittest.TestCase):
    def test_incomplete_and_refusal(self):
        with self.assertRaises(notes_fetch.FetchError):
            notes_fetch.parse_response({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}})
        r = {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}
        with self.assertRaises(notes_fetch.FetchError):
            notes_fetch.parse_response(r)

    def test_schema_is_strict_everywhere(self):
        p = notes_fetch.payload("gpt-5-nano", "q")
        self.assertEqual(p["text"]["format"]["type"], "json_schema")
        self.assertTrue(p["text"]["format"]["strict"])
        for s in (notes_fetch.TEAM_SCHEMA, notes_fetch.PLAYER_ITEM, notes_fetch.NEWS_ITEM):
            self.assertEqual(sorted(s["required"]), sorted(s["properties"]))   # strict mode needs all keys
            self.assertFalse(s["additionalProperties"])

    def test_unsupported_optional_parameter_is_dropped_and_retried(self):
        import urllib.error
        sent = []

        class _R:
            def __init__(self, b):
                self.b = b

            def read(self):
                return self.b

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(req, timeout=None):
            body = json.loads(req.data)
            sent.append(body)
            if "max_tool_calls" in body:
                msg = json.dumps({"error": {"message": "Unsupported parameter: 'max_tool_calls'"}}).encode()
                raise urllib.error.HTTPError(notes_fetch.API_URL, 400, "Bad Request", {}, io.BytesIO(msg))
            return _R(json.dumps(resp(answer())).encode())

        orig = notes_fetch.urllib.request.urlopen
        notes_fetch.urllib.request.urlopen = fake
        try:
            out = notes_fetch.LiveLLM("k")("team|X", notes_fetch.payload("gpt-5-nano", "q"))
        finally:
            notes_fetch.urllib.request.urlopen = orig
        self.assertEqual(len(sent), 2)
        self.assertNotIn("max_tool_calls", sent[1])
        self.assertEqual(sent[1]["tools"][0]["search_context_size"], "low")
        self.assertEqual(out["status"], "completed")


if __name__ == "__main__":
    unittest.main()
