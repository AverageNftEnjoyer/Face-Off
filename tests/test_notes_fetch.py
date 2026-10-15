"""collectors/notes_fetch.py: schema rejection, cache-first, idempotent merge,
failure policy, per-team isolation, manifest and cost. No network: the LLM and
HTTP are fakes, the bundle is a small synthetic one in a temp folder."""
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


def bundle(teams):
    """{team: [players]} -> a minimal bundle; every team plays on NOW + 2 days."""
    names = list(teams)
    fx = [{"day": "2026-10-16", "team_a": names[i], "team_b": names[i + 1] if i + 1 < len(names) else names[0]}
          for i in range(0, len(names), 2)]
    fx.append({"day": "2026-11-30", "team_a": names[0], "team_b": "Nobody"})   # outside the 7-day scope
    return {"fixtures": fx, "data": {
        "teams": {t: {"roster": [{"id": p, "coach": False} for p in ps] + [{"id": "coach_" + t, "coach": True}]}
                  for t, ps in teams.items()},
        "players": {}}}


def resp(obj=None, text=None, inp=1000, out=200, searches=1):
    o = [{"type": "web_search_call", "status": "completed"} for _ in range(searches)]
    o.append({"type": "message", "content": [{"type": "output_text",
                                               "text": text if text is not None else json.dumps(obj)}]})
    return {"status": "completed", "output": o, "usage": {"input_tokens": inp, "output_tokens": out}}


def form(team, player, r30=1.1, rb=1.0, maps=20, url=URL + "/form", found=True):
    return {"found": found, "team": team, "player": player, "rating_last30": r30, "rating_baseline": rb,
            "baseline": "last12m", "maps_last30": maps, "rating_version": "3.0", "source_url": url}


def news(team, items):
    return {"items": [dict({"type": "stand_in", "team": team, "player": "sub", "date": "2026-10-12",
                            "source_url": URL + "/news", "replaced": None, "until": None, "event": None,
                            "move": None, "other_team": None}, **it) for it in items]}


class FakeLLM:
    """key -> response (dict), exception (raised) or callable(key)."""

    def __init__(self, table, default=None):
        self.table, self.default, self.calls = table, default, []

    def __call__(self, key, body):
        self.calls.append((key, body))
        r = self.table.get(key, self.default)
        if callable(r):
            r = r(key)
        if isinstance(r, Exception):
            raise r
        if r is None:
            raise notes_fetch.FetchError("no answer")
        return r


def good_table(teams):
    t = {}
    for team, ps in teams.items():
        t[f"news|{team}"] = resp(news(team, []), searches=2)
        for p in ps:
            t[f"form|{team}|{p}"] = resp(form(team, p))
    return t


class Base(unittest.TestCase):
    TEAMS = {"Alpha": ["a1", "a2"], "Bravo Esports": ["b1", "b2"]}

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.bundle = os.path.join(self.tmp, "bundle.json")
        with open(self.bundle, "w", encoding="utf-8") as f:
            json.dump(bundle(self.TEAMS), f)
        self.out = os.path.join(self.tmp, "facts.json")
        self.man = os.path.join(self.tmp, "manifest.json")
        self.log = os.path.join(self.tmp, notes_fetch.LOG_NAME)

    def run_fetch(self, llm, now=NOW, extra=(), http=None, env=None):
        argv = ["--bundle", self.bundle, "--out", self.out, "--manifest", self.man, "--concurrency", "2"] + list(extra)
        with contextlib.redirect_stderr(io.StringIO()):
            return notes_fetch.main(argv, now=now, llm=llm, http=http or notes_fetch.Http(recorded={}, now=now),
                                    env={} if env is None else env)

    def manifest(self):
        with open(self.man, encoding="utf-8") as f:
            return json.load(f)

    def store(self):
        with open(self.out, encoding="utf-8") as f:
            return json.load(f)


class HappyPath(Base):
    def test_writes_valid_facts_and_manifest(self):
        llm = FakeLLM(good_table(self.TEAMS))
        self.assertEqual(self.run_fetch(llm), 0)
        s = self.store()
        self.assertEqual(len(s["form"]), 4)
        self.assertEqual(s["updated_at"], "2026-10-14T12:00:00Z")
        form, news_, bad = nf.clean(s, set(self.TEAMS))
        self.assertEqual((len(form), bad), (4, []))
        self.assertTrue(all(f["observed_at"] == "2026-10-14T12:00:00Z" and f["method"] == "web_search" for f in form))
        m = self.manifest()
        self.assertEqual(m["scope_teams"], ["Alpha", "Bravo Esports"])
        self.assertEqual(m["llm_calls"], 6)
        self.assertEqual(m["players_attempted"], 4)
        self.assertEqual(m["teams_attempted"], ["Alpha", "Bravo Esports"])
        self.assertEqual(m["facts_written"], 4)
        self.assertEqual(m["exit_code"], 0)
        # file format: sorted keys, indent 1, trailing newline
        with open(self.out, encoding="utf-8") as f:
            self.assertEqual(f.read(), json.dumps(s, sort_keys=True, indent=1, ensure_ascii=False) + "\n")

    def test_cost_arithmetic_and_log(self):
        t = good_table(self.TEAMS)
        self.run_fetch(FakeLLM(t))
        m = self.manifest()
        self.assertEqual(m["input_tokens"], 6000)
        self.assertEqual(m["output_tokens"], 1200)
        self.assertEqual(m["web_search_calls"], 2 * 2 + 4)
        want = 6000 / 1e6 * notes_fetch.PRICE_INPUT_PER_M + 1200 / 1e6 * notes_fetch.PRICE_OUTPUT_PER_M \
            + 8 * notes_fetch.PRICE_WEB_SEARCH_PER_CALL
        self.assertAlmostEqual(m["est_cost_usd"], round(want, 4))
        self.assertIn("ESTIMATES", m["pricing_estimate"]["note"])
        self.run_fetch(FakeLLM(t), now=NOW + timedelta(hours=1))
        with open(self.log, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f]
        self.assertEqual([x["llm_calls"] for x in lines], [6, 0])
        self.assertEqual(lines[1]["cache_hits"], 6)

    def test_team_names_map_onto_the_bundle(self):
        names = notes_fetch.TeamNames(["Team Falcons", "FURIA", "9z Team"], {"9z Team": ["9z"]})
        self.assertEqual(names("Falcons"), "Team Falcons")
        self.assertEqual(names("FURIA Esports"), "FURIA")
        self.assertEqual(names("9z"), "9z Team")
        self.assertIsNone(names("Vitality"))


class Rejections(Base):
    def test_rejection_paths(self):
        t = good_table(self.TEAMS)
        t["form|Alpha|a1"] = resp(form("Alpha", "a1", r30=3.4))                          # bad rating
        t["form|Alpha|a2"] = resp(form("Alpha", "a2", url=None))                         # no source
        t["form|Bravo Esports|b1"] = resp(form("Gamma", "b1"))                           # unknown team
        t["form|Bravo Esports|b2"] = resp(text="b2 is playing really well lately.")      # prose
        t["news|Alpha"] = resp(news("Alpha", [{"source_url": ""}, {"team": "Nobody FC"},
                                              {"player": "ok_sub"}]))
        self.assertEqual(self.run_fetch(FakeLLM(t)), 0)      # 1 failure of 6: not an outage
        m = self.manifest()
        reasons = {(r["kind"], r["team"], r.get("player")): r["reasons"] for r in m["facts_rejected"]
                   if r["kind"] == "form"}
        self.assertIn("rating_last30 3.4 outside", reasons[("form", "Alpha", "a1")][0])
        self.assertEqual(reasons[("form", "Alpha", "a2")], ["missing source_url"])
        self.assertEqual(reasons[("form", "Bravo Esports", "b1")], ["unknown team 'Gamma'"])
        news_rej = sorted(r["reasons"][0] for r in m["facts_rejected"] if r["kind"] == "news")
        self.assertEqual(news_rej, ["missing source_url", "unknown team 'Nobody FC'"])
        self.assertEqual(len(m["fetches_failed"]), 1)
        self.assertIn("not JSON", m["fetches_failed"][0]["reason"])
        s = self.store()
        self.assertEqual(s["form"], [])
        self.assertEqual([f["player"] for f in s["news"]], ["ok_sub"])
        # a failed unit is retried next run; rejected / empty ones are checked
        self.assertNotIn("form|Bravo Esports|b2", m["checked"])
        self.assertIn("form|Alpha|a1", m["checked"])

    def test_news_outside_the_window_is_rejected(self):
        t = good_table(self.TEAMS)
        t["news|Alpha"] = resp(news("Alpha", [{"type": "roster_rumor", "date": "2026-08-01", "move": "in"}]))
        self.run_fetch(FakeLLM(t))
        self.assertIn("outside the last 30 days", self.manifest()["facts_rejected"][0]["reasons"][0])


class CacheAndMerge(Base):
    def test_cache_skip_then_refetch_after_ttl(self):
        t = good_table(self.TEAMS)
        self.run_fetch(FakeLLM(t))
        before = open(self.out, "rb").read()
        llm = FakeLLM(t)
        self.assertEqual(self.run_fetch(llm, now=NOW + timedelta(hours=11)), 0)
        self.assertEqual(llm.calls, [])
        m = self.manifest()
        self.assertEqual((m["cache_hits"], m["cache_hit_rate"], m["store_changed"]), (6, 1.0, False))
        self.assertEqual(open(self.out, "rb").read(), before)          # byte-identical
        llm = FakeLLM(t)
        self.run_fetch(llm, now=NOW + timedelta(hours=13))
        self.assertEqual(len(llm.calls), 6)
        self.assertEqual(len(self.store()["form"]), 8)                  # append-only history

    def test_empty_news_is_not_searched_again_within_the_ttl(self):
        t = good_table(self.TEAMS)
        self.run_fetch(FakeLLM(t))
        llm = FakeLLM(t)
        self.run_fetch(llm, now=NOW + timedelta(hours=6))
        self.assertFalse([k for k, _ in llm.calls if k.startswith("news|")])

    def test_rerun_of_the_same_observations_writes_nothing(self):
        t = good_table(self.TEAMS)
        self.run_fetch(FakeLLM(t))
        before = open(self.out, "rb").read()
        self.run_fetch(FakeLLM(t), extra=["--refresh"])     # same observations fetched again ...
        m = self.manifest()
        self.assertEqual((m["llm_calls"], m["facts_written"], m["store_changed"]), (6, 0, False))
        self.assertEqual(open(self.out, "rb").read(), before)  # ... same ids, so merge adds nothing

    def test_dry_run_with_the_recorded_fixtures_is_idempotent(self):
        teams = {"PARIVISION": ["Jame", "xiELO", "zweih", "FL1T", "HObbit"],
                 "FURIA": ["yuurih", "KSCERATO", "FalleN", "molodoy", "YEKINDAR"], "Team Falcons": ["NiKo"]}
        with open(self.bundle, "w", encoding="utf-8") as f:
            json.dump(bundle(teams), f)
        argv = ["--bundle", self.bundle, "--out", self.out, "--manifest", self.man, "--dry-run",
                "--teams", "PARIVISION,FURIA"]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(notes_fetch.main(argv, now=NOW, env={}), 0)
        m = self.manifest()
        self.assertEqual((m["mode"], m["llm_calls"], m["facts_written"], len(m["facts_rejected"]),
                          len(m["fetches_failed"]), m["http_requests"]), ("dry-run", 12, 9, 3, 1, 0))
        rumor = [f for f in self.store()["news"] if f["type"] == "roster_rumor"][0]
        self.assertEqual(rumor["other_team"], "Team Falcons")
        before = open(self.out, "rb").read()
        with contextlib.redirect_stderr(io.StringIO()):
            notes_fetch.main(argv, now=NOW + timedelta(hours=1), env={})
        self.assertEqual(open(self.out, "rb").read(), before)
        self.assertEqual(self.manifest()["facts_written"], 0)


class FailurePolicy(Base):
    def test_more_than_half_failed_exits_non_zero(self):
        llm = FakeLLM({}, default=notes_fetch.FetchError("OpenAI HTTP 401: bad key"))
        self.assertEqual(self.run_fetch(llm), 1)
        m = self.manifest()
        self.assertEqual((m["units_failed"], m["units_attempted"], m["exit_code"]), (6, 6, 1))
        self.assertIn("6 of 6 units failed", m["error"])
        self.assertFalse(os.path.exists(self.out))

    def test_half_failed_is_not_an_outage(self):
        t = good_table(self.TEAMS)
        for k in ("form|Alpha|a1", "form|Alpha|a2", "news|Alpha"):
            t[k] = notes_fetch.FetchError("boom")
        self.assertEqual(self.run_fetch(FakeLLM(t)), 0)     # 3 of 6 is not > 50%

    def test_per_team_isolation(self):
        t = good_table(self.TEAMS)
        for k in list(t):
            if "|Alpha" in k:
                t[k] = RuntimeError("unexpected crash")
        self.run_fetch(FakeLLM(t))
        s = self.store()
        self.assertEqual(sorted({f["team"] for f in s["form"]}), ["Bravo Esports"])
        self.assertEqual(len(s["form"]), 2)
        self.assertTrue(all("RuntimeError" in x["reason"] for x in self.manifest()["fetches_failed"]))

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

    def test_call_cap_defers_the_rest(self):
        llm = FakeLLM(good_table(self.TEAMS))
        self.run_fetch(llm, extra=["--max-calls", "3"])
        m = self.manifest()
        self.assertEqual((m["llm_calls"], m["units_deferred"]), (3, 3))
        self.assertEqual([k for k, _ in llm.calls][:2], ["news|Alpha", "news|Bravo Esports"])  # news first


class ScrapePath(Base):
    TEAMS = {"Alpha": ["a1"]}
    HLTV = "https://www.hltv.org/stats/players/123/a1"

    def seed(self):
        old = nf.make_fact(team="Alpha", player="a1", observed_at="2026-10-10T00:00:00Z", source_url=self.HLTV,
                           source="HLTV", method="web_search", rating_last30=1.0, rating_baseline=1.0,
                           baseline="last12m", maps_last30=10)
        with open(self.out, "w", encoding="utf-8") as f:
            json.dump(nf.merge(None, [old], "2026-10-10T00:00:00Z"), f)

    def test_robots_disallow_falls_back_to_web_search(self):
        self.seed()
        robots = "User-agent: *\nDisallow: /stats*?*startDate*\n"
        http = notes_fetch.Http(recorded={"https://www.hltv.org/robots.txt": {"status": 200, "body": robots}}, now=NOW)
        llm = FakeLLM(good_table(self.TEAMS))
        self.run_fetch(llm, http=http)
        m = self.manifest()
        self.assertEqual((m["scrape"]["tried"], m["scrape"]["ok"], m["scrape"]["fell_back"]), (1, 0, 1))
        self.assertIn("robots.txt disallows /stats/players/123/a1?startDate=", list(m["scrape"]["reasons"])[0])
        self.assertIn("form|Alpha|a1", [k for k, _ in llm.calls])
        self.assertEqual(m["http_requests"], 1)                      # robots.txt only

    def test_cloudflare_403_falls_back(self):
        self.seed()
        lo = "2026-09-14"
        page = f"{self.HLTV}?startDate={lo}&endDate=2026-10-14"
        http = notes_fetch.Http(recorded={"https://www.hltv.org/robots.txt": {"status": 200, "body": ""},
                                          page: {"status": 403, "body": "<title>Just a moment...</title>"}}, now=NOW)
        self.run_fetch(FakeLLM(good_table(self.TEAMS)), http=http)
        self.assertIn("HTTP 403 Cloudflare challenge", list(self.manifest()["scrape"]["reasons"])[0])

    def test_scrape_ok_uses_the_extractor(self):
        self.seed()
        page = f"{self.HLTV}?startDate=2026-09-14&endDate=2026-10-14"
        html = "<html><head><script>x=1</script></head><body><div>Rating 3.0</div><div>1.17</div></body></html>"
        http = notes_fetch.Http(recorded={"https://www.hltv.org/robots.txt": {"status": 200, "body": ""},
                                          page: {"status": 200, "body": html}}, now=NOW)
        t = good_table(self.TEAMS)
        t["form_scrape|Alpha|a1"] = resp(form("Alpha", "a1", r30=1.17, url=None), searches=0)
        llm = FakeLLM(t)
        self.run_fetch(llm, http=http)
        key, body = [c for c in llm.calls if c[0].startswith("form")][0]
        self.assertEqual(key, "form_scrape|Alpha|a1")
        self.assertNotIn("tools", body)                              # extractor only, no web search
        self.assertIn("Rating 3.0\n1.17", body["input"])
        self.assertNotIn("x=1", body["input"])
        new = [f for f in self.store()["form"] if f["observed_at"] == "2026-10-14T12:00:00Z"][0]
        self.assertEqual((new["method"], new["source"], new["source_url"]), ("scrape", "HLTV", page))

    def test_robots_wildcards(self):
        rules = "User-agent: *\nDisallow: /stats*?*startDate*\nDisallow: /blog/*\n"
        self.assertFalse(notes_fetch.robots_allows(rules, "https://www.hltv.org/stats/players/1/x?startDate=2026-01-01"))
        self.assertTrue(notes_fetch.robots_allows(rules, "https://www.hltv.org/stats/players/1/x"))
        self.assertFalse(notes_fetch.robots_allows(rules, "https://www.hltv.org/blog/post"))


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

    def test_payload_uses_strict_json_schema(self):
        p = notes_fetch.payload("gpt-5-mini", "q", notes_fetch.NEWS_SCHEMA, "roster_news", True)
        self.assertEqual(p["text"]["format"]["type"], "json_schema")
        self.assertTrue(p["text"]["format"]["strict"])
        self.assertEqual(p["tools"][0]["type"], "web_search")
        item = notes_fetch.NEWS_SCHEMA["properties"]["items"]["items"]
        self.assertEqual(sorted(item["required"]), sorted(item["properties"]))  # strict mode needs all keys


if __name__ == "__main__":
    unittest.main()
