"""api/notes_api.py: the on-demand notes endpoint on a SYNTHETIC snapshot (made-up
teams; nothing read from data/): HTTP statuses, the error contract, cache
headers, determinism, version pinning and a real HTTP round trip."""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from datetime import date, timedelta
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401

import notes_api as API  # noqa: E402
import notes_data as ND  # noqa: E402

AS_OF = date(2026, 6, 1)
EMPTY = {"fact_schema": 1, "updated_at": None, "form": [], "news": []}


def snapshot():
    ms = []
    for i in range(6):
        d = (AS_OF - timedelta(days=12 - i)).isoformat()
        ms.append({"date": d, "event": "Cup", "team_a": "Alpha", "team_b": "Bravo", "winner": "Alpha",
                   "maps": [{"map": "Nuke", "winner": "Alpha"}, {"map": "Mirage", "winner": "Alpha"}]})
    sec = ND.section(AS_OF, ms, [], ["Alpha", "Bravo", "Charlie"], {}, [], {}, {}, [], EMPTY, {"Alpha": ["alp"]})
    sec["next"] = {}
    sec["inputs_version"] = ND.inputs_version(sec)
    return sec


class Base(unittest.TestCase):
    def setUp(self):
        self.sec = snapshot()
        self.v = self.sec["inputs_version"]
        self.saved = dict(API._state)
        API._state.update(bundle={"notes": self.sec}, version=self.v)
        API._lru.clear()

    def tearDown(self):
        API._state.clear()
        API._state.update(self.saved)
        API._lru.clear()


class AnswerTest(Base):
    def test_matchup_ok_and_cacheable(self):
        st, body, cache = API.answer(f"v={self.v}&a=Alpha&b=Bravo&asof=latest")
        self.assertEqual(st, 200)
        self.assertTrue(body["ok"])
        self.assertIn("STREAK", [n["rule"] for n in body["notes"]])
        self.assertEqual(body["inputs_version"], self.v)
        self.assertEqual(body["asof_resolved"], self.sec["live_cutoff"])
        self.assertEqual(cache, API.CACHE_OK)
        self.assertFalse(body["counted_in_win_probability"])

    def test_team_and_alias(self):
        st, body, _ = API.answer(f"v={self.v}&team=alp")
        self.assertEqual((st, body["team"]), (200, "Alpha"))

    def test_historical_asof_is_its_own_answer(self):
        _, now, _ = API.answer(f"v={self.v}&a=Alpha&b=Bravo&asof=latest")
        st, past, _ = API.answer(f"v={self.v}&a=Alpha&b=Bravo&asof={(AS_OF - timedelta(days=9)).isoformat()}")
        self.assertEqual(st, 200)
        self.assertNotEqual(json.dumps(now["notes"]), json.dumps(past["notes"]))   # fewer series known then
        self.assertEqual(past["asof_resolved"], (AS_OF - timedelta(days=9)).isoformat())

    def test_deterministic(self):
        q = f"v={self.v}&a=Alpha&b=Bravo&asof=latest"
        x = json.dumps(API.answer(q)[1], sort_keys=True)
        API._lru.clear()
        self.assertEqual(x, json.dumps(API.answer(q)[1], sort_keys=True))

    def test_errors_are_structured(self):
        cases = [("a=Alpah&b=Bravo", 404, "UNKNOWN_TEAM"), ("a=Alpha", 400, "BAD_REQUEST"),
                 ("a=Alpha&b=alpha", 400, "SAME_TEAM"), ("a=Alpha&b=Bravo&asof=June", 400, "BAD_DATE"),
                 ("a=Alpha&b=Bravo&day=x", 400, "BAD_DATE"), ("a=" + "x" * 200 + "&b=Bravo", 400, "BAD_REQUEST"),
                 ("v=nothex&a=Alpha&b=Bravo", 400, "BAD_REQUEST")]
        for q, status, code in cases:
            st, body, _ = API.answer(q if q.startswith("v=") else f"v={self.v}&{q}")
            self.assertEqual((st, body["ok"], body["error"]["code"]), (status, False, code), q)
        st, body, _ = API.answer(f"v={self.v}&a=Alpah&b=Bravo")
        self.assertIn("Alpha", body["error"]["close_matches"])

    def test_stale_version(self):
        st, body, cache = API.answer("v=000000000000&a=Alpha&b=Bravo")
        self.assertEqual((st, body["error"]["code"], body["error"]["current"]), (409, "STALE_VERSION", self.v))
        self.assertEqual(cache, API.CACHE_NONE)

    def test_unavailable_without_snapshot(self):
        API._state.update(bundle=None, version=None)
        saved = API.SNAPSHOT
        API.SNAPSHOT = os.path.join(os.path.dirname(__file__), "no-such-snapshot.json")
        try:
            st, body, cache = API.answer("a=Alpha&b=Bravo")
        finally:
            API.SNAPSHOT = saved
        self.assertEqual((st, body["error"]["code"], cache), (503, "UNAVAILABLE", API.CACHE_NONE))


class HttpTest(Base):
    def test_round_trip(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), API.handler)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        base = f"http://127.0.0.1:{srv.server_address[1]}/api/notes_api"
        try:
            with urllib.request.urlopen(f"{base}?v={self.v}&a=Alpha&b=Bravo&asof=latest") as r:
                self.assertEqual(r.status, 200)
                self.assertEqual(r.headers["Vercel-CDN-Cache-Control"], "max-age=86400")
                self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")
                self.assertTrue(json.loads(r.read())["ok"])
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(f"{base}?v={self.v}&a=Nope&b=Bravo")
            self.assertEqual(cm.exception.code, 404)
            self.assertEqual(json.loads(cm.exception.read())["error"]["code"], "UNKNOWN_TEAM")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
