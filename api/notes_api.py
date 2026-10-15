#!/usr/bin/env python3
"""
notes_api.py -- the on-demand matchup notes endpoint: GET /api/notes_api.
A Vercel Python serverless function (the `handler` class below) and, with
--serve, the same handler on localhost for development. Stdlib only.

Contract and the decision to serve notes on request (not pre-rendered):
docs/notes-contracts.md section 4. In short:

  GET /api/notes_api?v=<inputs_version>&a=<team>&b=<team>&asof=<YYYY-MM-DD|latest>
                     [&day=<YYYY-MM-DD>][&event=<name>][&limit=1..24]
  GET /api/notes_api?v=<inputs_version>&team=<team>&asof=...[&day=...][&event=...]

The body is model/notes.py's tool output (get_notes / team_notes) plus
`asof_resolved` and `inputs_version`. The engine is the same one the CLI, the
tests and the future chatbot call: this file only parses, validates, caches
and sets HTTP headers.

SNAPSHOT
  viewer/notes-inputs.json, written by viewer/build_bundle.py at build time and
  shipped with the function (vercel.json `includeFiles`). Fallback when it is
  not in the function bundle: the static copy public/notes/inputs-<v>.json the
  page build publishes, fetched once from this deployment. Loaded once per
  warm instance; a request for another version gets 409 STALE_VERSION.

CACHING
  200 responses: CDN for a day (the URL carries v, so a new snapshot is a new
  URL; nothing stale is served), browsers for 5 minutes. A small LRU inside
  the warm instance. Errors: short or none (section 4 table).

USAGE (local):  python api/notes_api.py --serve [--port 8787]
"""
import argparse
import json
import os
import re
import sys
import threading
import urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import notes as NT  # noqa: E402

SNAPSHOT = os.path.join(paths.VIEWER, "notes-inputs.json")
MAX_PARAM = 80          # longest team / event name accepted
LRU_SIZE = 512
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VER = re.compile(r"^[0-9a-f]{12}$")

STATUS = {"BAD_REQUEST": 400, "BAD_DATE": 400, "SAME_TEAM": 400, "UNKNOWN_TEAM": 404,
          "STALE_VERSION": 409, "MISSING_FIELDS": 503, "BAD_SCHEMA": 503, "UNAVAILABLE": 503}
CACHE_OK = ("public, max-age=300", "max-age=86400")          # (browser, CDN)
CACHE_SHORT = ("public, max-age=60", "max-age=300")
CACHE_NONE = ("no-store", "no-store")

_state = {"bundle": None, "version": None}
_lru = OrderedDict()
_lock = threading.Lock()      # the local server is threaded; Vercel runs one request at a time


def _load_local(path=None):
    path = path or SNAPSHOT
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_remote(version):
    """The page build's static copy of the snapshot, from this deployment."""
    for host in (os.environ.get("VERCEL_URL"), os.environ.get("VERCEL_PROJECT_PRODUCTION_URL")):
        if not host:
            continue
        try:
            with urllib.request.urlopen(f"https://{host}/notes/inputs-{version}.json", timeout=8) as r:
                return json.loads(r.read().decode("utf-8"))
        except (OSError, ValueError):
            continue
    return None


def snapshot(version):
    """-> (bundle dict {"notes": section}, its version) or raises NT.NotesError."""
    if _state["bundle"] is None:
        sec = _load_local()
        if sec is None and version:
            sec = _load_remote(version)
        if sec is None:
            raise NT.NotesError("UNAVAILABLE", "The notes snapshot could not be loaded.")
        _state["bundle"], _state["version"] = {"notes": sec}, sec.get("inputs_version")
    if version and version != _state["version"]:
        raise NT.NotesError("STALE_VERSION", "This page was built from another data snapshot; reload it.",
                            requested=version, current=_state["version"])
    return _state["bundle"], _state["version"]


def _param(q, k, required=False):
    v = (q.get(k) or [None])[0]
    if v is not None:
        v = v.strip()
        if not v or len(v) > MAX_PARAM:
            raise NT.NotesError("BAD_REQUEST", f"Parameter {k!r} must be 1-{MAX_PARAM} characters.", field=k)
    elif required:
        raise NT.NotesError("BAD_REQUEST", f"Missing parameter {k!r}.", field=k)
    return v


def answer(query):
    """Query string -> (HTTP status, body dict, (browser, CDN) cache headers).
    Every failure is the structured error contract, never an exception."""
    with _lock:
        return _answer(query)


def _answer(query):
    try:
        q = parse_qs(query or "", keep_blank_values=True)
        v = _param(q, "v")
        if v is not None and not _VER.match(v):
            raise NT.NotesError("BAD_REQUEST", "v must be the 12-hex inputs_version.", field="v")
        team, a, b = _param(q, "team"), _param(q, "a"), _param(q, "b")
        if not team and not (a and b):
            raise NT.NotesError("BAD_REQUEST", "Pass a and b (a matchup) or team.", field="a")
        asof = _param(q, "asof") or "latest"
        if asof != "latest" and not _DAY.match(asof):
            raise NT.NotesError("BAD_DATE", "asof must be YYYY-MM-DD or latest.", field="asof", value=asof)
        day, event = _param(q, "day"), _param(q, "event")
        lim = _param(q, "limit")
        if lim is not None:
            if not lim.isdigit():
                raise NT.NotesError("BAD_REQUEST", "limit must be an integer from 1 to 24.", field="limit")
            lim = int(lim)
        if day is not None and not _DAY.match(day):
            raise NT.NotesError("BAD_DATE", "day must be YYYY-MM-DD.", field="day", value=day)
        bundle, ver = snapshot(v)
        key = (ver, team, a, b, asof, day, event, lim)
        if key in _lru:
            _lru.move_to_end(key)
            return _lru[key]
        out = (NT.team_notes(team, asof, bundle, day, event, lim) if team
               else NT.get_notes(a, b, asof, bundle, match_day=day, event=event, limit=lim))
        if out["ok"]:
            out["asof_resolved"] = bundle["notes"]["live_cutoff"] if asof == "latest" else asof
            out["inputs_version"] = ver
            res = (200, out, CACHE_OK)
        else:
            res = (STATUS.get(out["error"]["code"], 400), out, CACHE_SHORT)
        _lru[key] = res
        if len(_lru) > LRU_SIZE:
            _lru.popitem(last=False)
        return res
    except NT.NotesError as e:
        code = STATUS.get(e.code, 400)
        return code, NT._fail(e), (CACHE_NONE if code >= 409 else CACHE_SHORT)


class handler(BaseHTTPRequestHandler):
    """Vercel's Python runtime calls this class (its name is the contract)."""

    def _send(self, status, body, cache):
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", cache[0])
        self.send_header("Vercel-CDN-Cache-Control", cache[1])
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def do_GET(self):
        status, body, cache = answer(urlparse(self.path).query)
        self._send(status, body, cache)

    do_HEAD = do_GET

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self):
        self._send(405, NT._fail(NT.NotesError("BAD_REQUEST", "Use GET.")), CACHE_NONE)

    def log_message(self, fmt, *args):   # quiet locally; Vercel logs requests itself
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the notes endpoint on localhost.")
    ap.add_argument("--serve", action="store_true", help="run a local server")
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args(argv)
    if not a.serve:
        ap.print_help()
        return 0
    if not os.path.exists(SNAPSHOT):
        print(f"no snapshot at {SNAPSHOT}: run python viewer/build_bundle.py first")
        return 1
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    print(f"notes endpoint on http://127.0.0.1:{a.port}/api/notes_api (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
