#!/usr/bin/env python3
"""Write public/status.json: a tiny public feed of the newest finished series.

The page's result alerts poll this file (a few KB) instead of re-downloading the
whole site. It is derived from the data bundle built just before it, so it can
never disagree with the page it ships beside.

  {"generated_at": ..., "data_through": ..., "data_hash": ...,
   "results": [{"id", "event", "stage", "day", "team_a", "team_b",
                "series_score", "winner"}, ...]}      newest first, capped

USAGE (repo root):  python viewer/write_status.py [OUT.json] [--bundle PATH]
Stdlib only.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 60


def result_id(r):
    raw = "|".join([r.get("event_slug") or "", r.get("day") or "", r.get("team_a") or "",
                    r.get("team_b") or "", r.get("stage") or ""])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build(bundle):
    results = []
    for r in (bundle.get("results") or [])[:CAP]:
        results.append({
            "id": result_id(r), "event": r.get("event"), "stage": r.get("stage"), "day": r.get("day"),
            "team_a": r.get("team_a"), "team_b": r.get("team_b"),
            "series_score": r.get("series_score"), "winner": r.get("winner"),
        })
    return {"generated_at": bundle.get("generated_at"), "data_through": bundle.get("data_through"),
            "data_hash": bundle.get("data_hash"), "results": results}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bundle_path = argv[argv.index("--bundle") + 1] if "--bundle" in argv else os.path.join(HERE, "data-bundle.json")
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--bundle")]
    out_path = pos[0] if pos else os.path.join(os.path.dirname(HERE), "public", "status.json")
    with open(bundle_path, encoding="utf-8") as f:
        bundle = json.load(f)
    status = build(bundle)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(status, f, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    print(f"wrote {out_path}: {len(status['results'])} results, data through {status['data_through']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
