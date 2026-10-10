#!/usr/bin/env python3
"""Emit viewer/data-bundle.json: the single versioned derived-data bundle.

Pure function of data/matches.json + cached pages + assets, via
viewer/build_viewer.py's build_data(). The frontend renders exclusively from
this bundle; the schema is the contract in docs/bundle-schema.md.

Envelope:
  bundle_version   int, bumped on any schema change
  generated_at     UTC timestamp of the build
  data_through     last match date ingested (honest staleness label)
  data_hash        sha1-12 of data/matches.json (same stamp the freshness
                   check reads off the deployed page)
  vrs_date         Valve standings date used
  fixtures         upcoming matches (both teams known, not finished) with full
                   pre-match prediction outputs
  results          recently finished matches with map scores + model verdicts
  data             the full computed payload (teams, pairs, h2h, events, ...)

Determinism: identical inputs produce byte-identical output when
FACEOFF_GENERATED_AT is pinned (acceptance: delete + rebuild = identical).

USAGE (repo root):  python viewer/build_bundle.py [OUT.json]
Stdlib only.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import build_viewer as bv  # noqa: E402

BUNDLE_VERSION = 1
RESULT_DAYS = 30       # results window: finished matches this recent
RESULT_CAP = 150


def data_hash():
    with open(os.path.join(ROOT, "data", "matches.json"), "rb") as f:
        return hashlib.sha1(f.read().replace(b"\r\n", b"\n")).hexdigest()[:12]


def generated_at():
    return os.environ.get("FACEOFF_GENERATED_AT") or \
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fixture_id(event_slug, day, t1, t2):
    raw = "|".join([event_slug, day or "", t1, t2])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def expand_pred(p):
    """Readable prediction from the compact page form. s = [p_2_0, p_2_1,
    p_1_2, p_0_2] from team A's perspective; v = coded veto steps."""
    s = p["s"]
    scoreline = {"2-0": s[0], "2-1": s[1], "1-2": s[2], "0-2": s[3]}
    return {
        "win_prob_a": p["p"],
        "ci_90": list(p["b"]),
        "scoreline_probs": scoreline,
        "modal_scoreline": max(scoreline, key=lambda k: scoreline[k]),
        "veto": decode_veto(p["v"]),
        "veto_maps": list(p["vm"]),
        "map_win_prob_a": list(p["vp"]),
        "factors": {name: pp for name, pp in zip(bv.FACTORS, p["f"])},
        "warnings": list(p["w"]),
        "reliability": p["r"],
        "pre_match_elo": p.get("e"),
    }


def decode_veto(steps):
    """['A-Dust2', 'B+Mirage', 'DAnubis'] -> [{team, action, map}]."""
    out = []
    for s in steps:
        if s.startswith("D"):
            out.append({"team": None, "action": "decider", "map": s[1:]})
        else:
            action = {"-": "ban", "+": "pick"}.get(s[1], "ban")
            out.append({"team": s[0], "action": action, "map": s[2:]})
    return out


def build_fixtures(events, data_through):
    # Fixtures dated long before the data is through are page rot, not upcoming
    # matches (e.g. PGL Bucharest 2026 still lists "upcoming" April matches that
    # never happened). A 7-day grace keeps recently-unreported matches visible.
    stale_before = (datetime.fromisoformat(data_through) - timedelta(days=7)).date().isoformat()
    out = []
    for e in events:
        for m in e["matches"]:
            if m.get("finished") or not m.get("t1") or not m.get("t2") or "pred" not in m:
                continue
            if (m.get("day") or "") < stale_before:
                continue
            t1, t2 = m["t1"], m["t2"]
            out.append({
                "fixture_id": fixture_id(e["slug"], m.get("day"), t1, t2),
                "event": e["name"], "event_slug": e["slug"],
                "stage": m.get("stage"), "day": m.get("day"), "when": m.get("when"),
                "best_of": m.get("bo"), "team_a": t1, "team_b": t2,
                "prediction": expand_pred(m["pred"]),
            })
    out.sort(key=lambda f: (f["day"] or "9999", f["when"] or "", f["event_slug"]))
    return out


def build_results(events, data_through):
    cutoff = (datetime.fromisoformat(data_through) - timedelta(days=RESULT_DAYS)).date().isoformat()
    out = []
    for e in events:
        for m in e["matches"]:
            if not m.get("finished") or not m.get("t1") or not m.get("t2"):
                continue
            if (m.get("day") or "") < cutoff:
                continue
            t1, t2 = m["t1"], m["t2"]
            w1, w2 = m.get("w1", 0), m.get("w2", 0)
            winner = t1 if w1 > w2 else t2
            maps = []
            for mp in m.get("maps") or []:
                maps.append({"map": mp.get("map"), "score_a": mp.get("s1"),
                             "score_b": mp.get("s2"),
                             "winner": t1 if mp.get("w") == 1 else t2})
            verdict = None
            if "pred" in m:
                p = m["pred"]["p"]
                predicted = t1 if p >= 0.5 else t2
                verdict = {"predicted": predicted, "win_prob": p, "hit": predicted == winner}
            out.append({
                "event": e["name"], "event_slug": e["slug"],
                "stage": m.get("stage"), "day": m.get("day"),
                "team_a": t1, "team_b": t2,
                "series_score": f"{w1}-{w2}", "winner": winner, "maps": maps,
                "verdict": verdict,
            })
    out.sort(key=lambda r: (r["day"] or "", r["event_slug"], r["stage"] or ""),
               reverse=True)
    return out[:RESULT_CAP]


def main(out_path=None):
    out_path = out_path or os.path.join(HERE, "data-bundle.json")
    data, info = bv.build_data()
    fixtures = build_fixtures(data["events"], info["data_through"])
    results = build_results(data["events"], info["data_through"])
    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "generated_at": generated_at(),
        "data_through": info["data_through"],
        "data_hash": data_hash(),
        "vrs_date": data.get("vrs_date"),
        "fixtures": fixtures,
        "results": results,
        "data": data,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    kb = os.path.getsize(out_path) // 1024
    print(f"wrote {out_path}: bundle v{BUNDLE_VERSION}, data through {info['data_through']}, "
          f"{len(fixtures)} fixtures, {len(results)} results, {kb} KB")
    return bundle


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
