"""Tests for the veto data the viewer page carries (viewer/build_viewer.py):
the compact pick / ban chances and the permaban-safe BO3 veto log.
Team and map names here are test data.

Run from the repo root:  python -m unittest tests.test_viewer_veto
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "viewer"))

import build_viewer as bv  # noqa: E402
import predictor as pr  # noqa: E402

POOL = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]


def inp(**kw):
    m = {"team_a": "TA", "team_b": "TB", "rating_a": 1.05, "rating_b": 1.0, "map_pool": list(POOL),
         "maps_a": {"M1": [0.6, 8], "M2": [0.4, 5], "M3": [0.5, 3]},
         "maps_b": {"M4": [0.55, 7], "M5": [0.45, 4], "M1": [0.4, 2]},
         "maps_raw_a": {}, "maps_raw_b": {}}
    m.update(kw)
    return m


def decode_chances(code, n_vals):
    """Inverse of vd_encode's marginal packing (what the page does in JS)."""
    out = []
    for j in range(0, len(code), 2):
        out.append((bv._B32.index(code[j]) * 32 + bv._B32.index(code[j + 1])) / 1000)
    return [out[i:i + n_vals] for i in range(0, len(out), n_vals)]


class TestVetoDist(unittest.TestCase):
    def setUp(self):
        bv.VMAPS.clear()

    def test_round_trip_and_conservation(self):
        raw = bv._vd_run(inp())
        self.assertEqual(sorted(raw), [1, 3])
        for bo, n_vals in ((3, 5), (1, 3)):
            enc = bv.vd_encode(raw[bo])
            rows = decode_chances(enc["m"], n_vals)
            self.assertEqual(len(rows), len(POOL))
            played = sum(r[0] + r[1] + r[2] for r in rows) if bo == 3 else sum(r[0] for r in rows)
            self.assertAlmostEqual(played, bo, delta=0.02)   # thousandths rounding only
            for r in rows:
                self.assertTrue(all(0 <= x <= 1 for x in r))
            # pool characters index the shared map list
            self.assertEqual([bv.VMAPS[bv._VCODE.index(c)] for c in enc["pl"]], sorted(POOL))

    def test_small_pool_is_left_out_not_raised(self):
        raw = bv._vd_run(inp(map_pool=["M1", "M2"]))
        self.assertEqual(raw, {})

    def test_cold_start_flags(self):
        raw = bv._vd_run(inp(maps_a={}, maps_b={}))
        enc = bv.vd_encode(raw[3])
        self.assertEqual(enc["cd"], "ab")
        self.assertIn("cold_start_both", enc["w"])

    def test_event_pool_overrides_only_when_usable(self):
        self.assertIsNone(bv.valid_pool(["M1", "M2"]))
        self.assertIsNone(bv.valid_pool(None))
        self.assertEqual(bv.valid_pool(["M2", "M1", "M2", "M3", "M4", "M5"]), ["M1", "M2", "M3", "M4", "M5"])

    def test_permaban_suffix_is_not_part_of_the_map(self):
        r = pr.predict_match(inp(permaban_a="M1"))
        self.assertTrue(any("(permaban)" in x for x in r["veto_log"]))
        c = bv.compact(r)
        self.assertFalse(any("permaban" in x for x in c["v"]))
        self.assertIn("A-M1", c["v"])


if __name__ == "__main__":
    unittest.main()
