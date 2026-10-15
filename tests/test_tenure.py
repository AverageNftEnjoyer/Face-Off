"""tenure.py: point-in-time behaviour on tiny fixtures (ported from the tenure-factor work)."""
import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

import tenure as TN  # noqa: E402

FIVE = ["a", "b", "c", "d", "e"]


def series(n, start="2025-01-01", step=7):
    """n series for team X with the same five, each published 3 days later."""
    out, d0 = {}, date.fromisoformat(start)
    for i in range(n):
        d = d0.fromordinal(d0.toordinal() + i * step)
        av = d.fromordinal(d.toordinal() + 3)
        out[f"{d.isoformat()}|X|Y{i}"] = {"a": FIVE, "b": None, "avail_a": av.isoformat()}
    return out


class TenureTest(unittest.TestCase):
    def test_unpublished_rows_are_invisible_in_a_backtest(self):
        lu = series(3)
        t = TN.Tenure(lu, {})
        # the third series is dated 2025-01-15, published 2025-01-18
        self.assertEqual(len(t.readable("X", "2025-01-17")), 2)
        self.assertEqual(len(t.readable("X", "2025-01-19")), 3)
        self.assertEqual(len(TN.Tenure(lu, {}, live=True).readable("X", "2025-01-17")), 3)

    def test_same_day_row_not_read(self):
        t = TN.Tenure(series(2), {}, live=True)
        self.assertEqual(len(t.readable("X", "2025-01-08")), 1)

    def test_unknown_is_neutral(self):
        t = TN.Tenure({}, {})
        x, z, (sa, sb) = t.factor("X", "Z", "2025-06-01", 0.3)
        self.assertEqual((x, z, sa["known"], sb["known"]), (0.0, 0.0, False, False))

    def test_stale_lineup_is_unknown(self):
        t = TN.Tenure(series(3), {})
        self.assertFalse(t.state("X", "2025-06-01")["known"])

    def test_one_new_player_resets_four_of_ten_pairs(self):
        lu = series(30)
        # an earlier series elsewhere pushes the start of the data back, so the
        # five's run is not censored by the start of the data
        lu["2023-01-01|Q|R"] = {"a": ["q1", "q2", "q3", "q4", "q5"], "b": None, "avail_a": "2023-01-03"}
        d = date.fromisoformat("2025-01-01").toordinal() + 30 * 7
        for i in range(3):
            day = date.fromordinal(d + i * 7)
            lu[f"{day.isoformat()}|X|Z{i}"] = {"a": FIVE[:4] + ["n"], "b": None,
                                               "avail_a": date.fromordinal(d + i * 7 + 2).isoformat()}
        t = TN.Tenure(lu, {})
        s = t.state("X", date.fromordinal(d + 3 * 7).isoformat())
        self.assertTrue(s["known"])
        self.assertAlmostEqual(s["cov"], 1.0)
        # 6 old pairs at the cap, 4 pairs with the newcomer a few weeks old
        self.assertLess(s["T"], 0.85)
        self.assertGreater(s["T"], 0.6)

    def test_run_at_the_start_of_the_data_is_not_guessed(self):
        s = TN.Tenure(series(30), {}).state("X", "2025-08-05")
        self.assertFalse(s["known"])             # no pair resolves: neutral, not a guess
        self.assertIsNone(s["days_together"])

    def test_squad_spell_joined_after_day_is_ignored(self):
        sq = {"X": [["a", "2025-03-01", ""]]}
        t = TN.Tenure({}, sq)
        self.assertIsNone(t.squad_join("X", "a", date(2025, 2, 1)))
        self.assertEqual(t.squad_join("X", "a", date(2025, 3, 2)), date(2025, 3, 1))


if __name__ == "__main__":
    unittest.main()
