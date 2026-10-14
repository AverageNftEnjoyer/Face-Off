"""Tests for pipeline/build_db.py (data/faceoff.db), on the real data files."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)
import build_db  # noqa: E402


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data", "matches.json")), "no data/matches.json")
class TestBuildDb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.path = os.path.join(cls.dir, "a.db")
        build_db.build(cls.path)
        cls.con = sqlite3.connect(cls.path)
        with open(os.path.join(ROOT, "data", "matches.json"), encoding="utf-8") as f:
            cls.matches = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

    def q(self, sql):
        return self.con.execute(sql).fetchall()

    def test_counts_match_json(self):
        self.assertEqual(self.q("SELECT COUNT(*) FROM matches")[0][0], len(self.matches))
        self.assertEqual(self.q("SELECT COUNT(*) FROM maps")[0][0], sum(len(m.get("maps") or []) for m in self.matches))
        self.assertEqual(self.q("SELECT COUNT(*) FROM elo_history WHERE source = 'main'")[0][0], 2 * len(self.matches))
        # every lower-tier series that moved Elo has its two elo_history rows
        moved = self.q("SELECT COUNT(*) FROM lower_matches WHERE elo_a_before IS NOT NULL")[0][0]
        self.assertEqual(self.q("SELECT COUNT(*) FROM elo_history WHERE source = 'lower'")[0][0], 2 * moved)
        self.assertEqual(self.q("SELECT COUNT(*) FROM team_matches")[0][0], 2 * len(self.matches))

    def test_every_column_documented(self):
        documented = {(t, c) for t, c in self.q("SELECT table_name, column_name FROM schema_doc")}
        tables = [t for (t,) in self.q("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")]
        for t in tables:
            self.assertIn((t, ""), documented, t)
            if t in build_db.VIEWS:
                continue
            for row in self.q(f"PRAGMA table_info({t})"):
                self.assertIn((t, row[1]), documented, f"{t}.{row[1]}")

    def test_elo_is_point_in_time(self):
        # elo_before of a team's next series equals elo_after of its previous one
        # when they are on different days
        rows = self.q("SELECT team, date, elo_before, elo_after FROM elo_history ORDER BY team, rowid")
        checked = 0
        for prev, cur in zip(rows, rows[1:]):
            if prev[0] == cur[0] and prev[1] < cur[1]:
                self.assertAlmostEqual(prev[3], cur[2], places=9)
                checked += 1
        self.assertGreater(checked, 1000)

    def test_deterministic(self):
        other = os.path.join(self.dir, "b.db")
        build_db.build(other)
        with open(self.path, "rb") as f1, open(other, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())


if __name__ == "__main__":
    unittest.main()
