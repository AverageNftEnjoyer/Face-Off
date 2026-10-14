"""Tests for the persistent veto-distribution cache (viewer/veto_cache.py).

The cache is correct only if (1) its key covers every field of an engine input
that predictor.veto_distribution can read, (2) its fingerprint invalidates the
whole file when the model changes, (3) a hit renders the same page data as a
recompute. Team and map names here are test data.

Run from the repo root:  python -m unittest tests.test_veto_cache
"""
import copy
import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

import build_viewer as bv  # noqa: E402
import predictor as pr  # noqa: E402
import veto_cache as vc  # noqa: E402

POOL = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]


def ev(seed):
    r = random.Random(seed)
    maps = {mp: [r.randint(0, 6), r.randint(0, 8), -r.random() * 3] for mp in POOL}
    return {"n": r.randint(0, 9), "maps": maps}


def base(seed=1):
    """An input with every field the engine can read set to something real."""
    r = random.Random(seed)
    return {"team_a": "TA", "team_b": "TB", "rating_a": 1.05, "rating_b": 0.97, "date": "2026-01-01",
            "map_pool": list(POOL),
            "maps_a": {mp: [round(r.random(), 3), r.randint(0, 9)] for mp in POOL[:5]},
            "maps_b": {mp: [round(r.random(), 3), r.randint(0, 9)] for mp in POOL[2:]},
            "maps_raw_a": {"M1": [0.5, 3]}, "maps_raw_b": {"M2": [0.5, 3]},
            "veto_ev_a": ev(seed), "veto_ev_b": ev(seed + 100),
            "permaban_a": "M3", "permaban_b": None,
            "veto_format": None, "veto_prefix": None,
            "h2h": {"a_wins": 1, "b_wins": 0}, "form30_a": 0.6, "form30_b": 0.4,
            "roster_a": {"standin": True}, "roster_b": {}}


class Recording(dict):
    """A dict that records every key an engine call looks at."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = set()

    def get(self, k, d=None):
        self.seen.add(k)
        return super().get(k, d)

    def __getitem__(self, k):
        self.seen.add(k)
        return super().__getitem__(k)

    def __contains__(self, k):
        self.seen.add(k)
        return super().__contains__(k)

    def __iter__(self):
        self.seen.add("<iter>")
        return super().__iter__()

    def keys(self):
        self.seen.add("<iter>")
        return super().keys()

    def items(self):
        self.seen.add("<iter>")
        return super().items()

    def values(self):
        self.seen.add("<iter>")
        return super().values()


def variants():
    out = [base(1), base(2)]
    v = base(3)
    v["veto_format"] = ["S_ban", "O_ban", "S_pick", "O_pick", "S_ban", "O_ban"]
    out.append(v)
    v = base(4)
    v["veto_prefix"] = [{"team": "a", "action": "ban", "map": "M1"}]
    out.append(v)
    v = base(5)
    v["map_pool"] = POOL[:6]
    v["permaban_a"] = v["permaban_b"] = None
    out.append(v)
    return out


class TestKeyCoverage(unittest.TestCase):
    def test_engine_reads_only_keyed_fields(self):
        # every engine run, in every variant and format, may look only at KEY_FIELDS
        for inp in variants():
            for bo in (3, 1):
                rec = Recording(inp)
                try:
                    pr.veto_distribution(rec, bo)
                except ValueError:
                    pass
                self.assertNotIn("<iter>", rec.seen)
                self.assertLessEqual(rec.seen, set(vc.KEY_FIELDS), sorted(rec.seen - set(vc.KEY_FIELDS)))

    def test_each_keyed_field_changes_the_key(self):
        inp = base(1)
        k0 = vc.input_key(inp)
        perturb = {
            "map_pool": lambda d: d.update(map_pool=POOL[:6]),
            "maps_a": lambda d: d["maps_a"].update(M1=[0.61, 4]),
            "maps_b": lambda d: d["maps_b"].update(M7=[0.2, 1]),
            "veto_ev_a": lambda d: d["veto_ev_a"]["maps"]["M1"].__setitem__(1, 99),
            "veto_ev_b": lambda d: d["veto_ev_b"].update(n=d["veto_ev_b"]["n"] + 1),
            "veto_format": lambda d: d.update(veto_format=["S_ban"]),
            "permaban_a": lambda d: d.update(permaban_a="M4"),
            "permaban_b": lambda d: d.update(permaban_b="M5"),
            "veto_prefix": lambda d: d.update(veto_prefix=[{"team": "a", "action": "ban", "map": "M1"}]),
        }
        self.assertEqual(set(perturb), set(vc.KEY_FIELDS))
        for f, fn in perturb.items():
            d = copy.deepcopy(inp)
            fn(d)
            self.assertNotEqual(vc.input_key(d), k0, f)
        # a tiny float change still changes the key (floats go through repr)
        d = copy.deepcopy(inp)
        d["maps_a"]["M1"][0] += 1e-12
        self.assertNotEqual(vc.input_key(d), k0)

    def test_unkeyed_fields_change_neither_key_nor_output(self):
        inp = base(1)
        k0 = vc.input_key(inp)
        out0 = [bv._vd_run(inp)]
        for f, val in (("team_a", "ZZ"), ("team_b", "YY"), ("rating_a", 1.4), ("rating_b", 0.5),
                       ("date", "2020-02-02"), ("maps_raw_a", {"M2": [0.9, 20]}), ("maps_raw_b", {}),
                       ("h2h", {"a_wins": 9, "b_wins": 9}), ("form30_a", 0.1), ("form30_b", 0.9),
                       ("roster_a", {}), ("roster_b", {"standin": True, "missing_igl": True}),
                       ("an_unknown_field", 1)):
            d = copy.deepcopy(inp)
            d[f] = val
            self.assertEqual(vc.input_key(d), k0, f)
            self.assertEqual([bv._vd_run(d)], out0, f)

    def test_non_list_prefix_is_ignored_like_the_engine(self):
        a = base(1)
        b = base(1)
        b["veto_prefix"] = ({"team": "a", "action": "ban", "map": "M1"},)   # tuple: engine ignores it
        self.assertEqual(vc.input_key(a), vc.input_key(b))
        self.assertEqual(bv._vd_run(a), bv._vd_run(b))

    def test_unserialisable_input_is_never_cached(self):
        d = base(1)
        d["maps_a"] = {"M1": {1, 2}}
        self.assertIsNone(vc.input_key(d))


class TestFingerprint(unittest.TestCase):
    def fp(self, **kw):
        return vc.fingerprint(kw.get("pr", pr), kw.get("veto", pr._veto), bv.VD_FORMATS, bv.MARG_KEYS)

    def test_stable(self):
        self.assertEqual(self.fp(), self.fp())

    def test_changes_with_config_entries(self):
        f0 = self.fp()
        names = [k for k in pr.CONFIG if k.startswith(("veto_", "map_"))]
        self.assertTrue(names)
        for k in names:
            old = copy.deepcopy(pr.CONFIG[k])
            try:
                pr.CONFIG[k] = [old, "x"]    # any change at all
                self.assertNotEqual(self.fp(), f0, k)
            finally:
                pr.CONFIG[k] = old
        self.assertEqual(self.fp(), f0)

    def test_changes_with_veto_source_and_formats(self):
        f0 = self.fp()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "veto.py")
            with open(pr._veto.__file__, "rb") as src, open(p, "wb") as dst:
                dst.write(src.read())
            fake = mock.Mock(__file__=p)
            self.assertEqual(self.fp(veto=fake), f0)        # same source, same fingerprint
            with open(p, "ab") as dst:
                dst.write(b"\n# a change\n")
            self.assertNotEqual(self.fp(veto=fake), f0)
        self.assertNotEqual(vc.fingerprint(pr, pr._veto, (3,), bv.MARG_KEYS), f0)

    def test_line_endings_do_not_matter(self):
        f0 = self.fp()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "veto.py")
            with open(pr._veto.__file__, "rb") as src:
                data = src.read().replace(b"\r\n", b"\n")
            with open(p, "wb") as dst:
                dst.write(data.replace(b"\n", b"\r\n"))
            self.assertEqual(self.fp(veto=mock.Mock(__file__=p)), f0)


class TestRoundTrip(unittest.TestCase):
    def test_hit_encodes_exactly_like_a_recompute(self):
        with tempfile.TemporaryDirectory() as d:
            c = vc.VetoCache(os.path.join(d, "c.json"), "fp", bv.VD_FORMATS, bv.MARG_KEYS)
            ins = variants() + [dict(base(9), map_pool=POOL[:2]), dict(base(10), maps_a={}, maps_b={})]
            for n, inp in enumerate(ins):
                raw = bv._vd_run(inp)
                c.put(f"k{n}", raw)
                got = c.get(f"k{n}")
                assert got is not None
                self.assertEqual(sorted(got), sorted(raw))
                for bo in raw:
                    bv.VMAPS.clear()
                    want = bv.vd_encode(raw[bo])
                    bv.VMAPS.clear()
                    self.assertEqual(bv.vd_encode(got[bo]), want)
            # and through the file
            self.assertEqual(c.save({f"k{n}" for n in range(len(ins))}), len(ins))
            c2 = vc.VetoCache(c.path, "fp", bv.VD_FORMATS, bv.MARG_KEYS)
            self.assertEqual(c2.loaded, len(ins))
            for n, inp in enumerate(ins):
                raw = bv._vd_run(inp)
                got = c2.get(f"k{n}")
                assert got is not None
                for bo in raw:
                    bv.VMAPS.clear()
                    want = bv.vd_encode(raw[bo])
                    bv.VMAPS.clear()
                    self.assertEqual(bv.vd_encode(got[bo]), want)

    def test_quantiser_matches_vd_encode_for_all_thousandths(self):
        for n in range(0, 1001):
            p = n / 1000
            self.assertEqual(vc._q(p), n)
            self.assertEqual(round(p * 1000), n)
        for p in (random.Random(3).random() for _ in range(2000)):
            n = vc._q(p)
            self.assertEqual(round((n / 1000) * 1000), n)


class TestFile(unittest.TestCase):
    def make(self, d, fp="fp"):
        return vc.VetoCache(os.path.join(d, "c.json"), fp, bv.VD_FORMATS, bv.MARG_KEYS)

    def test_prune_and_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.make(d)
            for n in range(3):
                c.put(f"k{n}", bv._vd_run(base(n + 1)))
            self.assertEqual(c.save({"k0", "k2", "unknown"}), 2)
            self.assertEqual(os.listdir(d), ["c.json"])      # no temp file left
            c2 = self.make(d)
            self.assertEqual(sorted(c2.entries), ["k0", "k2"])
            self.assertIsNone(c2.get("k1"))
            with open(c2.path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 1 + 2 + 1)          # header, 2 entries, close
            json.loads("\n".join(lines))

    def test_bad_files_are_ignored_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            good = self.make(d)
            good.put("k", bv._vd_run(base(1)))
            good.save({"k"})
            with open(p, encoding="utf-8") as f:
                text = f.read()
            cases = {"garbage": "{not json", "empty": "", "list": "[]",
                     "foreign version": text.replace('"version":1', '"version":99', 1),
                     "other fingerprint": text.replace('"fingerprint":"fp"', '"fingerprint":"zz"', 1),
                     "truncated": text[:len(text) // 2]}
            for name, body in cases.items():
                with open(p, "w", encoding="utf-8") as f:
                    f.write(body)
                with mock.patch("sys.stderr"):
                    c = self.make(d)
                self.assertEqual(c.loaded, 0, name)
                self.assertIsNone(c.get("k"), name)

    def test_malformed_entry_is_a_miss(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.make(d)
            c.entries["bad"] = ["nopool", [1], None]
            c.entries["short"] = ["x"]
            self.assertIsNone(c.get("bad"))
            self.assertIsNone(c.get("short"))

    def test_disabled_reads_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            c = vc.VetoCache(os.path.join(d, "c.json"), "fp", bv.VD_FORMATS, bv.MARG_KEYS, enabled=False)
            c.put("k", bv._vd_run(base(1)))
            self.assertIsNone(c.get("k"))
            self.assertEqual(c.save({"k"}), 0)
            self.assertEqual(os.listdir(d), [])


class TestRunVdist(unittest.TestCase):
    def queue(self):
        bv.VD_JOBS.clear()
        outs = []
        for n in (1, 2, 1, 3, 2):            # repeats exercise the within-run de-dup
            o = {}
            outs.append(o)
            bv.VD_JOBS.append((o, base(n)))
        return outs

    def run_once(self, env):
        bv.VMAPS.clear()
        outs = self.queue()
        with mock.patch.dict(os.environ, env, clear=False), mock.patch("builtins.print") as pr_:
            bv.run_vdist()
        return outs, pr_.call_args[0][0]

    def test_cold_warm_and_disabled_agree(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            env = {"FACEOFF_VD_CACHE_FILE": path}
            os.environ.pop("FACEOFF_NO_VD_CACHE", None)
            cold, msg = self.run_once(env)
            self.assertIn("0 cache hits, 3 computed, 2 repeats", msg)
            warm, msg = self.run_once(env)
            self.assertIn("5 cache hits, 0 computed", msg)
            off, msg = self.run_once({"FACEOFF_NO_VD_CACHE": "1", "FACEOFF_VD_CACHE_FILE": path})
            self.assertIn("5 computed", msg)
            self.assertEqual(cold, warm)
            self.assertEqual(cold, off)
            self.assertTrue(all("vd" in o for o in cold))
            self.assertEqual(vc.VetoCache(path, vc.fingerprint(pr, pr._veto, bv.VD_FORMATS, bv.MARG_KEYS),
                                          bv.VD_FORMATS, bv.MARG_KEYS).loaded, 3)


if __name__ == "__main__":
    unittest.main()
