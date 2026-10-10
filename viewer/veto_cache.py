"""Persistent, content-addressed cache of the per-matchup veto distribution
summary that build_viewer._vd_run returns.

A veto distribution is a pure function of (a) the veto inputs of one engine
input and (b) the model code + CONFIG. The key covers exactly (a); the
fingerprint, stored in the file header, covers (b) and discards the whole file
when it differs.

(a) predictor.veto_distribution(inp, bo) reads these keys of `inp` and no
others (predictor.veto_context / veto_distribution):
    map_pool, maps_a, maps_b, veto_ev_a, veto_ev_b,
    veto_format, permaban_a, permaban_b, veto_prefix
Ratings, h2h, form, rosters, team names, dates, maps_raw_* are never read
(invariant I9); tests/test_veto_cache.py proves this by access recording and by
perturbation. Whole values are hashed (never a sub-projection), so a field
inside veto_ev that a future flag starts to read is covered automatically.

(b) see fingerprint(): veto.py source, the predictor functions on the veto
path, every CONFIG entry whose name starts with veto_ or map_, VETO_ORDER, and
the cache layout / VD formats / marginal keys.

File: one JSON object, one entry per line, sorted keys (diff-friendly):
    {"version": 1, "fingerprint": "...", "pools": {...},
     "entries": {
    "<key>": [pool_id, <bo3>, <bo1>],
    ...}}
where <bo> is null (format not modelled) or
    [starter_thousandths, masked_a, masked_b, cold ("a"/"b"/"ab"/""), warnings,
     marginals as base-32 pairs in thousandths, pool order]
which is exactly the precision build_viewer.vd_encode keeps, so a hit and a
recompute give the same page. Stdlib only.
"""
import hashlib
import inspect
import json
import os
import sys

CACHE_VERSION = 1
KEY_FIELDS = ("map_pool", "maps_a", "maps_b", "veto_ev_a", "veto_ev_b",
              "veto_format", "permaban_a", "permaban_b", "veto_prefix")
# predictor functions on the veto path (their source is part of the fingerprint)
_PRED_FUNCS = ("veto_context", "veto_distribution", "veto_params", "_team_maps", "_comfort",
               "_pool", "_num", "_count", "_shrink", "_clamp")
_B32 = "0123456789abcdefghijklmnopqrstuv"


def _sha1(b):
    return hashlib.sha1(b).hexdigest()


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True)


def fingerprint(pr, veto_mod, formats, marg_keys):
    """Hash of everything but the inputs that decides a veto distribution."""
    with open(veto_mod.__file__, "rb") as f:
        veto_src = _sha1(f.read().replace(b"\r\n", b"\n"))
    funcs = {n: _sha1(inspect.getsource(getattr(pr, n)).replace("\r\n", "\n").encode("utf-8"))
             for n in _PRED_FUNCS}
    cfg = {k: v for k, v in pr.CONFIG.items() if k.startswith(("veto_", "map_"))}
    payload = {"cache_version": CACHE_VERSION, "key_fields": KEY_FIELDS, "veto_py": veto_src,
               "predictor_funcs": funcs, "veto_order": pr.VETO_ORDER, "config": cfg,
               "limits": {"MAX_COUNT": pr.MAX_COUNT},
               "formats": list(formats), "marg_keys": {str(k): list(v) for k, v in marg_keys.items()}}
    return _sha1(_canon(payload).encode("utf-8"))


def input_key(inp):
    """Content key of one engine input's veto-relevant fields, or None when it
    cannot be keyed safely (then it is always computed and never stored)."""
    sub = {k: inp.get(k) for k in KEY_FIELDS}
    if not isinstance(sub["veto_prefix"], list):
        sub["veto_prefix"] = None       # predictor.veto_distribution ignores a non-list prefix
    try:
        return _sha1(_canon(sub).encode("utf-8"))[:24]
    except (TypeError, ValueError):
        return None


def _q(p):
    return min(1000, max(0, round(p * 1000)))


def _pack_one(raw):
    pool, starter, masked, cold, warns, marg = raw
    m = "".join(_B32[_q(p) >> 5] + _B32[_q(p) & 31] for row in marg for p in row)
    return [round(starter * 1000), list(masked["a"]), list(masked["b"]),
            "".join(t for t in "ab" if cold.get(t)), list(warns), m]


def _unpack_one(pool, packed, width):
    st, ma, mb, cold, warns, m = packed
    vals = [(_B32.index(m[i]) << 5 | _B32.index(m[i + 1])) / 1000 for i in range(0, len(m), 2)]
    if len(vals) != width * len(pool):
        raise ValueError("marginal width")
    marg = [vals[i * width:(i + 1) * width] for i in range(len(pool))]
    return (list(pool), st / 1000, {"a": list(ma), "b": list(mb)},
            {"a": "a" in cold, "b": "b" in cold}, list(warns), marg)


class VetoCache:
    def __init__(self, path, fp, formats, marg_keys, enabled=True):
        self.path, self.fp, self.formats, self.marg_keys = path, fp, tuple(formats), marg_keys
        self.enabled = enabled
        self.entries = {}       # key -> packed entry (loaded or new)
        self.pools = {}         # pool id -> list of map names
        self.loaded = 0
        if enabled:
            self._load()

    def _warn(self, why):
        print(f"veto cache: ignoring {os.path.basename(self.path)} ({why})", file=sys.stderr)

    def _load(self):
        try:
            with open(self.path, "rb") as f:
                doc = json.loads(f.read().decode("utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as e:
            self._warn(f"unreadable: {type(e).__name__}")
            return
        if not (isinstance(doc, dict) and doc.get("version") == CACHE_VERSION
                and isinstance(doc.get("entries"), dict) and isinstance(doc.get("pools"), dict)):
            self._warn("unknown layout or version")
            return
        if doc.get("fingerprint") != self.fp:
            self._warn("model or CONFIG changed; rebuilding")
            return
        self.entries, self.pools = doc["entries"], doc["pools"]
        self.loaded = len(self.entries)

    def get(self, key):
        """{bo: raw} for a cached key, or None (a miss; a malformed entry is a miss)."""
        if not self.enabled or key is None:
            return None
        e = self.entries.get(key)
        if e is None:
            return None
        try:
            pool = self.pools[e[0]]
            out = {}
            for bo, packed in zip(self.formats, e[1:]):
                if packed is not None:
                    out[bo] = _unpack_one(pool, packed, len(self.marg_keys[bo]))
            if len(e) != 1 + len(self.formats):
                raise ValueError("format count")
            return out
        except (KeyError, IndexError, TypeError, ValueError):
            self.entries.pop(key, None)
            return None

    def put(self, key, res):
        if not self.enabled or key is None:
            return
        pool = next((r[0] for r in res.values()), [])
        pid = _sha1("\n".join(pool).encode("utf-8"))[:8]
        self.pools[pid] = list(pool)
        self.entries[key] = [pid] + [_pack_one(res[bo]) if bo in res else None for bo in self.formats]

    def save(self, used):
        """Write only the entries in `used` (prunes the rest), atomically."""
        if not self.enabled:
            return 0
        keep = sorted(k for k in used if k is not None and k in self.entries)
        pids = {self.entries[k][0] for k in keep}
        pools = {p: self.pools[p] for p in sorted(pids)}
        dump = lambda o: json.dumps(o, separators=(",", ":"), ensure_ascii=False)
        lines = [f'{{"version":{CACHE_VERSION},"fingerprint":{dump(self.fp)},"pools":{dump(pools)},"entries":{{']
        lines += [dump(k) + ":" + dump(self.entries[k]) + ("," if i < len(keep) - 1 else "")
                  for i, k in enumerate(keep)]
        lines.append("}}")
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"veto cache: not written ({e})", file=sys.stderr)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return 0
        return len(keep)
