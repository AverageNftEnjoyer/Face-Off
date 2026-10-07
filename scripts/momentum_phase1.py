"""Phase-1 measurement: map-1 dependence in completed BO3 series.

Point-in-time Elo only (same-day results are not visible). Round margins come
from the cached Liquipedia wikitext; a map with no round score is left out of
the margin split and is never imputed.

Run from the repo root:  python scripts/momentum_phase1.py
"""
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))

import backtest as bt  # noqa: E402
import collect_liquipedia as C  # noqa: E402
import lpfetch  # noqa: E402

# Regulation is first to 13. 13-3 is a 10-round margin; 13-11 is a 2-round margin.
BLOWOUT_MARGIN = 8     # 13-5 or more one-sided, includes 13-3
CLOSE_MARGIN = 2       # 13-11, 13-12 does not occur; overtime (16-14, ...) is close too


def clip(p):
    return min(1.0 - 1e-15, max(1e-15, p))


def largest_remainder(probs, places=2):
    """Percentages with `places` decimals that sum to exactly 100.00.

    probs are shares of one whole (they should already sum to 1). Returns a
    list of floats in percent units, e.g. 41.63.
    """
    factor = 10 ** places
    target = 100 * factor
    scaled = [p * target for p in probs]
    base = [math.floor(x + 1e-9) for x in scaled]
    left = int(round(target - sum(base)))
    order = sorted(range(len(probs)), key=lambda i: (scaled[i] - base[i], -i), reverse=True)
    for k in range(max(0, left)):
        base[order[k % len(base)]] += 1
    return [b / factor for b in base]


def pct(p):
    return f"{p * 100:.2f}%"


def pcts(probs):
    return "/".join(f"{x:.2f}" for x in largest_remainder(probs))


def map_rounds(blob):
    """{map, s1, s2, w} from one map parameter, or None. Scores stay None when
    the page recorded a winner but no round totals."""
    for _, body in C.find_templates(blob, "Map"):
        _, nm = C.split_params(body)
        name = nm.get("map", "").strip()
        if not name:
            return None
        name = C.MAP_NORMALIZE.get(name.lower(), name)
        s1, s2 = C._int(nm.get("score1")), C._int(nm.get("score2"))
        if s1 is None or s2 is None:
            def tot(team):
                keys = [k for k in nm if re.fullmatch(rf"(o\d+)?t{team}(t|ct)", k)]
                if not keys:
                    return None
                vals = [C._int(nm[k]) for k in keys]
                if any(v is None for v in vals):
                    return None
                return sum(vals)
            s1, s2 = tot(1), tot(2)
        w = C._int(nm.get("winner"))
        if w not in (1, 2):
            if s1 is None or s2 is None or s1 == s2:
                return None
            w = 1 if s1 > s2 else 2
        if s1 is not None and s2 is not None and s1 == s2:
            return None
        if s1 is not None and s2 is not None and ((s1 > s2) != (w == 1)):
            # winner flag and round totals disagree: keep the rounds, fix the winner
            w = 1 if s1 > s2 else 2
        return {"map": name, "s1": s1, "s2": s2, "w": w}
    return None


def index_scores(titles):
    """hltv id and (date, team, team) -> list of map dicts, from the cache only."""
    alias = C.cached_aliases()
    by_hltv, by_pair = {}, {}
    pages = lpfetch.cached_pages()
    found = 0
    for title in titles:
        rec = pages.get(title)
        if not rec:
            continue
        found += 1
        text = rec[1]
        for _, body in C.find_templates(text, "Match"):
            _, nm = C.split_params(body)
            k1, c1, _ = C.parse_opponent(nm.get("opponent1"))
            k2, c2, _ = C.parse_opponent(nm.get("opponent2"))
            if k1 != "TeamOpponent" or k2 != "TeamOpponent" or not c1 or not c2:
                continue
            date = C.parse_date(nm.get("date"))
            slots = sorted((k for k in nm if re.fullmatch(r"map\d+", k)), key=lambda k: int(k[3:]))
            maps = []
            for k in slots:
                pm = map_rounds(nm[k])
                if pm:
                    maps.append(pm)
            if not maps:
                continue
            hltv = (nm.get("hltv") or "").strip()
            a, b = alias.get(c1, c1), alias.get(c2, c2)
            if hltv and (hltv not in by_hltv or _score_count(maps) > _score_count(by_hltv[hltv])):
                by_hltv[hltv] = maps
            if date:
                key = (date, a, b)
                if key not in by_pair or _score_count(maps) > _score_count(by_pair[key]):
                    by_pair[key] = maps
    return by_hltv, by_pair, found


def _score_count(maps):
    return sum(1 for m in maps if m["s1"] is not None)


def hltv_of(m):
    src = m.get("source") or ""
    hit = re.search(r"hltv match (\d+)", src)
    return hit.group(1) if hit else None


def page_of(m):
    src = m.get("source") or ""
    if not src.startswith("liquipedia:"):
        return None
    return src[len("liquipedia:"):].split(" (hltv")[0]


def attach_scores(m, by_hltv, by_pair):
    maps = by_hltv.get(hltv_of(m) or "")
    if not maps:
        maps = by_pair.get((m["date"], m["team_a"], m["team_b"]))
    if not maps:
        swapped = by_pair.get((m["date"], m["team_b"], m["team_a"]))
        if swapped:
            maps = [{"map": x["map"], "s1": x["s2"], "s2": x["s1"],
                     "w": 1 if x["w"] == 2 else 2} for x in swapped]
    return maps


def margin_bin(winner_rounds, loser_rounds):
    """'blowout', 'close', 'mid', or None. Overtime finals count as close."""
    if winner_rounds is None or loser_rounds is None:
        return None
    if winner_rounds <= loser_rounds:
        return None
    margin = winner_rounds - loser_rounds
    overtime = winner_rounds > 13 or loser_rounds > 11
    if overtime or margin <= CLOSE_MARGIN:
        return "close"
    if margin >= BLOWOUT_MARGIN:
        return "blowout"
    return "mid"


def walk(matches):
    """Yield (match, elo_a, elo_b, n_a, n_b) with Elos from strictly earlier days."""
    ms = sorted(matches, key=lambda m: (m["date"], m.get("team_a", "")))
    h = bt.History()
    i = 0
    while i < len(ms):
        j = i
        while j < len(ms) and ms[j]["date"] == ms[i]["date"]:
            j += 1
        for m in ms[i:j]:
            a, b = m["team_a"], m["team_b"]
            yield m, h.elo[a], h.elo[b], h.n_prior(a), h.n_prior(b)
        for m in ms[i:j]:
            h.add(m)
        i = j


class Cell:
    def __init__(self):
        self.n = 0
        self.hits = 0
        self.base = 0.0

    def add(self, hit, expectation):
        self.n += 1
        self.hits += 1 if hit else 0
        self.base += expectation

    def obs(self):
        return self.hits / self.n if self.n else float("nan")

    def ind(self):
        return self.base / self.n if self.n else float("nan")

    def gap(self):
        return self.obs() - self.ind()


def role_of(elo_self, elo_opp):
    if elo_self > elo_opp:
        return "favorite"
    if elo_self < elo_opp:
        return "underdog"
    return "even"


def q_of(elo_self, elo_opp):
    return bt.map_prob_from_series(bt.elo_expected(elo_self, elo_opp), 3)


def series_ok(m):
    """A completed BO3 whose map list determines 2-0 / 2-1 / 1-2 / 0-2."""
    if m.get("best_of") != 3:
        return False
    maps = m.get("maps") or []
    if len(maps) < 2 or len(maps) > 3:
        return False
    a, b = m["team_a"], m["team_b"]
    if any(x.get("winner") not in (a, b) for x in maps):
        return False
    wa = sum(1 for x in maps if x["winner"] == a)
    wb = len(maps) - wa
    if max(wa, wb) != 2:
        return False
    winner = a if wa == 2 else b
    return m.get("winner") == winner


def main():
    matches = bt.load_matches(os.path.join(ROOT, "data", "matches.json")) if hasattr(bt, "load_matches") else None
    if matches is None:
        import json
        with open(os.path.join(ROOT, "data", "matches.json"), encoding="utf-8") as f:
            matches = json.load(f)

    bo3 = [m for m in matches if m.get("best_of") == 3]
    with_maps = [m for m in bo3 if m.get("maps")]
    usable = [m for m in bo3 if series_ok(m)]
    print("DATA CHECK")
    print(f"  matches {len(matches)}  bo3 {len(bo3)}  bo3 with any maps {len(with_maps)}")
    print(f"  bo3 whose map winners determine the series {len(usable)}")
    print(f"  bo3 with empty map list {sum(1 for m in bo3 if not m.get('maps'))}")
    lens = {}
    for m in usable:
        lens[len(m["maps"])] = lens.get(len(m["maps"]), 0) + 1
    print(f"  usable map-count: {lens}")

    titles = {page_of(m) for m in usable if page_of(m)}
    print(f"  distinct liquipedia pages referenced: {len(titles)}")
    by_hltv, by_pair, pages_found = index_scores(titles)
    print(f"  pages found in cache: {pages_found}  hltv keys with maps: {len(by_hltv)}")

    scored = 0
    rows = []
    for m, ea, eb, na, nb in walk(matches):
        if not series_ok(m):
            continue
        sm = attach_scores(m, by_hltv, by_pair)
        m1_score = None
        if sm and sm[0]["map"] == m["maps"][0]["map"] and sm[0]["s1"] is not None:
            s1, s2 = sm[0]["s1"], sm[0]["s2"]
            if m["maps"][0]["winner"] == m["team_a"]:
                m1_score = (s1, s2)
            else:
                m1_score = (s2, s1)
            scored += 1
        maps = m["maps"]
        w1 = maps[0]["winner"]
        l1 = m["team_b"] if w1 == m["team_a"] else m["team_a"]
        w2 = maps[1]["winner"]
        went3 = len(maps) == 3
        w3 = maps[2]["winner"] if went3 else None
        elo = {m["team_a"]: ea, m["team_b"]: eb}
        hist = {m["team_a"]: na, m["team_b"]: nb}
        rows.append({
            "m": m, "elo": elo, "hist": hist,
            "w1": w1, "l1": l1, "w2": w2, "w3": w3, "went3": went3,
            "series_winner": m["winner"],
            "m1": m1_score,
            "date": m["date"],
        })
    print(f"  usable series with a numeric map-1 round score: {scored} / {len(rows)}")
    if scored == 0:
        print("  MARGIN: no round totals in the cached pages. Margin split cannot be computed.")

    # ---- cells ----
    # series | won map 1, and comeback | lost map 1, overall and by role of THAT team
    series_cells = {k: Cell() for k in ("all", "favorite", "underdog", "even")}
    back_cells = {k: Cell() for k in ("all", "favorite", "underdog", "even")}
    # map 2 win, conditional on map 1, by the role of the team we are asking about
    m2_win = {k: Cell() for k in ("favorite", "underdog", "even")}
    m2_loss = {k: Cell() for k in ("favorite", "underdog", "even")}
    # map 3, from the map-2 winner's side and the map-2 loser's side
    m3_after_w2 = {k: Cell() for k in ("favorite", "underdog", "even")}
    m3_after_l2 = {k: Cell() for k in ("favorite", "underdog", "even")}
    # margin of the map-1 loss: does the loser win map 2?
    margin_m2 = {k: Cell() for k in ("close", "mid", "blowout")}
    margin_m2_role = {(role, b): Cell() for role in ("favorite", "underdog") for b in ("close", "blowout")}
    margin_back = {k: Cell() for k in ("close", "mid", "blowout")}

    for r in rows:
        elo, hist = r["elo"], r["hist"]
        for team, other, won_m1 in ((r["w1"], r["l1"], True), (r["l1"], r["w1"], False)):
            q = q_of(elo[team], elo[other])
            role = role_of(elo[team], elo[other])
            won_series = r["series_winner"] == team
            won_m2 = r["w2"] == team
            if won_m1:
                series_cells["all"].add(won_series, q * (2.0 - q))
                series_cells[role].add(won_series, q * (2.0 - q))
                m2_win[role].add(won_m2, q)
            else:
                back_cells["all"].add(won_series, q * q)
                back_cells[role].add(won_series, q * q)
                m2_loss[role].add(won_m2, q)
            if r["went3"]:
                won_m3 = r["w3"] == team
                won_m2_flag = r["w2"] == team
                if won_m2_flag:
                    m3_after_w2[role].add(won_m3, q)
                else:
                    m3_after_l2[role].add(won_m3, q)

        # margin is from the map-1 winner's round totals (winner, loser)
        if r["m1"]:
            wr, lr = r["m1"]
            b = margin_bin(wr, lr)
            if b:
                loser = r["l1"]
                q_l = q_of(elo[loser], elo[r["w1"]])
                role_l = role_of(elo[loser], elo[r["w1"]])
                margin_m2[b].add(r["w2"] == loser, q_l)
                margin_back[b].add(r["series_winner"] == loser, q_l * q_l)
                if role_l in ("favorite", "underdog") and b in ("close", "blowout"):
                    margin_m2_role[(role_l, b)].add(r["w2"] == loser, q_l)

    def show(title, cells, keys):
        print()
        print(title)
        print(f"  {'slice':<12} {'n':>6} {'observed':>10} {'independ.':>10} {'gap':>9}")
        for k in keys:
            c = cells[k]
            if not c.n:
                print(f"  {k:<12} {0:6d} {'—':>10} {'—':>10} {'—':>9}")
                continue
            print(f"  {k:<12} {c.n:6d} {pct(c.obs()):>10} {pct(c.ind()):>10} {c.gap() * 100:+8.2f}pp")

    print()
    print("PHASE 1  (one row per completed BO3; favorite = higher pre-match Elo)")
    show("When a team wins map 1, how often do they win the series?",
         series_cells, ("all", "favorite", "underdog", "even"))
    show("When a team goes down 0-1, how often do they win the series 2-1?",
         back_cells, ("all", "favorite", "underdog", "even"))
    show("P(win map 2 | won map 1) by that team's role",
         m2_win, ("favorite", "underdog", "even"))
    show("P(win map 2 | lost map 1) by that team's role",
         m2_loss, ("favorite", "underdog", "even"))
    show("Decider: P(win map 3 | won map 2) at 1-1, by that team's role",
         m3_after_w2, ("favorite", "underdog", "even"))
    show("Decider: P(win map 3 | lost map 2) at 1-1, by that team's role",
         m3_after_l2, ("favorite", "underdog", "even"))
    show("After a map-1 loss, P(win map 2) by map-1 margin",
         margin_m2, ("close", "mid", "blowout"))
    show("After a map-1 loss, P(win series 2-1) by map-1 margin",
         margin_back, ("close", "mid", "blowout"))
    print()
    print("Map-2 response after a map-1 loss, favorite and underdog x margin")
    print(f"  {'slice':<24} {'n':>6} {'observed':>10} {'independ.':>10} {'gap':>9}")
    for role in ("favorite", "underdog"):
        for b in ("close", "blowout"):
            c = margin_m2_role[(role, b)]
            label = f"{role} lost {b}"
            if not c.n:
                print(f"  {label:<24} {0:6d}")
                continue
            print(f"  {label:<24} {c.n:6d} {pct(c.obs()):>10} {pct(c.ind()):>10} {c.gap() * 100:+8.2f}pp")

    # dependence beyond the strength mix: residual(win map1) - residual(lose map1)
    print()
    print("Dependence (map-2 residual after a map-1 win, minus residual after a map-1 loss)")
    for role in ("favorite", "underdog"):
        a, b = m2_win[role], m2_loss[role]
        if a.n and b.n:
            extra = a.gap() - b.gap()
            print(f"  {role:<12} {extra * 100:+.2f}pp   "
                  f"(win-map1 residual {a.gap() * 100:+.2f}pp, n={a.n}; "
                  f"loss-map1 residual {b.gap() * 100:+.2f}pp, n={b.n})")

    # chronological split for a later model: first 60% of usable series, no day cut
    ordered = sorted(rows, key=lambda r: r["date"])
    # match backtest: eval rows are min history 5; split those
    ev = [r for r in ordered if min(r["hist"].values()) >= 5 and role_of(
        r["elo"][r["m"]["team_a"]], r["elo"][r["m"]["team_b"]]) != "even"]
    k = max(1, min(len(ev) - 1, int(len(ev) * 0.6)))
    while 0 < k < len(ev) and ev[k]["date"] == ev[k - 1]["date"]:
        k += 1
    train, test = ev[:k], ev[k:]
    print()
    print(f"SPLIT  min-history-5, strict Elo edge: train {len(train)} ({train[0]['date']}..{train[-1]['date']})  "
          f"test {len(test)} ({test[0]['date']}..{test[-1]['date']})")

    def residuals(sample):
        acc = {("favorite", "win"): Cell(), ("favorite", "loss"): Cell(),
               ("underdog", "win"): Cell(), ("underdog", "loss"): Cell(),
               ("favorite", "m3w"): Cell(), ("favorite", "m3l"): Cell(),
               ("underdog", "m3w"): Cell(), ("underdog", "m3l"): Cell()}
        for r in sample:
            elo = r["elo"]
            for team, other, won_m1 in ((r["w1"], r["l1"], True), (r["l1"], r["w1"], False)):
                role = role_of(elo[team], elo[other])
                if role == "even":
                    continue
                q = q_of(elo[team], elo[other])
                key = (role, "win" if won_m1 else "loss")
                acc[key].add(r["w2"] == team, q)
                if r["went3"]:
                    mk = (role, "m3w" if r["w2"] == team else "m3l")
                    acc[mk].add(r["w3"] == team, q)
        return acc

    tr = residuals(train)
    print("TRAIN residuals (these are the only numbers a model may use)")
    for key in (("favorite", "win"), ("favorite", "loss"), ("underdog", "win"), ("underdog", "loss"),
                ("favorite", "m3w"), ("favorite", "m3l"), ("underdog", "m3w"), ("underdog", "m3l")):
        c = tr[key]
        if c.n:
            print(f"  {key[0]:<10} {key[1]:<5} n={c.n:<5} obs {pct(c.obs())}  q {pct(c.ind())}  "
                  f"residual {c.gap() * 100:+.2f}pp")

    # scoreline shift on the TEST set, Elo-q binomial vs conditional residuals from TRAIN
    dw = {role: tr[(role, "win")].gap() for role in ("favorite", "underdog")}
    dl = {role: tr[(role, "loss")].gap() for role in ("favorite", "underdog")}
    d3w = {role: tr[(role, "m3w")].gap() for role in ("favorite", "underdog")}
    d3l = {role: tr[(role, "m3l")].gap() for role in ("favorite", "underdog")}

    def binomial(q):
        p20 = q * q
        p02 = (1 - q) * (1 - q)
        reach = 1.0 - p20 - p02
        return (p20, reach * q, reach * (1 - q), p02)

    def conditional(q, role, use_m3):
        p2w = clip(q + dw[role])
        p2l = clip(q + dl[role])
        p3w = clip(q + (d3w[role] if use_m3 else 0.0))
        p3l = clip(q + (d3l[role] if use_m3 else 0.0))
        p20 = q * p2w
        p02 = (1 - q) * (1 - p2l)
        p21 = q * (1 - p2w) * p3l + (1 - q) * p2l * p3w
        p12 = q * (1 - p2w) * (1 - p3l) + (1 - q) * p2l * (1 - p3w)
        s = p20 + p21 + p12 + p02
        return (p20 / s, p21 / s, p12 / s, p02 / s)

    def actual_dist(rows_s):
        labels = ("2-0", "2-1", "1-2", "0-2")
        cnt = {s: 0 for s in labels}
        n = 0
        for r in rows_s:
            sc = bt.actual_scoreline(r["m"])
            if sc not in cnt:
                continue
            cnt[sc] += 1
            n += 1
        return n, [cnt[s] / n for s in labels]

    def logloss(rows_s, use_m3, constrain):
        labels = {"2-0": 0, "2-1": 1, "1-2": 2, "0-2": 3}
        lb = lc = 0.0
        shift = 0.0
        n = 0
        sum_b = [0.0, 0.0, 0.0, 0.0]
        sum_c = [0.0, 0.0, 0.0, 0.0]
        for r in rows_s:
            m = r["m"]
            a = m["team_a"]
            q = q_of(r["elo"][a], r["elo"][m["team_b"]])
            role = role_of(r["elo"][a], r["elo"][m["team_b"]])
            if role == "even":
                continue
            y = labels[bt.actual_scoreline(m)]
            b = binomial(q)
            if constrain:
                p_series = b[0] + b[1]
                c = conditional_constrained(p_series, role, use_m3)
            else:
                c = conditional(q, role, use_m3)
            lb -= math.log(clip(b[y]))
            lc -= math.log(clip(c[y]))
            shift += max(abs(c[i] - b[i]) for i in range(4))
            for i in range(4):
                sum_b[i] += b[i]
                sum_c[i] += c[i]
            n += 1
        return n, lb / n, lc / n, shift / n, [x / n for x in sum_b], [x / n for x in sum_c]

    def conditional_constrained(p_series, role, use_m3):
        """Same series win probability as the binomial; only the scoreline shape moves."""
        lo, hi = 1e-6, 1.0 - 1e-6
        for _ in range(60):
            mid = (lo + hi) / 2.0
            s = conditional(mid, role, use_m3)
            if s[0] + s[1] < p_series:
                lo = mid
            else:
                hi = mid
        return conditional((lo + hi) / 2.0, role, use_m3)

    n_act, act = actual_dist(test)
    print()
    print(f"TEST actual scoreline distribution n={n_act}: {pcts(act)}")
    te = residuals(test)
    print("TEST residuals (not used to fit; replication check)")
    for key in (("favorite", "win"), ("favorite", "loss"), ("underdog", "win"), ("underdog", "loss"),
                ("favorite", "m3w"), ("favorite", "m3l")):
        c = te[key]
        if c.n:
            print(f"  {key[0]:<10} {key[1]:<5} n={c.n:<5} obs {pct(c.obs())}  q {pct(c.ind())}  "
                  f"residual {c.gap() * 100:+.2f}pp")

    for constrain, tag in ((False, "unconstrained"), (True, "series-prob held fixed")):
        for name, flag in (("map2 only", False), ("map2 + decider", True)):
            n, lb, lc, shift, mb, mc = logloss(test, flag, constrain)
            print()
            print(f"TEST {tag}, {name}: n={n}")
            print(f"  log loss binomial {lb:.6f}  conditional {lc:.6f}  delta {lc - lb:+.6f}")
            print(f"  mean max |component shift| {shift * 100:.2f}pp")
            print(f"  mean binomial    {pcts(mb)}")
            print(f"  mean conditional {pcts(mc)}")
    n, lb, lc, shift, mb, mc = logloss(train, False, True)
    print()
    print(f"TRAIN series-prob held fixed, map2 only: n={n}")
    print(f"  log loss binomial {lb:.6f}  conditional {lc:.6f}  delta {lc - lb:+.6f}")
    print(f"  mean max |component shift| {shift * 100:.2f}pp")


if __name__ == "__main__":
    main()
