#!/usr/bin/env python3
"""
Fetch the viewer's visual assets and roster data from Liquipedia, politely.

  * team pages (wikitext): logo file names (light + dark), location, active
    CS2 roster (id, real name, flag, IGL / coach), stand-ins table rows
  * map pages: the map's infobox image
  * Template:Flag/<cc>: flag image per nationality
  * imageinfo with iiurlwidth: Liquipedia serves a server-side thumbnail URL,
    so we download small images (logos ~160px, maps ~640px, flags ~36px)

All API calls go through data/lpfetch.py (descriptive UA, >= 2.5 s apart,
cached under data/raw/liquipedia/). Image downloads are cached under
data/raw/liquipedia_media/ and spaced >= 1.5 s apart. Writes
viewer/assets.json (no image bytes; the build step embeds them).

Logos and map images are the property of their owners (team trademarks,
Valve). Used here in a non-commercial fan page with attribution to Liquipedia.

USAGE:
    python viewer/fetch_assets.py [--offline]
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, ROOT)

import collect_liquipedia as C  # noqa: E402
import lpfetch  # noqa: E402

OFFLINE = "--offline" in sys.argv
MEDIA = os.path.join(ROOT, "data", "raw", "liquipedia_media")
EVENT = "ESL/Pro League/Season 24"
# dataset map name -> Liquipedia map page, only where the two differ
# (any other map is looked up under its own name)
MAP_PAGES = {"Dust2": "Dust II"}
# square icons whose names don't follow the " full" pattern (found on Liquipedia
# commons; each is the same mark as the team's current full logo). Vitality is
# left on its 2026 wordmark: the only square icons predate the 2026 rebrand.
ICON_OVERRIDES = {
    "FURIA": "FURIA Esports allmode.png",
    "Team Falcons": "Team Falcons 2022 allmode.png",
    "Legacy": "Legacy allmode.png",
    # Liquipedia has no text-free icon under the infobox file name for these two;
    # use their emblem files (no wordmark) instead
    "G2 Esports": "G2 Esports 2020 lightmode.png",
    "Team Vitality": "Team Vitality 2023 darkmode.png",
    "HEROIC": "HEROIC 2024 allmode.png",
    "Luminosity Gaming": "Luminosity Gaming 2018 allmode.png",
    "BC.Game Esports": "BC.Game Esports nov 2025 allmode.png",
    "Nemiga Gaming": "Nemiga Gaming 2020logo.png",
}
_last_dl = [0.0]


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def strip_markup(s):
    s = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", s or "", flags=re.S)
    s = re.sub(r"\{\{abbr\|([^|}]*)\|[^}]*\}\}", r"\1", s, flags=re.I)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    return s.replace("&nbsp;", " ").strip()


def infobox(txt, name="Infobox team"):
    # editors write both "Infobox team" and "Infobox Team"
    for variant in (name, name[:8] + name[8:].capitalize(), name[:8] + name[8:].lower()):
        for _, body in C.find_templates(txt, variant):
            return C.split_params(body)[1]
    return {}


def active_roster(txt):
    """Players + coach from the first `{{Squad|status=active` block (the CS2 tab
    comes first on Liquipedia team pages)."""
    out = []
    for _, body in C.find_templates(txt, "Squad"):
        pos, nm = C.split_params(body)
        if nm.get("status", "").lower() != "active":
            continue
        for _, pb in C.find_templates(body, "Person"):
            _, p = C.split_params(pb)
            out.append({
                "id": strip_markup(p.get("id", "")),
                "name": strip_markup(p.get("name", "")),
                "flag": p.get("flag", "").strip().lower(),
                "igl": p.get("igl", "").strip().lower() in ("y", "yes", "true"),
                "role": strip_markup(p.get("role", "")),
                "joined": strip_markup(p.get("joindate", ""))[:10],
            })
        break
    return out


def team_record(title, txt):
    ib = infobox(txt)
    return {
        "page": title,
        "logo": ib.get("image", "").strip(),
        "logo_dark": ib.get("imagedark", "").strip() or ib.get("image", "").strip(),
        "location": strip_markup(ib.get("location", "")),
        "region": ib.get("region", "").strip(),
        "roster": active_roster(txt),
    }


def cached_event_text(title=EVENT):
    """Tournament pages were cached by data/collect_liquipedia.py in batches of 8
    from lp_titles_selected.txt; read them back the same way (no network)."""
    titles = [l.strip() for l in open(os.path.join(ROOT, "data", "raw", "lp_titles_selected.txt"),
                                      encoding="utf-8") if l.strip()]
    for b in batched(titles, 8):
        if title in b:
            return lpfetch.wikitext(b, offline=True)[title]
    raise SystemExit(f"{title} is not in lp_titles_selected.txt")


def event_teams():
    txt = cached_event_text()
    codes = set()
    for _, body in C.find_templates(txt, "Match"):
        _, nm = C.split_params(body)
        for k in ("opponent1", "opponent2"):
            kind, code, _ = C.parse_opponent(nm.get(k))
            if kind == "TeamOpponent" and code:
                codes.add(code)
    alias = C.cached_aliases()
    return sorted(alias.get(c, c) for c in codes)


def cached_imageinfo(width):
    """Every cached imageinfo response at this thumbnail width, newest fetch
    last. Offline runs read files from these whatever batch they were fetched
    in (a shorter team list shifts the batches of 50, and so the cache keys)."""
    out = []
    if os.path.isdir(lpfetch.CACHE):
        for fn in os.listdir(lpfetch.CACHE):
            with open(os.path.join(lpfetch.CACHE, fn), encoding="utf-8") as f:
                rec = json.load(f)
            url = rec.get("url", "")
            if "prop=imageinfo" in url and f"iiurlwidth={width}&" in url + "&":
                out.append(rec)
    return [r["response"] for r in sorted(out, key=lambda r: r.get("fetched", ""))]


def imageinfo(files, width):
    """{file title: thumb url} for File: names, one API call per 50."""
    out = {}
    for b in batched(sorted(set(f for f in files if f)), 50):
        resp = lpfetch.query({"action": "query", "prop": "imageinfo", "iiprop": "url",
                              "iiurlwidth": str(width), "redirects": "1",
                              "titles": "|".join("File:" + f for f in b)}, offline=OFFLINE)
        if resp is None:
            # offline batch miss: answer from every cached response at this width
            seen = {}
            for r in cached_imageinfo(width):
                seen.update(_imageinfo_urls(r))
            missing = [f for f in b if f not in seen]
            if missing:
                raise SystemExit("offline: imageinfo not cached for %r" % missing)
            out.update({f: seen[f] for f in b if seen[f]})
            continue
        out.update({k: v for k, v in _imageinfo_urls(resp).items() if v})
    return out


def _imageinfo_urls(resp):
    """{file title (as requested and as resolved): thumb url, or None for a
    file the response says does not exist} from one imageinfo response."""
    out = {}
    q = resp.get("query", {})
    norm = {n["to"]: n["from"] for n in q.get("normalized", [])}
    for p in q.get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        title = norm.get(p["title"], p["title"])
        for t in (title, p["title"]):
            if url or not out.get(t[len("File:"):]):
                out[t[len("File:"):]] = url
    return out


def download(url):
    os.makedirs(MEDIA, exist_ok=True)
    ext = os.path.splitext(url.split("?")[0])[1].lower() or ".png"
    path = os.path.join(MEDIA, hashlib.sha1(url.encode()).hexdigest()[:16] + ext)
    if os.path.exists(path):
        return path
    if OFFLINE:
        raise SystemExit("offline: media not cached: " + url)
    wait = 1.5 - (time.time() - _last_dl[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": lpfetch.UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    _last_dl[0] = time.time()
    with open(path, "wb") as f:
        f.write(data)
    return path


def main():
    import backtest as bt
    from datetime import timedelta

    # teams: everyone in the event + the strongest active teams the viewer offers
    h = bt.History()
    ms = bt.load_matches()
    for m in sorted(ms, key=lambda m: m["date"]):
        h.add(m)
    as_of = max(bt._d(m["date"]) for m in ms) + timedelta(days=1)
    active = [t for t, g in h.games.items() if g and g[-1]["date"] >= as_of - timedelta(days=45) and len(g) >= 15]
    top = sorted(active, key=lambda t: -h.elo[t])[:16]
    import events as E
    evs = E.discover()
    teams = sorted(set(E.event_teams(evs)) | set(top) | set(E.vrs_top_teams(set(h.games))))

    texts = {}
    for b in batched(teams, 8):
        texts.update(lpfetch.wikitext(b, offline=OFFLINE))
    recs = {t: team_record(t, texts[t]) for t in teams if texts.get(t)}

    # every map that appears in the results or in a tournament's map pool gets its
    # Liquipedia page (picture, location): a map entering the pool is picked up on
    # the next refresh. Only names that differ from the page title need MAP_PAGES.
    seen = {x["map"] for m in ms for x in m.get("maps", []) if x.get("map")}
    seen |= {mp for e in evs for mp in e.get("pool", [])}
    map_pages = {mp: MAP_PAGES.get(mp, mp) for mp in sorted(seen | set(MAP_PAGES))}
    maps_txt = lpfetch.wikitext(list(map_pages.values()), offline=OFFLINE, strict=False)
    map_files, map_info = {}, {}
    for mp, page in map_pages.items():
        ib = infobox(maps_txt.get(page) or "", "Infobox map")
        f = ib.get("image", "").strip()
        if f:
            map_files[mp] = f
            map_info[mp] = {"theme": strip_markup(ib.get("theme", "")),
                            "location": strip_markup(ib.get("location", ""))}

    # flags are rendered by a Lua module, so Template:Flag/<cc> has no wikitext;
    # Liquipedia's flag files follow the "<Cc> hd.png" naming instead
    flags = sorted({p["flag"] for r in recs.values() for p in r["roster"] if p["flag"]}
                   | {cc for e in evs for v in e["participants"].values() for cc in v["flags"].values()})
    flag_files = {c: c.capitalize() + " hd.png" for c in flags}

    # square icons usually share the infobox file name minus " full"
    # ("Team Spirit 2022 full lightmode.png" -> "Team Spirit 2022 lightmode.png");
    # fall back to the full wordmark when no icon exists
    for r in recs.values():
        r["icon"] = r["logo"].replace(" full", "")
        r["icon_dark"] = r["logo_dark"].replace(" full", "")
        if r["page"] in ICON_OVERRIDES:
            r["icon"] = r["icon_dark"] = ICON_OVERRIDES[r["page"]]
    logo_urls = imageinfo([r[k] for r in recs.values()
                           for k in ("logo", "logo_dark", "icon", "icon_dark") if r[k]], 160)
    ev_urls = imageinfo([e["logo_file"] for e in evs if e["logo_file"]], 240)
    for r in recs.values():
        for k in ("icon", "icon_dark"):
            if r[k] not in logo_urls:
                r[k] = r["logo" if k == "icon" else "logo_dark"]
    map_urls = imageinfo(list(map_files.values()), 640)
    flag_urls = imageinfo(list(flag_files.values()), 36)

    def fetch(url):
        return os.path.relpath(download(url), ROOT).replace(os.sep, "/") if url else None

    for r in recs.values():
        r["logo_path"] = fetch(logo_urls.get(r["icon"]))
        r["logo_dark_path"] = fetch(logo_urls.get(r["icon_dark"]))
    maps = {mp: {"file": f, "path": fetch(map_urls.get(f)), **map_info[mp]} for mp, f in map_files.items()}
    flag_paths = {c: fetch(flag_urls.get(f)) for c, f in flag_files.items()}

    out = {"fetched_for": as_of.isoformat(), "event": EVENT,
           "event_logos": {e["slug"]: fetch(ev_urls.get(e["logo_file"])) for e in evs},
           "teams": recs,
           "maps": maps, "flags": flag_paths}
    with open(os.path.join(HERE, "assets.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    missing = [t for t, r in recs.items() if not r["logo_path"]]
    print(f"teams {len(recs)} (no logo: {missing}), roster sizes "
          f"{sorted(len(r['roster']) for r in recs.values())}, maps {sorted(k for k, v in maps.items() if v['path'])}, "
          f"flags {len([p for p in flag_paths.values() if p])}/{len(flags)}")


if __name__ == "__main__":
    main()
