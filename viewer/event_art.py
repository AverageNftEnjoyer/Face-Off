#!/usr/bin/env python3
"""
Generated tournament key art: one small SVG per event, styled from its logo.

The page shows each event's logo over a darkened piece of art (list thumbnails,
the preview banner, the tournament card). Events without a hand-supplied image
(viewer/media/events/<slug>.<ext>) get this generated artwork instead:

  * palette   from the event logo's pixels (team_colors' stdlib PNG decoder):
              hue families of its saturated pixels, ranked by area;
              the dominant one, a secondary (the logo's own second
              hue, else a seeded hue shift of the first) and a dark base in the
              same hue. A logo with no real hue (white / grey marks) starts from
              the site accent #2FD4A2 turned by a seeded hue.
  * layout    seeded from the slug (same slug -> same art): beam angle and count,
              glow position, background pattern (hex grid, circuit traces,
              halftone dots or a stage floor grid), accent slashes and sparks.

Pure string building, no rasters and no filters (glows are radial gradients),
so every SVG is a few KB. build_viewer.py writes them content-hashed beside the
page and inlines their URLs as IMG.evgen.

USAGE (repo root, preview one event):  python viewer/event_art.py SLUG LOGO.png OUT.svg
"""

import colorsys
import hashlib
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402,F401  (puts every code folder on sys.path)

import team_colors as TC  # noqa: E402

W, H = 1600, 600
ACCENT = "#2FD4A2"
# hue turns (of a full circle) for logos with no hue of their own: the site accent first,
# then the blues, violets, reds and ambers of event key art
NEUTRAL_TURNS = (0.0, 0.08, 0.16, 0.26, 0.36, 0.47, 0.56, 0.62)
PATTERNS = ("hex", "circuit", "dots", "floor")


def _rgb(hx):
    return tuple(int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _hx(r, g, b):
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c * 255))) for c in (r, g, b))


def _hsv(hx):
    return colorsys.rgb_to_hsv(*_rgb(hx))


def _from_hsv(h, s, v):
    return _hx(*colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v))))


def _logo_hues(path):
    """The logo's saturated colours, most common hue family first (hex). Unlike
    team_colors.logo_colors, a small accent (a gold crest on a white wordmark) counts."""
    if not (path and path.lower().endswith(".png") and os.path.exists(path)):
        return []
    try:
        img = TC.read_png(path)
    except Exception:  # a logo we cannot decode just gets the fallback palette
        return []
    if not img:
        return []
    fam, total = {}, 0
    for r, g, b, al in img[2]:
        if al < 160:
            continue
        total += 1
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if s >= 0.25 and v >= 0.3:
            acc = fam.setdefault(int(h * 18) % 18, [0, 0, 0, 0])
            acc[0] += 1; acc[1] += r; acc[2] += g; acc[3] += b
    ranked = sorted((a for a in fam.values() if a[0] >= 0.02 * total), key=lambda a: -a[0])
    return ["#%02x%02x%02x" % (a[1] // a[0], a[2] // a[0], a[3] // a[0]) for a in ranked]


def palette(logo_path, rng):
    """(primary, secondary, base, deep) hex colours for the art of one event."""
    sat = _logo_hues(logo_path)
    if sat:
        h, s, v = _hsv(sat[0])
    else:
        h, s, v = _hsv(ACCENT)
        h += rng.choice(NEUTRAL_TURNS) + rng.uniform(-0.03, 0.03)
    # the primary glows: keep it saturated and bright whatever the logo's shade
    prim = _from_hsv(h, max(s, 0.7), max(v, 0.85))
    if len(sat) > 1 and abs(((_hsv(sat[1])[0] - h + 0.5) % 1) - 0.5) > 0.06:
        h2, s2, v2 = _hsv(sat[1])
    else:
        h2, s2, v2 = h + rng.choice((-1, 1)) * rng.uniform(0.07, 0.14), s, v
    sec = _from_hsv(h2, max(s2, 0.6), max(v2, 0.8))
    base = _from_hsv(h, 0.8, 0.24)
    deep = _from_hsv(h2, 0.7, 0.08)
    return prim, sec, base, deep


def _f(x):
    return f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def _pattern(kind, rng, sec):
    """(<pattern> def, its id) for the faint background texture."""
    if kind == "hex":
        r = rng.choice((26, 32, 38))
        w, hh = r * 3, r * 1.732
        d = (f"M0 {_f(hh/2)}L{_f(r/2)} 0H{_f(r*1.5)}L{_f(r*2)} {_f(hh/2)}L{_f(r*1.5)} {_f(hh)}H{_f(r/2)}Z"
             f"M{_f(r*2)} {_f(hh/2)}H{_f(w)}")
        body = f'<path d="{d}" fill="none" stroke="{sec}" stroke-width="1.2"/>'
        size = (w, hh)
    elif kind == "circuit":
        s = rng.choice((90, 110))
        body = (f'<path d="M0 {s*.3:.0f}H{s*.35:.0f}L{s*.55:.0f} {s*.5:.0f}V{s:.0f}M{s*.55:.0f} {s*.5:.0f}H{s:.0f}'
                f'M{s*.2:.0f} 0V{s*.15:.0f}L{s*.35:.0f} {s*.3:.0f}" fill="none" stroke="{sec}" stroke-width="1.4"/>'
                f'<circle cx="{s*.55:.0f}" cy="{s*.5:.0f}" r="3" fill="{sec}"/>'
                f'<circle cx="{s*.2:.0f}" cy="{s*.15:.0f}" r="2.5" fill="none" stroke="{sec}" stroke-width="1.2"/>')
        size = (s, s)
    elif kind == "dots":
        s = rng.choice((14, 18))
        body = f'<circle cx="{s/2:.0f}" cy="{s/2:.0f}" r="{s*.16:.1f}" fill="{sec}"/>'
        size = (s, s)
    else:
        return "", None
    tilt = rng.choice((0, 0, 30, -30)) if kind != "hex" else 0
    tr = f' patternTransform="rotate({tilt})"' if tilt else ""
    return (f'<pattern id="p" width="{_f(size[0])}" height="{_f(size[1])}" patternUnits="userSpaceOnUse"{tr}>'
            f'{body}</pattern>'), "p"


def render(slug, logo_path=None):
    """The SVG text of one event's art (deterministic per slug)."""
    rng = random.Random(int(hashlib.sha256(slug.encode()).hexdigest()[:12], 16))
    prim, sec, base, deep = palette(logo_path, rng)
    kind = PATTERNS[rng.randrange(len(PATTERNS))]
    side = rng.choice((-1, 1))                    # beams lean left or right
    ang = side * rng.uniform(22, 38)
    gx, gy = rng.uniform(0.5, 0.66) * W, rng.uniform(0.42, 0.56) * H
    out = []
    a = out.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice">')
    a("<defs>")
    a(f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{base}"/>'
      f'<stop offset=".55" stop-color="{deep}"/><stop offset="1" stop-color="#030409"/></linearGradient>')
    a(f'<radialGradient id="gl" cx="{gx/W:.3f}" cy="{gy/H:.3f}" r=".42">'
      f'<stop offset="0" stop-color="{prim}" stop-opacity=".9"/><stop offset=".3" stop-color="{prim}" stop-opacity=".38"/>'
      f'<stop offset="1" stop-color="{prim}" stop-opacity="0"/></radialGradient>')
    hx2, hy2 = rng.choice((0.12, 0.88)) * W, rng.choice((0.1, 0.9)) * H
    a(f'<radialGradient id="g2" cx="{hx2:.0f}" cy="{hy2:.0f}" r="520" gradientUnits="userSpaceOnUse">'
      f'<stop offset="0" stop-color="{sec}" stop-opacity=".45"/><stop offset="1" stop-color="{sec}" stop-opacity="0"/></radialGradient>')
    # beam fill: bright core along the beam, fading toward both ends
    a(f'<linearGradient id="bm" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{prim}" stop-opacity="0"/>'
      f'<stop offset=".5" stop-color="{prim}" stop-opacity=".9"/><stop offset="1" stop-color="{prim}" stop-opacity="0"/></linearGradient>')
    a(f'<linearGradient id="bs" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{sec}" stop-opacity="0"/>'
      f'<stop offset=".45" stop-color="{sec}" stop-opacity=".9"/><stop offset="1" stop-color="{sec}" stop-opacity="0"/></linearGradient>')
    a(f'<linearGradient id="ln" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/>'
      f'<stop offset=".5" stop-color="#fff" stop-opacity=".95"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>')
    pat, pid = _pattern(kind, rng, sec)
    a(pat)
    # the texture shows in a soft ring around the glow and fades out toward the edges
    a(f'<radialGradient id="mk" cx="{gx:.0f}" cy="{gy:.0f}" r="{rng.uniform(560, 760):.0f}" gradientUnits="userSpaceOnUse">'
      f'<stop offset="0" stop-color="#fff" stop-opacity=".25"/><stop offset=".35" stop-color="#fff" stop-opacity="1"/>'
      f'<stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>')
    a('<mask id="m"><rect width="100%" height="100%" fill="url(#mk)"/></mask>')
    a('<radialGradient id="vg" cx=".5" cy=".5" r=".75"><stop offset=".55" stop-color="#000" stop-opacity="0"/>'
      '<stop offset="1" stop-color="#000" stop-opacity=".7"/></radialGradient>')
    a("</defs>")
    a(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
    a(f'<rect width="{W}" height="{H}" fill="url(#g2)"/>')
    # texture
    if pid:
        a(f'<rect width="{W}" height="{H}" fill="url(#{pid})" opacity=".32" mask="url(#m)"/>')
    else:  # stage floor: lines running to a vanishing point behind the logo, plus horizon rungs
        vy = gy - 30
        lines = []
        for i in range(-14, 15):
            lines.append(f"M{gx:.0f} {vy:.0f}L{gx + i * 150:.0f} {H}")
        for k in range(1, 7):
            y = vy + (H - vy) * (k / 6) ** 1.8
            lines.append(f"M0 {y:.0f}H{W}")
        a(f'<path d="{"".join(lines)}" stroke="{sec}" stroke-width="1.2" fill="none" opacity=".42" mask="url(#m)"/>')
    # light beams: long thin bars through the glow, rotated as one group
    n = rng.randint(4, 7)
    a(f'<g transform="rotate({ang:.1f} {gx:.0f} {gy:.0f})">')
    for i in range(n):
        off = rng.uniform(-620, 620)
        w = rng.choice((6, 14, 26, 48, 90, 140))
        op = rng.uniform(0.14, 0.3) if w > 40 else rng.uniform(0.3, 0.65)
        fill = "url(#bm)" if i % 3 else "url(#bs)"
        a(f'<rect x="{gx + off - w / 2:.0f}" y="{gy - 900:.0f}" width="{w}" height="1800" fill="{fill}" opacity="{op:.2f}"/>')
    a("</g>")
    a(f'<rect width="{W}" height="{H}" fill="url(#gl)"/>')
    # angular slashes: two filled parallelograms framing the glow, in the logo colours
    sl = []
    for j, col in enumerate((prim, sec)):
        x0 = gx + side * (rng.uniform(300, 520) + j * 70) * (1 if j else -1)
        t = rng.uniform(16, 40)
        k = -math.tan(math.radians(ang)) * (H + 20)
        sl.append(f'<path d="M{x0:.0f} -10l{t:.0f} 0l{k:.0f} {H + 20}l{-t:.0f} 0z" fill="{col}" opacity="{rng.uniform(.2, .34):.2f}"/>')
    a("".join(sl))
    # thin accent lines across the art, parallel to the beams' cross direction
    a(f'<g transform="rotate({-ang / 3:.1f} {gx:.0f} {gy:.0f})">')
    for _ in range(rng.randint(2, 3)):
        y = gy + rng.uniform(-220, 220)
        x = rng.uniform(-200, 400)
        a(f'<rect x="{x:.0f}" y="{y:.0f}" width="{rng.uniform(700, 1300):.0f}" height="{rng.choice((1.5, 2, 3))}" '
          f'fill="url(#ln)" opacity="{rng.uniform(.35, .6):.2f}"/>')
    a("</g>")
    # corner brackets: a HUD frame mark at two opposite corners
    c = rng.choice(((60, 50, 1, 1), (W - 60, 50, -1, 1)))
    cx, cy, sx, sy = c
    a(f'<path d="M{cx:.0f} {cy + sy * 70:.0f}V{cy:.0f}H{cx + sx * 70:.0f}M{W - cx:.0f} {H - cy - sy * 70:.0f}'
      f'V{H - cy:.0f}H{W - cx - sx * 70:.0f}" fill="none" stroke="{prim}" stroke-width="3" opacity=".55"/>')
    # sparks: a few small specks drifting around the glow
    sp = []
    for _ in range(rng.randint(14, 22)):
        px, py = gx + rng.gauss(0, 380), gy + rng.gauss(0, 170)
        if 0 < px < W and 0 < py < H:
            sp.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="{rng.choice((1, 1.5, 2, 3))}" opacity="{rng.uniform(.3, .9):.1f}"/>')
    a(f'<g fill="{prim}">{"".join(sp)}</g>')
    a(f'<rect width="{W}" height="{H}" fill="url(#vg)"/>')
    a("</svg>")
    return "".join(out)


def all_event_art(assets, root):
    """{event slug: SVG text} for every event with a logo in assets["event_logos"]."""
    return {slug: render(slug, os.path.join(root, p)) for slug, p in sorted(assets["event_logos"].items()) if p}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        f.write(render(sys.argv[1], sys.argv[2]))
