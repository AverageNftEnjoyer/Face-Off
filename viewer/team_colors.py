"""
Team colours from team logos (stdlib only: a small PNG decoder + colour counting).

For each logo (the dark-mode variant, since the hub is dark) we count opaque
pixels in coarse colour buckets. The primary colour is the most common
SATURATED colour if it covers enough of the logo (FaZe red, Falcons green);
logos with no real hue fall back to their most common colour (Spirit white).
A secondary colour is kept for clashes: when two teams' primaries are too
close, the second team is drawn in its secondary colour (or a neutral
fallback). Colours too dark to read on the dark UI are lightened.
"""

import colorsys
import os
import struct
import zlib

MIN_HUE_SHARE = 0.08     # a saturated colour must cover >= 8% of opaque pixels to win
MIN_LUMA = 0.11          # lighten only genuinely dark colours (pure red is 0.21 and stays red)


# ------------------------------------------------------------------ PNG decoding
def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def read_png(path):
    """Return (width, height, [(r, g, b, a), ...]) or None if unsupported."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, idat, plte, trns = 8, b"", None, None
    w = h = depth = ctype = interlace = None
    while pos < len(data):
        n, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if typ == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif typ == b"PLTE":
            plte = [tuple(body[i:i + 3]) for i in range(0, len(body), 3)]
        elif typ == b"tRNS":
            trns = body
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
    if interlace or depth not in (8, 16) or ctype not in (0, 2, 3, 4, 6):
        if not (ctype == 3 and depth in (1, 2, 4, 8)) or interlace:
            return None
    raw = zlib.decompress(idat)
    chans = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    bpp_bits = chans * depth
    stride = (w * bpp_bits + 7) // 8
    bpp = max(1, bpp_bits // 8)
    rows, prev, i = [], bytearray(stride), 0
    for _ in range(h):
        ft = raw[i]
        line = bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if ft == 1:
                line[x] = (line[x] + a) & 255
            elif ft == 2:
                line[x] = (line[x] + b) & 255
            elif ft == 3:
                line[x] = (line[x] + ((a + b) >> 1)) & 255
            elif ft == 4:
                line[x] = (line[x] + _paeth(a, b, c)) & 255
        rows.append(line)
        prev = line
    px = []
    for line in rows:
        if ctype == 3:
            per = 8 // depth if depth < 8 else 1
            for x in range(w):
                if depth == 8:
                    idx = line[x]
                else:
                    byte = line[x // per]
                    shift = 8 - depth * (x % per + 1)
                    idx = (byte >> shift) & ((1 << depth) - 1)
                r, g, b = plte[idx] if plte and idx < len(plte) else (0, 0, 0)
                al = trns[idx] if trns and idx < len(trns) else 255
                px.append((r, g, b, al))
            continue
        step = depth // 8
        for x in range(w):
            o = x * chans * step
            v = [line[o + k * step] for k in range(chans)]   # high byte for 16-bit
            if ctype == 0:
                px.append((v[0], v[0], v[0], 255))
            elif ctype == 4:
                px.append((v[0], v[0], v[0], v[1]))
            elif ctype == 2:
                px.append((v[0], v[1], v[2], 255))
            else:
                px.append(tuple(v))
    return w, h, px


# ------------------------------------------------------------------ colours
def _luma(rgb):
    def ch(c):
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _lighten(rgb):
    r, g, b = rgb
    for _ in range(20):
        if _luma((r, g, b)) >= MIN_LUMA:
            break
        r, g, b = (int(r + (255 - r) * 0.12), int(g + (255 - g) * 0.12), int(b + (255 - b) * 0.12))
    return r, g, b


def _hex(rgb):
    return "#%02X%02X%02X" % rgb


def logo_colors(path):
    """(primary, secondary) hex colours from a logo, or (None, None)."""
    try:
        img = read_png(path)
    except (OSError, zlib.error, struct.error, IndexError, KeyError, ValueError):
        return None, None
    if not img:
        return None, None
    _, _, px = img
    buckets = {}
    total = 0
    for r, g, b, a in px:
        if a < 160:
            continue
        total += 1
        key = (r >> 4, g >> 4, b >> 4)
        acc = buckets.setdefault(key, [0, 0, 0, 0])
        acc[0] += 1; acc[1] += r; acc[2] += g; acc[3] += b
    if not total:
        return None, None
    cols = []
    for n, rs, gs, bs in buckets.values():
        rgb = (rs // n, gs // n, bs // n)
        h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
        cols.append((n, rgb, s, v, h))
    # merge buckets of the same hue family so anti-aliased edges don't split a colour
    chroma = {}
    for n, rgb, s, v, h in cols:
        if s >= 0.35 and v >= 0.25:
            k = int(h * 18) % 18
            cur = chroma.setdefault(k, [0, None, 0])
            cur[0] += n
            if n > cur[2]:
                cur[1], cur[2] = rgb, n
    ranked = sorted(chroma.values(), key=lambda x: -x[0])
    neutral = [c for c in cols if not (c[2] >= 0.35 and c[3] >= 0.25)]
    # brightest neutral that is a real part of the logo (>= 15% of pixels),
    # so white-on-black marks read as white, not as an anti-aliased grey
    big = [c for c in neutral if c[0] / total >= 0.15]
    picks = [c[1] for c in ranked if c[0] / total >= MIN_HUE_SHARE]
    if big:
        picks.append(max(big, key=lambda c: c[3])[1])
    elif neutral:
        picks.append(max(neutral, key=lambda c: c[0])[1])
    picks += [c[1] for c in ranked if c[1] not in picks]
    picks = [_lighten(p) for p in picks]
    if not picks:
        return None, None
    return _hex(picks[0]), _hex(picks[1]) if len(picks) > 1 else None


def all_team_colors(assets, root):
    out = {}
    for team, rec in assets["teams"].items():
        path = rec.get("logo_dark_path") or rec.get("logo_path")
        if path:
            p, s = logo_colors(os.path.join(root, path))
            if p:
                out[team] = [p, s]
    return out
