"""
Visual Peer Inspector - presentation video, Vaibhav's sections.

Renders the motion-graphics clips V01-V05 and the credit roll (C01) at 1920x1080, 30 fps.
Every frame is drawn in code with Pillow and piped to ffmpeg; no generative video is used.
Inputs: the team's bottle photographs and app screenshot (assets/), bottle masks (masks/),
the internal-lock results table extracted from the project repository (data/), and the
sentence timings measured from the narration recordings (narration.py).

Usage:
    python render.py                      # render every clip to out/
    python render.py --only V02,V03       # render selected clips
    python render.py --stills V03 --times 4,14,30   # save preview frames as PNG
"""
import os, sys, math, json as _json, argparse, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import narration as N

W, H, FPS = 1920, 1080, 30
SRC = os.path.join(HERE, "assets") + os.sep
FD = os.path.join(HERE, "fonts") + os.sep

# ---- App theme (colours sampled from the Visual Peer Inspector web interface) ----
A_BG = (250, 249, 247)        # page background
A_TITLE = (61, 18, 16)        # maroon heading
A_INK = (25, 18, 17)          # body text
A_MUTED = (100, 96, 93)       # secondary text
A_LINE = (238, 230, 227)      # hairline rules / card borders
A_TERRA = (193, 99, 74)       # orange-red accent rule
A_TEAL = (46, 118, 110)       # object chip
A_TINT_BG, A_TINT_TX = (255, 240, 219), (102, 86, 61)   # "Review required" style chip

STOCK_CREDITS = []          # add "Creator / Site" strings here if stock footage is used
VOICE_CREDIT = None         # e.g. "ElevenLabs text-to-speech" if a synthetic voice is used

_fc = {}
def F(weight, size):
    k = (weight, size)
    if k not in _fc:
        name = {"r": "Inter-Regular", "m": "Inter-Medium", "s": "Inter-SemiBold", "b": "Inter-Bold",
                "d": "InterDisplay-Bold", "x": "InterDisplay-ExtraBold"}[weight]
        _fc[k] = ImageFont.truetype(FD + name + ".otf", size)
    return _fc[k]

def load(n): return Image.open(SRC + n).convert("RGB")
V1, V2 = load("V01-01-bottle-view-1.jpeg"), load("V01-02-bottle-view-2.jpeg")
APP = load("V04-02-review-required.jpg")

# ---------- animation helpers ----------
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(x): x = clamp(x); return x * x * (3 - 2 * x)
def ramp(t, a, d=0.6): return ease((t - a) / d) if d > 0 else float(t >= a)
def lerp(a, b, k): return a + (b - a) * k
def lerpv(a, b, k): return tuple(lerp(x, y, k) for x, y in zip(a, b))
def rgba(c, a): return (*(int(round(x)) for x in c[:3]), int(255 * clamp(a)))

class Frame:
    def __init__(self, bg=A_BG):
        self.im = Image.new("RGBA", (W, H), (*bg, 255))
    def layer(self):
        l = Image.new("RGBA", (W, H), (0, 0, 0, 0)); return l, ImageDraw.Draw(l)
    def comp(self, l): self.im.alpha_composite(l)
    def rgb(self): return self.im.convert("RGB")

def mapper(rect, crop):
    """Map source-image pixel coordinates to on-screen coordinates."""
    x, y, w, h = rect; cx0, cy0, cx1, cy1 = crop
    return lambda sx, sy: (x + (sx - cx0) / (cx1 - cx0) * w, y + (sy - cy0) / (cy1 - cy0) * h)

def text(d, xy, s, f, c, a=1.0, anchor="la", spacing=10, align="left"):
    if a <= 0.01: return
    d.multiline_text(xy, s, font=f, fill=rgba(c, a), anchor=anchor, spacing=spacing, align=align)

def wrap(s, f, maxw):
    out = []
    for para in s.split("\n"):
        line = ""
        for wd in para.split():
            t = (line + " " + wd).strip()
            if f.getlength(t) <= maxw: line = t
            else: out.append(line); line = wd
        out.append(line)
    return "\n".join(out)

def chip(d, xy, s, f, fg, bg, a=1.0, anchor="l"):
    if a <= 0.01: return
    x, y = xy; tw = f.getlength(s); ph, pv = 18, 10
    asc, desc = f.getmetrics(); th = asc
    if anchor == "r": x -= tw + 2 * ph
    if anchor == "c": x -= (tw + 2 * ph) / 2
    d.rounded_rectangle([x, y, x + tw + 2 * ph, y + th + 2 * pv], radius=(th + 2 * pv) / 2, fill=rgba(bg, a))
    d.text((x + ph, y + pv), s, font=f, fill=rgba(fg, a))

def ring(d, box, k, c=A_TERRA, a=1.0, width=5, pad=14):
    """Rounded box that draws itself in as k goes 0->1."""
    if k <= 0: return
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    per = 2 * ((x1 - x0) + (y1 - y0)); rem = per * clamp(k)
    for p, q in zip(pts, pts[1:]):
        seg = math.dist(p, q)
        if rem <= 0: break
        f = min(1, rem / seg); d.line([p, (lerp(p[0], q[0], f), lerp(p[1], q[1], f))], fill=rgba(c, a), width=width)
        rem -= seg
    if k >= 1: d.rounded_rectangle([x0, y0, x1, y1], radius=10, outline=rgba(c, a), width=width)

def dashed(d, p, q, c, a, width=4, dash=16, gap=12, k=1.0):
    L = math.dist(p, q) * clamp(k); n = 0.0
    while n < L:
        e = min(n + dash, L)
        d.line([lerpv(p, q, n / math.dist(p, q)), lerpv(p, q, e / math.dist(p, q))], fill=rgba(c, a), width=width)
        n += dash + gap

def TS(seg_id):
    """Segment record, sentence start times and (start, end) pairs from narration.py."""
    seg = next(s for s in N.SEGMENTS if s["id"] == seg_id)
    return seg, [a for a, b in N.timings(seg)], N.timings(seg)

# ---------- V01 ----------
V1_NOCAP = (352, 338, 562, 912)      # uncapped bottle, view 1 (source px)
V2_NOCAP = (128, 250, 422, 858)      # uncapped bottle, view 2

def paste_card(fr, img, rect, crop, alpha=1.0, radius=14):
    """Photo with rounded corners and a hairline border, like the app's cards."""
    if alpha <= 0.003: return
    x, y, w, h = rect
    iw, ih = max(1, int(round(w))), max(1, int(round(h)))
    r = img.resize((iw, ih), Image.BICUBIC, box=crop).convert("RGBA")
    m = Image.new("L", (iw, ih), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, iw - 1, ih - 1], radius=radius, fill=int(255 * alpha))
    r.putalpha(m)
    l, d = fr.layer()
    d.rounded_rectangle([x - 2, y - 2, x + w + 1, y + h + 1], radius=radius + 2, outline=rgba(A_LINE, alpha), width=2)
    fr.comp(l); fr.im.alpha_composite(r, (int(round(x)), int(round(y))))

def v01(t):
    seg, S, T = TS("V01"); fr = Frame(A_BG)
    ar = V1.width / V1.height
    # layouts: A single v1 zoom; B two panels; C two panels right + title left
    hA = H - 80; A1 = ((W - hA * ar) / 2, 40, hA * ar, hA)
    hB = 860; wB = hB * ar; B1 = (960 - 18 - wB, 140, wB, hB); B2 = (960 + 18, 140, wB, hB)
    hC = 700; wC = hC * ar; C1 = (770, 190, wC, hC); C2 = (770 + wC + 24, 190, wC, hC)
    kB = ramp(t, S[1] - 0.2, 0.9); kC = ramp(t, S[2] - 0.1, 1.0)
    r1 = lerpv(lerpv(A1, B1, kB), C1, kC)
    r2 = lerpv(lerpv((W + 40, 140, wB, hB), B2, kB), C2, kC)
    # zoom on v1 during phase A (toward the uncapped bottle), released in B
    z = lerp(1.0, 1.18, ease(t / max(0.1, S[1]))) if t < S[1] else lerp(1.18, 1.0, kB)
    fx, fy = 460, 640
    cw, ch = V1.width / z, V1.height / z
    cx0 = clamp(fx - cw / 2, 0, V1.width - cw); cy0 = clamp(fy - ch / 2, 0, V1.height - ch)
    crop1 = (cx0, cy0, cx0 + cw, cy0 + ch)
    paste_card(fr, V1, r1, crop1, alpha=ramp(t, 0, 0.8))
    if kB > 0: paste_card(fr, V2, r2, (0, 0, V2.width, V2.height), alpha=kB)
    m1 = mapper(r1, crop1); m2 = mapper(r2, (0, 0, V2.width, V2.height))
    b1 = (*m1(*V1_NOCAP[:2]), *m1(*V1_NOCAP[2:])); b2 = (*m2(*V2_NOCAP[:2]), *m2(*V2_NOCAP[2:]))
    l, d = fr.layer()
    ra = 1 - 0.35 * kC
    ring(d, b1, ramp(t, S[0] + 0.9, 0.8), c=A_TERRA, a=ra, pad=10)
    ftag = F("s", 26)
    if t > S[0] + 1.5: chip(d, (b1[0] - 10, b1[1] - 62), "No cap", ftag, (255, 255, 255), A_TEAL, ramp(t, S[0] + 1.5, 0.4) * ra)
    if kB > 0.5:
        ring(d, b2, ramp(t, S[1] + 1.6, 0.9), c=A_TERRA, a=ra, pad=10)
        if t > S[1] + 2.2: chip(d, (b2[0] - 10, b2[1] - 62), "Same bottle", ftag, (255, 255, 255), A_TEAL, ramp(t, S[1] + 2.2, 0.4) * ra)
        p = ((b1[2] + 10), (b1[1] + b1[3]) / 2); q = (b2[0] - 10, (b2[1] + b2[3]) / 2)
        dashed(d, p, q, A_TERRA, 0.95 * ra * (1 - kC), k=ramp(t, S[1] + 2.4, 0.8))
        for r, s in ((r1, "View 1"), (r2, "View 2")):
            text(d, (r[0] + r[2] / 2, r[1] - 20), s, F("s", 30), A_INK, kB * (1 - kC), anchor="md")
            text(d, (r[0] + r[2] / 2, r[1] - 16), s, F("m", 24), A_MUTED, kC, anchor="md")
    # title block (phase C), styled like the app header
    if kC > 0:
        x = 110; a = kC; dy = (1 - kC) * 30
        text(d, (x, 250 + dy), "Visual Peer\nInspector", F("x", 104), A_TITLE, a, spacing=6)
        d.rectangle([x, 488 + dy, x + 560 * ramp(t, S[2] + 1.2, 0.8), 491 + dy], fill=rgba(A_TERRA, a))
        sub = wrap("Identify the unusual item among similar products, and show the evidence across several photographs.", F("r", 34), 600)
        text(d, (x, 520 + dy), sub, F("r", 34), A_MUTED, ramp(t, S[2] + 2.2, 0.8), spacing=12)
        text(d, (x, 745 + dy), "Sreekar Reddy Edulapalli\nVaibhav Bairathi", F("s", 30), A_INK, ramp(t, S[2] + 4.2, 0.8), spacing=12)
        if t > S[3] - 0.2:
            chip(d, (x, 870), "Intended setting:  small-batch inspection", F("m", 28), A_TINT_TX, A_TINT_BG, ramp(t, S[3] - 0.2, 0.5))
    fr.comp(l)
    return fr

# ---------- V02 ----------

# ---------- V02 (process illustration on the real photos, app theme) ----------
import numpy as _np
MASKDIR = os.path.join(HERE, "masks") + os.sep
OBJ_ORDER = ["A", "B", "C"]          # A = left capped, B = uncapped, C = right capped (same physical object in both views)
ODD = "B"

def _load_obj(view, img):
    out = {}
    for n in OBJ_ORDER:
        m = Image.open(f"{MASKDIR}{view}_{n}.png").convert("L")
        bb = m.point(lambda v: 255 if v > 127 else 0).getbbox()
        crop = img.crop(bb).convert("RGBA"); crop.putalpha(m.crop(bb))
        # patch grid (source px) restricted to the mask, used for the "features" step
        cs = 46; cells = []
        ma = _np.array(m)
        for y in range(bb[1], bb[3], cs):
            for x in range(bb[0], bb[2], cs):
                if ma[y:y + cs, x:x + cs].mean() > 150: cells.append((x, y, x + cs, y + cs))
        # feature strip: 28 bands of the masked object, colour from mean brightness/saturation (illustrative)
        hsv = _np.array(img.crop(bb).convert("HSV")).astype(float); mk = _np.array(m.crop(bb)) > 127
        bands = []
        for i in range(28):
            y0 = int(i * hsv.shape[0] / 28); y1 = int((i + 1) * hsv.shape[0] / 28)
            sel = mk[y0:y1]; px = hsv[y0:y1][sel]
            bands.append(float(px[:, 1].mean() / 255) if len(px) else 0.0)
        out[n] = dict(mask=m, bbox=bb, crop=crop, cells=cells, bands=bands)
    return out

OBJ = {"v1": _load_obj("v1", V1), "v2": _load_obj("v2", V2)}
UNION = {}
for v in OBJ:
    u = Image.new("L", V1.size, 0)
    for n in OBJ_ORDER: u = ImageChops.lighter(u, OBJ[v][n]["mask"])
    UNION[v] = u
def _dimmed(img):
    g = img.convert("L").convert("RGB")
    return Image.blend(g, Image.new("RGB", img.size, A_BG), 0.62)
DIMMED = {"v1": _dimmed(V1), "v2": _dimmed(V2)}
SRCIMG = {"v1": V1, "v2": V2}

STEPS = [("Find objects", "LocateAnything-3B"), ("Isolate foreground", "SAM 2.1"),
         ("Compare appearance", "DINOv2-small"), ("Combine views", "Our pipeline")]

def _stepper(d, active, a):
    if a <= 0.01: return
    n = len(STEPS); w = 400; x0 = (W - n * w) / 2; y = 62
    for i, (lab, mod) in enumerate(STEPS):
        x = x0 + i * w; on = (i == active); done = i < active
        c = A_TERRA if on else (A_TEAL if done else (200, 192, 188))
        d.ellipse([x, y, x + 44, y + 44], fill=rgba(c, a))
        text(d, (x + 22, y + 22), str(i + 1), F("b", 24), (255, 255, 255), a, anchor="mm")
        text(d, (x + 60, y + 2), lab, F("b" if on else "s", 28), A_TITLE if on else A_MUTED, a)
        text(d, (x + 60, y + 38), mod, F("r", 21), A_MUTED, a)
        if i < n - 1: d.line([(x + 60 + F("s", 28).getlength(lab) + 14, y + 22), (x + w - 14, y + 22)], fill=rgba(A_LINE, a), width=3)
    d.line([(60, 150), (W - 60, 150)], fill=rgba(A_LINE, a), width=2)

def _photo(fr, v, rect, kdim, alpha):
    img = SRCIMG[v]
    if kdim > 0:
        bgimg = Image.blend(img, DIMMED[v], kdim) if kdim < 1 else DIMMED[v]
        img = Image.composite(img, bgimg, UNION[v])
    paste_card(fr, img, rect, (0, 0, img.width, img.height), alpha=alpha)

def _feature_strip(d, x, y, w, h, bands, k, a, hi=False):
    n = len(bands); bw = w / n
    for i, b in enumerate(bands):
        if i / n > k: break
        c = lerpv((236, 228, 224), A_TERRA if hi else A_TEAL, 0.25 + 0.75 * clamp(b * 1.4))
        d.rectangle([x + i * bw + 1, y, x + (i + 1) * bw - 1, y + h], fill=rgba(c, a))

def v02(t):
    seg, S, T = TS("V02"); fr = Frame(A_BG)
    ar = V1.width / V1.height
    kB = ramp(t, S[1] - 0.25, 0.8)          # two views
    kC = ramp(t, S[2] - 0.1, 0.8)           # isolate foreground
    kD = ramp(t, S[3] - 0.2, 0.9)           # move left + objects panel
    kE = ramp(t, S[4] - 0.1, 0.8)           # combine views
    kF = ramp(t, S[5] - 0.35, 0.6)          # handover
    a_main = 1 - kF
    # --- photo layouts
    hA = 800; A1 = ((W - hA * ar) / 2, 205, hA * ar, hA)
    hB = 800; wB = hB * ar; B1 = (960 - 23 - wB, 205, wB, hB); B2 = (960 + 23, 205, wB, hB)
    hD = 640; wD = hD * ar; D1 = (70, 260, wD, hD); D2 = (70 + wD + 26, 260, wD, hD)
    r1 = lerpv(lerpv(A1, B1, kB), D1, kD); r2 = lerpv(lerpv((W + 30, 205, wB, hB), B2, kB), D2, kD)
    if a_main > 0:
        _photo(fr, "v1", r1, kC, a_main * ramp(t, 0, 0.6))
        if kB > 0: _photo(fr, "v2", r2, kC, kB * a_main)
    l, d = fr.layer()
    maps = {"v1": mapper(r1, (0, 0, V1.width, V1.height)), "v2": mapper(r2, (0, 0, V2.width, V2.height))}
    def dbb(v, n):
        bb = OBJ[v][n]["bbox"]; m = maps[v]; return (*m(*bb[:2]), *m(*bb[2:]))
    if a_main > 0:
        # phase A: peers as reference (view 1 only)
        kAo = 1 - kB
        if kAo > 0:
            text(d, (W / 2, 112), "Each object is compared with its peers", F("x", 50), A_TITLE, kAo * ramp(t, 0.3, 0.6), anchor="mm")
            for i, n in enumerate(OBJ_ORDER):
                odd = n == ODD
                ring(d, dbb("v1", n), ramp(t, S[0] + 0.5 + 0.3 * i, 0.6), c=A_TERRA if odd else A_TEAL,
                     a=kAo, width=5 if odd else 4, pad=8)
            bo = dbb("v1", ODD); co = ((bo[0] + bo[2]) / 2, (bo[1] + bo[3]) / 2 + 60)
            for n in ("A", "C"):
                b = dbb("v1", n); dashed(d, co, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), A_TERRA, 0.9 * kAo, width=4, k=ramp(t, S[0] + 1.6, 0.8))
            if t > S[0] + 2.2: chip(d, (bo[0] - 30, bo[1] - 62), "Compared with peers", F("s", 24), (255, 255, 255), A_TERRA, ramp(t, S[0] + 2.2, 0.4) * kAo)
        # stepper
        active = 0 if t < S[2] - 0.1 else 1 if t < S[3] - 0.2 else 2 if t < S[4] - 0.1 else 3
        _stepper(d, active, kB)
        # phase B+: candidate boxes in both views
        if kB > 0:
            for vi, v in enumerate(("v1", "v2")):
                for i, n in enumerate(OBJ_ORDER):
                    k = ramp(t, S[1] + 0.25 + 0.18 * (i + 3 * vi), 0.5)
                    hi = kE > 0.5 and n == ODD
                    ring(d, dbb(v, n), k, c=A_TERRA if hi else A_TEAL, a=kB * (1 - 0.25 * kC), width=5 if hi else 4, pad=6)
            for r, s in ((r1, "View 1"), (r2, "View 2")):
                text(d, (r[0] + r[2] / 2, r[1] - 16), s, F("s", 26), A_INK, kB, anchor="md")
        # phase D: patch grid sweeping over each object
        if kD > 0 and kE < 1:
            for v in ("v1", "v2"):
                m = maps[v]
                for n in OBJ_ORDER:
                    cells = OBJ[v][n]["cells"]
                    for j, c in enumerate(cells):
                        kk = ramp(t, S[3] + 0.1 + 1.4 * (c[1] - OBJ[v][n]["bbox"][1]) / max(1, OBJ[v][n]["bbox"][3] - OBJ[v][n]["bbox"][1]), 0.25)
                        if kk <= 0: continue
                        p0 = m(*c[:2]); p1 = m(*c[2:])
                        d.rectangle([p0[0], p0[1], p1[0], p1[1]], outline=rgba(A_TEAL, 0.55 * kk * (1 - kE)), width=1)
        # phase E: link the same object across views
        if kE > 0:
            for n in OBJ_ORDER:
                b1, b2 = dbb("v1", n), dbb("v2", n)
                p = (b1[2] + 6, (b1[1] + b1[3]) / 2); q = (b2[0] - 6, (b2[1] + b2[3]) / 2)
                dashed(d, p, q, A_TERRA if n == ODD else A_TEAL, 0.9, width=3, dash=10, gap=8, k=ramp(t, S[4] + 0.2, 0.9))
        # objects-across-views panel (right side)
        if kD > 0:
            a = kD; px0, py0, px1, py1 = 1080, 230, 1850, 990
            d.rounded_rectangle([px0, py0 + (1 - kD) * 30, px1, py1], radius=16, fill=rgba((255, 255, 255), a), outline=rgba(A_LINE, a), width=2)
            text(d, (px0 + 34, py0 + 30), "Objects across views", F("b", 32), A_INK, a)
            text(d, (px0 + 34, py0 + 76), "Same object, matched across both photos", F("r", 22), A_MUTED, a)
            d.line([(px0 + 34, py0 + 118), (px1 - 34, py0 + 118)], fill=rgba(A_TERRA, a), width=3)
            order_start = {n: i for i, n in enumerate(OBJ_ORDER)}
            order_end = {"B": 0, "A": 1, "C": 2}
            kr = ramp(t, S[4] + 1.2, 0.9)
            fr.comp(l); l, d = fr.layer()
            for n in OBJ_ORDER:
                row = lerp(order_start[n], order_end[n], kr)
                ry = py0 + 140 + row * 205; odd = n == ODD
                ka = ramp(t, S[3] + 0.3 + 0.25 * order_start[n], 0.5) * a
                if ka <= 0: continue
                if odd and kE > 0:
                    d.rounded_rectangle([px0 + 18, ry - 6, px1 - 18, ry + 190], radius=12, fill=rgba((245, 236, 231), ka * kr))
                # thumbnails (view 1, view 2)
                for vi, v in enumerate(("v1", "v2")):
                    cimg = OBJ[v][n]["crop"]; th = 170; tw = th * cimg.width / cimg.height
                    if tw > 110: tw = 110; th = tw * cimg.height / cimg.width
                    tx = px0 + 40 + vi * 130 + (110 - tw) / 2; ty = ry + 6 + (170 - th) / 2
                    im = cimg.resize((max(1, int(tw)), max(1, int(th))), Image.BICUBIC)
                    if ka < 1:
                        al = im.getchannel("A").point(lambda q: int(q * ka)); im.putalpha(al)
                    fr.comp(l); fr.im.alpha_composite(im, (int(tx), int(ty))); l, d = fr.layer()
                    text(d, (px0 + 95 + vi * 130, ry + 200), f"View {vi + 1}", F("r", 18), A_MUTED, ka, anchor="mb")
                # feature strips from both views
                sx = px0 + 320; sw = px1 - 40 - sx
                ks = ramp(t, S[3] + 0.6 + 0.3 * order_start[n], 1.2)
                for vi, v in enumerate(("v1", "v2")):
                    _feature_strip(d, sx, ry + 40 + vi * 46, sw, 30, OBJ[v][n]["bands"], ks, ka, hi=(odd and kE > 0.5))
                text(d, (sx, ry + 6), "visual features", F("r", 19), A_MUTED, ka * ramp(t, S[3] + 0.6, 0.4))
                if odd and t > S[4] + 1.8:
                    chip(d, (sx, ry + 140), "Highest combined unusualness", F("m", 20), A_TINT_TX, A_TINT_BG, ramp(t, S[4] + 1.8, 0.5) * ka)
            # honest label
            chip(d, (px1, py1 + 20), "Illustration of the steps, not live model output", F("m", 19), A_MUTED, (241, 236, 233), a * ramp(t, S[3], 0.5), anchor="r")
    # --- handover card
    if kF > 0:
        text(d, (W / 2, 390), "NEXT  ·  WEB APP DEMONSTRATION", F("s", 30), A_TERRA, kF, anchor="mm")
        text(d, (W / 2, 495), "Sreekar Reddy Edulapalli", F("x", 96), A_TITLE, kF, anchor="mm")
        d.rectangle([W / 2 - 300 * ramp(t, S[5], 0.8), 560, W / 2 + 300 * ramp(t, S[5], 0.8), 563], fill=rgba(A_TERRA, kF))
        text(d, (W / 2, 610), "Two views of the same bottle arrangement", F("r", 34), A_MUTED, ramp(t, S[5] + 0.4, 0.6), anchor="mm")
    fr.comp(l)
    return fr

# ---------- V03 ----------

# ---------- V03 (main measured outcome, app theme, drawn from the repo's internal-lock tables) ----------
import json as _json
RES = _json.load(open(os.path.join(HERE, "data", "results_internal_lock400.json")))
R_AS = RES["routes"]["assisted_correspondence"]; R_AU = RES["routes"]["automatic_predicted"]
A_DEV, A_CAL = (217, 207, 202), (191, 176, 170)
# cue points inside the recording (seconds)
CUE3 = dict(dev=5.98, cal=7.59, internal=9.55, five=11.80, assisted=12.95, automatic=18.16)

def _chart(fr, t, a, S, kdim=0.0):
    """Line chart of tie-aware top-1 vs views with 95% asset-group intervals."""
    if a <= 0.01: return
    l, d = fr.layer()
    cx0, cy0, cx1, cy1 = 80, 150, 1270, 975
    d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=16, fill=rgba((255, 255, 255), a), outline=rgba(A_LINE, a), width=2)
    text(d, (cx0 + 40, cy0 + 32), "Tie-aware top-1 accuracy  ·  400 internal-evaluation scenes", F("b", 30), A_INK, a)
    text(d, (cx0 + 40, cy0 + 76), "Exposure-qualified: historical exposure to related dataset objects remains a limitation", F("r", 21), A_MUTED, a)
    d.line([(cx0 + 40, cy0 + 116), (cx1 - 40, cy0 + 116)], fill=rgba(A_TERRA, a), width=3)
    # plot area
    px0, py0, px1, py1 = cx0 + 120, cy0 + 190, cx1 - 70, cy1 - 110
    def X(v): return px0 + (v - 1) / 4 * (px1 - px0)
    def Y(p): return py1 - p / 80 * (py1 - py0)
    for g in (0, 20, 40, 60, 80):
        d.line([(px0, Y(g)), (px1, Y(g))], fill=rgba(A_LINE, a), width=2)
        text(d, (px0 - 18, Y(g)), f"{g}%", F("r", 22), A_MUTED, a, anchor="rm")
    for v in (1, 2, 3, 5):
        text(d, (X(v), py1 + 22), str(v), F("m", 24), A_INK, a, anchor="ma")
    text(d, (X(1), py1 + 54), "diagnostic", F("r", 18), A_MUTED, a, anchor="ma")
    text(d, ((px0 + px1) / 2, py1 + 62), "Views per scene", F("m", 22), A_MUTED, a, anchor="ma")
    # legend
    for i, (c, s) in enumerate(((A_TEAL, "Assisted  ·  dataset-supplied object matches"), (A_TERRA, "Automatic  ·  predicted object matches"))):
        lx = px0 + 10; ly = cy0 + 140 + i * 0
        lx = px0 + (0 if i == 0 else 560)
        d.line([(lx, ly + 2), (lx + 34, ly + 2)], fill=rgba(c, a), width=5)
        text(d, (lx + 46, ly + 2), s, F("m", 21), A_INK, a, anchor="lm")
    fr.comp(l); l, d = fr.layer()
    kline = ramp(t, CUE3["five"] + 0.1, 1.1)
    for route, c in ((R_AS, A_TEAL), (R_AU, A_TERRA)):
        pts = [(X(int(v)), Y(m)) for v, (m, lo, hi) in sorted(route["views"].items(), key=lambda kv: int(kv[0]))]
        band = [(X(int(v)), Y(hi)) for v, (m, lo, hi) in sorted(route["views"].items(), key=lambda kv: int(kv[0]))] + \
               [(X(int(v)), Y(lo)) for v, (m, lo, hi) in sorted(route["views"].items(), key=lambda kv: -int(kv[0]))]
        d.polygon(band, fill=rgba(c, 0.13 * a * kline))
        n = kline * (len(pts) - 1); i = int(n)
        path = pts[:i + 1] + ([lerpv(pts[i], pts[i + 1], n - i)] if i < len(pts) - 1 else [])
        if len(path) > 1: d.line(path, fill=rgba(c, a), width=5, joint="curve")
        for j, p in enumerate(pts):
            if j <= n + 0.01: d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=rgba(c, a), outline=rgba((255, 255, 255), a), width=2)
    fr.comp(l)
    if kdim > 0:
        l, d = fr.layer(); d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=16, fill=rgba(A_BG, 0.55 * kdim)); fr.comp(l)
    return X, Y

def _eyebrow(d, a):
    text(d, (W / 2, 70), "FIVE VIEWS  ·  400 INTERNAL-EVALUATION SCENES  ·  TIE-AWARE TOP-1", F("s", 24), A_TERRA, a, anchor="mm")

def v03(t):
    seg, S, T = TS("V03"); fr = Frame(A_BG)
    k1o = ramp(t, CUE3["five"] - 0.45, 0.35)          # split bar out
    k2 = ramp(t, CUE3["five"] - 0.1, 0.5)              # chart in
    k3 = ramp(t, S[2] - 0.2, 0.6)                      # averages panel
    k4 = ramp(t, S[3] - 0.25, 0.45)                    # assisted benchmark scene
    k4o = ramp(t, S[4] - 0.55, 0.35); k5 = ramp(t, S[4] - 0.2, 0.45)
    l, d = fr.layer()
    # ---- phase 1: data split
    if k1o < 1:
        a = 1 - k1o
        text(d, (160, 290), "MAIN STUDY", F("s", 30), A_TERRA, a * ramp(t, 0.2, 0.5))
        text(d, (160, 330), "2,000 selected ToysAD-8K scenes", F("x", 80), A_TITLE, a * ramp(t, 0.4, 0.6))
        x0, y0, bw, bh = 160, 560, 1600, 116
        parts = [(1000, "Development", A_DEV, CUE3["dev"]), (600, "Calibration", A_CAL, CUE3["cal"]),
                 (400, "Internal evaluation", A_TERRA, CUE3["internal"])]
        x = x0
        for n, lab, c, st in parts:
            w = bw * n / 2000; k = ramp(t, st - 0.25, 0.6)
            if k > 0:
                d.rectangle([x, y0, max(x + 1, x + w * k - 5), y0 + bh], fill=rgba(c, a))
                ka = a * ramp(t, st + 0.1, 0.35)
                text(d, (x + 24, y0 + bh / 2), f"{n:,}", F("x", 54), (255, 255, 255) if c == A_TERRA else A_INK, ka, anchor="lm")
                text(d, (x, y0 + bh + 30), lab, F("m", 30), A_TERRA if c == A_TERRA else A_MUTED, ka)
            x += w
        kb = ramp(t, CUE3["internal"] + 0.8, 0.5) * a
        if kb > 0:
            bx0 = x0 + bw * 1600 / 2000; bx1 = x0 + bw - 5
            d.line([(bx0, y0 - 22), (bx0, y0 - 36), (bx1, y0 - 36), (bx1, y0 - 22)], fill=rgba(A_TERRA, kb), width=3)
            text(d, ((bx0 + bx1) / 2, y0 - 52), "Results that follow", F("s", 24), A_TERRA, kb, anchor="md")
    fr.comp(l)
    # ---- phase 2 + 3: chart
    ka = k2 * (1 - k4)
    if ka > 0.01:
        XY = _chart(fr, t, ka, S, kdim=0.35 * k3)
        X, Y = XY
        l, d = fr.layer()
        pulse = 0.5 + 0.5 * math.sin(t * 4)
        # callouts (phase 2)
        kc = ka * (1 - k3)
        if kc > 0:
            text(d, (1330, 230), "FIVE VIEWS", F("s", 26), A_MUTED, kc * ramp(t, CUE3["five"] + 0.3, 0.4))
            for route, c, lab, val, ty, st in ((R_AS, A_TEAL, "Assisted", "61.72%", 330, CUE3["assisted"]),
                                               (R_AU, A_TERRA, "Automatic", "22.00%", 640, CUE3["automatic"])):
                k = ramp(t, st, 0.5) * kc
                if k <= 0: continue
                ex, ey = X(5), Y(route["views"]["5"][0]); rr = 15 + 5 * pulse
                d.ellipse([ex - rr, ey - rr, ex + rr, ey + rr], outline=rgba(c, k), width=4)
                d.line([(ex + rr, ey), (1310, ty + 50)], fill=rgba(c, k), width=3)
                text(d, (1330, ty), lab, F("s", 32), c, k)
                text(d, (1330, ty + 40), val, F("x", 92), A_TITLE, k)
                text(d, (1330, ty + 150), "95% interval " + f"{route['views']['5'][1]:.2f}–{route['views']['5'][2]:.2f}%", F("r", 21), A_MUTED, k)
            text(d, (1330, 540), "compared with", F("r", 26), A_MUTED, ramp(t, 17.18, 0.4) * kc)
        # phase 3: averaging across predefined five-view combinations
        if k3 > 0:
            a3 = k3 * (1 - k4)
            px = 1320; pw = 520
            text(d, (px, 175), "Averaged over the predefined\nfive-view combinations", F("b", 32), A_TITLE, a3, spacing=8)
            kmerge = ramp(t, S[2] + 2.3, 0.9)
            for i, (route, c, lab, lo, hi) in enumerate(((R_AS, A_TEAL, "Assisted", 58, 65), (R_AU, A_TERRA, "Automatic", 18, 26))):
                y = 330 + i * 230
                text(d, (px, y), lab, F("s", 26), c, a3)
                ax0, ax1 = px + 10, px + pw - 30; ay = y + 95
                def SX(v): return ax0 + (v - lo) / (hi - lo) * (ax1 - ax0)
                d.line([(ax0, ay), (ax1, ay)], fill=rgba(A_LINE, a3), width=3)
                for g in range(lo + (5 - lo % 5) % 5, hi + 1, 5):
                    text(d, (SX(g), ay + 16), f"{g}%", F("r", 18), A_MUTED, a3, anchor="ma")
                mean = route["views"]["5"][0]
                sch = list(route["five_view_schedules"].values())
                for j, v in enumerate(sch):
                    kd = ramp(t, S[2] + 0.3 + 0.2 * j, 0.4) * a3
                    xv = lerp(SX(v), SX(mean), kmerge); yv = ay - 30 - j * 0
                    d.ellipse([xv - 10, ay - 10, xv + 10, ay + 10], fill=rgba(c, kd * (1 - 0.6 * kmerge)), outline=rgba((255, 255, 255), kd), width=2)
                    if kmerge < 0.3:
                        srt = sorted(sch); hy = 18 + 24 * (srt.index(v) % 2)
                        text(d, (SX(v), ay - hy), f"{v:.2f}", F("r", 18), A_MUTED, kd * (1 - kmerge / 0.3), anchor="md")
                km = kmerge * a3
                d.line([(SX(mean), ay - 34), (SX(mean), ay + 12)], fill=rgba(A_TITLE, km), width=4)
                text(d, (SX(mean), ay - 40), f"{mean:.2f}%", F("x", 34), A_TITLE, km, anchor="md")
                text(d, (ax1, y + 4), "4 combinations", F("r", 19), A_MUTED, a3, anchor="ra")
            # tie-aware note
            kt = ramp(t, S[2] + 2.2, 0.5) * a3
            if kt > 0:
                bx, by = px, 790
                d.rounded_rectangle([bx - 10, by, bx + pw, by + 170], radius=14, fill=rgba(A_TINT_BG, kt))
                text(d, (bx + 14, by + 18), "Tie-aware scoring", F("b", 24), A_TINT_TX, kt)
                text(d, (bx + 14, by + 54), "If k objects share the top score,\nthe scene counts as 1/k correct.", F("r", 22), A_TINT_TX, kt, spacing=6)
                for j in range(3):
                    gx = bx + 360 + j * 46; gy = by + 70
                    d.rounded_rectangle([gx, gy, gx + 28, gy + 66], radius=8, fill=rgba(A_TERRA if j == 1 else (205, 190, 170), kt))
                    d.rounded_rectangle([gx + 8, gy - 10, gx + 20, gy], radius=3, fill=rgba(A_TERRA if j == 1 else (205, 190, 170), kt))
                text(d, (bx + 405, by + 150), "tied → 1/3", F("s", 20), A_TINT_TX, kt, anchor="ms")
        fr.comp(l)
    # ---- phase 4: what the assisted benchmark measured
    l, d = fr.layer()
    a4 = k4 * (1 - k4o)
    if a4 > 0:
        _eyebrow(d, a4)
        text(d, (W / 2, 175), "What the assisted benchmark measured", F("x", 62), A_TITLE, a4, anchor="mm")
        boxes = [("Dataset annotations", "ToysAD-8K reference"), ("Object matches", "same object across views"),
                 ("Assisted scorer", "object-bank top-2"), ("61.72%", "five views, tie-aware")]
        bw, bh, gap = 340, 170, 70; x0 = (W - 4 * bw - 3 * gap) / 2; y0 = 330
        for i, (h1, h2) in enumerate(boxes):
            k = ramp(t, S[3] + 0.15 + 0.45 * i, 0.4) * a4
            if k <= 0: continue
            x = x0 + i * (bw + gap); last = i == 3
            d.rounded_rectangle([x, y0, x + bw, y0 + bh], radius=16, fill=rgba(A_TEAL if last else (255, 255, 255), k), outline=rgba(A_TEAL if i < 3 else A_TEAL, k), width=3)
            text(d, (x + bw / 2, y0 + 70), h1, F("x" if last else "b", 52 if last else 32), (255, 255, 255) if last else A_INK, k, anchor="mm")
            text(d, (x + bw / 2, y0 + 122), h2, F("r", 22), (225, 240, 238) if last else A_MUTED, k, anchor="mm")
            if i > 0:
                ax = x - gap + 12; ay = y0 + bh / 2
                d.line([(ax, ay), (x - 14, ay)], fill=rgba(A_TEAL, k), width=4)
                d.polygon([(x - 12, ay), (x - 26, ay - 10), (x - 26, ay + 10)], fill=rgba(A_TEAL, k))
        # not measured
        kn = ramp(t, S[3] + 2.7, 0.5) * a4
        if kn > 0:
            nx0, ny0, nx1, ny1 = x0 + bw + gap - 100, 610, x0 + 3 * bw + 2 * gap + 160, 830
            dashed(d, (nx0, ny0), (nx1, ny0), (190, 180, 175), kn, width=3); dashed(d, (nx1, ny0), (nx1, ny1), (190, 180, 175), kn, width=3)
            dashed(d, (nx1, ny1), (nx0, ny1), (190, 180, 175), kn, width=3); dashed(d, (nx0, ny1), (nx0, ny0), (190, 180, 175), kn, width=3)
            hx, hy = nx0 + 110, ny0 + 110
            d.ellipse([hx - 24, hy - 70, hx + 24, hy - 22], fill=rgba((190, 180, 175), kn))
            d.rounded_rectangle([hx - 44, hy - 14, hx + 44, hy + 60], radius=22, fill=rgba((190, 180, 175), kn))
            text(d, (nx0 + 200, ny0 + 66), "Real users confirming matches in the app", F("b", 30), A_MUTED, kn)
            text(d, (nx0 + 200, ny0 + 116), "Not measured: this was not a human-user study", F("s", 28), A_TERRA, kn)
            ks = ramp(t, S[3] + 3.4, 0.4) * kn
            d.line([(nx0 + 60, ny0 + 40), (lerp(nx0 + 60, nx0 + 160, ks), lerp(ny0 + 40, ny0 + 180, ks))], fill=rgba(A_TERRA, kn), width=6)
    # ---- phase 5: two routes, two differences
    if k5 > 0:
        a5 = k5
        _eyebrow(d, a5)
        text(d, (W / 2, 175), "The two routes differ in two ways", F("x", 62), A_TITLE, a5, anchor="mm")
        tx0, tx1 = 200, 1720; c1, c2 = 880, 1420; ty = 300
        heads = [("Automatic", "22.00%", A_TERRA, c1), ("Assisted", "61.72%", A_TEAL, c2)]
        for lab, val, c, cx in heads:
            text(d, (cx, ty), lab, F("s", 32), c, a5, anchor="mm")
            text(d, (cx, ty + 62), val, F("x", 64), A_TITLE, a5, anchor="mm")
        rows = [("Object matching", "Predicted by the system", "Supplied by the dataset", S[4] + 0.2),
                ("Scoring rule", "Winner-cluster consensus", "Object-bank top-2", S[4] + 1.5)]
        for i, (lab, v1, v2, st) in enumerate(rows):
            k = ramp(t, st, 0.45) * a5
            if k <= 0: continue
            ry = 440 + i * 170
            d.rounded_rectangle([tx0, ry, tx1, ry + 140], radius=16, fill=rgba((255, 255, 255), k), outline=rgba(A_LINE, k), width=2)
            text(d, (tx0 + 40, ry + 70), lab, F("b", 32), A_INK, k, anchor="lm")
            text(d, (c1, ry + 70), v1, F("m", 30), A_INK, k, anchor="mm")
            text(d, (c2, ry + 70), v2, F("m", 30), A_INK, k, anchor="mm")
            kd = ramp(t, st + 0.5, 0.35) * a5
            mx = (c1 + c2) / 2
            d.ellipse([mx - 28, ry + 42, mx + 28, ry + 98], fill=rgba(A_TINT_BG, kd))
            text(d, (mx, ry + 70), "≠", F("b", 40), A_TINT_TX, kd, anchor="mm")
        kc = ramp(t, S[4] + 2.6, 0.5) * a5
        if kc > 0:
            s = "So matching alone does not explain the gap"
            f = F("b", 34); w = f.getlength(s) + 80
            d.rounded_rectangle([W / 2 - w / 2, 820, W / 2 + w / 2, 900], radius=40, fill=rgba(A_TINT_BG, kc))
            text(d, (W / 2, 860), s, f, A_TINT_TX, kc, anchor="mm")
    fr.comp(l)
    return fr

# ---------- V04 ----------
# ---------- V04 (limits and potential value, app theme) ----------
CUE4 = dict(supp=2.66, s1=5.10, s2=9.50, falarm=10.54, normal=11.79, optime=13.41,
            s3=15.42, clearer=16.56, but=18.76, certify=19.88, defectfree=21.54)
LIMITS = ["Matching", "Wider validation", "All-normal testing"]

def _limits_tracker(d, active, a):
    if a <= 0.01: return
    text(d, (110, 62), "LIMITS", F("s", 24), A_TERRA, a)
    x = 230
    for i, lab in enumerate(LIMITS):
        on = i == active; done = i < active
        c = A_TERRA if on else (A_TEAL if done else (200, 192, 188))
        d.ellipse([x, 48, x + 40, 88], fill=rgba(c, a))
        text(d, (x + 20, 68), str(i + 1), F("b", 22), (255, 255, 255), a, anchor="mm")
        f = F("b" if on else "s", 28)
        text(d, (x + 54, 68), lab, f, A_TITLE if on else A_MUTED, a, anchor="lm")
        x += 54 + f.getlength(lab) + 60
    d.line([(60, 130), (W - 60, 130)], fill=rgba(A_LINE, a), width=2)

def _bottle(d, x, y, s, c, a, cap=True):
    """Simple bottle glyph; (x, y) is the bottom-left corner, s the height."""
    w = s * 0.38
    d.rounded_rectangle([x, y - s * 0.78, x + w, y], radius=w * 0.28, fill=rgba(c, a))
    d.rounded_rectangle([x + w * 0.3, y - s * 0.92, x + w * 0.7, y - s * 0.74], radius=4, fill=rgba(c, a))
    if cap: d.rounded_rectangle([x + w * 0.26, y - s, x + w * 0.74, y - s * 0.9], radius=4, fill=rgba(A_TERRA if c != A_TERRA else A_TITLE, a))

def v04(t):
    seg, S, T = TS("V04"); fr = Frame(A_BG)
    k1o = ramp(t, CUE4["s1"] + 0.25, 0.35); k2 = ramp(t, CUE4["s1"] + 0.55, 0.4)
    k2o = ramp(t, CUE4["s2"] - 0.45, 0.35); k3 = ramp(t, CUE4["s2"] - 0.1, 0.4)
    k3o = ramp(t, CUE4["s3"] - 0.45, 0.35); k4 = ramp(t, CUE4["s3"] - 0.1, 0.5)
    active = 0 if t < CUE4["s1"] - 0.2 else 1 if t < CUE4["s2"] - 0.2 else 2
    l, d = fr.layer()
    _limits_tracker(d, active, ramp(t, 0.1, 0.4) * (1 - k3o))
    # ---- 1. Matching: the automatic route falls as views are added (repo values)
    a1 = 1 - k1o
    if a1 > 0.01:
        cx0, cy0, cx1, cy1 = 110, 200, 1040, 960
        d.rounded_rectangle([cx0, cy0, cx1, cy1], radius=16, fill=rgba((255, 255, 255), a1), outline=rgba(A_LINE, a1), width=2)
        text(d, (cx0 + 36, cy0 + 30), "Automatic route  ·  tie-aware top-1", F("b", 28), A_INK, a1)
        text(d, (cx0 + 36, cy0 + 70), "400 internal-evaluation scenes, predicted object matches", F("r", 21), A_MUTED, a1)
        px0, py0, px1, py1 = cx0 + 110, cy0 + 150, cx1 - 60, cy1 - 100
        def X(v): return px0 + (v - 1) / 4 * (px1 - px0)
        def Y(p): return py1 - p / 60 * (py1 - py0)
        for g in (0, 20, 40, 60):
            d.line([(px0, Y(g)), (px1, Y(g))], fill=rgba(A_LINE, a1), width=2)
            text(d, (px0 - 16, Y(g)), f"{g}%", F("r", 21), A_MUTED, a1, anchor="rm")
        for v in (1, 2, 3, 5): text(d, (X(v), py1 + 20), str(v), F("m", 23), A_INK, a1, anchor="ma")
        text(d, ((px0 + px1) / 2, py1 + 54), "Views per scene", F("m", 21), A_MUTED, a1, anchor="ma")
        vals = sorted(R_AU["views"].items(), key=lambda kv: int(kv[0]))
        pts = [(X(int(v)), Y(m)) for v, (m, lo, hi) in vals]
        band = [(X(int(v)), Y(hi)) for v, (m, lo, hi) in vals] + [(X(int(v)), Y(lo)) for v, (m, lo, hi) in reversed(vals)]
        kl = ramp(t, 0.7, 1.6)
        d.polygon(band, fill=rgba(A_TERRA, 0.13 * a1 * kl))
        n = kl * (len(pts) - 1); i = int(n)
        path = pts[:i + 1] + ([lerpv(pts[i], pts[i + 1], n - i)] if i < len(pts) - 1 else [])
        if len(path) > 1: d.line(path, fill=rgba(A_TERRA, a1), width=6, joint="curve")
        for j, (v, (m, lo, hi)) in enumerate(vals):
            if j <= n + 0.01:
                p = pts[j]; d.ellipse([p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8], fill=rgba(A_TERRA, a1), outline=rgba((255, 255, 255), a1), width=2)
                text(d, (p[0], p[1] - 18), f"{m:.2f}%", F("s", 22), A_TITLE, a1, anchor="md")
        # right-hand message
        text(d, (1110, 300), "Automatic matching\nremains unreliable", F("x", 58), A_TITLE, a1 * ramp(t, 0.7, 0.5), spacing=6)
        text(d, (1110, 470), wrap("Accuracy falls as more views are added, because each extra view needs the same object to be matched again.", F("r", 30), 700),
             F("r", 30), A_MUTED, a1 * ramp(t, 1.3, 0.5), spacing=10)
        ks = ramp(t, CUE4["supp"], 0.4) * a1
        if ks > 0:
            bx0, by0, bx1, by1 = 1110, 610, 1830, 900
            d.rounded_rectangle([bx0, by0, bx1, by1], radius=16, fill=rgba(A_TINT_BG, ks))
            text(d, (bx0 + 30, by0 + 26), "Supplementary results were also lower", F("b", 28), A_TINT_TX, ks)
            text(d, (bx0 + 30, by0 + 66), "Official seen / unseen roles  ·  five views, primary schedule", F("r", 21), A_TINT_TX, ks)
            cA, cU = bx0 + 400, bx0 + 590
            text(d, (cA, by0 + 120), "Assisted", F("s", 22), A_TEAL, ks, anchor="mm")
            text(d, (cU, by0 + 120), "Automatic", F("s", 22), A_TERRA, ks, anchor="mm")
            # Final report, Section 7.5 and Appendix D (Table 16): predefined primary five-view schedule
            rows = [("Seen", "715 scenes", "22.93%", "11.39%"), ("Unseen", "2,437 scenes", "23.35%", "9.26%")]
            for i, (r1, r2, va, vu) in enumerate(rows):
                ry = by0 + 175 + i * 64
                kr = ramp(t, CUE4["supp"] + 0.3 + 0.35 * i, 0.4) * ks
                text(d, (bx0 + 30, ry), r1, F("b", 28), A_INK, kr, anchor="lm")
                text(d, (bx0 + 150, ry), r2, F("r", 22), A_MUTED, kr, anchor="lm")
                text(d, (cA, ry), va, F("x", 36), A_TITLE, kr, anchor="mm")
                text(d, (cU, ry), vu, F("x", 36), A_TITLE, kr, anchor="mm")
    # ---- 2. Wider validation: exposure of the 109 asset groups
    a2 = k2 * (1 - k2o)
    if a2 > 0.01:
        text(d, (110, 210), "Historical exposure\nlimits the evaluation", F("x", 58), A_TITLE, a2, spacing=6)
        text(d, (110, 375), wrap("The 400 internal-evaluation scenes come from 109 asset groups. Earlier access to related dataset objects is incomplete, so every group is potentially exposed.", F("r", 30), 900),
             F("r", 30), A_MUTED, a2 * ramp(t, CUE4["s1"] + 0.6, 0.5), spacing=10)
        gx0, gy0, cs, gap = 1190, 230, 50, 10
        for i in range(109):
            r, c = divmod(i, 11)
            k = ramp(t, CUE4["s1"] + 0.3 + i * 0.022, 0.25) * a2
            if k <= 0: continue
            x = gx0 + c * (cs + gap); y = gy0 + r * (cs + gap)
            col = lerpv((236, 230, 226), A_TINT_BG, ramp(t, CUE4["s1"] + 2.2, 0.6))
            d.rounded_rectangle([x, y, x + cs, y + cs], radius=8, fill=rgba(col, k), outline=rgba((222, 196, 160), k * ramp(t, CUE4["s1"] + 2.2, 0.6)), width=2)
        kc = ramp(t, CUE4["s1"] + 2.4, 0.5) * a2
        if kc > 0:
            # internal-lock README: 400 scenes in 109 asset groups, all potentially exposed
            text(d, (110, 590), "109 / 109", F("x", 96), A_TITLE, kc)
            text(d, (110, 710), "asset groups potentially exposed", F("s", 32), A_TINT_TX, kc)
            s = "Not independent confirmation"
            f = F("b", 30); w = f.getlength(s) + 70
            d.rounded_rectangle([110, 790, 110 + w, 860], radius=35, fill=rgba(A_TINT_BG, kc))
            text(d, (110 + w / 2, 825), s, f, A_TINT_TX, kc, anchor="mm")
    # ---- 3. All-normal testing and time savings: not yet measured
    a3 = k3 * (1 - k3o)
    if a3 > 0.01:
        text(d, (W / 2, 230), "Not yet measured", F("x", 62), A_TITLE, a3, anchor="mm")
        cards = [(CUE4["falarm"] - 0.3, "False-alarm rate", "on all-normal batches",
                  "The project dataset contains no all-normal scenes, so this could not be estimated."),
                 (CUE4["optime"] - 0.2, "Operator time savings", "compared with manual checking",
                  "No timing study has been run with real operators.")]
        cw, ch = 700, 520; x0 = (W - 2 * cw - 60) / 2; y0 = 330
        for i, (st, h1, h2, body) in enumerate(cards):
            k = ramp(t, st, 0.45) * a3
            if k <= 0: continue
            x = x0 + i * (cw + 60); y = y0 + (1 - ramp(t, st, 0.45)) * 30
            d.rounded_rectangle([x, y, x + cw, y + ch], radius=18, fill=rgba((255, 255, 255), k), outline=rgba(A_LINE, k), width=2)
            if i == 0:
                for j in range(4): _bottle(d, x + 60 + j * 70, y + 230, 150, (205, 190, 170), k)
                text(d, (x + 380, y + 150), "?", F("x", 120), A_TERRA, k, anchor="mm")
            else:
                cx, cy, r = x + 150, y + 150, 72
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=rgba((205, 190, 170), k), width=10)
                d.rounded_rectangle([cx - 14, cy - r - 30, cx + 14, cy - r - 8], radius=4, fill=rgba((205, 190, 170), k))
                d.line([(cx, cy), (cx, cy - 46)], fill=rgba((205, 190, 170), k), width=8)
                d.line([(cx, cy), (cx + 34, cy + 14)], fill=rgba((205, 190, 170), k), width=8)
                text(d, (x + 380, y + 150), "?", F("x", 120), A_TERRA, k, anchor="mm")
            chip(d, (x + cw - 36, y + 36), "NOT MEASURED", F("s", 20), A_TINT_TX, A_TINT_BG, k, anchor="r")
            text(d, (x + 50, y + 290), h1, F("b", 40), A_INK, k)
            text(d, (x + 50, y + 345), h2, F("m", 28), A_MUTED, k)
            text(d, (x + 50, y + 405), wrap(body, F("r", 26), cw - 100), F("r", 26), A_MUTED, k, spacing=8)
    fr.comp(l)
    # ---- 4. Potential value, with the review-required screenshot
    if k4 > 0.01:
        a = k4; sc = 860 / APP.height; r = (960, 110, APP.width * sc, 860)
        paste_card(fr, APP, r, (0, 0, APP.width, APP.height), alpha=a)
        l, d = fr.layer()
        m = mapper(r, (0, 0, APP.width, APP.height))
        ring(d, (*m(14, 246), *m(121, 270)), ramp(t, CUE4["certify"], 0.6), c=A_TERRA, a=a, width=5, pad=8)
        kr = ramp(t, CUE4["defectfree"] - 0.3, 0.5) * a
        if kr > 0:
            ring(d, (*m(14, 368), *m(510, 420)), 1, c=A_TEAL, a=kr, width=4, pad=6)
            p = m(14, 420)
            chip(d, (p[0] - 6, p[1] + 14), "Relative scores, not defect probabilities", F("m", 22), (255, 255, 255), A_TEAL, kr)
        chip(d, (r[0] + r[2] - 14, r[1] + r[3] - 60), "Previously recorded demonstration", F("m", 20), A_INK, (241, 236, 233), 0.95 * a, anchor="r")
        text(d, (110, 250), "POTENTIAL VALUE", F("s", 28), A_TERRA, a)
        text(d, (110, 295), wrap("A clearer inspection review process", F("x", 66), 780), F("x", 66), A_TITLE, a * ramp(t, CUE4["clearer"] - 0.4, 0.5), spacing=6)
        text(d, (110, 470), wrap("Evidence across views, a highlighted object and a separate human decision.", F("r", 30), 760),
             F("r", 30), A_MUTED, a * ramp(t, CUE4["clearer"] + 0.6, 0.5), spacing=10)
        kb = ramp(t, CUE4["but"], 0.45) * a
        if kb > 0:
            d.rounded_rectangle([110, 640, 880, 800], radius=18, fill=rgba(A_TINT_BG, kb))
            d.rectangle([110, 660, 117, 780], fill=rgba(A_TERRA, kb))
            text(d, (150, 672), "But the prototype cannot certify", F("b", 34), A_TINT_TX, kb)
            text(d, (150, 722), "that a batch is defect-free", F("b", 34), A_TINT_TX, ramp(t, CUE4["certify"], 0.4) * kb)
            text(d, (110, 830), "A person records the final decision.", F("m", 28), A_MUTED, ramp(t, CUE4["certify"] + 0.4, 0.5) * kb)
        fr.comp(l)
    return fr

# ---------- V05 ----------
# ---------- V05 (next steps and conclusion, app theme, no screenshot) ----------
CUE5 = dict(nxt=1.23, improve=1.83, indep=2.88, normal=3.89, unusual=4.55, s1=5.82,
            measured=7.60, turning=8.74, evidence=9.77, operator=10.48, review=11.25, end=11.57)

def _cutout(fr, v, n, x, y, h, a):
    """Paste the segmented bottle (from the V02 masks) with its bottom-centre at (x, y)."""
    c = OBJ[v][n]["crop"]; w = h * c.width / c.height
    im = c.resize((max(1, int(w)), max(1, int(h))), Image.BICUBIC)
    if a < 1: im.putalpha(im.getchannel("A").point(lambda q: int(q * a)))
    fr.im.alpha_composite(im, (int(x - w / 2), int(y - h)))
    return (x - w / 2, y - h, x + w / 2, y)

def v05(t):
    seg, S, T = TS("V05"); fr = Frame(A_BG)
    kAo = ramp(t, CUE5["s1"] - 0.35, 0.35); kB = ramp(t, CUE5["s1"] - 0.05, 0.5)
    # ---- phase A: next steps
    aA = 1 - kAo
    if aA > 0.01:
        l, d = fr.layer()
        text(d, (W / 2, 150), "NEXT STEPS", F("s", 30), A_TERRA, aA * ramp(t, 0.2, 0.4), anchor="mm")
        cw, ch = 780, 640; x0 = (W - 2 * cw - 60) / 2; y0 = 250
        k1 = ramp(t, CUE5["improve"] - 0.3, 0.45) * aA; k2 = ramp(t, CUE5["indep"] - 0.3, 0.45) * aA
        for i, k in enumerate((k1, k2)):
            if k <= 0: continue
            x = x0 + i * (cw + 60)
            d.rounded_rectangle([x, y0, x + cw, y0 + ch], radius=18, fill=rgba((255, 255, 255), k), outline=rgba(A_LINE, k), width=2)
            d.ellipse([x + 40, y0 + 40, x + 96, y0 + 96], fill=rgba(A_TERRA, k))
            text(d, (x + 68, y0 + 68), str(i + 1), F("b", 30), (255, 255, 255), k, anchor="mm")
        if k1 > 0:
            text(d, (x0 + 120, y0 + 46), "Improve object matching", F("b", 38), A_INK, k1)
            text(d, (x0 + 120, y0 + 98), "Link the same object reliably across views", F("r", 26), A_MUTED, k1)
        if k2 > 0:
            x = x0 + cw + 60
            text(d, (x + 120, y0 + 46), "Independent testing", F("b", 38), A_INK, k2)
            text(d, (x + 120, y0 + 98), "On new normal and unusual groups", F("r", 26), A_MUTED, k2)
        fr.comp(l)
        # card 1 illustration: the same bottle in two views, linked
        if k1 > 0:
            bx = x0 + 230; by = y0 + 560
            b1 = _cutout(fr, "v1", ODD, bx, by, 330, k1)
            b2 = _cutout(fr, "v2", ODD, bx + 330, by, 330, k1)
            l, d = fr.layer()
            ring(d, b1, 1, c=A_TERRA, a=k1, width=4, pad=8); ring(d, b2, 1, c=A_TERRA, a=k1, width=4, pad=8)
            p = (b1[2] + 12, (b1[1] + b1[3]) / 2); q = (b2[0] - 12, (b2[1] + b2[3]) / 2)
            dashed(d, p, q, A_TERRA, k1, width=4, k=ramp(t, CUE5["improve"] + 0.3, 0.6))
            text(d, (bx, by + 30), "View 1", F("m", 22), A_MUTED, k1, anchor="ma")
            text(d, (bx + 330, by + 30), "View 2", F("m", 22), A_MUTED, k1, anchor="ma")
            fr.comp(l)
        # card 2 illustration: an all-normal group and a group with an unusual item
        if k2 > 0:
            x = x0 + cw + 60
            groups = [("All-normal group", ["A", "C", "A"], CUE5["normal"] - 0.2),
                      ("Group with an unusual item", ["A", ODD, "C"], CUE5["unusual"] - 0.2)]
            for gi, (lab, objs, st) in enumerate(groups):
                kg = ramp(t, st, 0.4) * k2
                if kg <= 0: continue
                gy = y0 + 170 + gi * 230
                l, d = fr.layer()
                d.rounded_rectangle([x + 40, gy, x + cw - 40, gy + 200], radius=14, fill=rgba((248, 244, 241), kg))
                text(d, (x + 70, gy + 100), lab, F("s", 28), A_INK, kg, anchor="lm")
                fr.comp(l)
                for j, n in enumerate(objs):
                    bb = _cutout(fr, "v1", n, x + 520 + j * 80, gy + 188, 170, kg)
                    if n == ODD:
                        l, d = fr.layer(); ring(d, bb, 1, c=A_TERRA, a=kg, width=3, pad=5); fr.comp(l)
    # ---- phase B: closing, photographs -> evidence -> operator review
    if kB > 0.01:
        a = kB
        l, d = fr.layer()
        text(d, (150, 110), "Visual Peer Inspector", F("x", 84), A_TITLE, a)
        d.rectangle([150, 222, 150 + 520 * ramp(t, CUE5["s1"] + 0.4, 0.8), 225], fill=rgba(A_TERRA, a))
        text(d, (150, 250), "A working prototype with measured limits", F("r", 36), A_MUTED, ramp(t, CUE5["measured"] - 0.3, 0.5) * a)
        fr.comp(l)
        cw, ch, gap = 480, 400, 90; x0 = (W - 3 * cw - 2 * gap) / 2; y0 = 380
        stages = [("Photographs", CUE5["turning"]), ("Evidence", CUE5["evidence"]), ("Operator review", CUE5["operator"])]
        for i, (lab, st) in enumerate(stages):
            k = ramp(t, st - 0.15, 0.45) * a
            if k <= 0: continue
            x = x0 + i * (cw + gap); y = y0 + (1 - ramp(t, st - 0.15, 0.45)) * 25
            l, d = fr.layer()
            d.rounded_rectangle([x, y, x + cw, y + ch], radius=18, fill=rgba((255, 255, 255), k), outline=rgba(A_LINE, k), width=2)
            text(d, (x + cw / 2, y + ch - 44), lab, F("b", 32), A_INK, k, anchor="mm")
            if i > 0:
                ax = x - gap + 18; ay = y + ch / 2 - 30
                d.line([(ax, ay), (x - 22, ay)], fill=rgba(A_TERRA, k), width=4)
                d.polygon([(x - 18, ay), (x - 34, ay - 11), (x - 34, ay + 11)], fill=rgba(A_TERRA, k))
            fr.comp(l)
            if i == 0:      # the two original photos
                ph = 250; pw = ph * V1.width / V1.height
                paste_card(fr, V1, (x + cw / 2 - pw - 10, y + 50, pw, ph), (0, 0, V1.width, V1.height), alpha=k, radius=10)
                paste_card(fr, V2, (x + cw / 2 + 10, y + 50, pw, ph), (0, 0, V2.width, V2.height), alpha=k, radius=10)
            elif i == 1:    # the selected object, isolated in both views
                b1 = _cutout(fr, "v1", ODD, x + cw / 2 - 80, y + 310, 220, k)
                b2 = _cutout(fr, "v2", ODD, x + cw / 2 + 80, y + 310, 220, k)
                l, d = fr.layer()
                ring(d, b1, 1, c=A_TERRA, a=k, width=4, pad=6); ring(d, b2, 1, c=A_TERRA, a=k, width=4, pad=6)
                chip(d, (x + cw / 2, y + 16), "Same object, both views", F("m", 20), A_TINT_TX, A_TINT_BG, k, anchor="c")
                fr.comp(l)
            else:           # a person recording the decision
                l, d = fr.layer()
                hx, hy = x + cw / 2 - 70, y + 150
                d.ellipse([hx - 40, hy - 110, hx + 40, hy - 30], fill=rgba(A_TEAL, k))
                d.rounded_rectangle([hx - 70, hy - 18, hx + 70, hy + 100], radius=40, fill=rgba(A_TEAL, k))
                cx, cy = x + cw / 2 + 110, y + 150
                kc = ramp(t, CUE5["review"], 0.4) * k
                d.ellipse([cx - 56, cy - 56, cx + 56, cy + 56], fill=rgba(A_TINT_BG, kc))
                d.line([(cx - 26, cy + 2), (cx - 6, cy + 24), (cx + 30, cy - 22)], fill=rgba(A_TINT_TX, kc), width=9, joint="curve")
                text(d, (x + cw / 2, y + 300), "Decision recorded by a person", F("r", 22), A_MUTED, kc, anchor="mm")
                fr.comp(l)
        l, d = fr.layer()
        text(d, (W / 2, 900), "Sreekar Reddy Edulapalli   ·   Vaibhav Bairathi", F("s", 36), A_INK, ramp(t, CUE5["end"] + 0.1, 0.6) * a, anchor="mm")
        fr.comp(l)
    return fr

# ---------- Credits (movie-style roll, app theme) ----------
ROLL = [("title", "Visual Peer Inspector"),
        ("gap", 70), ("name", "Sreekar Reddy Edulapalli"), ("name", "Vaibhav Bairathi"),
        ("gap", 110), ("role", "PROJECT SUPERVISOR"), ("name", "Dr. Maoying Qiao"),
        ("gap", 90), ("role", "SUBJECT COORDINATOR"), ("name", "Rene Leveaux"),
        ("gap", 90), ("role", "DATASET"), ("name", "ToysAD-8K, Bhunia et al. (2025)")]
ROLL += [("gap", 90), ("role", "STOCK FOOTAGE")] + [("name", x) for x in STOCK_CREDITS] if STOCK_CREDITS else []
ROLL += [("gap", 90), ("role", "NARRATION")] + [("name", VOICE_CREDIT)] if VOICE_CREDIT else []
ROLL += [("gap", 130), ("small", "University of Technology Sydney"), ("small", "32040 Industry Project")]
_roll_cache = {}

def _roll_image():
    if "img" in _roll_cache: return _roll_cache["img"]
    items, y = [], 0
    for kind, v in ROLL:
        if kind == "gap": y += v; continue
        f, c, h = {"title": (F("x", 112), A_TITLE, 150), "name": (F("r", 46), A_INK, 72),
                   "role": (F("s", 26), A_TERRA, 50), "small": (F("m", 32), A_MUTED, 50)}[kind]
        items.append((kind, v, f, c, y)); y += h
    img = Image.new("RGBA", (W, y + 40), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    for kind, v, f, c, yy in items:
        d.text((W / 2, yy), v, font=f, fill=(*c, 255), anchor="ma")
        if kind == "title": d.rectangle([W / 2 - 300, yy + 132, W / 2 + 300, yy + 135], fill=(*A_TERRA, 255))
    _roll_cache["img"] = img; _roll_cache["last"] = items[-1][4]
    return img

def credits(t):
    fr = Frame(A_BG); img = _roll_image()
    # scroll from below the screen until the last line sits in the middle, then hold
    stop = 9.0; y_start = H - 70; y_end = H / 2 - _roll_cache["last"] - 25
    k = clamp(t / stop)
    y = lerp(y_start, y_end, k)              # constant speed, like a film roll
    fr.im.alpha_composite(img, (0, int(round(y))))
    # soft fade at the top and bottom edges
    l, d = fr.layer()
    for i in range(120):
        a = (1 - i / 120) ** 1.6
        d.line([(0, i), (W, i)], fill=rgba(A_BG, a)); d.line([(0, H - 1 - i), (W, H - 1 - i)], fill=rgba(A_BG, a))
    fr.comp(l)
    return fr

# ---------- placeholder for Sreekar's section (used only when his clips are not supplied) ----------
def slate(t):
    fr = Frame(); l, d = fr.layer()
    text(d, (W / 2, 470), "SREEKAR'S APP DEMONSTRATION  ·  S01 to S05", F("s", 36), A_TERRA, 1, anchor="mm")
    text(d, (W / 2, 560), "Replace this slot with the five clips (0:50 to 3:00, 130 s)", F("r", 36), A_MUTED, 1, anchor="mm")
    fr.comp(l); return fr

SEGS = {"V01": (v01, 25), "V02": (v02, 25), "V03": (v03, 40), "V04": (v04, 25), "V05": (v05, 15),
        "C01": (credits, 10), "SLATE": (slate, 130)}

def render_video(fn, dur, out):
    """Pipe raw RGB frames into ffmpeg (H.264, yuv420p) with a silent stereo track."""
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                          "-map", "0:v", "-map", "1:a", "-shortest", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out],
                         stdin=subprocess.PIPE)
    for i in range(int(dur * FPS)):
        p.stdin.write(fn(i / FPS).rgb().tobytes())
    p.stdin.close(); p.wait()
    if p.returncode: raise SystemExit(f"ffmpeg failed for {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", help="comma-separated clip ids, e.g. V01,V03 (default: V01-V05 and C01)")
    ap.add_argument("--stills", help="clip id to export preview PNGs for")
    ap.add_argument("--times", default="2,8,14", help="comma-separated seconds for --stills")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.stills:
        fn, _ = SEGS[args.stills]
        for ts in [float(x) for x in args.times.split(",")]:
            path = os.path.join(args.out, f"{args.stills}_{ts:05.1f}.png"); fn(ts).rgb().save(path); print(path)
    else:
        for n in (args.only.split(",") if args.only else ["V01", "V02", "V03", "V04", "V05", "C01"]):
            fn, dur = SEGS[n]
            render_video(fn, dur, os.path.join(args.out, f"{n}_silent.mp4" if n.startswith("V") else f"{n}.mp4"))
            print("rendered", n, flush=True)
