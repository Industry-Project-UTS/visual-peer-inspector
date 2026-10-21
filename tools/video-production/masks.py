"""
Foreground cut-outs of the six bottles used in V02 and V05 (illustration only).

The masks come from OpenCV GrabCut seeded with hand-placed boxes and a few hint rectangles.
They are NOT output from SAM 2.1; the clip labels them as an illustration. Output: masks/*.png
"""
import os, cv2, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "assets") + os.sep
BOX = {
 "v1": [("A", (55, 445, 425, 1262)), ("B", (352, 338, 562, 912)), ("C", (748, 415, 1033, 1135))],
 "v2": [("B", (128, 250, 422, 858)), ("A", (420, 425, 705, 1160)), ("C", (720, 215, 925, 825))],
}
HINTS = {  # extra definite-foreground / background rectangles (x0, y0, x1, y1)
 ("v1", "A"): dict(fg=[(140, 465, 245, 555)], bg=[(40, 430, 125, 560), (258, 430, 440, 560), (40, 1000, 140, 1275)]),
 ("v1", "B"): dict(fg=[(432, 352, 478, 520), (400, 420, 510, 520)], bg=[(552, 340, 600, 455)]),
 ("v2", "C"): dict(fg=[(795, 222, 865, 268)], bg=[]),
}
files = {"v1": "V01-01-bottle-view-1.jpeg", "v2": "V01-02-bottle-view-2.jpeg"}
for v, items in BOX.items():
    img = cv2.imread(SRC + files[v])
    for name, (x0, y0, x1, y1) in items:
        pad = 10
        mask = np.full(img.shape[:2], cv2.GC_BGD, np.uint8)
        mask[y0 - pad:y1 + pad, x0 - pad:x1 + pad] = cv2.GC_PR_BGD
        w, h = x1 - x0, y1 - y0
        # probable foreground: the box shrunk a little; definite foreground: central body strip
        mask[y0 + int(.04*h):y1 - int(.02*h), x0 + int(.12*w):x1 - int(.12*w)] = cv2.GC_PR_FGD
        cx0, cx1 = x0 + int(.35*w), x1 - int(.35*w)
        mask[y0 + int(.35*h):y1 - int(.08*h), cx0:cx1] = cv2.GC_FGD
        hint = HINTS.get((v, name), {})
        for a, b, c, d in hint.get("bg", []): mask[b:d, a:c] = cv2.GC_BGD
        for a, b, c, d in hint.get("fg", []): mask[b:d, a:c] = cv2.GC_FGD
        seed_fg = (mask == cv2.GC_FGD).astype(np.uint8)
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, None, bg, fg, 8, cv2.GC_INIT_WITH_MASK)
        m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        # keep the largest component, fill holes, smooth edges
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            seeds = set(np.unique(lab[seed_fg > 0])) - {0}
            seeds.add(1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
            m = np.where(np.isin(lab, list(seeds)), 255, 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m2 = np.zeros_like(m); cv2.drawContours(m2, cnts, -1, 255, -1)
        m2 = cv2.GaussianBlur(m2, (5, 5), 0)
        cv2.imwrite(os.path.join(BASE, "masks", f"{v}_{name}.png"), m2)
        print(v, name, round((m2 > 127).mean() * 100, 2), "% of image")
