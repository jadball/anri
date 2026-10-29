"""Draw the SVG figures of the theory pages' reference frames. Standard library only.

Run from anywhere: ``python docs/source/theory/figures/make_figures.py``. Each 3D sketch is an orthographic projection
of real vectors in the frame it shows, so the geometry (directions, angles, handedness) is right by construction.
"""

from __future__ import annotations

import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "font-family='DejaVu Sans, Helvetica, Arial, sans-serif'"
COL = {"lab": "#222222", "sample": "#1f5fbf", "crystal": "#2e8b3d", "recip": "#7b2fa8", "det": "#d9822b",
       "beam": "#c62828", "ray": "#e65100", "grey": "#888888", "fill": "#fdf1e3"}  # fmt: skip


# ----------------------------------------------------------------------------------------------- vectors
def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scale(a, s):
    return [x * s for x in a]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def unit(a):
    return scale(a, 1 / math.sqrt(dot(a, a)))


def rot_z(t, v):
    c, s = math.cos(t), math.sin(t)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


class Camera:
    """Orthographic view from direction ``eye`` (towards the viewer), z up, mapped to SVG pixels."""

    def __init__(self, eye, px, origin):
        e = unit(eye)
        self.right = unit(cross([0, 0, 1], e))
        self.up = cross(e, self.right)
        self.px, self.origin = px, origin

    def __call__(self, p):
        return (self.origin[0] + self.px * dot(p, self.right), self.origin[1] - self.px * dot(p, self.up))


# ----------------------------------------------------------------------------------------------- svg
class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def line(self, p, q, col, width=2.0, arrow=False, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        m = f" marker-end='url(#a{col[1:]})'" if arrow else ""
        self.items.append(f"<line x1='{p[0]:.1f}' y1='{p[1]:.1f}' x2='{q[0]:.1f}' y2='{q[1]:.1f}' stroke='{col}' "
                          f"stroke-width='{width}' stroke-linecap='round'{d}{m}/>")  # fmt: skip
        if arrow:
            self._markers.add(col)

    def path(self, pts, col, width=1.5, arrow=False, dash=None, fill="none", close=False):
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + (" Z" if close else "")
        da = f" stroke-dasharray='{dash}'" if dash else ""
        m = f" marker-end='url(#a{col[1:]})'" if arrow else ""
        self.items.append(f"<path d='{d}' stroke='{col}' stroke-width='{width}' fill='{fill}' "
                          f"stroke-linejoin='round'{da}{m}/>")  # fmt: skip
        if arrow:
            self._markers.add(col)

    def text(self, p, s, col="#222222", size=15, anchor="middle", weight="normal", style="normal"):
        """Text with simple markup: _x_ for a subscript x."""
        parts = s.split("_")
        out = ""
        for k, part in enumerate(parts):
            if k % 2:
                out += f"<tspan baseline-shift='sub' font-size='72%'>{part}</tspan>"
            else:
                out += part
        self.items.append(f"<text x='{p[0]:.1f}' y='{p[1]:.1f}' fill='{col}' font-size='{size}' "
                          f"text-anchor='{anchor}' font-weight='{weight}' font-style='{style}' {FONT}>{out}</text>")  # fmt: skip

    def rect(self, x, y, w, h, col, fill="none", width=1.5, rx=6):
        self.items.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' stroke='{col}' "
                          f"stroke-width='{width}' fill='{fill}'/>")  # fmt: skip

    def save(self, name):
        markers = "".join(
            f"<marker id='a{c[1:]}' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
            f"orient='auto-start-reverse'><path d='M 0 0 L 10 5 L 0 10 z' fill='{c}'/></marker>"
            for c in sorted(self._markers)
        )
        body = "\n".join(self.items)
        svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {self.w} {self.h}' width='{self.w}' "
               f"height='{self.h}'>\n<defs>{markers}</defs>\n"
               f"<rect x='0' y='0' width='{self.w}' height='{self.h}' rx='10' fill='white'/>\n{body}\n</svg>\n")  # fmt: skip
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            f.write(svg)

    _markers: set


def new_svg(w, h):
    s = Svg(w, h)
    s._markers = set()
    return s


def arc3(cam, centre, a, b, r, n=32):
    """Projected arc of radius r about centre, from direction a to direction b (both unit, the shorter way)."""
    a, b = unit(a), unit(b)
    t = math.acos(max(-1.0, min(1.0, dot(a, b))))
    perp = unit(add(b, scale(a, -dot(a, b))))
    return [cam(add(centre, scale(add(scale(a, math.cos(t * k / n)), scale(perp, math.sin(t * k / n))), r)))
            for k in range(n + 1)]  # fmt: skip


# ----------------------------------------------------------------------------------------------- figures
def chain():
    """The chain of frames: Miller indices to detector pixels."""
    s = new_svg(1050, 190)
    boxes = [("Miller indices", "(h, k, l)", COL["recip"]), ("crystal Cartesian", "g_c_ = B h", COL["crystal"]),
             ("sample", "g_s_ = U g_c_", COL["sample"]), ("lab", "W C R(ω) g_s_", COL["lab"]),
             ("detector", "(slow, fast)", COL["det"])]  # fmt: skip
    arrows = ["B", "U", "ω, χ, wedge, dty", "ray trace"]
    w, gap, y = 166, 44, 50
    for k, (name, eq, col) in enumerate(boxes):
        x = 20 + k * (w + gap)
        s.rect(x, y, w, 76, col, fill="#ffffff", width=2.2)
        s.text((x + w / 2, y + 30), name, col, 16, weight="bold")
        s.text((x + w / 2, y + 56), eq, "#222222", 15, style="italic")
        if k < len(arrows):
            s.line((x + w + 4, y + 38), (x + w + gap - 4, y + 38), "#555555", 2.2, arrow=True)
            s.text((x + w + gap / 2, y - 8), arrows[k], "#555555", 13)
    s.text((525, 165), "reciprocal-lattice vectors carry no 2π: |g| = 1/d; UBI = (U B)⁻¹ holds orientation and strain",
           "#555555", 13)  # fmt: skip
    s.save("frames_chain.svg")


def crystal():
    """Reciprocal lattice vectors and the crystal Cartesian frame (B matrix)."""
    s = new_svg(620, 470)
    cam = Camera([2.0, 1.0, 0.8], 150, (250, 300))
    o = [0, 0, 0]
    # a triclinic reciprocal cell, built as the B matrix builds it: a* along x, b* in the xy plane
    ast = [1.0, 0, 0]
    bst = scale([math.cos(math.radians(70)), math.sin(math.radians(70)), 0], 0.85)
    cst = [0.30, 0.22, 0.95]
    for v, c in ((ast, bst), (bst, cst), (cst, ast)):  # the cell's faces at the origin, faint
        s.path([cam(o), cam(v), cam(add(v, c)), cam(c)], COL["grey"], 0.8, dash="3 3", close=True)
    g = add(add(ast, bst), cst)
    for v in (add(ast, bst), add(bst, cst), add(cst, ast)):
        s.line(cam(v), cam(g), COL["grey"], 0.8, dash="3 3")
    for v, lab, off in (([1.5, 0, 0], "x_c_", (-14, 10)), ([0, 1.4, 0], "y_c_", (14, 6)), ([0, 0, 1.5], "z_c_", (14, 4))):
        s.line(cam(o), cam(v), COL["crystal"], 1.6, arrow=True, dash="6 4")
        s.text(add(cam(v), off), lab, COL["crystal"], 17)
    s.line(cam(o), cam(g), "#222222", 2.4, arrow=True)
    s.text(add(cam(g), (12, -6)), "g_c_ = B (1, 1, 1) = a* + b* + c*", "#222222", 15, anchor="start", style="italic")
    for v, lab, off in ((ast, "a*", (16, 18)), (bst, "b*", (4, 26)), (cst, "c*", (-22, 4))):
        s.line(cam(o), cam(v), COL["recip"], 3, arrow=True)
        s.text(add(cam(v), off), lab, COL["recip"], 18, weight="bold")
    for k, t in enumerate(["crystal Cartesian frame (the B matrix):", "x_c_ along a*", "y_c_ in the plane of a* and b*",
                           "z_c_ perpendicular to a* and b*"]):  # fmt: skip
        s.text((30 if k else 20, 40 + 22 * k), ("• " if k else "") + t,
               COL["crystal"] if not k else "#222222", 14, anchor="start", weight="bold" if not k else "normal")  # fmt: skip
    s.save("frames_crystal.svg")


def lab(eye=(-0.45, -1.0, 0.55), eta_deg=130.0, tth_deg=24.0):
    """Lab and sample frames, the beam, omega, dty, and a diffracted ray to the detector."""
    s = new_svg(900, 600)
    L, half = 6.0, 2.4
    corners = [[L, -half, -half], [L, half, -half], [L, half, half], [L, -half, half]]
    tth, eta = math.radians(tth_deg), math.radians(eta_deg)
    d = [math.cos(tth), -math.sin(tth) * math.sin(eta), math.sin(tth) * math.cos(eta)]  # anri's k_out direction
    P = scale(d, L / d[0])
    # fit the sketch into the canvas above the caption
    probe = Camera(list(eye), 1, (0, 0))
    pts = [probe(c) for c in corners] + [probe(q) for q in ([-3.2, 0, 0], [0, 0, 2.7], [0, 2.4, 0], [0, -1.9, -0.7], [-3.2, 0, -0.6])]
    xs_, ys_ = [q[0] for q in pts], [q[1] for q in pts]
    px = min(820 / (max(xs_) - min(xs_)), 440 / (max(ys_) - min(ys_)))
    cam = Camera(list(eye), px, (40 - px * min(xs_), 40 - px * min(ys_)))
    o = [0, 0, 0]
    s.path([cam(c) for c in corners], COL["det"], 2, fill=COL["fill"], close=True)
    s.text(add(cam([L, half, half]), (-10, -10)), "detector", COL["det"], 16, anchor="end", weight="bold")
    # beam, through the origin to the beam centre
    s.line(cam([-3.2, 0, 0]), cam([-0.15, 0, 0]), COL["beam"], 4, arrow=True)
    s.text(add(cam([-3.2, 0, 0]), (6, 30)), "beam, k_in_", COL["beam"], 16, anchor="start", weight="bold")
    s.line(cam(o), cam([L, 0, 0]), COL["beam"], 1.4, dash="6 5")
    bc = cam([L, 0, 0])
    s.path([(bc[0] - 6, bc[1]), (bc[0] + 6, bc[1])], COL["beam"], 2)
    s.path([(bc[0], bc[1] - 6), (bc[0], bc[1] + 6)], COL["beam"], 2)
    # dty under the turntable, then the turntable centred on the origin, and the sample frame turned by omega
    p1, p2 = cam([0.0, -1.9, -0.7]), cam([0.0, 1.9, -0.7])
    s.line(p2, p1, "#555555", 1.6, arrow=True)
    s.line(p1, p2, "#555555", 1.6, arrow=True)
    s.text(add(p1, (-10, 4)), "dty", "#555555", 15, anchor="end")
    w = math.radians(35)
    ring = [cam([1.15 * math.cos(t / 40 * 2 * math.pi), 1.15 * math.sin(t / 40 * 2 * math.pi), 0]) for t in range(41)]
    s.path(ring, COL["sample"], 1.2, fill="#eaf1fb")
    xs, ys = rot_z(w, [1.9, 0, 0]), rot_z(w, [0, 1.9, 0])
    s.line(cam(o), cam(xs), COL["sample"], 2.2, arrow=True, dash="7 4")
    s.line(cam(o), cam(ys), COL["sample"], 2.2, arrow=True, dash="7 4")
    s.text(add(cam(scale(xs, 1.12)), (4, 16)), "x_s_", COL["sample"], 16, anchor="start")
    s.text(add(cam(scale(ys, 1.12)), (6, -4)), "y_s_", COL["sample"], 16, anchor="start")
    s.path(arc3(cam, o, [1, 0, 0], xs, 1.3), COL["sample"], 1.8, arrow=True)
    s.text(add(cam(rot_z(w / 2, [1.55, 0, 0])), (8, 10)), "ω", COL["sample"], 18, anchor="start", weight="bold")
    # lab axes
    for v, labl, off, anc in (([2.4, 0, 0], "x_l_", (2, -12), "start"), ([0, 2.3, 0], "y_l_", (-6, -8), "end"),
                              ([0, 0, 2.6], "z_l_ = ω axis", (10, 4), "start")):  # fmt: skip
        s.line(cam(o), cam(v), COL["lab"], 2.4, arrow=True)
        s.text(add(cam(v), off), labl, COL["lab"], 16, anchor=anc)
    # the diffracted ray at 2theta, eta
    s.line(cam(o), cam(P), COL["ray"], 2.6, arrow=True)
    s.text(add(cam(P), (10, 4)), "k_out_", COL["ray"], 16, anchor="start", weight="bold")
    s.path(arc3(cam, o, [1, 0, 0], d, 3.4), COL["ray"], 1.8)
    s.text(add(cam(scale(unit(add([1, 0, 0], d)), 3.6)), (8, 6)), "2θ", COL["ray"], 17, anchor="start", weight="bold")
    # eta on the detector: from up (z), clockwise looking downstream, to the spot
    r = math.hypot(P[1], P[2])
    up = cam([L, 0, half * 0.92])
    s.line(bc, up, COL["det"], 1.2, dash="4 3")
    s.line(bc, cam(P), COL["det"], 1.2, dash="4 3")
    eta_arc = [cam([L, -0.45 * r * math.sin(t), 0.45 * r * math.cos(t)]) for t in [eta * k / 40 for k in range(41)]]
    s.path(eta_arc, COL["det"], 1.8, arrow=True)
    lab_eta = cam([L, -0.62 * r * math.sin(eta / 2), 0.62 * r * math.cos(eta / 2)])
    s.text(add(lab_eta, (4, 6)), "η", COL["det"], 18, anchor="start", weight="bold")
    # caption block
    for k, t in enumerate(["lab: x_l_ along the beam, z_l_ up, y_l_ to the left looking downstream",
                           "sample: turned by ω about z_l_ (positive anticlockwise seen from above)",
                           f"η: from up, positive clockwise looking downstream (here {eta_deg:.0f}°); 2θ: from the beam"]):  # fmt: skip
        s.text((20, 526 + 22 * k), t, "#333333", 14, anchor="start")
    s.save("frames_lab.svg")


def gonio(eye=(-0.45, -1.0, 0.42)):
    """The goniometer stack in 3D, exploded: dty on the floor, then wedge (about y), chi (about x), omega (about z).

    Each curved arrow turns the positive (right-handed) way about its axis: it is drawn in the plane of two unit
    vectors u, v with u x v along the axis, from u towards v.
    """
    s = new_svg(900, 700)
    cam = Camera(list(eye), 62, (300, 225))
    grey = "#555555"
    label_x = 640

    def slab(z0, z1, hx, hy, col, fill):  # a box's faces seen from this side: front (-y), left (-x), top
        top = [[-hx, -hy, z1], [hx, -hy, z1], [hx, hy, z1], [-hx, hy, z1]]
        front = [[-hx, -hy, z0], [hx, -hy, z0], [hx, -hy, z1], [-hx, -hy, z1]]
        left = [[-hx, -hy, z0], [-hx, hy, z0], [-hx, hy, z1], [-hx, -hy, z1]]
        for face in (front, left, top):
            s.path([cam(q) for q in face], col, 1.4, fill=fill, close=True)

    def turn(centre, u, v, r, col, label, t0=-50.0, t1=220.0):
        arc = [cam(add(centre, add(scale(u, r * math.cos(math.radians(t))), scale(v, r * math.sin(math.radians(t))))))
               for t in [t0 + (t1 - t0) * k / 48 for k in range(49)]]  # fmt: skip
        s.path(arc, col, 2.6, arrow=True)
        right = max(arc, key=lambda q: q[0])
        s.line((right[0] + 6, right[1]), (label_x - 8, right[1]), col, 0.8, dash="2 3")
        s.text((label_x, right[1] + 5), label, col, 16, anchor="start", weight="bold")

    def axis(p, q, col):
        s.line(cam(p), cam(q), col, 1.2, dash="6 4")

    # floor and dty
    slab(-4.6, -4.35, 2.2, 2.6, COL["lab"], "#eeeeee")
    a, b = cam([2.2, -2.3, -4.35]), cam([2.2, 2.3, -4.35])
    s.line(a, b, grey, 2.4, arrow=True)
    s.line(b, a, grey, 2.4, arrow=True)
    s.line((max(a[0], b[0]) + 6, a[1]), (label_x - 8, a[1]), grey, 0.8, dash="2 3")
    s.text((label_x, a[1] + 5), "dty: translation along y", grey, 16, anchor="start", weight="bold")
    # wedge: rotation about y
    slab(-3.5, -3.0, 1.6, 1.9, grey, "#f4f4f4")
    axis([0, -3.0, -3.25], [0, 2.6, -3.25], grey)
    turn([0, -2.6, -3.25], [0, 0, 1], [1, 0, 0], 0.55, grey, "wedge: rotation about y")
    # chi: rotation about x
    slab(-2.1, -1.6, 1.3, 1.5, grey, "#f4f4f4")
    axis([-2.0, 0, -1.85], [2.6, 0, -1.85], grey)
    turn([2.2, 0, -1.85], [0, 1, 0], [0, 0, 1], 0.55, grey, "χ: rotation about x")
    # omega: a turntable rotating about z, the sample on it at the lab origin
    disc = [cam([1.1 * math.cos(t / 40 * 2 * math.pi), 1.1 * math.sin(t / 40 * 2 * math.pi), -0.6]) for t in range(41)]
    s.path(disc, COL["sample"], 1.6, fill="#eaf1fb")
    axis([0, 0, -0.6], [0, 0, 1.6], COL["sample"])
    s.path([cam([0.3 * math.cos(t / 30 * 2 * math.pi), 0.3 * math.sin(t / 30 * 2 * math.pi), 0])
            for t in range(31)], COL["sample"], 1.6, fill="#cfe0f7")  # fmt: skip
    turn([0, 0, 1.0], [1, 0, 0], [0, 1, 0], 0.7, COL["sample"], "ω: rotation about z")
    s.text(add(cam([0.3, 0, 0]), (8, 18)), "sample", COL["sample"], 14, anchor="start")
    # the beam onto the sample at the lab origin
    s.line(cam([-4.4, 0, 0]), cam([-0.4, 0, 0]), COL["beam"], 4, arrow=True)
    s.text(add(cam([-4.4, 0, 0]), (0, -14)), "beam", COL["beam"], 16, anchor="start", weight="bold")
    # lab axes, off to the side
    o = [-3.6, -1.0, -3.4]
    for v, labl in (([1.2, 0, 0], "x"), ([0, 1.2, 0], "y"), ([0, 0, 1.2], "z")):
        s.line(cam(o), cam(add(o, v)), COL["lab"], 2.2, arrow=True)
        s.text(add(cam(add(o, scale(v, 1.25))), (0, 6)), labl, COL["lab"], 16)
    s.text(add(cam(o), (0, 26)), "lab axes", COL["lab"], 13)
    for k, t in enumerate(["Arrows show the positive (right-handed) sense of each rotation: ω anticlockwise seen from",
                           "above, χ anticlockwise seen from +x, wedge anticlockwise seen from +y. R(ω) acts first, so χ",
                           "and wedge tilt the ω axis: v_lab_ = (0, dty − y_0_, 0) + W C R(ω) v_s_"]):  # fmt: skip
        s.text((20, 626 + 22 * k), t, "#333333", 14, anchor="start")
    s.save("frames_gonio.svg")


def detector():
    """The detector seen from the sample, looking downstream: lab y and z, eta, and the raw pixel axes."""
    s = new_svg(820, 470)
    x0, y0, W, H = 140, 40, 420, 380  # detector face
    s.rect(x0, y0, W, H, COL["det"], fill=COL["fill"], width=2.2, rx=2)
    bc = (x0 + 0.45 * W, y0 + 0.5 * H)  # beam centre
    # Debye-Scherrer ring and a spot
    r = 130
    ring = [(bc[0] + r * math.cos(t / 60 * 2 * math.pi), bc[1] + r * math.sin(t / 60 * 2 * math.pi)) for t in range(61)]
    s.path(ring, COL["ray"], 1.4, dash="5 4")
    eta = math.radians(50)
    spot = (bc[0] + r * math.sin(eta), bc[1] - r * math.cos(eta))  # +eta turns clockwise as seen here
    s.items.append(f"<circle cx='{spot[0]:.1f}' cy='{spot[1]:.1f}' r='7' fill='{COL['ray']}'/>")
    s.line(bc, (bc[0], bc[1] - r - 25), COL["lab"], 1.2, dash="4 3")
    s.line(bc, spot, COL["ray"], 1.4)
    arc = [(bc[0] + 70 * math.sin(eta * k / 30), bc[1] - 70 * math.cos(eta * k / 30)) for k in range(31)]
    s.path(arc, COL["det"], 1.8, arrow=True)
    s.text((bc[0] + 38, bc[1] - 80), "η", COL["det"], 19, weight="bold")
    s.text((bc[0] + 70, bc[1] - 25), "2θ grows outwards", COL["ray"], 13, anchor="start")
    # lab axes at the beam centre, as seen looking downstream: y to the left, z up
    s.line(bc, (bc[0] - 95, bc[1]), COL["lab"], 2.2, arrow=True)
    s.line(bc, (bc[0], bc[1] - 95), COL["lab"], 2.2, arrow=True)
    s.text((bc[0] - 100, bc[1] + 20), "y_l_", COL["lab"], 16)
    s.text((bc[0] - 14, bc[1] - 100), "z_l_", COL["lab"], 16)
    s.items.append(f"<circle cx='{bc[0]:.1f}' cy='{bc[1]:.1f}' r='5' fill='none' stroke='{COL['beam']}' stroke-width='2'/>")
    s.items.append(f"<circle cx='{bc[0]:.1f}' cy='{bc[1]:.1f}' r='1.8' fill='{COL['beam']}'/>")
    s.text((bc[0] + 10, bc[1] + 22), "beam centre (y_center_, z_center_)", COL["beam"], 13, anchor="start")
    # raw pixel axes for the identity o-matrix: pixel (0, 0) at the lower right, fast along +y, slow along +z
    c0 = (x0 + W - 12, y0 + H - 12)
    s.line(c0, (c0[0] - 120, c0[1]), COL["det"], 2.4, arrow=True)
    s.line(c0, (c0[0], c0[1] - 120), COL["det"], 2.4, arrow=True)
    s.text((c0[0] - 125, c0[1] - 8), "fast", COL["det"], 15, anchor="end", weight="bold")
    s.text((c0[0] - 8, c0[1] - 128), "slow", COL["det"], 15, anchor="end", weight="bold")
    s.text((c0[0] + 4, c0[1] + 30), "pixel (0, 0)", COL["det"], 13, anchor="end")
    for k, t in enumerate(["seen from the sample,", "looking downstream", "", "o-matrix = identity:",
                           "slow → z_det_ (up)", "fast → y_det_ (left)", "", "other o-matrices flip",
                           "or transpose the image"]):  # fmt: skip
        s.text((x0 + W + 25, y0 + 20 + 22 * k), t, "#333333", 14, anchor="start")
    s.save("frames_detector.svg")


if __name__ == "__main__":
    chain()
    crystal()
    lab()
    gonio()
    detector()
