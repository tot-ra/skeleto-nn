"""Pictures of the line muscles (``muscles.py``) as the physics has them, for every body family.

    python muscle_atlas.py [out_dir]     (default showcase/motor)

  muscle_atlas_<family>.png   orthographic views of the left side: every muscle's path from its origin (filled dot)
                              to its insertion (ring), the wrapping surfaces as grey circles; line width goes with the
                              muscle's strength (diameter ~ square root of its force), named with its strength
  muscle_twitch_<family>.gif  the body pinned in the air by its pelvis; each left-side muscle in turn is excited and let
                              go: what it does on its own against gravity (no network involved). Red and thick while
                              it contracts, pale and thin when relaxed
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from muscles import SETS
from physics import PhysChar

PALETTE = [(214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (255, 127, 14), (44, 160, 44), (23, 190, 207),
           (31, 119, 180), (188, 189, 34), (127, 127, 127), (120, 160, 220), (100, 190, 90), (240, 120, 120), (170, 140, 200),
           (180, 130, 110), (230, 150, 200), (190, 190, 90), (90, 200, 215), (99, 99, 99), (250, 170, 90)]
COLORS = {"ILIO": (214, 39, 40), "GMAX": (148, 103, 189), "HAM": (140, 86, 75), "RF": (227, 119, 194), "VAS": (255, 127, 14),
          "GAS": (44, 160, 44), "SOL": (23, 190, 207), "TA": (31, 119, 180), "GMED": (188, 189, 34), "ADD": (127, 127, 127)}
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
WING = ("PECT", "SUPRA", "PROT", "RETR", "BICW", "TRIW", "EMR", "FCU")


def font(n):
    try:
        return ImageFont.truetype(FONT, n)
    except Exception:
        return ImageFont.load_default()


def color(key, k):
    return COLORS.get(key, PALETTE[k % len(PALETTE)])


def creature(family, species=None):
    from bodies import human, quadruped, bird
    if family == "human":
        return human()
    if family == "quadruped":
        return quadruped(species or "dog")
    return bird(species or "crow")


def build(family, species=None):
    c = creature(family, species)
    return c, PhysChar(c, torque=True, muscles=SETS[family](c))


def pose(pc, q):
    d = pc.data
    d.qpos[:] = pc.model.qpos0; d.qpos[2] = 2.0; d.qpos[pc.qadr] = q
    mujoco.mj_forward(pc.model, d)


def panel(pc, view, title, W=520, H=600, labels_on=True, only=None, side="L"):
    """One orthographic view: ``view`` "side" (x right, z up), "front" (y right, z up) or "top" (y right, x up)."""
    im = Image.new("RGB", (W, H), (252, 251, 247)); dr = ImageDraw.Draw(im)
    d = pc.data; sk = pc.sk; m = pc.model
    ax = {"side": (0, 2), "front": (1, 2), "top": (1, 0)}[view]
    S, E, _R = pc.bone_ends()
    pts = np.concatenate([S, E])
    lo, hi = pts[:, ax].min(0), pts[:, ax].max(0)
    scale = min((W - 60) / max(hi[0] - lo[0], 1e-3), (H - 150) / max(hi[1] - lo[1], 1e-3))
    cen = 0.5 * (lo + hi)
    to = lambda p: (W * 0.5 + (p[ax[0]] - cen[0]) * scale, 40 + (H - 120) * 0.5 - (p[ax[1]] - cen[1]) * scale)
    for i, b in enumerate(sk.bones):                          # bones
        if view == "side" and b.side not in ("C", side):
            continue
        w = max(3, int(2 * b.radius * scale * (0.95 if b.group == "trunk" else 0.6)))
        dr.line([to(S[i]), to(E[i])], fill=(208, 201, 184), width=w)
        r = max(2, w // 3); c0 = to(S[i])
        dr.ellipse([c0[0] - r, c0[1] - r, c0[0] + r, c0[1] + r], fill=(172, 162, 142))
    for g in range(m.ngeom):                                  # wrapping surfaces
        nm = m.geom(g).name
        if nm.startswith("w_") and nm.endswith("_" + side) and view == "side":
            c0 = to(d.geom_xpos[g]); r = m.geom_size[g][0] * scale
            dr.ellipse([c0[0] - r, c0[1] - r, c0[0] + r, c0[1] + r], outline=(150, 150, 150), width=1)
    paths = pc.muscle_paths(); f = font(12); labels = []
    wmax = max(pc.m_thick.max(), 1e-6)
    keys = sorted({n[:-2] for n in pc.mnames})
    for k, (nm, pts_) in enumerate(zip(pc.mnames, paths)):
        if not nm.endswith("_" + side) or (only and nm[:-2] not in only):
            continue
        key = nm[:-2]; col = color(key, keys.index(key)); xy = [to(p) for p in pts_]
        w = int(round(2 + 9 * pc.m_thick[k] / wmax))
        dr.line(xy, fill=col, width=w, joint="curve")
        o, e = xy[0], xy[-1]
        dr.ellipse([o[0] - 4, o[1] - 4, o[0] + 4, o[1] + 4], fill=col)
        dr.ellipse([e[0] - 5, e[1] - 5, e[0] + 5, e[1] + 5], outline=col, width=2)
        mid = xy[len(xy) // 2] if len(xy) > 2 else ((o[0] + e[0]) / 2, (o[1] + e[1]) / 2)
        if labels_on:
            F = pc.F0[k]
            labels.append((mid, f"{pc.ms.muscles[k].title} " + (f"{F / 1000:.1f} kN" if F >= 1000 else f"{F:.0f} N"), col))
    left = sorted([l for l in labels if l[0][0] < W * 0.5], key=lambda t: t[0][1])
    right = sorted([l for l in labels if l[0][0] >= W * 0.5], key=lambda t: t[0][1])
    for col_labels, is_right in ((left, False), (right, True)):
        n = len(col_labels)
        for j, (mid, text, col) in enumerate(col_labels):
            tw = dr.textlength(text, font=f)
            x = W - 6 - tw if is_right else 6
            y = 40 + j * (H - 120) / max(n, 1)
            dr.line([mid, (x - 3, y + 7) if is_right else (x + tw + 3, y + 7)], fill=col, width=1)
            dr.text((x, y), text, fill=col, font=f)
    dr.text((8, 8), title, fill=(30, 30, 30), font=font(15))
    return im


def strip(panels, note, path):
    W = sum(p.width for p in panels); H = max(p.height for p in panels)
    im = Image.new("RGB", (W, H + 34), (252, 251, 247)); x = 0
    for p in panels:
        im.paste(p, (x, 0)); x += p.width
    ImageDraw.Draw(im).text((8, H + 9), note, fill=(60, 60, 60), font=font(13))
    im.save(path)


NOTE = ("Filled dot: origin. Ring: insertion. Grey circles: wrapping surfaces. Line width ~ square root of the muscle's "
        "strength F0 (its cross-section). skeleto-nn muscles.py")


def leg_force(pc, fore):
    """Summed strength of a quadruped's left hind or fore leg muscles, in body weights."""
    hind = lambda n: n.endswith("H_L")
    trunk = ("EPAX", "RECT", "OBLQ")
    sel = [k for k, n in enumerate(pc.mnames) if n.endswith("_L") and not n.startswith(trunk) and (hind(n) != fore)]
    return sum(pc.F0[k] for k in sel) / (pc.total_mass * 9.81)


def atlas(out):
    c, pc = build("human"); pose(pc, np.zeros(pc.nj))
    n = sum(1 for x in pc.mnames if x.endswith("_L"))
    strip([panel(pc, "side", f"Human, left side: {n} muscles per side", W=640, H=720),
           panel(pc, "front", "From the front: hip, shoulder and trunk", W=520, H=720,
                 only={"GMED", "ADD", "DELTM", "PEC", "LAT", "EO", "ILIO", "RA"})],
          NOTE, os.path.join(out, "muscle_atlas_human.png"))
    panels = []
    for sp in ("dog", "horse", "cat"):
        c, pc = build("quadruped", sp); pose(pc, np.zeros(pc.nj))
        panels.append(panel(pc, "side", f"{sp}: muscle strength, hind leg {leg_force(pc, False):.0f}, fore leg "
                                        f"{leg_force(pc, True):.0f} body weights", W=600, H=560, labels_on=(sp == "dog")))
    strip(panels, "Quadrupeds, left side: 11 muscles per leg, back and belly; the hind legs push, the fore legs carry. " + NOTE,
          os.path.join(out, "muscle_atlas_quadruped.png"))
    c, pc = build("bird", "crow"); pose(pc, np.zeros(pc.nj))
    strip([panel(pc, "front", "Crow from the front: flight muscles on the keel", W=640, H=480, only={"PECT", "SUPRA", "BICW", "TRIW"}),
           panel(pc, "top", "From above: sweep, elbow and wrist", W=560, H=480, only={"PROT", "RETR", "BICW", "TRIW", "EMR", "FCU"}),
           panel(pc, "side", "Side: the leg", W=420, H=480, only={k[:-2] for k in pc.mnames if k[:-2] not in WING})],
          "The supracoracoideus lifts the wing through the triosseal canal, a pulley over the shoulder. " + NOTE,
          os.path.join(out, "muscle_atlas_bird.png"))


def twitch(out, family, species=None, each=1.4, level=0.6, keys=None, fname=None):
    """Each left-side muscle excited in turn, the pelvis pinned in the air: the limb hangs, the muscle moves it, it
    falls back. The joints the muscle does not cross are held still, so what moves is what it moves."""
    from render import Scene, Actor, caption, save_gif
    c = creature(family, species)
    z = c.z0 + 0.45 * c.leg_len()
    pc = PhysChar(c, torque=True, muscles=SETS[family](c), fixed_root=(0.0, 0.0, z))   # the pelvis pinned in the air
    nf = len(pc.free)
    m = pc.model
    jid = np.array([m.dof_jntid[v] for v in pc.vadr])
    stiff0, damp0 = m.jnt_stiffness.copy(), m.dof_damping.copy()

    def hold_except(spanned):
        """The joints the muscle does not cross held by joint springs (MuJoCo's own, integrated stably)."""
        m.jnt_stiffness[:] = stiff0; m.dof_damping[:] = damp0
        k = np.where(spanned, 0.0, 15.0 * pc.tau)
        m.jnt_stiffness[jid] = k; m.dof_damping[pc.vadr] = damp0[pc.vadr] + 0.3 * k
        m.qpos_spring[pc.qadr] = 0.0

    span = c.leg_len() * (3.4 if family == "quadruped" else 4.0 if family == "bird" else 3.3)
    sc = Scene(None, [Actor(c, "a", muscles=pc.nm, skeleton=True)], 480, 400)
    imgs = []; hz = 30
    left = [k for k, n in enumerate(pc.mnames) if n.endswith("_L") and (keys is None or n[:-2] in keys)]
    for k in left:
        pc.set_state(None, None, np.zeros(pc.nj))
        hold_except(np.abs(pc.arm0[k]) > 1e-4)
        T = int(each * hz)
        for t in range(T):
            a = np.zeros(nf + pc.nm)
            a[nf + k] = level if t < 0.4 * T else 0.0          # contract, then relax
            pc.step(a)
            if t % 2 == 0:
                S, E, R = pc.bone_ends(); sc.pose(sc.actors[0], S, E, R)
                sc.pose_muscles(sc.actors[0], pc.muscle_paths(), pc.m_act, thick=pc.m_thick)
                mid = S[0] if family == "human" else 0.5 * (S[0] + E[pc.sk.idx["chest"]])
                look = np.array([mid[0], mid[1], mid[2] - (0.25 if family == "quadruped" else 0.0 if family == "human" else 0.1) * c.leg_len()])
                state = "contracts" if t < 0.4 * T else "relaxes"
                az, el = {"human": (-120, -5), "quadruped": (-90, -3), "bird": (-150, -25)}[family]
                imgs.append(caption(sc.render(look, span, az, el), pc.ms.muscles[k].title, f"{state}  (activation {pc.m_act[k]:.2f})"))
    sc.close()
    save_gif(imgs, os.path.join(out, fname or f"muscle_twitch_{family}.gif"), fps=15)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "showcase/motor"
    os.makedirs(out, exist_ok=True)
    atlas(out)
    twitch(out, "human", keys={"ILIO", "GMAX", "HAM", "VAS", "GAS", "TA", "DELTA", "DELTM", "BIC", "TRI", "RA", "ESL"})
    twitch(out, "quadruped", "dog", keys={"GLUTH", "ILIOH", "QUADH", "GASH", "TCH", "BRC", "TRIF", "BICF", "FCU", "ECR", "EPAX"})
    twitch(out, "bird", "crow", keys={"PECT", "SUPRA", "PROT", "RETR", "TRIW", "BICW", "EMR", "FCU"})
    print("wrote", out)
