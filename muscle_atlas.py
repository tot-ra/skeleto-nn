"""Pictures of the leg muscles (``muscles.py``) as the physics has them.

    python muscle_atlas.py [out_dir]     (default showcase/motor)

  muscle_atlas.png    the left leg from the side and from the front, standing and with hip and knee bent: every
                      muscle's path from its origin (filled dot) to its insertion (ring), the wrapping surfaces
                      (kneecap, femoral condyles, femoral head) as circles, and its name
  muscle_twitch.gif   the body hung by the pelvis in the air; each left-leg muscle in turn is excited and let go:
                      what it does on its own against gravity (the network is not involved)
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from bodies import human
from muscles import human_legs
from physics import PhysChar

COLORS = {"ILIO": (214, 39, 40), "GMAX": (148, 103, 189), "HAM": (140, 86, 75), "RF": (227, 119, 194), "VAS": (255, 127, 14),
          "GAS": (44, 160, 44), "SOL": (23, 190, 207), "TA": (31, 119, 180), "GMED": (188, 189, 34), "ADD": (127, 127, 127)}
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(n):
    try:
        return ImageFont.truetype(FONT, n)
    except Exception:
        return ImageFont.load_default()


def pose(pc, q):
    d = pc.data
    d.qpos[:] = pc.model.qpos0; d.qpos[2] = 1.0; d.qpos[pc.qadr] = q
    mujoco.mj_forward(pc.model, d)


def panel(pc, view, title, W=420, H=600, scale=440.0, labels_on=True):
    """One orthographic view of the left leg: ``view`` "side" (x right, z up) or "front" (y right, z up)."""
    im = Image.new("RGB", (W, H), (252, 251, 247)); dr = ImageDraw.Draw(im)
    d = pc.data; sk = pc.sk
    hip = d.xpos[pc.bid[sk.idx["thigh_L"]]].copy()
    ax = (0, 2) if view == "side" else (1, 2)
    flip = 1.0                                                # the front view faces the body: its left on the picture's right
    to = lambda p: (W * 0.5 + flip * (p[ax[0]] - hip[ax[0]]) * scale, H * 0.18 - (p[ax[1]] - hip[ax[1]]) * scale)
    S, E, _R = pc.bone_ends()
    legs = [n for n in ("pelvis", "thigh_L", "shank_L", "foot_L", "toes_L")]
    if view == "front":
        legs += ["thigh_R", "shank_R", "foot_R"]
    for n in legs:                                            # bones
        i = sk.idx[n]
        dr.line([to(S[i]), to(E[i])], fill=(205, 198, 180), width=14 if n != "pelvis" else 30)
        dr.ellipse([to(S[i])[0] - 6, to(S[i])[1] - 6, to(S[i])[0] + 6, to(S[i])[1] + 6], fill=(170, 160, 140))
    m = pc.model
    for g in range(m.ngeom):                                  # wrapping surfaces (cylinders across the joint axis)
        nm = m.geom(g).name
        if nm.startswith("w_") and nm.endswith("_L") and view == "side":
            c = to(d.geom_xpos[g]); r = m.geom_size[g][0] * scale
            dr.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=(150, 150, 150), width=1)
    paths = pc.muscle_paths(); f = font(13); labels = []
    for k, (nm, pts) in enumerate(zip(pc.mnames, paths)):
        if not nm.endswith("_L"):
            continue
        key = nm[:-2]
        if view == "front" and key not in ("GMED", "ADD", "ILIO", "GMAX"):
            continue
        col = COLORS[key]; xy = [to(p) for p in pts]
        dr.line(xy, fill=col, width=5, joint="curve")
        o, e = xy[0], xy[-1]
        dr.ellipse([o[0] - 5, o[1] - 5, o[0] + 5, o[1] + 5], fill=col)
        dr.ellipse([e[0] - 6, e[1] - 6, e[0] + 6, e[1] + 6], outline=col, width=3)
        mid = xy[len(xy) // 2] if len(xy) > 2 else ((o[0] + e[0]) / 2, (o[1] + e[1]) / 2)
        if labels_on:
            labels.append((mid, pc.ms.muscles[k].title, col))
    # labels in two columns at the sides, with leader lines
    labels.sort(key=lambda t: t[0][1])
    for j, (mid, text, col) in enumerate(labels):
        right = mid[0] > W * 0.5 if view == "side" else j % 2 == 1
        x = W - 8 - dr.textlength(text, font=f) if right else 8
        y = 70 + j * (H - 110) / max(len(labels), 1)
        dr.line([mid, (x if not right else x - 4, y + 7) if right else (x + dr.textlength(text, font=f) + 4, y + 7)], fill=col, width=1)
        dr.text((x, y), text, fill=col, font=f)
    dr.text((8, 8), title, fill=(30, 30, 30), font=font(16))
    return im


def atlas(out):
    c = human(); pc = PhysChar(c, torque=True, muscles=human_legs(c))
    idx = {(pc.sk.bones[i].name, "XYZ"[a]): k for k, (i, a) in enumerate(pc.dof_list)}
    q = np.zeros(pc.nj); pose(pc, q)
    a = panel(pc, "side", "Left leg from the side, standing")
    b = panel(pc, "front", "From the front (hip abductors, adductors)")
    q2 = q.copy(); q2[idx[("thigh_L", "Y")]] = np.radians(-60); q2[idx[("shank_L", "Y")]] = np.radians(80); q2[idx[("foot_L", "Y")]] = np.radians(15)
    pose(pc, q2)
    cpanel = panel(pc, "side", "Hip and knee bent: paths wrap the kneecap", labels_on=False)
    im = Image.new("RGB", (a.width * 3, a.height + 40), (252, 251, 247))
    for k, p in enumerate((a, b, cpanel)):
        im.paste(p, (k * a.width, 0))
    ImageDraw.Draw(im).text((8, a.height + 10), "Filled dot: origin. Ring: insertion. Grey circles: wrapping surfaces (femoral head, kneecap, "
                            "femoral condyles). 10 muscles per leg; skeleto-nn muscles.py", fill=(60, 60, 60), font=font(14))
    im.save(os.path.join(out, "muscle_atlas.png"))


def twitch(out, each=1.6, level=0.6):
    """Each left-leg muscle excited in turn, the pelvis held in the air: the leg hangs, the muscle moves it, it falls back."""
    from render import Scene, Actor, caption, save_gif
    c = human(); pc = PhysChar(c, torque=True, muscles=human_legs(c))
    q = np.zeros(pc.nj)
    pc.set_state(np.array([0, 0, c.z0 + 0.3]), np.eye(3), q)
    root = pc.data.qpos[0:7].copy()

    def hold():                                              # a stiff spring and damper on the pelvis (zeroing its
        d = pc.data                                          # speed instead breaks the coupled dynamics of the legs)
        d.qfrc_applied[:] = 0.0; d.xfrc_applied[:] = 0.0
        R = d.xmat[pc.bid[0]].reshape(3, 3); w = R @ d.qvel[3:6]
        rot = 0.5 * (np.cross(R[:, 0], [1, 0, 0]) + np.cross(R[:, 1], [0, 1, 0]) + np.cross(R[:, 2], [0, 0, 1]))
        d.xfrc_applied[pc.bid[0], :3] = 4000.0 * (root[:3] - d.qpos[:3]) - 300.0 * d.qvel[:3] + [0, 0, pc.total_mass * 9.81]
        d.xfrc_applied[pc.bid[0], 3:] = 800.0 * rot - 40.0 * w
    sc = Scene(None, [Actor(c, "a", muscles=pc.nm, skeleton=True)], 480, 400)
    imgs = []; nf = len(pc.free); hz = 30
    left = [k for k, n in enumerate(pc.mnames) if n.endswith("_L")]
    for k in left:
        pc.set_state(root[:3], np.eye(3), q)
        T = int(each * hz)
        for t in range(T):
            a = np.zeros(nf + pc.nm)
            a[nf + k] = level if t < 0.4 * T else 0.0          # contract, then relax
            # the joint motors hold the rest of the body still (the trunk and the other leg)
            pc.step(a, pre=hold)
            if t % 2 == 0:
                S, E, R = pc.bone_ends(); sc.pose(sc.actors[0], S, E, R)
                sc.pose_muscles(sc.actors[0], pc.muscle_paths(), pc.m_act)
                look = np.array([S[0][0] + 0.1, S[0][1], S[0][2] - 0.4])
                state = "contracts" if t < 0.4 * T else "relaxes"
                imgs.append(caption(sc.render(look, 1.9, -90, -3), pc.ms.muscles[k].title, f"{state}  (activation {pc.m_act[k]:.2f})"))
    sc.close()
    save_gif(imgs, os.path.join(out, "muscle_twitch.gif"), fps=15)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "showcase/motor"
    os.makedirs(out, exist_ok=True)
    atlas(out); twitch(out)
    print("wrote", os.path.join(out, "muscle_atlas.png"), os.path.join(out, "muscle_twitch.gif"))
