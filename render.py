"""Shaded 3D rendering of kinematic poses (MuJoCo renderer, mocap capsules) and GIF/filmstrip output."""
from __future__ import annotations
import os, subprocess, tempfile, shutil
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

PALETTE = {
    "human": [(0.55, 0.57, 0.62), (0.30, 0.45, 0.78), (0.93, 0.55, 0.20)],
    "dog": [(0.62, 0.45, 0.30), (0.30, 0.45, 0.78), (0.93, 0.55, 0.20)],
}
SIDE_TINT = {"L": (0.22, 0.42, 0.80), "R": (0.92, 0.52, 0.18), "C": (0.55, 0.57, 0.62)}

def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q

def z_to_quat(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    q = np.zeros(4); mujoco.mju_quatZ2Vec(q, v / max(n, 1e-9)); return q

class Actor:
    """One creature instance in a scene."""
    def __init__(self, creature, tag, tint_mode="sides", base=None):
        self.c = creature; self.tag = tag; self.tint_mode = tint_mode; self.base = base

class Scene:
    def __init__(self, terrain=None, actors=(), width=480, height=300, props=(), light_dir=(-0.4, 0.5, -1.0), sky=(0.66, 0.80, 0.94)):
        self.terrain = terrain; self.actors = list(actors); self.W, self.H = width, height
        parts = []
        for a in self.actors:
            sk = a.c.skel
            for i, b in enumerate(sk.bones):
                col = a.base if (a.base is not None and a.tint_mode != "sides") else SIDE_TINT[b.side]
                if a.base is not None and a.tint_mode == "sides":
                    col = tuple(0.5 * x + 0.5 * y for x, y in zip(SIDE_TINT[b.side], a.base))
                if b.group in ("foot",): col = tuple(0.55 * x for x in col)
                rgba = f'{col[0]:.3f} {col[1]:.3f} {col[2]:.3f} 1'
                nm = f"{a.tag}_{i}"
                if b.shape == "box":
                    hs = b.size
                    geom = f'<geom type="box" size="{hs[0]:.4f} {hs[2] if len(hs)>2 else hs[1]:.4f} {hs[1]:.4f}" rgba="{rgba}"/>'
                    # box local frame: x along bone, so size order is (half length, half height, half width): see pose()
                    geom = f'<geom type="box" size="{hs[0]:.4f} {hs[1]:.4f} {hs[2]:.4f}" rgba="{rgba}"/>'
                elif b.shape == "ellipsoid":
                    hs = b.size
                    geom = f'<geom type="ellipsoid" size="{hs[0]:.4f} {hs[1]:.4f} {hs[2]:.4f}" rgba="{rgba}"/>'
                else:
                    geom = f'<geom type="capsule" fromto="0 0 0 0 0 {b.length:.5f}" size="{max(b.radius, 0.004):.4f}" rgba="{rgba}"/>'
                parts.append(f'<body name="{nm}" mocap="true">{geom}</body>')
                # joint knob so that bent limbs read as joints
                if b.group in ("leg", "arm") and b.shape == "capsule":
                    parts.append(f'<body name="{nm}_j" mocap="true"><geom type="sphere" size="{b.radius * 1.12:.4f}" rgba="{rgba}"/></body>')
            for k, e in enumerate(sk.extra):
                parts.append(f'<body name="{a.tag}_x{k}" mocap="true"><geom type="sphere" size="{0.045 * max(e[3], 1.0) ** (1/3):.4f}" rgba="0.55 0.45 0.35 0.9"/></body>')
        for k, p in enumerate(props):
            parts.append(f'<body name="prop{k}" mocap="true">{p["geom"]}</body>')
        tgeoms = "\n".join(terrain.mjcf_geoms()) if terrain else ""
        ground = terrain.ground_mjcf() if terrain else '<geom name="floor" type="plane" size="200 200 0.1" material="grid" contype="1" conaffinity="1"/>'
        xml = f'''<mujoco><visual><headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.05 0.05 0.05"/>
<global offwidth="{width}" offheight="{height}"/><quality shadowsize="2048"/><map znear="0.02"/></visual>
<asset><texture name="sky" type="skybox" builtin="gradient" rgb1="{sky[0]} {sky[1]} {sky[2]}" rgb2="0.95 0.96 0.98" width="64" height="64"/>
<texture name="grid" type="2d" builtin="checker" rgb1="0.80 0.78 0.70" rgb2="0.70 0.68 0.60" width="256" height="256" mark="edge" markrgb="0.6 0.58 0.5"/>
<material name="grid" texture="grid" texrepeat="40 40" reflectance="0.0"/></asset>
<worldbody><light pos="2 -3 6" dir="{light_dir[0]} {light_dir[1]} {light_dir[2]}" directional="true" diffuse="0.65 0.65 0.65" castshadow="true"/>
{ground}
{tgeoms}
{''.join(parts)}
</worldbody></mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.r = mujoco.Renderer(self.model, height, width)
        self.mid = {}
        for a in self.actors:
            for i in range(a.c.skel.n):
                self.mid[(a.tag, i)] = self.model.body_mocapid[self.model.body(f"{a.tag}_{i}").id]
            for i, b in enumerate(a.c.skel.bones):
                if b.group in ("leg", "arm") and b.shape == "capsule":
                    self.mid[(a.tag, i, "j")] = self.model.body_mocapid[self.model.body(f"{a.tag}_{i}_j").id]
            for k in range(len(a.c.skel.extra)):
                self.mid[(a.tag, "x", k)] = self.model.body_mocapid[self.model.body(f"{a.tag}_x{k}").id]
        for k in range(len(props)):
            self.mid[("prop", k)] = self.model.body_mocapid[self.model.body(f"prop{k}").id]
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def pose(self, actor, S, E, R, extra_pts=()):
        sk = actor.c.skel; d = self.data
        for i, b in enumerate(sk.bones):
            m = self.mid[(actor.tag, i)]
            if b.shape == "capsule":
                d.mocap_pos[m] = S[i]; d.mocap_quat[m] = z_to_quat(E[i] - S[i])
            elif b.shape == "box":
                # box centred half way along the bone, axes follow the bone's world rotation
                d.mocap_pos[m] = 0.5 * (S[i] + E[i]) + R[i] @ np.array([0, 0, 0]); d.mocap_quat[m] = mat2quat(R[i])
            else:  # ellipsoid: long axis along the bone (local z)
                d.mocap_pos[m] = 0.5 * (S[i] + E[i]); d.mocap_quat[m] = z_to_quat(E[i] - S[i])
            if (actor.tag, i, "j") in self.mid:
                d.mocap_pos[self.mid[(actor.tag, i, "j")]] = S[i]
        for k, e in enumerate(sk.extra):
            i = sk.idx[e[0]]
            d.mocap_pos[self.mid[(actor.tag, "x", k)]] = S[i] + (E[i] - S[i]) * e[1] + R[i] @ np.array(e[2])

    def set_prop(self, k, pos, quat=(1, 0, 0, 0)):
        m = self.mid[("prop", k)]; self.data.mocap_pos[m] = pos; self.data.mocap_quat[m] = quat

    def render(self, look, dist=4.0, azimuth=90.0, elevation=-10.0):
        mujoco.mj_forward(self.model, self.data)
        self.cam.lookat[:] = look; self.cam.distance = dist; self.cam.azimuth = azimuth; self.cam.elevation = elevation
        self.r.update_scene(self.data, camera=self.cam)
        return self.r.render()

    def close(self):
        self.r.close()

def caption(img, text, sub=None, size=13):
    im = Image.fromarray(img); dr = ImageDraw.Draw(im)
    try: f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception: f = None
    hh = size + 6 + (size + 2 if sub else 0)
    dr.rectangle([0, 0, im.width, hh], fill=(250, 250, 250))
    dr.text((5, 3), text, fill=(20, 20, 20), font=f)
    if sub: dr.text((5, size + 5), sub, fill=(80, 80, 80), font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size - 2) if f else None)
    return np.asarray(im)

def save_gif(frames, path, fps=20, colors=64, max_w=None):
    """ffmpeg palette GIF (smaller and cleaner than PIL's adaptive palette)."""
    tmp = tempfile.mkdtemp()
    try:
        for k, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(tmp, f"f{k:05d}.png"))
        vf = f"fps={fps},split[s0][s1];[s0]palettegen=max_colors={colors}:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%05d.png"), "-vf", vf, "-loop", "0", path], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def filmstrip(frames, n=6, path=None):
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    strip = np.concatenate([frames[i] for i in idx], axis=1)
    if path: Image.fromarray(strip).save(path)
    return strip

def grid(frames, cols):
    rows = []
    for i in range(0, len(frames), cols):
        row = list(frames[i:i + cols])
        while len(row) < cols: row.append(np.full_like(row[0], 255))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)
