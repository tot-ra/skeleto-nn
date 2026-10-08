"""Water around a physical body: buoyancy and drag on every segment, flat plates for hands and feet, a free surface with air above.

Each bone is a capsule (or, for the hands and feet, a capsule plus a flat plate). Along the bone a few sample points carry
  * buoyancy  rho g V / n  where the sample is below the surface (smooth over the thickness of the segment),
  * drag      -0.5 rho Cd A |v_n| v_n  for the velocity component normal to the bone (Cd 1.1) and a small axial drag,
  * plates    -0.5 rho Cp A |v.n| (v.n) n  for the palm (area 0.014 m^2) and the sole (0.02 m^2) with Cp 1.3: the hand and the foot push on the water,
                and the water pushes back, which is the only way a swimmer moves.
The forces are applied at the sample points through MuJoCo's applyFT every physics sub-step. Added mass is not modelled (it is
roughly the mass of the displaced water, and matters for fast limb reversals); drag is clipped for stability."""
from __future__ import annotations
import math
import numpy as np
import mujoco

RHO = 1000.0; G = 9.81
HUMAN_DENSITY = 985.0            # kg/m^3 with lungs half full: slightly buoyant

class Water:
    def __init__(self, pc, surface_z, density=RHO, n_samples=3, palm_area=0.014, sole_area=0.02, cd_n=1.1, cd_a=0.25, cd_plate=1.3, body_density=HUMAN_DENSITY):
        self.pc = pc; self.W = surface_z; self.rho = density; sk = pc.sk; c = pc.c
        self.plate_cd = cd_plate; self.cd_n = cd_n; self.cd_a = cd_a
        hands = {a.chain[-1] for a in c.arms}; feet = {l.chain[-1] for l in c.legs}
        body, pt, vol, area_n, area_a, plate_n, plate_a = [], [], [], [], [], [], []
        for i, b in enumerate(sk.bones):
            L = float(sk.length[i]); r = max(float(b.radius), 0.008); dvec = sk.dir[i]
            V = float(sk.mass[i]) / body_density                                  # the body displaces its own mass at this density
            n = n_samples if L > 0.12 else 1
            for k in range(n):
                s = (k + 0.5) / n
                body.append(i); pt.append(dvec * L * s); vol.append(V / n); area_n.append(2 * r * L / n); area_a.append(math.pi * r * r / n)
                pn = np.zeros(3); pa = 0.0
                if i in hands: pn = np.array([1.0, 0.0, 0.0]) if k == n - 1 else pn; pa = palm_area if k == n - 1 else 0.0
                if i in feet: pn = np.array([0.0, 0.0, 1.0]) if k == n - 1 else pn; pa = sole_area if k == n - 1 else 0.0
                plate_n.append(pn); plate_a.append(pa)
        self.body = np.array(body); self.pt = np.array(pt); self.vol = np.array(vol); self.An = np.array(area_n); self.Aa = np.array(area_a)
        self.pn = np.array(plate_n); self.pa = np.array(plate_a); self.r = np.array([max(float(sk.bones[i].radius), 0.008) for i in body])
        self.bid = np.array([pc.bid[i] for i in body]); self.axis_local = np.array([sk.dir[i] for i in body])
        self.n = len(body); self.root = pc.bid[0]
        self.last_power = 0.0

    def apply(self):
        """Add the water forces to qfrc_applied for the current state (call before each mj_step)."""
        pc = self.pc; m = pc.model; d = pc.data
        R = d.xmat[self.bid].reshape(-1, 3, 3); P = d.xpos[self.bid] + np.einsum("nij,nj->ni", R, self.pt)
        com = d.subtree_com[self.root]; cv = d.cvel[self.bid]                              # [rot; lin] about the whole-body COM
        v = cv[:, 3:] + np.cross(cv[:, :3], P - com)
        ax = np.einsum("nij,nj->ni", R, self.axis_local)
        v_a = np.sum(v * ax, axis=1)[:, None] * ax; v_n = v - v_a
        # submerged fraction (smooth over the segment's thickness)
        sub = np.clip((self.W - P[:, 2]) / (2 * self.r) + 0.5, 0.0, 1.0)
        F = np.zeros((self.n, 3))
        F[:, 2] += self.rho * G * self.vol * sub                                           # buoyancy
        vn = np.linalg.norm(v_n, axis=1)[:, None]; va = np.linalg.norm(v_a, axis=1)[:, None]
        F += -0.5 * self.rho * self.cd_n * self.An[:, None] * vn * v_n * sub[:, None]
        F += -0.5 * self.rho * self.cd_a * self.Aa[:, None] * va * v_a * sub[:, None]
        has = self.pa > 0
        if has.any():
            nw = np.einsum("nij,nj->ni", R[has], self.pn[has]); vp = np.sum(v[has] * nw, axis=1)[:, None]
            F[has] += -0.5 * self.rho * self.plate_cd * self.pa[has][:, None] * np.abs(vp) * vp * nw * sub[has][:, None]
        # air above the surface: negligible; keep the forces finite
        mag = np.linalg.norm(F, axis=1)[:, None]; F = F * np.minimum(1.0, 900.0 / np.maximum(mag, 1e-9))
        d.qfrc_applied[:] = 0.0
        for k in range(self.n):
            mujoco.mj_applyFT(m, d, F[k], np.zeros(3), P[k], int(self.bid[k]), d.qfrc_applied)

class Breath:
    """Oxygen: spent while the mouth is under water, faster the harder the muscles work; refilled when the mouth is in the air.
    Muscle strength falls as the reserve runs low, and below zero the swimmer blacks out."""
    def __init__(self, pc, surface_z, rest_time=120.0, power_scale=6000.0, refill=0.45):
        self.pc = pc; self.W = surface_z; self.s = 1.0; self.rest = 1.0 / rest_time; self.ps = power_scale; self.refill = refill; self.blackout = False
        self.hi = pc.sk.idx["head"]

    def mouth(self):
        d = self.pc.data; i = self.hi
        R = d.xmat[self.pc.bid[i]].reshape(3, 3); return d.xpos[self.pc.bid[i]] + R @ np.array([0.085, 0.0, 0.05])

    def step(self, dt, power_w):
        out = self.mouth()[2] > self.W - 0.01
        if out: self.s = min(1.0, self.s + self.refill * dt)
        else: self.s -= (self.rest + max(power_w, 0.0) / self.ps) * dt
        if self.s <= 0.0: self.blackout = True; self.s = 0.0
        return out

    def strength(self):
        return 0.0 if self.blackout else 0.35 + 0.65 * float(np.clip(self.s / 0.3, 0.0, 1.0))
