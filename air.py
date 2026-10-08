"""Air around a physical bird: lift and drag on wings and tail as flat plates (blade elements), drag on the body.

Every wing bone is cut into a few strips. A strip is a plate with area chord x width and a normal (the bone's local z). With the velocity of the
air relative to the strip (the strip's own velocity: the body's flight plus the flapping) and the angle of attack alpha between the plate and
that velocity, the force is
    N = 0.5 rho A |v|^2 C_N(alpha),  C_N = 1.2 sin(a) + (2 pi - 1.2) sin(a) exp(-(a/0.2)^4)    (lift slope 2 pi up to the stall at ~11 degrees, then a flat plate),
acting along the plate normal against the normal velocity, plus a small skin drag along the plate. Lift, induced drag and stall are not
put in by hand: they come out of the plate being tilted to the flow. The body is a few drag samples (Cd 0.5 on the frontal area). The
wing downwash is not modelled (a lone wing sees undisturbed air); thrust comes from the flapping strips seeing a velocity that tilts the
force forward. The forces are applied at the strip centres through applyFT every physics sub-step."""
from __future__ import annotations
import math
import numpy as np
import mujoco

RHO_AIR = 1.2

class Air:
    def __init__(self, pc, rho=RHO_AIR, strips=4, wind=(0.0, 0.0, 0.0)):
        self.pc = pc; self.rho = rho; self.wind = np.array(wind, float); sk = pc.sk
        body, pt, area, normal, kind, camber, axial, aarea = [], [], [], [], [], [], [], []
        for i, b in enumerate(sk.bones):
            L = float(sk.length[i]); dvec = sk.dir[i]
            if b.group == "wing" or b.group == "tail":
                if b.shape == "box" and b.size: chord = 2 * b.size[0]
                else: chord = 0.10 * 0.9 if b.group == "tail" else 0.1
                n = strips if b.group == "wing" else 2
                if b.group == "tail": chord = 0.13
                for k in range(n):
                    s = (k + 0.5) / n
                    body.append(i); pt.append(dvec * L * s); area.append(chord * L / n); normal.append(np.array([0.0, 0.0, 1.0])); kind.append(1); camber.append(0.08 if b.group == "wing" else 0.0); axial.append(dvec); aarea.append(0.0)
            else:
                r = max(float(b.radius), 0.008)
                body.append(i); pt.append(dvec * L * 0.5); area.append(2 * r * L); normal.append(np.zeros(3)); kind.append(0); camber.append(0.0); axial.append(dvec); aarea.append(math.pi * r * r)
        self.body = np.array(body); self.pt = np.array(pt); self.A = np.array(area); self.n = np.array(normal); self.kind = np.array(kind); self.camber = np.array(camber); self.axial = np.array(axial); self.aarea = np.array(aarea)
        self.bid = np.array([pc.bid[i] for i in body]); self.root = pc.bid[0]; self.N = len(body)

    def apply(self):
        pc = self.pc; m = pc.model; d = pc.data
        R = d.xmat[self.bid].reshape(-1, 3, 3); P = d.xpos[self.bid] + np.einsum("nij,nj->ni", R, self.pt)
        com = d.subtree_com[self.root]; cv = d.cvel[self.bid]
        v = cv[:, 3:] + np.cross(cv[:, :3], P - com) - self.wind
        sp = np.linalg.norm(v, axis=1) + 1e-9
        F = np.zeros((self.N, 3))
        pl = self.kind == 1
        if pl.any():
            nw = np.einsum("nij,nj->ni", R[pl], self.n[pl]); vp = v[pl]; spp = sp[pl]
            s = np.sum(vp * nw, axis=1) / spp
            cam = self.camber[pl]                                         # a cambered wing lifts a little at zero angle of attack
            a_eff = np.arcsin(np.clip(-s, -1, 1)) + cam                   # signed: + = the flow hits the underside
            alpha = np.abs(a_eff)
            cn = 1.2 * np.sin(alpha) + (2 * math.pi - 1.2) * np.sin(alpha) * np.exp(-(alpha / 0.30) ** 4)
            FN = (0.5 * self.rho * self.A[pl] * spp * spp * cn * np.sign(a_eff))[:, None] * nw
            vt = vp - (s * spp)[:, None] * nw
            FT = -0.5 * self.rho * 0.02 * self.A[pl][:, None] * spp[:, None] * vt
            F[pl] = FN + FT
        bd = ~pl
        axw = np.einsum("nij,nj->ni", R[bd], self.axial[bd]); va = np.sum(v[bd] * axw, axis=1)[:, None] * axw; vn = v[bd] - va
        F[bd] = -0.5 * self.rho * (1.0 * self.A[bd][:, None] * np.linalg.norm(vn, axis=1)[:, None] * vn + 0.25 * self.aarea[bd][:, None] * np.linalg.norm(va, axis=1)[:, None] * va)
        mag = np.linalg.norm(F, axis=1)[:, None]; F = F * np.minimum(1.0, 60.0 / np.maximum(mag, 1e-9))
        d.qfrc_applied[:] = 0.0
        for k in range(self.N): mujoco.mj_applyFT(m, d, F[k], np.zeros(3), P[k], int(self.bid[k]), d.qfrc_applied)
