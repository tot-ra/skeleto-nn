"""Air around a physical bird: lift and drag on wings and tail as flat plates (blade elements), drag on the body.

Every wing bone is cut into a few strips. A strip is a plate with area chord x width and a normal (the bone's local z). With the velocity of the
air relative to the strip (the strip's own velocity: the body's flight plus the flapping) and the angle of attack alpha between the plate and
that velocity: below the stall (about 35 degrees for a beating wing) the force is lift perpendicular to the relative wind (2 pi sin a, saturating at 1.6) with induced drag (C_L^2 / (pi AR e)); after it, a flat
plate's force along its normal; a small skin drag along the plate either way. Lift perpendicular to the wind, not to the plate, is
what makes a beating wing push forward (the Knoller-Betz effect). The body is a drag sample per bone: bare legs are thin cylinders
(Cd 1.0 on the real tarsus, 0.3 on the thicker capsule drawn); the feathered trunk, thighs, neck and head are a streamlined body (Cd 0.3 across, 0.08 end-on), which keeps
the whole bird's lift-to-drag ratio near a real crow's instead of a bundle of sticks'. The
wing's downwash is one velocity for the whole bird (momentum theory: w = L / (2 rho pi b^2/4 V), following the lift over about
a beat): the plates see the air descending, which tilts their lift back and is the induced drag. Thrust comes from the flapping
strips seeing a velocity that tilts the force forward. The forces are applied at the strip centres through applyFT every physics sub-step."""
from __future__ import annotations
import math
import numpy as np
import mujoco

RHO_AIR = 1.2
STALL = 0.6       # rad: about 35 degrees; a beating wing holds a leading-edge vortex far past the steady stall
CLMAX = 1.6       # lift coefficient it saturates at (flapping birds and bats: 1.5 to 2, Muijres et al. 2012)
ASPECT = 6.0      # aspect ratio of a whole bird wing (crow, gull 6 to 9)
DOWNWASH_TAU = 0.2  # s: the induced downwash follows the wings' total lift over about a wingbeat

class Air:
    def __init__(self, pc, rho=RHO_AIR, strips=4, wind=(0.0, 0.0, 0.0)):
        self.pc = pc; self.rho = rho; self.wind = np.array(wind, float); sk = pc.sk
        body, pt, area, normal, kind, camber, axial, aarea, cd = [], [], [], [], [], [], [], [], []
        for i, b in enumerate(sk.bones):
            L = float(sk.length[i]); dvec = sk.dir[i]
            if b.group == "wing" or b.group == "tail":
                if b.shape == "box" and b.size: chord = 2 * b.size[0]
                else: chord = 0.10 * 0.9 if b.group == "tail" else 0.1
                n = strips if b.group == "wing" else 2
                if b.group == "tail": chord = 0.13
                for k in range(n):
                    s = (k + 0.5) / n
                    body.append(i); pt.append(dvec * L * s); area.append(chord * L / n); normal.append(np.array([0.0, 0.0, 1.0])); kind.append(1); camber.append(0.08 if b.group == "wing" else 0.0); axial.append(dvec); aarea.append(0.0); cd.append((0.0, 0.0))
            else:
                r = max(float(b.radius), 0.008)
                body.append(i); pt.append(dvec * L * 0.5); area.append(2 * r * L); normal.append(np.zeros(3)); kind.append(0); camber.append(0.0); axial.append(dvec); aarea.append(math.pi * r * r)
                # the thigh is inside the feathers; the bare shank and foot capsules are drawn about 3x thicker than a
                # crow's tarsus, so their cylinder drag is scaled to the real thickness
                feathered = b.group in ("trunk", "neck", "head") or b.name.startswith("thigh")
                cd.append((0.3, 0.08) if feathered else (0.3, 0.1))
        self.body = np.array(body); self.pt = np.array(pt); self.A = np.array(area); self.n = np.array(normal); self.kind = np.array(kind); self.camber = np.array(camber); self.axial = np.array(axial); self.aarea = np.array(aarea)
        self.cd = np.array(cd, float)
        self.wing = np.array([sk.bones[i].group == "wing" and k == 1 for i, k in zip(body, kind)])
        self.bid = np.array([pc.bid[i] for i in body]); self.root = pc.bid[0]; self.N = len(body)
        # induced downwash of the whole wing (momentum theory, forward flight): w = L / (2 rho S V), S = pi span^2 / 4
        wl = [i for i, b in enumerate(sk.bones) if b.group == "wing"]
        span = sum(float(sk.length[i]) for i in wl)        # both wings' bones, tip to tip
        self.disk = math.pi * span * span / 4.0; self.w_ind = 0.0
        self.lifting = self.kind == 1                       # wing and tail plates see the downwash

    def apply(self):
        pc = self.pc; m = pc.model; d = pc.data
        R = d.xmat[self.bid].reshape(-1, 3, 3); P = d.xpos[self.bid] + np.einsum("nij,nj->ni", R, self.pt)
        com = d.subtree_com[self.root]; cv = d.cvel[self.bid]
        v = cv[:, 3:] + np.cross(cv[:, :3], P - com) - self.wind
        # the plates sit in the air their own lift pushes down: relative to it they move up by w
        v[self.lifting, 2] += self.w_ind
        sp = np.linalg.norm(v, axis=1) + 1e-9
        F = np.zeros((self.N, 3))
        pl = self.kind == 1
        if pl.any():
            nw = np.einsum("nij,nj->ni", R[pl], self.n[pl]); vp = v[pl]; spp = sp[pl]
            s = np.sum(vp * nw, axis=1) / spp
            cam = self.camber[pl]                                         # a cambered wing lifts a little at zero angle of attack
            a_eff = np.arcsin(np.clip(-s, -1, 1)) + cam                   # signed: + = the flow hits the underside
            alpha = np.abs(a_eff)
            # a beating wing keeps its lift far past the steady stall angle (a leading-edge vortex: lift coefficients of
            # 1.5 to 2 up to 35 to 45 degrees, Dickinson et al. 1999); the lift saturates at CLMAX, the stall comes at STALL
            att = np.exp(-(alpha / STALL) ** 4)                             # 1 in attached flow, 0 after the stall
            # a folded wing has less area: the feathers slide over each other, so a strip's area is its spanwise
            # extent (its bone across the body); a wing swept or folded back on the upstroke lifts less
            Rr = d.xmat[self.root].reshape(3, 3)
            across = np.abs(np.einsum("nij,nj->ni", R[pl], self.axial[pl]) @ Rr[:, 1])
            area = self.A[pl] * np.where(self.wing[pl], np.maximum(0.15, across), 1.0)
            q = 0.5 * self.rho * area * spp * spp
            vh = vp / spp[:, None]
            # attached flow: lift perpendicular to the relative wind (with the leading-edge suction of a real wing, the force
            # leans forward when the wing beats down: that is where the thrust of flapping comes from) and induced drag
            ld = nw - np.sum(nw * vh, axis=1)[:, None] * vh
            ld = ld / (np.linalg.norm(ld, axis=1)[:, None] + 1e-9) * np.sign(a_eff)[:, None]
            cl = CLMAX * np.tanh(2 * math.pi * np.sin(alpha) / CLMAX) * att
            # no induced drag per strip: it comes from the downwash (the lift tilts back in the descending air). Counted
            # per strip from its instantaneous lift coefficient it made every stroke pay as if the whole wing were at
            # maximum lift, and flapping cost about fifteen times a real crow's power
            cdi = 0.0 * cl
            # separated flow: a flat plate, the force along its normal
            cn = 1.2 * np.sin(alpha) * (1.0 - att)
            FN = (q * cl)[:, None] * ld - (q * cdi)[:, None] * vh + (q * cn * np.sign(a_eff))[:, None] * nw
            vt = vp - (s * spp)[:, None] * nw
            FT = -0.5 * self.rho * 0.02 * area[:, None] * spp[:, None] * vt
            F[pl] = FN + FT
        bd = ~pl
        axw = np.einsum("nij,nj->ni", R[bd], self.axial[bd]); va = np.sum(v[bd] * axw, axis=1)[:, None] * axw; vn = v[bd] - va
        cx, ca = self.cd[bd, 0][:, None], self.cd[bd, 1][:, None]
        F[bd] = -0.5 * self.rho * (cx * self.A[bd][:, None] * np.linalg.norm(vn, axis=1)[:, None] * vn + ca * self.aarea[bd][:, None] * np.linalg.norm(va, axis=1)[:, None] * va)
        # Stability: the forces are explicit (once per physics step), so no sample may push harder than would remove half
        # of its own relative airspeed within one step, sharing its body's mass with the other samples on that body.
        # Without this, a heron's long, light wing segments flipped their airspeed every step and the simulation blew up.
        if not hasattr(self, "_mcap"):
            share = np.bincount(self.bid, minlength=int(self.bid.max()) + 1)[self.bid]
            self._mcap = 0.5 * m.body_mass[self.bid] / share / m.opt.timestep
        cap = np.minimum(60.0, self._mcap * sp)[:, None]
        mag = np.linalg.norm(F, axis=1)[:, None]; F = F * np.minimum(1.0, cap / np.maximum(mag, 1e-9))
        d.qfrc_applied[:] = 0.0
        for k in range(self.N): mujoco.mj_applyFT(m, d, F[k], np.zeros(3), P[k], int(self.bid[k]), d.qfrc_applied)
        if self.disk > 0:                                     # the downwash follows the lift over about a beat
            lift = max(0.0, float(F[self.lifting, 2].sum()))
            V = max(2.0, float(np.linalg.norm(d.subtree_linvel[self.root] - self.wind)))
            w = lift / (2 * self.rho * self.disk * V)
            self.w_ind += (w - self.w_ind) * min(1.0, m.opt.timestep / DOWNWASH_TAU)
