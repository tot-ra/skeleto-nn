"""Inverse kinematics helpers: planar leg chains (any number of links) and a 3D two-bone arm."""
from __future__ import annotations
import numpy as np
from skeleton import rot, euler_from_R

def basis(d, n):
    """Orthonormal frame with first axis d and second axis along n projected perpendicular to d."""
    e1 = np.asarray(d, float); e1 = e1 / np.linalg.norm(e1)
    e2 = np.asarray(n, float) - (np.asarray(n, float) @ e1) * e1
    nn = np.linalg.norm(e2)
    if nn < 1e-6:
        e2 = np.cross(e1, [0, 0, 1.0]); nn = np.linalg.norm(e2)
        if nn < 1e-6: e2 = np.cross(e1, [1.0, 0, 0]); nn = np.linalg.norm(e2)
    e2 = e2 / nn
    return np.stack([e1, e2, np.cross(e1, e2)], axis=1)

def clamp_lim(sk, bone, axis, ang):
    lim = sk.bones[bone].lim.get(axis)
    if not lim: return ang
    return float(np.clip(ang, np.radians(lim[0]), np.radians(lim[1])))

class PlanarLeg:
    """Planar chain in the leg plane. Link i has rest direction (sin phi_i, -cos phi_i) in (x forward, z up).
    World angle of link i = phi_i - Theta_i with Theta_i the cumulative flexion (positive moves the distal end back)."""
    def __init__(self, sk, leg):
        self.flex = leg.chain[:-1]
        self.L = np.array([sk.length[i] for i in self.flex])
        self.phi = np.array([np.arctan2(sk.dir[i][0], -sk.dir[i][2]) for i in self.flex])
        self.lo = np.array([np.radians(sk.bones[i].lim.get("Y", (-180, 180))[0]) for i in self.flex])
        self.hi = np.array([np.radians(sk.bones[i].lim.get("Y", (-180, 180))[1]) for i in self.flex])
        self.theta = np.zeros(len(self.flex))
        self.prior = np.zeros(len(self.flex))
        self.reach = float(self.L.sum())
        self.w_prior = 0.02

    def fwd(self, th):
        Th = np.cumsum(th); psi = self.phi - Th
        pts = np.cumsum(np.stack([self.L * np.sin(psi), -self.L * np.cos(psi)], axis=1), axis=0)
        return pts

    def solve(self, px, pz, warm=None):
        n = len(self.flex)
        D = np.hypot(px, pz)
        if n == 2:
            L1, L2 = self.L
            Dc = float(np.clip(D, abs(L1 - L2) + 1e-4, (L1 + L2) * 0.9995))
            ca = np.clip((L1 * L1 + Dc * Dc - L2 * L2) / (2 * L1 * Dc), -1, 1); delta = np.arccos(ca)
            ck = np.clip((L1 * L1 + L2 * L2 - Dc * Dc) / (2 * L1 * L2), -1, 1); kappa = np.arccos(ck)
            psi_t = np.arctan2(px, -pz)
            psi1 = psi_t + delta; psi2 = psi1 - (np.pi - kappa)
            Th = np.array([self.phi[0] - psi1, self.phi[1] - psi2])
            th = np.array([Th[0], Th[1] - Th[0]])
        else:
            th = self.theta.copy() if warm is None else warm.copy()
            # damped least squares towards the target, with a pull to the rest zig-zag
            for _ in range(12):
                pts = self.fwd(th); e = np.array([px, pz]) - pts[-1]
                J = np.zeros((2, n))
                Th = np.cumsum(th); psi = self.phi - Th
                for k in range(n):
                    # d/dtheta_k of end point = sum over links j >= k of d(link j)/dTheta_j
                    s = np.zeros(2)
                    for j in range(k, n):
                        s += np.array([-self.L[j] * np.cos(psi[j]), -self.L[j] * np.sin(psi[j])])
                    J[:, k] = s
                lam = 0.01
                A = J.T @ J + (lam + self.w_prior) * np.eye(n)
                g = J.T @ e - self.w_prior * (th - self.prior)
                th = th + np.linalg.solve(A, g)
                th = np.clip(th, self.lo, self.hi)
                if np.linalg.norm(e) < 1e-4: break
        th = np.clip(th, self.lo, self.hi)
        self.theta = th
        return th

def leg_ik(sk, leg, pl: PlanarLeg, parent_R, hip_w, ankle_w, gamma=0.0):
    """Returns angle rows for the flex bones [(X,Y,Z)...]: hip (alpha, theta1, gamma), then (0, theta_k, 0)."""
    F = parent_R @ rot("Z", gamma)
    v = F.T @ (ankle_w - hip_w)
    alpha = np.arctan2(v[1], -v[2]) if v[2] < 0 else np.arctan2(v[1], 1e-3)
    alpha = clamp_lim(sk, leg.chain[0], "X", alpha)
    rp = np.sqrt(v[1] ** 2 + v[2] ** 2) * (1.0 if v[2] < 0 else -1.0)
    th = pl.solve(v[0], -rp)
    rows = [np.array([alpha, th[0], gamma])]
    for k in range(1, len(th)):
        rows.append(np.array([0.0, th[k], 0.0]))
    return rows

def arm_ik(sk, arm, parent_R, shoulder_w, target_w, pole_w):
    """3D two-bone arm. Returns (q_upper, q_lower, elbow_world). Upper arm rest dir (0,0,-1), hinge axis Y."""
    u, f = arm.chain[0], arm.chain[1]
    L1, L2 = sk.length[u], sk.length[f]
    w = target_w - shoulder_w; D = np.linalg.norm(w)
    Dc = float(np.clip(D, abs(L1 - L2) + 1e-4, (L1 + L2) * 0.999))
    wn = w / max(D, 1e-9)
    p = pole_w - (pole_w @ wn) * wn; pn = np.linalg.norm(p)
    p = p / pn if pn > 1e-6 else np.cross(wn, [0, 0, 1.0])
    a = (L1 * L1 + Dc * Dc - L2 * L2) / (2 * L1 * Dc); a = np.clip(a, -1, 1)
    E = shoulder_w + wn * (L1 * a) + p * (L1 * np.sqrt(max(0.0, 1 - a * a)))
    hand = shoulder_w + wn * Dc
    d_ua = (E - shoulder_w) / L1; d_fa = (hand - E) / L2
    c = np.cross(d_ua, d_fa); cn = np.linalg.norm(c)
    ang = np.arccos(np.clip(d_ua @ d_fa, -1, 1))
    n_w = -c / cn if cn > 1e-6 else -np.cross(d_ua, p)
    B_rest = basis(sk.dir[u], [0, 1.0, 0]); B_new = basis(d_ua, n_w)
    R_ua = B_new @ B_rest.T
    qu = euler_from_R(parent_R.T @ R_ua, sk.axes_order[u])
    el = -ang
    el = clamp_lim(sk, f, "Y", el)
    return qu, np.array([0.0, el, 0.0]), E, R_ua
