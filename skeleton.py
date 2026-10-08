"""Skeleton description shared by the planner, the physics executor and the retargeter.

Convention (creature frame at rest): x forward, y left, z up. In the rest pose every bone has a
world rotation of identity and points along `dir`; a pose is therefore a set of *delta* rotations,
which is exactly what a retargeter needs. Each bone owns up to three hinge angles (about the
rest-frame axes X, Y, Z); `order` says in which order they compose, leftmost outermost.
Only the left side of a body is written by hand, `mirror` builds the right side.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}

def rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == "X": return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y": return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

def rot_axis(u, a):
    u = np.asarray(u, float); u = u / (np.linalg.norm(u) + 1e-12)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K

def align(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if c < -0.999999:
        p = np.cross(a, [1, 0, 0]); p = p if np.linalg.norm(p) > 1e-6 else np.cross(a, [0, 1, 0])
        return rot_axis(p, np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)

def euler_from_R(R: np.ndarray, order: str) -> np.ndarray:
    """Angles (a1, a2, a3) with R = rot(order[0], a1) @ rot(order[1], a2) @ rot(order[2], a3). Returned
    in the order of the axes X, Y, Z (not of composition)."""
    perm = {"XYZ": (0, 1, 2), "XZY": (0, 2, 1), "YXZ": (1, 0, 2), "YZX": (1, 2, 0), "ZXY": (2, 0, 1), "ZYX": (2, 1, 0)}[order]
    i, j, k = perm
    sign = 1.0 if (j - i) % 3 == 1 else -1.0
    # R = Ri(a1) Rj(a2) Rk(a3); standard Tait-Bryan extraction
    s2 = sign * R[i, k]
    a2 = np.arcsin(np.clip(s2, -1, 1))
    if abs(s2) < 0.99999:
        a1 = np.arctan2(-sign * R[j, k], R[k, k])
        a3 = np.arctan2(-sign * R[i, j], R[i, i])
    else:
        a1 = np.arctan2(sign * R[k, j], R[j, j]); a3 = 0.0
    out = np.zeros(3); out[i], out[j], out[k] = a1, a2, a3
    return out

@dataclass
class Bone:
    name: str
    parent: str | None
    dir: tuple                 # rest direction (unit, creature frame)
    length: float
    mass: float
    radius: float = 0.04
    t: float = 1.0             # attach point along the parent bone (0 start, 1 end)
    offset: tuple = (0.0, 0.0, 0.0)
    order: str = "YXZ"         # composition order of the hinge angles
    lim: dict = field(default_factory=dict)   # axis -> (lo, hi) in degrees; missing axis = locked
    side: str = "C"
    group: str = "trunk"       # trunk | leg | arm | neck | head | tail | wing | foot
    tau: float = 100.0         # peak torque per active axis for the physics executor (N m)
    shape: str = "capsule"     # capsule | box | ellipsoid (render only)
    size: tuple = ()           # box half sizes or ellipsoid radii

def mirror_v(v): return (v[0], -v[1], v[2])

def mirror_lim(lim):
    """Axial vector rule for a reflection through y=0: rotations about X and Z flip sign."""
    out = {}
    for ax, (lo, hi) in lim.items():
        out[ax] = (-hi, -lo) if ax in "XZ" else (lo, hi)
    return out

class Skeleton:
    def __init__(self, name: str, bones: list[Bone], extra: list | None = None):
        self.name = name
        full = []
        for b in bones:
            full.append(b)
            if b.side == "L":
                full.append(Bone(b.name[:-1] + "R" if b.name.endswith("L") else b.name + "_R",
                                 (b.parent[:-1] + "R") if (b.parent and b.parent.endswith("L")) else b.parent,
                                 mirror_v(b.dir), b.length, b.mass, b.radius, b.t, mirror_v(b.offset), b.order,
                                 mirror_lim(b.lim), "R", b.group, b.tau, b.shape, b.size))
        # parents first
        names = [b.name for b in full]; placed = set(); order = []
        def place(b):
            if b.name in placed: return
            if b.parent: place(full[names.index(b.parent)])
            placed.add(b.name); order.append(b)
        for b in full: place(b)
        self.bones = order
        self.n = len(order)
        self.idx = {b.name: i for i, b in enumerate(order)}
        self.parent = np.array([self.idx[b.parent] if b.parent else -1 for b in order])
        self.dir = np.array([np.array(b.dir, float) / np.linalg.norm(b.dir) for b in order])
        self.length = np.array([b.length for b in order])
        self.mass = np.array([b.mass for b in order])
        self.children = [[j for j in range(self.n) if self.parent[j] == i] for i in range(self.n)]
        # rest attach vector: parent start -> own start, creature frame
        self.attach = np.zeros((self.n, 3))
        for i, b in enumerate(order):
            if b.parent:
                p = self.idx[b.parent]
                self.attach[i] = self.dir[p] * self.length[p] * b.t + np.array(b.offset)
        self.extra = list(extra or [])   # (bone name, t along bone, offset in rest frame, kg, label)
        self.axes_order = [b.order for b in order]

    def refresh(self):
        """Recompute the arrays after a Bone field was changed (length, mass, direction)."""
        self.length = np.array([b.length for b in self.bones]); self.mass = np.array([b.mass for b in self.bones])
        for i, b in enumerate(self.bones):
            if b.parent:
                p = self.idx[b.parent]; self.attach[i] = self.dir[p] * self.length[p] * b.t + np.array(b.offset)

    def zeros(self):
        return np.zeros((self.n, 3))

    def fk(self, root_pos, root_R, q):
        """q: (n,3) angles about X,Y,Z per bone (root row ignored). Returns starts, ends, world rotations."""
        n = self.n
        R = np.empty((n, 3, 3)); S = np.empty((n, 3)); E = np.empty((n, 3))
        for i in range(n):
            p = self.parent[i]
            if p < 0:
                R[i] = root_R; S[i] = root_pos
            else:
                o = self.axes_order[i]; a = {"X": q[i, 0], "Y": q[i, 1], "Z": q[i, 2]}
                Rl = rot(o[0], a[o[0]]) @ rot(o[1], a[o[1]]) @ rot(o[2], a[o[2]])
                R[i] = R[p] @ Rl
                S[i] = S[p] + R[p] @ self.attach[i]
            E[i] = S[i] + R[i] @ (self.dir[i] * self.length[i])
        return S, E, R

    def com(self, S, E, R=None, extra_world=None):
        m = self.mass[:, None]
        pts = 0.5 * (S + E)
        tot = self.mass.sum(); c = (pts * m).sum(0)
        for (bn, t, off, kg, _lab) in self.extra:
            i = self.idx[bn]
            Ri = R[i] if R is not None else np.eye(3)
            c = c + kg * (S[i] + (E[i] - S[i]) * t + Ri @ np.array(off)); tot += kg
        return c / tot

    def total_mass(self):
        return float(self.mass.sum() + sum(e[3] for e in self.extra))

    def group_indices(self, group, side=None):
        return [i for i, b in enumerate(self.bones) if b.group == group and (side is None or b.side == side)]
