"""Adversarial motion priors (Peng et al. 2021, AMP) from motion capture: a discriminator learns to tell the policy's
motion from recorded human motion; how human it looks is a reward during training. Nothing of the recordings is
used at run time: the network still drives the muscles in the physics, the recordings only teach the critic.

    python -m motor.amp data/mocap/cmu            build data/mocap/cmu_amp.npz from CMU ASF/AMC files

Data: the CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu; free for any use, not for resale): walks,
runs, turns and stops of five people (subjects 02, 07, 08, 09, 16, 35). ``data/mocap/get_cmu.sh`` downloads them.

The features compare poses, not skeletons: the direction of every leg and arm segment and of the trunk in the heading
frame (x forward, y left, z up), and the hip velocity in sqrt(g leg) (leg: thigh plus shank) with the turn rate. A transition is the features at two
consecutive control steps (30 Hz). Left and right mirrored copies double the data.
"""
from __future__ import annotations
import glob, math, os, sys
import numpy as np

G = 9.81
HZ = 30
# segment directions (unit vectors, from the first point to the second): independent of a body's proportions, so a
# long-legged recording and a short-legged model compare by pose alone (positions over leg length did not: the
# model's chest stood three standard deviations above the recordings' while standing like them)
SEGS = [(f"{seg}_{s}", f"{a}_{s}", f"{b}_{s}") for s in "LR" for seg, a, b in
        (("thigh", "hip", "knee"), ("shank", "knee", "ankle"), ("foot", "ankle", "ball"), ("uarm", "shoulder", "elbow"),
         ("farm", "elbow", "wrist"))] + [("trunk", "hip_mid", "chest")]
DIM = 3 * len(SEGS) + 4


def heading_frame(hip_L, hip_R):
    """Rotation from world to heading frame and its yaw: y along the hips (left), z up, x = y x z (forward)."""
    y = hip_L - hip_R; y[2] = 0.0; y /= max(np.linalg.norm(y), 1e-9)
    x = np.cross(y, [0.0, 0.0, 1.0])
    return np.stack([x, y, [0.0, 0.0, 1.0]]), math.atan2(x[1], x[0])


def features(pts, vel, yaw_rate, leg):
    """``pts``: world points named in SEGS (hip_mid is made here), ``vel``: world hip velocity, ``yaw_rate``: rad/s."""
    H, _yaw = heading_frame(pts["hip_L"], pts["hip_R"])
    pts = dict(pts, hip_mid=0.5 * (pts["hip_L"] + pts["hip_R"]))
    f = []
    for _n, a, b in SEGS:
        v = H @ (pts[b] - pts[a]); f.append(v / max(np.linalg.norm(v), 1e-9))
    f.append(H @ vel / math.sqrt(G * leg)); f.append([yaw_rate * 0.3])
    return np.concatenate(f).astype(np.float32)


def mirror(F):
    """Mirror features (..., DIM): y negated, left and right swapped, yaw rate negated."""
    F = F.copy(); n = len(SEGS)
    P = F[..., :3 * n].reshape(F.shape[:-1] + (n, 3)).copy()
    P[..., 1] *= -1
    names = [g[0] for g in SEGS]
    idx = [names.index(m[:-1] + ("R" if m[-1] == "L" else "L")) if m[-2:] in ("_L", "_R") else names.index(m) for m in names]
    F[..., :3 * n] = P[..., idx, :].reshape(F.shape[:-1] + (-1,))
    F[..., 3 * n + 1] *= -1; F[..., -1] *= -1
    return F


# ---------------------------------------------------------------------------------------------------- CMU ASF / AMC
def _rot(ax, a):
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[ax]


def _euler(angles, order="xyz"):
    """Rotation of successive rotations about fixed axes in ``order`` (ASF: R = Rz Ry Rx for 'xyz')."""
    R = np.eye(3)
    for ax, a in zip(order, angles):
        R = _rot(ax, a) @ R
    return R


def read_asf(path):
    lines = [l.strip() for l in open(path)]
    bones = {"root": dict(dir=np.zeros(3), length=0.0, axis=np.eye(3), dof=[], parent=None)}
    unit = 1.0
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("length") and unit == 1.0 and ":units" in "".join(lines[max(0, i - 3):i]):
            unit = float(l.split()[1])
        if l == "begin":
            b = {}; i += 1
            while lines[i] != "end":
                t = lines[i].split()
                if t[0] == "name": b["name"] = t[1]
                elif t[0] == "direction": b["dir"] = np.array([float(x) for x in t[1:4]])
                elif t[0] == "length": b["length"] = float(t[1])
                elif t[0] == "axis": b["axis"] = _euler(np.radians([float(x) for x in t[1:4]]), t[4].lower())
                elif t[0] == "dof": b["dof"] = [x.lower() for x in t[1:]]
                i += 1
            b.setdefault("dof", []); bones[b["name"]] = b
        if l.startswith(":hierarchy"):
            i += 2
            while lines[i] != "end":
                t = lines[i].split()
                for ch in t[1:]:
                    bones[ch]["parent"] = t[0]
                i += 1
        i += 1
    scale = 0.0254 / unit                                  # ASF lengths are in inches times ``length`` (0.45)
    for b in bones.values():
        b["length"] = b.get("length", 0.0) * scale
    return bones, scale


def read_amc(path):
    frames = []; cur = None
    for l in open(path):
        t = l.split()
        if not t or l.startswith(("#", ":")):
            continue
        if len(t) == 1 and t[0].isdigit():
            cur = {}; frames.append(cur); continue
        if cur is not None:
            cur[t[0]] = [float(x) for x in t[1:]]
    return frames


def fk(bones, scale, fr):
    """World start and end of every bone (metres, CMU's y-up frame)."""
    order = []
    def visit(n):
        order.append(n)
        for k, b in bones.items():
            if b.get("parent") == n:
                visit(k)
    visit("root")
    M = {}; S = {}; E = {}
    r = fr["root"]
    M["root"] = _euler(np.radians(r[3:6]), "xyz")
    E["root"] = S["root"] = np.array(r[0:3]) * scale
    for n in order[1:]:
        b = bones[n]; vals = fr.get(n, [])
        ang = {"rx": 0.0, "ry": 0.0, "rz": 0.0}
        for d, v in zip(b["dof"], vals):
            ang[d] = math.radians(v)
        C = b["axis"]; L = C @ _euler([ang["rx"], ang["ry"], ang["rz"]], "xyz") @ C.T
        M[n] = M[b["parent"]] @ L
        S[n] = E[b["parent"]]; E[n] = S[n] + b["length"] * (M[n] @ b["dir"])
    return S, E


def to_zup(p):
    return np.array([p[2], p[0], p[1]])                    # CMU (x, y up, z) -> (x', y', z' up): a cyclic swap


def clip_features(asf, amc, hz=HZ):
    bones, scale = asf
    frames = read_amc(amc)
    leg = bones["lfemur"]["length"] + bones["ltibia"]["length"]
    step = max(1, int(round(120 / hz)))
    out = []
    for off in range(step):                                # every phase of the 120 Hz recording
        P = []
        for fr in frames[off::step]:
            S, E = fk(bones, scale, fr)
            z = lambda v: to_zup(v)
            P.append(dict(hip_L=z(S["lfemur"]), hip_R=z(S["rfemur"]), knee_L=z(E["lfemur"]), knee_R=z(E["rfemur"]),
                          ankle_L=z(E["ltibia"]), ankle_R=z(E["rtibia"]), ball_L=z(E["lfoot"]), ball_R=z(E["rfoot"]),
                          shoulder_L=z(S["lhumerus"]), shoulder_R=z(S["rhumerus"]), elbow_L=z(E["lhumerus"]),
                          elbow_R=z(E["rhumerus"]), wrist_L=z(E["lradius"]), wrist_R=z(E["rradius"]), chest=z(E["thorax"])))
        F = []
        for t in range(1, len(P) - 1):
            mid = lambda p: 0.5 * (p["hip_L"] + p["hip_R"])
            vel = (mid(P[t + 1]) - mid(P[t - 1])) * hz / 2
            ya = [heading_frame(P[k]["hip_L"], P[k]["hip_R"])[1] for k in (t - 1, t + 1)]
            yr = math.remainder(ya[1] - ya[0], 2 * math.pi) * hz / 2
            F.append(features(P[t], vel, yr, leg))
        if len(F) > 2:
            out.append(np.array(F))
    return out


def build(src, dst):
    pairs = []
    for asf_path in sorted(glob.glob(os.path.join(src, "*.asf"))):
        subj = os.path.basename(asf_path)[:-4]; asf = read_asf(asf_path)
        for amc in sorted(glob.glob(os.path.join(src, subj + "_*.amc"))):
            for F in clip_features(asf, amc):
                for G_ in (F, mirror(F)):
                    pairs.append(np.concatenate([G_[:-1], G_[1:]], 1))
            print(amc, flush=True)
    X = np.concatenate(pairs).astype(np.float32)
    np.savez_compressed(dst, X=X)
    print("pairs", X.shape)


class Discriminator:
    """LSGAN discriminator on feature transitions with a gradient penalty on the recordings (AMP)."""

    def __init__(self, data_path, lr=1e-4, hidden=256, gp=5.0):
        import torch, torch.nn as nn
        self.torch = torch
        X = np.load(data_path)["X"]
        self.real = torch.tensor(X)
        self.mean = torch.tensor(X.mean(0)); self.std = torch.tensor(X.std(0) + 1e-3)
        self.net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gp = gp

    def _d(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)

    def reward(self, fake):
        with self.torch.no_grad():
            d = self._d(self.torch.tensor(fake))
            return self.torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0).numpy(), float(d.mean())

    def update(self, fake, epochs=2, mb=1024):
        torch = self.torch; F = torch.tensor(fake); n = len(F); out = []
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, mb):
                f = F[perm[s:s + mb]]
                r = self.real[torch.randint(len(self.real), (len(f),))].clone().requires_grad_(True)
                dr = self._d(r); df = self._d(f)
                g = torch.autograd.grad(dr.sum(), r, create_graph=True)[0]
                loss = ((dr - 1) ** 2).mean() + ((df + 1) ** 2).mean() + self.gp * (g ** 2).sum(-1).mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                out.append((float(dr.detach().mean()), float(df.detach().mean())))
        return np.mean(out, 0)

    def state(self):
        return dict(net=self.net.state_dict(), opt=self.opt.state_dict())

    def load(self, st):
        self.net.load_state_dict(st["net"]); self.opt.load_state_dict(st["opt"])


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/mocap/cmu"
    build(src, os.path.join(os.path.dirname(src.rstrip("/")), os.path.basename(src.rstrip("/")) + "_amp.npz"))
