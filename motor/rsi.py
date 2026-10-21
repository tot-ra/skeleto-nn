"""Reference starts (reference state initialisation, Peng et al. 2018, DeepMimic): training episodes that begin in a
pose and with the velocities of a recorded walk or run, so the network meets the middle of a fast stride from the
first round instead of having to find it from a standstill. Nothing of the recordings is used at run time, and they
are never a target: an episode that starts from one is scored like any other.

    python -m motor.rsi data/mocap/cmu data/mocap/cmu_rsi.npz      build the start poses (people)

Every 30 Hz frame of the CMU walks and runs (``motor.amp``) is fitted to the default human body: root position and
rotation and every joint angle, by damped least squares on the joints the recording marks (hips, knees, ankles,
balls, toe tips, shoulders, elbows, wrists, hands, chest, neck and head) scaled by leg length, within the joint
ranges, with a light pull of the spine, neck and the joints the markers leave open toward neutral. Velocities are
central differences of the fitted frames. Poses are stored in the frame of the hips' heading of travel (x forward),
with mirrored copies, and the speed of every frame.
"""
from __future__ import annotations
import glob, math, os, sys
import numpy as np
import mujoco

from motor.amp import read_asf, read_amc, fk, to_zup

HZ = 30
# (CMU point, our point, weight): a point is (bone, "S" start | "E" end)
MARKS = [(("lfemur", "S"), ("thigh_L", "S"), 1.0), (("rfemur", "S"), ("thigh_R", "S"), 1.0),
         (("lfemur", "E"), ("thigh_L", "E"), 1.0), (("rfemur", "E"), ("thigh_R", "E"), 1.0),
         (("ltibia", "E"), ("shank_L", "E"), 1.0), (("rtibia", "E"), ("shank_R", "E"), 1.0),
         (("lfoot", "E"), ("toes_L", "S"), 1.0), (("rfoot", "E"), ("toes_R", "S"), 1.0),
         (("ltoes", "E"), ("toes_L", "E"), 0.5), (("rtoes", "E"), ("toes_R", "E"), 0.5),
         (("lhumerus", "S"), ("uarm_L", "S"), 0.7), (("rhumerus", "S"), ("uarm_R", "S"), 0.7),
         (("lhumerus", "E"), ("uarm_L", "E"), 0.7), (("rhumerus", "E"), ("uarm_R", "E"), 0.7),
         (("lradius", "E"), ("farm_L", "E"), 0.7), (("rradius", "E"), ("farm_R", "E"), 0.7),
         (("lhand", "E"), ("hand_L", "E"), 0.3), (("rhand", "E"), ("hand_R", "E"), 0.3),
         (("thorax", "E"), ("chest", "E"), 1.0), (("upperneck", "E"), ("head", "S"), 0.7), (("head", "E"), ("head", "E"), 0.5)]
NEUTRAL = {"lumbar": 0.3, "lumbar1": 0.3, "lumbar2": 0.3, "thorax": 0.3, "chest": 0.3, "neck": 0.2, "head": 0.2,
           "clav": 0.3, "hand": 0.1}       # bone name prefix -> pull toward 0 (residual per radian)


def rotvec_to_R(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def R_to_rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2)); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th > math.pi - 1e-4:                                   # near half a turn: from the symmetric part
        A = (R + np.eye(3)) / 2; k = np.sqrt(np.maximum(np.diag(A), 0)); i = int(np.argmax(k))
        k = A[:, i] / max(k[i], 1e-9); return th * k / np.linalg.norm(k)
    return th * v / (2 * math.sin(th))


def yaw_R(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Fitter:
    def __init__(self, pc):
        self.pc = pc; sk = pc.sk; self.m = pc.model; self.d = mujoco.MjData(pc.model)
        self.ours = [(sk.idx[b], se) for _c, (b, se), _w in MARKS]
        self.w = np.repeat([w for *_x, w in MARKS], 3)
        self.reg = np.array([max([v for k, v in NEUTRAL.items() if sk.bones[i].name.startswith(k)] or [0.0])
                             for i, _a in pc.dof_list])
        self.lo, self.hi = pc.lo, pc.hi

    def points(self, x):
        pc = self.pc; d = self.d; sk = pc.sk
        d.qpos[:] = self.m.qpos0; d.qpos[0:3] = x[0:3]
        q4 = np.zeros(4); mujoco.mju_mat2Quat(q4, rotvec_to_R(x[3:6]).ravel()); d.qpos[3:7] = q4
        d.qpos[pc.qadr] = x[6:]
        mujoco.mj_kinematics(self.m, d)
        out = []
        for i, se in self.ours:
            b = pc.bid[i]; S = d.xpos[b]
            out.append(S if se == "S" else S + d.xmat[b].reshape(3, 3) @ (sk.dir[i] * sk.length[i]))
        return np.concatenate(out)

    def residual(self, x, target, x_prev):
        r = [(self.points(x) - target) * self.w, self.reg * x[6:]]
        if x_prev is not None:
            r.append(0.05 * (x[6:] - x_prev[6:]))            # frame to frame: the open joints stay where they were
        return np.concatenate(r)

    def fit(self, target, x0, x_prev=None, iters=12):
        x = x0.copy(); mu = 1e-2; eps = 1e-5
        r = self.residual(x, target, x_prev); cost = r @ r
        for _ in range(iters):
            J = np.empty((len(r), len(x)))
            for k in range(len(x)):
                xp = x.copy(); xp[k] += eps
                J[:, k] = (self.residual(xp, target, x_prev) - r) / eps
            A = J.T @ J; g = J.T @ r
            while True:
                dx = -np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-6), g)
                xn = x + dx; xn[6:] = np.clip(xn[6:], self.lo, self.hi)
                rn = self.residual(xn, target, x_prev); cn = rn @ rn
                if cn < cost:
                    x, r, cost = xn, rn, cn; mu = max(mu / 3, 1e-6); break
                mu *= 4
                if mu > 1e4:
                    return x, cost
            if np.abs(dx).max() < 1e-4:
                break
        return x, cost


def cmu_points(bones, scale, fr):
    S, E = fk(bones, scale, fr)
    return np.concatenate([to_zup((S if se == "S" else E)[b]) for (b, se), _o, _w in MARKS])


def heading_of(vxy):
    return math.atan2(vxy[1], vxy[0])


def build(src, dst):
    os.environ.setdefault("BIPED_MUSCLES", "1")
    from motor.ppo import make_env
    env = make_env("biped", seed=0); pc = env.pc; sk = pc.sk; i = sk.idx
    ft = Fitter(pc)
    leg_ours = sk.length[i["thigh_L"]] + sk.length[i["shank_L"]]
    rows = dict(z=[], R=[], q=[], v=[], w=[], qd=[], speed=[], err=[])
    for asf_path in sorted(glob.glob(os.path.join(src, "*.asf"))):
        subj = os.path.basename(asf_path)[:-4]; bones, scale = read_asf(asf_path)
        leg_cmu = bones["lfemur"]["length"] + bones["ltibia"]["length"]; s = leg_ours / leg_cmu
        for amc in sorted(glob.glob(os.path.join(src, subj + "_*.amc"))):
            frames = read_amc(amc)[::120 // HZ]
            if len(frames) < 12:
                continue
            T = [cmu_points(bones, scale, fr) * s for fr in frames]
            X = []; x = None; errs = []
            for t, tgt in enumerate(T):
                if x is None:                                  # the first frame: upright at the hips, joints neutral
                    P = tgt.reshape(-1, 3); x0 = np.zeros(6 + pc.nj); x0[0:3] = 0.5 * (P[0] + P[1])
                    hip = P[1] - P[0]; x0[3:6] = [0, 0, math.atan2(hip[1], hip[0]) + math.pi / 2]
                    x, _c = ft.fit(tgt, x0, None, iters=60)
                else:
                    x, _c = ft.fit(tgt, x, x, iters=12)
                X.append(x.copy())
                errs.append(float(np.sqrt(np.mean(((ft.points(x) - tgt).reshape(-1, 3)) ** 2 * 3))))
            X = np.array(X)
            hipmid = np.array([0.5 * (t_.reshape(-1, 3)[0] + t_.reshape(-1, 3)[1]) for t_ in T])
            for t in range(1, len(X) - 1):
                a, b = max(0, t - 8), min(len(X) - 1, t + 8)          # heading of travel over about half a second
                vh = (hipmid[b] - hipmid[a]) / ((b - a) / HZ)
                sp = float(np.linalg.norm(vh[:2]))
                Rt = rotvec_to_R(X[t, 3:6])
                yaw = heading_of(vh[:2]) if sp > 0.2 else math.atan2(Rt[1, 0], Rt[0, 0])
                Hi = yaw_R(-yaw)
                v = (X[t + 1, 0:3] - X[t - 1, 0:3]) * HZ / 2
                dR = rotvec_to_R(X[t + 1, 3:6]) @ rotvec_to_R(X[t - 1, 3:6]).T
                w = R_to_rotvec(dR) * HZ / 2                           # world angular velocity
                qd = (X[t + 1, 6:] - X[t - 1, 6:]) * HZ / 2
                rows["z"].append(X[t, 2]); rows["R"].append(Hi @ Rt); rows["q"].append(X[t, 6:])
                rows["v"].append(Hi @ v); rows["w"].append(Hi @ w); rows["qd"].append(qd)
                rows["speed"].append(sp); rows["err"].append(errs[t])
            print(f"{amc} frames {len(X)} err {np.mean(errs)*100:.1f} cm (max {np.max(errs)*100:.1f})", flush=True)
    D = {k: np.array(v) for k, v in rows.items()}
    # mirrored copies: the body reflected through its sagittal plane (y -> -y)
    M = np.diag([1.0, -1.0, 1.0])
    Dm = dict(z=D["z"], R=np.einsum("ij,njk,kl->nil", M, D["R"], M), q=D["q"][:, env.m_perm] * env.m_sign,
              v=D["v"] @ M, w=-(D["w"] @ M), qd=D["qd"][:, env.m_perm] * env.m_sign, speed=D["speed"], err=D["err"])
    out = {k: np.concatenate([D[k], Dm[k]]).astype(np.float32) for k in D}
    out["dofs"] = np.array([f"{sk.bones[b].name}:{'XYZ'[a]}" for b, a in pc.dof_list])
    out["leg"] = np.float32(env.leg)                           # leg length of the body fitted (FamilyEnv.leg)
    np.savez_compressed(dst, **out)
    print("frames", len(out["z"]), "mean err cm", float(out["err"].mean() * 100),
          "speeds", np.percentile(out["speed"], [10, 50, 90]).round(2))


class Starts:
    """The start poses of a family, picked by speed (``FamilyEnv.reset``)."""

    def __init__(self, path, pc):
        D = np.load(path)
        names = [f"{pc.sk.bones[b].name}:{'XYZ'[a]}" for b, a in pc.dof_list]
        assert list(D["dofs"]) == names, "start poses were built for other joints"
        self.leg = float(D["leg"])
        ok = D["err"] < 0.06                                   # frames the fit could not follow are left out
        self.D = {k: D[k][ok] for k in ("z", "R", "q", "v", "w", "qd", "speed")}

    def pick(self, rng, speed, tol=0.2):
        sp = self.D["speed"]
        idx = np.nonzero(np.abs(sp - speed) < tol * max(speed, 0.5))[0]
        if len(idx) == 0:
            return None
        k = int(rng.choice(idx))
        return {key: self.D[key][k] for key in self.D}


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
