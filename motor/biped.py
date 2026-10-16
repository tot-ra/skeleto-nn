"""The biped family: any human body, in any environment, with a goal and a motivation profile, in MuJoCo.

Input (observation), all in the pelvis frame or the heading frame:
  state       gravity direction, pelvis velocity and spin, pelvis height / leg length, joint angles and speeds,
              the previous command, foot contacts, the energy reserve
  goal        the velocity to hold, in the heading frame (stage 1: walk, run, turn, stand still)
  body        height, log mass, leg length, muscle, flexibility, pack, belly and armour (share of the mass), sex
  environment ground friction, wind in the heading frame
  sores       pain per region (legs, arms, torso, head)
  motivation  the weights of the evaluation terms (``evaluate.profile_vector``)
Output: a target angle for every joint muscle (PD, torque limited by strength and reserve), as an offset from the
standing pose, 30 times a second.

``mirror_obs`` / ``mirror_act`` reflect an observation and an action through the sagittal plane (left <-> right);
the trainer uses them for the symmetry loss. The harness (``assist``) holds the pelvis up and pushes it toward the
goal velocity early in training; the trainer fades it out (curriculum).
"""
from __future__ import annotations
import math
import numpy as np
import mujoco

from bodies import human, HumanBody
from physics import PhysChar, CTRL_HZ
from motor import evaluate as ev

G = 9.81
RHO_AIR = 1.2
SORES = ("leg_L", "leg_R", "arm_L", "arm_R", "torso", "head")
ALIVE = 1.0          # reward per second for staying upright (the motivation to stay whole)
FALL_COST = 5.0      # a fall ends the episode: about five seconds of a met goal


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def heading(R):
    c, s = math.cos(yaw_of(R)), math.sin(yaw_of(R))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sample_body(rng, spread=1.0):
    """A random person: child to tall adult, starved to athlete, stiff to supple, with or without a load."""
    h = float(np.clip(1.72 + rng.normal(0, 0.12) * spread - (0.45 * spread if rng.random() < 0.12 * spread else 0.0), 1.1, 2.0))
    bmi = float(np.clip(rng.normal(23.5, 3.0 * spread), 16.0, 34.0))
    mass = bmi * h * h
    muscle = float(np.clip(1.0 + rng.normal(0, 0.18) * spread, 0.5, 1.4))
    flex = float(np.clip(1.0 + rng.normal(0, 0.1) * spread, 0.7, 1.25))
    load = rng.random()
    pack = float(rng.uniform(4, 25)) if load < 0.15 * spread else 0.0
    armour = float(rng.uniform(5, 20)) if 0.15 * spread <= load < 0.25 * spread else 0.0
    belly = float(rng.uniform(2, 10)) if rng.random() < 0.1 * spread else 0.0
    sex = "f" if rng.random() < 0.5 else "m"
    return HumanBody(height=h, mass=mass + pack + armour + belly, pack=pack, armour=armour, belly=belly,
                     muscle=muscle, strength=muscle, flex=flex, sex=sex)


def body_vector(b: HumanBody, leg):
    return np.array([b.height - 1.7, math.log(b.mass / 75.0), leg - 0.86, b.muscle - 1.0, b.flex - 1.0,
                     b.pack / b.mass, b.belly / b.mass, b.armour / b.mass, 1.0 if b.sex == "f" else 0.0], np.float32)


class Wind:
    """Drag of a steady wind on every segment (frontal area 2 r L), applied at the segment centre each sub-step."""

    def __init__(self, pc, wind):
        self.pc = pc; self.w = np.array(wind, float); sk = pc.sk
        self.A = np.array([2 * max(b.radius, 0.02) * sk.length[i] for i, b in enumerate(sk.bones)])
        self.mid = np.array([sk.dir[i] * sk.length[i] * 0.5 for i in range(sk.n)])

    def apply(self):
        pc = self.pc; d = pc.data; m = pc.model
        if not self.w.any():
            return
        R = d.xmat[pc.bid].reshape(-1, 3, 3); P = d.xpos[pc.bid] + np.einsum("nij,nj->ni", R, self.mid)
        com = d.subtree_com[pc.bid[0]]; cv = d.cvel[pc.bid]
        v = cv[:, 3:] + np.cross(cv[:, :3], P - com)
        rel = self.w[None, :] - v
        F = 0.5 * RHO_AIR * 1.0 * self.A[:, None] * np.linalg.norm(rel, axis=1)[:, None] * rel
        for k in range(len(F)):
            mujoco.mj_applyFT(m, d, F[k], np.zeros(3), P[k], int(pc.bid[k]), d.qfrc_applied)


class BipedEnv:
    family = "biped"

    def __init__(self, seed=0, spread=1.0, act_scale=0.7, max_len=450):
        self.rng = np.random.default_rng(seed)
        self.spread = spread; self.act_scale_base = act_scale; self.max_len = max_len
        self.assist = 0.0
        self._build(HumanBody())
        self.obs_dim = len(self.reset())
        self.act_dim = self.pc.nj

    # ------------------------------------------------------------------ body and episode
    def _build(self, b: HumanBody):
        self.b = b
        self.c = human(b)
        self.pc = PhysChar(self.c)
        self.J = self.pc.nj
        self.base_force = self.pc.model.actuator_forcerange[self.pc.aid].copy()
        # a = +-1 reaches the end of the joint's range; 0 is the standing pose
        sc = np.maximum(np.abs(self.pc.lo), np.abs(self.pc.hi)) * self.act_scale_base
        for j, (i, _a) in enumerate(self.pc.dof_list):
            if self.pc.sk.bones[i].group in ("neck", "head"):
                sc[j] *= 0.5
        self.act_scale = sc
        self.leg = self.c.leg_len()
        self.bvec = body_vector(b, self.leg)
        self._mirror_tables()

    def _mirror_tables(self):
        pc = self.pc; sk = pc.sk
        index = {(sk.bones[i].name, a): j for j, (i, a) in enumerate(pc.dof_list)}
        perm = np.zeros(self.J, int); sign = np.ones(self.J)
        for j, (i, a) in enumerate(pc.dof_list):
            name = sk.bones[i].name
            other = name[:-1] + ("R" if name.endswith("L") else "L") if name.endswith(("_L", "_R")) else name
            perm[j] = index[(other, a)]
            sign[j] = -1.0 if a in (0, 2) else 1.0            # rotations about X and Z flip in a reflection through y = 0
        self.m_perm, self.m_sign = perm, sign

    def reset(self, body=None, friction=None, wind=None, sores=None, goal=None, profile=None):
        rng = self.rng
        if body is not None or rng.random() < 0.25 or not hasattr(self, "ev"):
            self._build(body or (sample_body(rng, self.spread) if rng.random() < 0.8 else HumanBody()))
        m = self.pc.model
        self.friction = float(friction if friction is not None else (rng.uniform(0.08, 0.3) if rng.random() < 0.2 * self.spread else rng.uniform(0.6, 1.2)))
        m.geom_friction[:, 0] = self.friction
        if wind is None:
            wind = np.zeros(3)
            if rng.random() < 0.25 * self.spread:
                a = rng.uniform(0, 2 * math.pi); s = rng.uniform(3, 16)
                wind = np.array([math.cos(a) * s, math.sin(a) * s, 0.0])
        self.wind = Wind(self.pc, wind)
        if sores is None:
            sores = {k: 0.0 for k in SORES}
            if rng.random() < 0.3 * self.spread:
                sores[SORES[int(rng.integers(len(SORES)))]] = float(rng.uniform(0.3, 1.0))
        self.sores = sores
        self.profile = profile or ev.sample_profile(rng, self.spread)
        if goal is None:
            r = rng.random()
            speed = 0.0 if r < 0.15 else (rng.uniform(0.3, 1.8) if r < 0.75 else rng.uniform(1.8, 4.0))
            ang = rng.uniform(-math.pi, math.pi) if rng.random() < 0.3 else rng.normal(0, 0.3)
            goal = np.array([speed * math.cos(ang), speed * math.sin(ang)])
        self.goal_world = np.array(goal, float)           # velocity to hold, world frame
        self.ev = ev.Evaluator(self.pc, sores)
        # start: standing, a little noise, facing a random yaw
        q0 = self.rng.normal(0, 0.03, self.J)
        yaw = rng.uniform(-math.pi, math.pi)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        self.goal_world = Rz[:2, :2] @ self.goal_world
        self.pc.set_state(np.array([0, 0, self.c.z0 + 0.01]), Rz, np.clip(q0, self.pc.lo, self.pc.hi))
        self.steps = 0; self.prev_a = np.zeros(self.J); self.ep_ret = 0.0; self.ep_goal = 0.0
        self.strength = 1.0
        return self.obs()

    # ------------------------------------------------------------------ observation
    def obs(self):
        pc = self.pc; d = pc.data
        p, R = pc.root_pose()
        H = heading(R)
        g_loc = R.T @ np.array([0, 0, -1.0])
        v_loc = R.T @ d.qvel[0:3]; w_loc = d.qvel[3:6]
        cmd = H.T @ np.array([self.goal_world[0], self.goal_world[1], 0.0])
        wind_h = H.T @ self.wind.w
        con = pc.foot_contacts()
        o = np.concatenate([
            g_loc, v_loc, w_loc * 0.2, [p[2] / self.leg - 1.0], pc.qvec(), pc.qdvec() * 0.1, self.prev_a, con, [self.ev.reserve],
            cmd[:2], self.bvec, [self.friction, wind_h[0] / 10.0, wind_h[1] / 10.0],
            [self.sores[k] for k in SORES], ev.profile_vector(self.profile)])
        return o.astype(np.float32)

    def obs_layout(self):
        J = self.J
        parts = [("g", 3), ("v", 3), ("w", 3), ("z", 1), ("q", J), ("qd", J), ("prev", J), ("contact", 2), ("reserve", 1),
                 ("cmd", 2), ("body", 9), ("env", 3), ("sores", len(SORES)), ("profile", len(ev.TERMS))]
        out, k = {}, 0
        for name, n in parts:
            out[name] = slice(k, k + n); k += n
        return out

    def mirror_obs(self, o):
        """Reflect a batch of observations through the sagittal plane (numpy or torch, last axis = features)."""
        L = self.obs_layout(); m = o.clone() if hasattr(o, "clone") else o.copy()
        def neg(sl, idx):
            m[..., sl.start + idx] = -o[..., sl.start + idx]
        neg(L["g"], 1); neg(L["v"], 1); neg(L["w"], 0); neg(L["w"], 2); neg(L["cmd"], 1); neg(L["env"], 2)
        for key in ("q", "qd", "prev"):
            sl = L[key]
            block = o[..., sl][..., self.m_perm]
            sign = self.m_sign if not hasattr(o, "clone") else __import__("torch").tensor(self.m_sign, dtype=o.dtype)
            m[..., sl] = block * sign
        c = L["contact"]; m[..., c.start] = o[..., c.start + 1]; m[..., c.start + 1] = o[..., c.start]
        s = L["sores"]
        for a, b in ((0, 1), (2, 3)):
            m[..., s.start + a] = o[..., s.start + b]; m[..., s.start + b] = o[..., s.start + a]
        return m

    def mirror_act(self, a):
        sign = self.m_sign if not hasattr(a, "clone") else __import__("torch").tensor(self.m_sign, dtype=a.dtype)
        return a[..., self.m_perm] * sign

    # ------------------------------------------------------------------ step
    def _harness(self):
        """Curriculum help: hold the pelvis up and upright (it never pushes toward the goal)."""
        if self.assist <= 0:
            return None
        pc = self.pc; d = pc.data
        p, R = pc.root_pose()
        mass = pc.total_mass
        v = d.qvel[0:3]
        Fz = self.assist * (0.5 * mass * G + 3000.0 * (self.c.z0 - p[2]) * (mass / 75.0) - 300.0 * v[2] * (mass / 75.0))
        # No push toward the goal: a pushing harness teaches the body to lean and skate instead of stepping.
        Fxy = np.zeros(2)
        up = np.cross(R[:, 2], np.array([0, 0, 1.0]))
        wv = d.qvel[3:6]; wv_w = R @ wv
        tq = self.assist * (mass / 75.0) * (800.0 * up - 60.0 * wv_w * np.array([1.0, 1.0, 0.3]))
        return np.array([Fxy[0], Fxy[1], max(Fz, 0.0), tq[0], tq[1], tq[2]])

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        pc = self.pc
        # strength: muscle, and a drained reserve weakens every muscle
        self.strength = 0.55 + 0.45 * self.ev.reserve
        pc.model.actuator_forcerange[pc.aid] = self.base_force * self.strength
        target = np.clip(a * self.act_scale, pc.lo, pc.hi)
        def pre():                                           # forces are recomputed every physics sub-step
            d = pc.data
            d.qfrc_applied[:] = 0.0
            d.xfrc_applied[:] = 0.0
            self.wind.apply()
            wrench = self._harness()
            if wrench is not None:
                d.xfrc_applied[pc.bid[0], :] = wrench
        pc.step(target, pre=pre, probe=self.ev.probe)
        self.steps += 1
        p, R = pc.root_pose()
        v = pc.data.qvel[0:2]
        err = v - self.goal_world
        sp = float(np.linalg.norm(self.goal_world))
        # 1 when the velocity is held; partial progress along the wanted direction already counts
        goal = 0.7 * math.exp(-float(err @ err) / (0.25 + 0.15 * sp * sp))
        if sp > 0.1:
            along = float(v @ (self.goal_world / sp)) / sp
            goal += 0.3 * float(np.clip(along, 0.0, 1.0))
            fwd = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)           # face where you go
            goal *= 0.75 + 0.25 * max(0.0, float(fwd @ (self.goal_world / sp)))
        else:
            goal += 0.3
        # Staying upright and whole is the first motivation of every body: without it, ending an episode early
        # (falling) would look cheaper than paying the costs of staying up.
        r, costs, bad = self.ev.step(a, 1.0 / CTRL_HZ, self.profile, goal * 2.0 + ALIVE, self.strength)
        self.prev_a = a
        tilt = math.acos(max(-1.0, min(1.0, float(R[2, 2]))))
        unstable = pc.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.all(np.isfinite(pc.data.qpos))
        if unstable:                                         # a blown-up simulation is a fall, never a reward
            pc.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 0
            r = -2.0 / CTRL_HZ
        fell = bad or unstable or p[2] < 0.5 * self.c.z0 or tilt > 1.2
        if fell:
            r -= FALL_COST
        trunc = self.steps >= self.max_len
        self.ep_ret += r; self.ep_goal += goal
        info = dict(trunc=trunc and not fell, costs=costs, goal_now=goal, speed=float(np.linalg.norm(v)))
        if fell or trunc:
            info.update(ep_ret=self.ep_ret, ep_len=self.steps, fell=fell, goal=self.ep_goal / self.steps)
        return self.obs(), r, fell, info
