"""Shared machinery of every body-family environment: physics body, observation, mirror maps, harness, wind, reward.

A family subclass says only what differs (``motor/biped.py`` and ``motor/quadruped.py``):

  default_body() / sample_body(rng, spread)   the individuals of the family
  make_creature(body)                         bones, joint ranges, masses, torques (``bodies.py``)
  body_vector(body, creature)                 the numbers that describe the individual to the policy (BODY_DIM)
  sores()                                     named sore regions (pain inputs)
  stride_pose(rng, speed)                     joint angles of a mid-stride start, {(bone, axis): angle}
  speed_unit()                                goal speeds are sampled as max_speed x this (m/s)

Observation (the same layout for every family): gravity in the root frame, root velocity and spin, root height over
leg length, joint angles, speeds and the muscle targets (activation state), foot contacts, the energy reserve, the goal velocity in the
heading frame (in speed units), the body vector, ground friction and wind in the heading frame, sores, the motivation
profile. Action: a target angle for every joint muscle, as a share of the joint's range, 30 times a second.
"""
from __future__ import annotations
import math
import numpy as np
import mujoco

from physics import PhysChar, CTRL_HZ
from motor import evaluate as ev

G = 9.81
RHO_AIR = 1.2
ALIVE = 1.0          # reward per second for staying upright (the motivation to stay whole)
ACTIVATION_TAU = 0.05  # s: a muscle's target follows the command like muscle activation, not instantly
FALL_COST = 5.0      # a fall ends the episode: about five seconds of a met goal


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def heading(R):
    c, s = math.cos(yaw_of(R)), math.sin(yaw_of(R))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mirror_name(name, known=None):
    """The name of the other side: thigh_L <-> thigh_R, leg_HL <-> leg_HR; names without a side are their own mirror.
    With ``known``, a swapped name that does not exist there means the name has no side."""
    if name.endswith(("_L", "_R")):
        other = name[:-1] + ("R" if name[-1] == "L" else "L")
    elif name[-1:] in ("L", "R"):
        other = name[:-1] + ("R" if name[-1] == "L" else "L")
    else:
        return name
    return other if known is None or other in known else name


class Wind:
    """Drag of a steady wind on every segment (frontal area 2 r L), at the segment centre; computed once per control
    step and added every sub-step."""

    def __init__(self, pc, wind):
        self.pc = pc; self.w = np.array(wind, float); sk = pc.sk; self.qfrc = None
        self.A = np.array([2 * max(b.radius, 0.02) * sk.length[i] for i, b in enumerate(sk.bones)])
        self.mid = np.array([sk.dir[i] * sk.length[i] * 0.5 for i in range(sk.n)])

    def update(self):
        pc = self.pc; d = pc.data; m = pc.model
        self.qfrc = None
        if not self.w.any():
            return
        R = d.xmat[pc.bid].reshape(-1, 3, 3); P = d.xpos[pc.bid] + np.einsum("nij,nj->ni", R, self.mid)
        com = d.subtree_com[pc.bid[0]]; cv = d.cvel[pc.bid]
        v = cv[:, 3:] + np.cross(cv[:, :3], P - com)
        rel = self.w[None, :] - v
        F = 0.5 * RHO_AIR * self.A[:, None] * np.linalg.norm(rel, axis=1)[:, None] * rel
        q = np.zeros(m.nv)
        for k in range(len(F)):
            mujoco.mj_applyFT(m, d, F[k], np.zeros(3), P[k], int(pc.bid[k]), q)
        self.qfrc = q

    def apply(self):
        if self.qfrc is not None:
            self.pc.data.qfrc_applied[:] += self.qfrc


class FamilyEnv:
    family = "?"
    BODY_DIM = 0

    # ------------------------------------------------------------------ hooks
    def default_body(self): raise NotImplementedError
    def sample_body(self, rng, spread): raise NotImplementedError
    def make_creature(self, body): raise NotImplementedError
    def body_vector(self, body, creature): raise NotImplementedError
    def sores(self): raise NotImplementedError
    def stride_pose(self, rng, speed): return {}
    def speed_unit(self): return 1.0

    MAX_SPEED = 3.0      # the curriculum never asks for more (in speed units): a run for people, a gallop for horses
    STALL_GOAL = 0.15    # a moving goal met this little for about a second ends the episode, as a fall does

    def _stalled(self, goal, sp):
        """``goal``: the progress toward the goal this step (the goal score, or turning toward it). Standing still when asked to move is a dead end, not a safe choice: without this, staying up earns
        ALIVE every step while trying to walk risks the fall cost, so the safest policy ignores the goal."""
        self.goal_ema += (goal - self.goal_ema) / CTRL_HZ
        return sp > 0.2 and self.steps > 1.5 * CTRL_HZ and self.goal_ema < self.STALL_GOAL

    # ------------------------------------------------------------------ setup
    def __init__(self, seed=0, spread=1.0, act_scale=0.7, max_len=450, alive=None, fall_cost=None, cost_scale=1.0):
        self.rng = np.random.default_rng(seed)
        self.alive = ALIVE if alive is None else alive
        self.fall_cost = FALL_COST if fall_cost is None else fall_cost
        self.cost_scale = cost_scale
        self.spread = spread; self.act_scale_base = act_scale; self.max_len = max_len
        self.assist = 0.0
        self.max_speed = 4.0
        self.moving_start = 0.7
        self.SORES = tuple(self.sores())
        self._build(self.default_body())
        self.obs_dim = len(self.reset())

    def _build(self, body):
        self.b = body
        self.c = self.make_creature(body)
        self.pc = PhysChar(self.c)
        self.J = self.pc.nj
        self.act_dim = self.J
        self.base_force = self.pc.model.actuator_forcerange[self.pc.aid].copy()
        sc = np.maximum(np.abs(self.pc.lo), np.abs(self.pc.hi)) * self.act_scale_base
        for j, (i, _a) in enumerate(self.pc.dof_list):
            if self.pc.sk.bones[i].group in ("neck", "head", "tail"):
                sc[j] *= 0.5
        self.act_scale = sc
        self.leg = self.c.leg_len()
        self.bvec = np.asarray(self.body_vector(body, self.c), np.float32)
        assert len(self.bvec) == self.BODY_DIM
        pc = self.pc; sk = pc.sk
        index = {(sk.bones[i].name, a): j for j, (i, a) in enumerate(pc.dof_list)}
        self.dof_index = index
        perm = np.zeros(self.J, int); sign = np.ones(self.J)
        for j, (i, a) in enumerate(pc.dof_list):
            perm[j] = index[(mirror_name(sk.bones[i].name), a)]
            sign[j] = -1.0 if a in (0, 2) else 1.0            # rotations about X and Z flip in a reflection through y = 0
        self.m_perm, self.m_sign = perm, sign
        self.a_perm, self.a_sign = perm, sign                 # actions mirror like joints unless a family adds outputs
        names = [l.name for l in self.c.legs]
        self.leg_perm = np.array([names.index(mirror_name(n, names)) for n in names])
        self.sore_perm = np.array([self.SORES.index(mirror_name(k, self.SORES)) for k in self.SORES])

    # ------------------------------------------------------------------ episode
    def reset(self, body=None, friction=None, wind=None, sores=None, goal=None, profile=None):
        rng = self.rng
        if body is not None or not hasattr(self, "ev") or (rng.random() < 0.25 and rng.random() < self.spread):
            self._build(body or (self.sample_body(rng, self.spread) if rng.random() < self.spread else self.default_body()))
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
            sores = {k: 0.0 for k in self.SORES}
            if rng.random() < 0.3 * self.spread:
                sores[self.SORES[int(rng.integers(len(self.SORES)))]] = float(rng.uniform(0.3, 1.0))
        self.sores_now = sores
        self.profile = profile or ev.sample_profile(rng, self.spread)
        if goal is None:
            speed = 0.0 if rng.random() < 0.15 else rng.uniform(0.3, max(0.4, self.max_speed)) * self.speed_unit()
            ang = rng.uniform(-math.pi, math.pi) if rng.random() < 0.3 * self.spread else rng.normal(0, 0.05 + 0.25 * self.spread)
            goal = np.array([speed * math.cos(ang), speed * math.sin(ang)])
        self.goal_world = np.array(goal, float)
        self.ev = ev.Evaluator(self.pc, sores)
        q0 = rng.normal(0, 0.03, self.J)
        yaw = rng.uniform(-math.pi, math.pi)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        self.goal_world = Rz[:2, :2] @ self.goal_world
        sp = float(np.linalg.norm(self.goal_world))
        v0 = np.zeros(3)
        if sp > 0.1 and rng.random() < self.moving_start:
            v0[:2] = self.goal_world * rng.uniform(0.6, 1.0)
            for key, val in self.stride_pose(rng, sp).items():
                j = self.dof_index.get(key)
                if j is not None:
                    q0[j] = val
        self.pc.set_state(np.array([0, 0, self.c.z0 + 0.01]), Rz, np.clip(q0, self.pc.lo, self.pc.hi), root_vel=v0)
        # muscle state: the current target as a share of the range, starting from the pose the body starts in
        self.prev_a = np.clip(np.clip(q0, self.pc.lo, self.pc.hi) / np.maximum(self.act_scale, 1e-6), -1, 1)
        self.steps = 0; self.ep_ret = 0.0; self.ep_goal = 0.0; self.goal_ema = 1.0
        self.strength = 1.0
        return self.obs()

    # ------------------------------------------------------------------ observation
    def obs(self):
        pc = self.pc; d = pc.data
        p, R = pc.root_pose()
        H = heading(R)
        g_loc = R.T @ np.array([0, 0, -1.0])
        v_loc = R.T @ d.qvel[0:3] / self.speed_unit(); w_loc = d.qvel[3:6]
        cmd = H.T @ np.array([self.goal_world[0], self.goal_world[1], 0.0]) / self.speed_unit()
        wind_h = H.T @ self.wind.w
        o = np.concatenate([
            g_loc, v_loc, w_loc * 0.2, [p[2] / self.leg - 1.0], pc.qvec(), pc.qdvec() * 0.1, self.prev_a, pc.foot_contacts(),
            [self.ev.reserve], cmd[:2], self.bvec, [self.friction, wind_h[0] / 10.0, wind_h[1] / 10.0],
            [self.sores_now[k] for k in self.SORES], ev.profile_vector(self.profile)])
        return o.astype(np.float32)

    def obs_layout(self):
        J = self.J
        parts = [("g", 3), ("v", 3), ("w", 3), ("z", 1), ("q", J), ("qd", J), ("prev", J), ("contact", len(self.c.legs)),
                 ("reserve", 1), ("cmd", 2), ("body", self.BODY_DIM), ("env", 3), ("sores", len(self.SORES)),
                 ("profile", len(ev.TERMS))]
        out, k = {}, 0
        for name, n in parts:
            out[name] = slice(k, k + n); k += n
        return out

    def mirror_obs(self, o):
        """Reflect a batch of observations through the sagittal plane (numpy or torch, last axis = features)."""
        L = self.obs_layout(); torch_ = hasattr(o, "clone"); m = o.clone() if torch_ else o.copy()

        def neg(sl, idx):
            m[..., sl.start + idx] = -o[..., sl.start + idx]
        neg(L["g"], 1); neg(L["v"], 1); neg(L["w"], 0); neg(L["w"], 2); neg(L["cmd"], 1); neg(L["env"], 2)
        sign = __import__("torch").tensor(self.m_sign, dtype=o.dtype) if torch_ else self.m_sign
        for key in ("q", "qd", "prev"):
            sl = L[key]
            m[..., sl] = o[..., sl][..., self.m_perm] * sign
        for key, perm in (("contact", self.leg_perm), ("sores", self.sore_perm)):
            sl = L[key]
            m[..., sl] = o[..., sl][..., perm]
        return m

    def mirror_act(self, a):
        sign = self.a_sign if not hasattr(a, "clone") else __import__("torch").tensor(self.a_sign, dtype=a.dtype)
        return a[..., self.a_perm] * sign

    # ------------------------------------------------------------------ step
    def _harness(self):
        """Curriculum help: hold the root up and upright (it never pushes toward the goal)."""
        if self.assist <= 0:
            return None
        pc = self.pc; d = pc.data
        p, R = pc.root_pose()
        k = self.assist * pc.total_mass / 75.0
        v = d.qvel[0:3]
        Fz = self.assist * 0.5 * pc.total_mass * G + k * (3000.0 * (self.c.z0 - p[2]) - 300.0 * v[2])
        zx, zy = R[0, 2], R[1, 2]
        wv_w = R @ d.qvel[3:6]
        return np.array([0.0, 0.0, max(Fz, 0.0), k * (800.0 * zy - 60.0 * wv_w[0]), k * (-800.0 * zx - 60.0 * wv_w[1]),
                         k * (-18.0 * wv_w[2])])

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        pc = self.pc
        self.strength = 0.55 + 0.45 * self.ev.reserve
        pc.model.actuator_forcerange[pc.aid] = self.base_force * self.strength
        # activation dynamics: the muscle target moves toward the command with a 50 ms time constant; commands
        # that flip every 33 ms cannot make the body vibrate (and cost energy no real muscle could spend)
        alpha = 1.0 - math.exp(-1.0 / (CTRL_HZ * ACTIVATION_TAU))
        act = self.prev_a + alpha * (a - self.prev_a)
        target = np.clip(act * self.act_scale, pc.lo, pc.hi)
        self.wind.update()

        def pre():
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
        v = pc.data.qvel[0:2] / self.speed_unit()
        goal_v = self.goal_world / self.speed_unit()
        err = v - goal_v
        sp = float(np.linalg.norm(goal_v))
        # the tolerance grows with the wanted speed, but standing still must not earn much of a slow walk's goal
        goal = 0.7 * math.exp(-float(err @ err) / (0.05 + 0.1 * sp * sp))
        progress = goal
        if sp > 0.1:
            goal += 0.3 * float(np.clip(float(v @ (goal_v / sp)) / sp, 0.0, 1.0))
            fwd = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)
            facing = float(fwd @ (goal_v / sp))
            goal *= 0.75 + 0.25 * max(0.0, facing)
            progress = goal
            if facing < 0.9:
                # turning toward a goal behind or beside the body is progress too: without this a body that turns
                # on the spot before it sets off scores nothing and is ended as a stall halfway round
                side = fwd[0] * goal_v[1] - fwd[1] * goal_v[0]            # > 0: the goal is to the left
                yaw_rate = float((R @ pc.data.qvel[3:6])[2])                # the free joint gives it in the body frame
                turn = float(np.clip(math.copysign(1.0, side) * yaw_rate / 0.8, 0.0, 1.0))
                goal += 0.2 * turn
                progress = max(progress, turn)
        else:
            goal += 0.3
            progress = goal
        weights = {t: w * self.cost_scale for t, w in self.profile.items()}
        r, costs, bad = self.ev.step(a, 1.0 / CTRL_HZ, weights, goal * 2.0 + self.alive, self.strength)
        self.prev_a = act
        tilt = math.acos(max(-1.0, min(1.0, float(R[2, 2]))))
        unstable = pc.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.all(np.isfinite(pc.data.qpos))
        if unstable:
            pc.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 0
            r = -2.0 / CTRL_HZ
        fell = bad or unstable or p[2] < 0.5 * self.c.z0 or tilt > 1.2
        stalled = not fell and self._stalled(progress, sp)
        if fell or stalled:
            r -= self.fall_cost
        done = fell or stalled
        trunc = self.steps >= self.max_len
        self.ep_ret += r; self.ep_goal += goal
        info = dict(trunc=trunc and not done, costs=costs, goal_now=goal, speed=float(np.linalg.norm(pc.data.qvel[0:2])),
                    stalled=stalled)
        if done or trunc:
            info.update(ep_ret=self.ep_ret, ep_len=self.steps, fell=fell, goal=self.ep_goal / self.steps)
        return self.obs(), r, done, info
