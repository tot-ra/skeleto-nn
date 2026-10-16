"""The bird family in the air: crow, gull, heron, sparrow, duck and variations, one policy (stage 1: flight).

The skeleton is the same for every species (``bodies.bird``: four neck joints, two tail joints, three wing segments and
four leg segments a side), so one network drives all of them; species differ in mass, size, wing, neck, leg and tail
length. The wings and tail are flat plates in the air (``air.py``: lift with stall, drag, wind); nothing about flapping
is authored. The policy sets every joint, so whether it flaps, glides or soars is what the evaluation system and the
goal make worthwhile.

Goal: a velocity to hold in the air, horizontal (in the heading frame) and a climb rate, in units of
sqrt(g x wingspan) (a crow's cruise of about 10 m/s is about 3.3 units). Episodes start in the air at about the goal
velocity, 3 to 6 wingspans up; touching the ground ends the episode (its contact force is scored as injury).

Not yet: take-off, landing, perching, walking and hopping, swimming birds.
"""
from __future__ import annotations
import dataclasses
import math
import numpy as np
import mujoco

import air as AIR
from bodies import bird, BIRDS
from physics import CTRL_HZ
from motor.family import FamilyEnv, G, ACTIVATION_TAU, heading
from motor import evaluate as ev

SPECIES = ("crow", "gull", "heron", "sparrow", "duck")


class BirdEnv(FamilyEnv):
    family = "bird"
    BODY_DIM = 7

    # ------------------------------------------------------------------ family hooks
    def default_body(self):
        return BIRDS["crow"]

    def sample_body(self, rng, spread):
        sp = BIRDS[SPECIES[int(rng.integers(len(SPECIES)))]] if rng.random() < spread else BIRDS["crow"]
        j = lambda: math.exp(rng.normal(0, 0.08 * spread))
        return dataclasses.replace(sp, mass=sp.mass * j() ** 3, scale=sp.scale * j(), wing_len=sp.wing_len * j())

    def make_creature(self, body):
        return bird(body)

    def body_vector(self, sp, c):
        return [math.log(sp.mass / 0.45), math.log(sp.scale), sp.leg_scale - 1.0, sp.neck_len - 1.0, sp.wing_len - 1.0,
                sp.tail_len - 1.0, math.log(self.wingspan(c))]

    def sores(self):
        return ("wing_L", "wing_R", "leg_L", "leg_R", "torso", "head")

    def wingspan(self, c=None):
        c = c or self.c; sk = c.skel
        return float(2 * sum(sk.length[i] for i in c.wings[0]) + 0.1 * sk.length[0])

    def speed_unit(self):
        return math.sqrt(G * self.wingspan())

    # ------------------------------------------------------------------ episode
    def reset(self, body=None, friction=None, wind=None, sores=None, goal=None, profile=None, climb=None):
        """``goal``: horizontal velocity in speed units (forward, left) relative to the start heading; ``climb`` in units."""
        rng = self.rng
        super().reset(body=body, friction=friction, wind=wind, sores=sores, goal=np.zeros(2), profile=profile)
        u = self.speed_unit()
        if goal is None:                                   # cruise: forward speed in units, a little sideways at most
            goal = np.array([rng.uniform(1.5, max(1.6, 1.5 + 2.0 * self.max_speed / 4.0)), rng.normal(0, 0.1 + 0.4 * self.spread)])
        self.climb = float(climb if climb is not None else rng.normal(0, 0.05 + 0.25 * self.spread)) * u
        # start in the air at about the goal velocity, heading roughly where the goal points
        yaw = rng.uniform(-math.pi, math.pi)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        self.goal_world = Rz[:2, :2] @ (np.asarray(goal, float) * u)
        self.alt0 = self.wingspan() * rng.uniform(3.0, 6.0)
        v0 = np.array([self.goal_world[0], self.goal_world[1], self.climb]) * rng.uniform(0.8, 1.0)
        self.pc.set_state(np.array([0, 0, self.alt0]), Rz, self.pc.qvec(), root_vel=v0)
        self.air = AIR.Air(self.pc, wind=self.wind.w)
        return self.obs()

    # ------------------------------------------------------------------ observation: + altitude and climb goal
    def obs(self):
        o = super().obs()
        if not hasattr(self, "climb"):
            return np.concatenate([o, [0.0, 0.0]]).astype(np.float32)
        p, _R = self.pc.root_pose()
        return np.concatenate([o, [p[2] / self.wingspan() - 4.0, self.climb / self.speed_unit()]]).astype(np.float32)

    def obs_layout(self):
        out = super().obs_layout()
        k = max(sl.stop for sl in out.values())
        out["alt"] = slice(k, k + 2)
        return out

    # ------------------------------------------------------------------ step: air forces, flight goal
    def _harness(self):
        """Curriculum help in the air: part of the weight carried, the body held level; never pushes forward."""
        if self.assist <= 0:
            return None
        pc = self.pc; d = pc.data
        p, R = pc.root_pose()
        m = pc.total_mass; k = self.assist
        Fz = k * (0.7 * m * G + m * (6.0 * (self.alt0 - p[2]) - 3.0 * d.qvel[2]))
        wv_w = R @ d.qvel[3:6]; I = m * self.wingspan() ** 2 * 0.02
        return np.array([0.0, 0.0, max(Fz, 0.0), k * I * (40.0 * R[1, 2] - 8.0 * wv_w[0]),
                         k * I * (-40.0 * R[0, 2] - 8.0 * wv_w[1]), k * I * (-3.0 * wv_w[2])])

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        pc = self.pc
        self.strength = 0.55 + 0.45 * self.ev.reserve
        pc.model.actuator_forcerange[pc.aid] = self.base_force * self.strength
        alpha = 1.0 - math.exp(-1.0 / (CTRL_HZ * ACTIVATION_TAU))
        act = self.prev_a + alpha * (a - self.prev_a)
        target = np.clip(act * self.act_scale, pc.lo, pc.hi)

        def pre():
            d = pc.data
            d.xfrc_applied[:] = 0.0
            self.air.apply()                                  # resets qfrc_applied, then lift and drag (with the wind)
            wrench = self._harness()
            if wrench is not None:
                d.xfrc_applied[pc.bid[0], :] = wrench
        pc.step(target, pre=pre, probe=self.ev.probe)
        self.steps += 1
        p, R = pc.root_pose()
        u = self.speed_unit()
        v = pc.data.qvel[0:3] / u
        goal_v = np.array([self.goal_world[0], self.goal_world[1], self.climb]) / u
        err = v - goal_v
        sp = float(np.linalg.norm(goal_v[:2]))
        goal = 0.7 * math.exp(-float(err @ err) / (0.25 + 0.15 * sp * sp))
        goal += 0.3 * float(np.clip(float(v[:2] @ (goal_v[:2] / max(sp, 1e-6))) / max(sp, 1e-6), 0.0, 1.0))
        # in the air the capture point means nothing: balance is scored by tilt alone, through the term's weight
        weights = {t: w * self.cost_scale for t, w in self.profile.items()}
        weights["balance"] = 0.0
        r, costs, bad = self.ev.step(a, 1.0 / CTRL_HZ, weights, goal * 2.0 + self.alive, self.strength)
        self.prev_a = act
        unstable = pc.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 or not np.all(np.isfinite(pc.data.qpos))
        if unstable:
            pc.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 0
            r = -2.0 / CTRL_HZ
        grounded = pc.data.ncon > 0                          # stage 1: any touch of the ground ends a flight
        upside_down = float(R[2, 2]) < -0.2
        fell = unstable or grounded or upside_down or p[2] < 0.15 * self.wingspan()
        if fell:
            r -= self.fall_cost
        trunc = self.steps >= self.max_len
        self.ep_ret += r; self.ep_goal += goal
        info = dict(trunc=trunc and not fell, costs=costs, goal_now=goal, speed=float(np.linalg.norm(pc.data.qvel[0:3])))
        if fell or trunc:
            info.update(ep_ret=self.ep_ret, ep_len=self.steps, fell=fell, goal=self.ep_goal / self.steps)
        return self.obs(), r, fell, info

    def stride_pose(self, rng, speed):
        return {}
