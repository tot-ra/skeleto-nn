"""The bird family in the air: crow, gull, heron, duck and variations (not the sparrow yet, see SPECIES), one policy (stage 1: flight).

The skeleton is the same for every species (``bodies.bird``: four neck joints, two tail joints, three wing segments and
four leg segments a side), so one network drives all of them; species differ in mass, size, wing, neck, leg and tail
length. The wings and tail are flat plates in the air (``air.py``: lift with stall, drag, wind); nothing about flapping
is authored. The policy sets every joint, so whether it flaps, glides or soars is what the evaluation system and the
goal make worthwhile.

Goal: a velocity to hold in the air, horizontal (in the heading frame) and a climb rate, in units of
sqrt(g x wingspan) (a crow's cruise of about 10 m/s is about 3.3 units). Episodes start in the air at about the goal
velocity, 3 to 6 wingspans up; cruise goals run from 2.6 units (just above the minimum flying speed) up; touching the ground ends the episode (its contact force is scored as injury).

The network reads a wingbeat clock (sine and cosine of a phase) and sets, as one extra output, how fast the phase
advances (half to twice the species' natural beat frequency, which scales inversely with wingspan); what the wings
do at each phase, and whether they beat at all, is the network's.

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

# Not the sparrow yet: it beats at about 13 Hz, and at 30 control steps a second the network has two commands per beat,
# too few to shape one. Small birds need a faster control rate first.
SPECIES = ("crow", "gull", "heron", "duck")


class BirdEnv(FamilyEnv):
    family = "bird"
    BODY_DIM = 7
    MAX_SPEED = 4.0
    ASSIST_RECOVER = 1.0   # a flier that cannot yet stay up gets the whole harness back (stall ends a hover under it)
    GOAL_OK = 0.35         # the climb rate is scored tightly (CLIMB_TOL), so a met flight scores lower than a met walk
    FAIL_OK = 0.5          # a flight ends at any touch of the ground, so more of them fail while the goals are met
    # exploration noise floor: 0.2 (log -1.6) until the wingbeat was found (a glider's 0.08 never found one); 0.12 once
    # it flies, so the noise does not cost the precision of climbs and turns
    MIN_LOGSTD = -2.1
    PHASE_NOISE = 0.6      # share of the exploration noise locked to the wingbeat clock (motor/ppo.py)
    CLIMB_TOL = 0.15       # climb-rate tolerance of the goal score (speed units)
    HOLD = 1.0             # harness attitude stiffness in m g x wingspan per radian
    GLIDE_SINK = 0.45      # sink rate (units) of the first goals: a glide at a crow's glide ratio of about 7

    # ------------------------------------------------------------------ family hooks
    def default_body(self):
        return BIRDS["crow"]

    def level(self):
        """How level the cruise goals are (0: glides, 1: level flight), from the curriculum's speed knob."""
        return float(np.clip(self.max_speed - 1.0, 0.0, 1.0))

    def sample_body(self, rng, spread):
        sp = BIRDS[SPECIES[int(rng.integers(len(SPECIES)))]] if rng.random() < spread else BIRDS["crow"]
        j = lambda: math.exp(rng.normal(0, 0.08 * spread))
        return dataclasses.replace(sp, mass=sp.mass * j() ** 3, scale=sp.scale * j(), wing_len=sp.wing_len * j())

    def make_creature(self, body):
        return bird(body)

    def _build(self, body):
        super()._build(body)
        J = self.J                                            # one extra output: the wingbeat frequency
        self.a_perm = np.append(self.m_perm, J); self.a_sign = np.append(self.m_sign, 1.0)
        self.act_dim = J + 1

    def beat_hz(self):
        return 4.5 * 0.9 / self.wingspan()                   # a crow (0.9 m) beats at about 4.5 Hz, a sparrow at 15

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
        full = self.spread
        self.spread = min(full, self.level() + 0.2)          # other species, wind and turns after flapping, not before
        try:
            return self._reset(body, friction, wind, sores, goal, profile, climb)
        finally:
            self.spread = full

    def _reset(self, body, friction, wind, sores, goal, profile, climb):
        rng = self.rng
        super().reset(body=body, friction=friction, wind=wind, sores=sores, goal=np.zeros(2), profile=profile)
        u = self.speed_unit()
        if goal is None:                                   # cruise: forward speed in units, a little sideways at most
            # no slower than a bird's minimum flying speed (a crow: about 7.5 m/s, 2.5 units)
            goal = np.array([rng.uniform(2.6, 3.0 + 0.8 * self.spread), rng.normal(0, 0.1 + 0.4 * self.spread)])
        if climb is None:
            # curriculum: the first goals are glides (gravity pays for the speed); as the curriculum's speed knob
            # grows they level out, so flapping has to pay for the speed
            level = self.level()
            climb = rng.normal(0, 0.05 + 0.25 * self.spread) - (1.0 - level) * self.GLIDE_SINK
        self.climb = float(climb) * u
        # start in the air at about the goal velocity, heading roughly where the goal points
        yaw = rng.uniform(-math.pi, math.pi)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        self.goal_world = Rz[:2, :2] @ (np.asarray(goal, float) * u)
        # high enough that the whole episode's asked-for sink still ends in the air
        self.alt0 = self.wingspan() * rng.uniform(3.0, 6.0) + max(0.0, -self.climb) * self.max_len / CTRL_HZ
        self.phase = rng.uniform(0, 2 * math.pi)
        wing = [self.pc.sk.bones[i].group == "wing" for i, _a in self.pc.dof_list]
        self.ev.flight(self.wingspan(), wing, beat_s=1.0 / self.beat_hz())
        v0 = np.array([self.goal_world[0], self.goal_world[1], self.climb]) * rng.uniform(0.8, 1.0)
        self.pc.set_state(np.array([0, 0, self.alt0]), Rz, self.pc.qvec(), root_vel=v0)
        self.air = AIR.Air(self.pc, wind=self.wind.w)
        return self.obs()

    # ------------------------------------------------------------------ observation: + altitude and climb goal
    def obs(self):
        o = super().obs()
        if not hasattr(self, "climb"):
            return np.concatenate([o, [0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
        p, _R = self.pc.root_pose()
        return np.concatenate([o, [p[2] / self.wingspan() - 4.0, self.climb / self.speed_unit(),
                                   math.sin(self.phase), math.cos(self.phase)]]).astype(np.float32)

    def obs_layout(self):
        out = super().obs_layout()
        k = max(sl.stop for sl in out.values())
        out["alt"] = slice(k, k + 2)
        out["clock"] = slice(k + 2, k + 4)                    # the same for both wings: unchanged by the mirror
        return out

    # ------------------------------------------------------------------ step: air forces, flight goal
    def _harness(self):
        """Curriculum help in the air: held on the asked-for path and level; never pushes forward."""
        if self.assist <= 0:
            return None
        pc = self.pc; d = pc.data
        p, R = pc.root_pose()
        m = pc.total_mass; k = self.assist
        alt = self.alt0 + self.climb * self.steps / CTRL_HZ    # the harness follows the asked-for climb or glide
        # no standing weight support: in a glide the weight is what drives the bird forward, and carrying it stalled
        # every glide; the harness only corrects departures from the path
        Fz = k * m * (6.0 * (alt - p[2]) - 3.0 * (d.qvel[2] - self.climb))
        # attitude held with a stiffness of m g x wingspan per radian: the air's torques on a spread wing are of that
        # order (a stiffness scaled by the body's inertia was ten times too weak and let it roll over)
        # The damping is explicit (applied each physics sub-step), so it is capped for the trunk alone: a bird's
        # trunk is light, and the uncapped damping blew the simulation up as soon as the wings beat
        wv_w = R @ d.qvel[3:6]; K = k * m * G * self.wingspan() * self.HOLD
        c = min(0.15 * K, 0.5 * float(pc.model.body_inertia[pc.bid[0]].min()) / pc.model.opt.timestep)
        # heading held toward the goal too (a turn is not a push: the speed is still the bird's to make)
        g = self.goal_world / max(float(np.linalg.norm(self.goal_world)), 1e-6)
        yaw_err = float(R[0, 0] * g[1] - R[1, 0] * g[0])        # sine of the angle from the body's forward to the goal
        if R[0, 0] * g[0] + R[1, 0] * g[1] < 0:
            yaw_err = math.copysign(1.0, yaw_err)
        return np.array([0.0, 0.0, max(Fz, 0.0), K * R[1, 2] - c * wv_w[0], -K * R[0, 2] - c * wv_w[1],
                         0.5 * K * yaw_err - c * wv_w[2]])

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        self.phase = (self.phase + 2 * math.pi * self.beat_hz() * math.exp(0.7 * a[self.J]) / CTRL_HZ) % (2 * math.pi)
        a = a[:self.J]
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
        # the climb rate is scored on its own, tightly: inside the speed tolerance a glide sinking 0.05 units faster than
        # asked lost under 1 % of the score, so holding height never paid and flapping was never learned
        goal = 0.7 * math.exp(-float(err[:2] @ err[:2]) / (0.05 + 0.1 * sp * sp)) * math.exp(-(float(err[2]) / self.CLIMB_TOL) ** 2)
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
        stalled = not fell and self._stalled(goal, sp)
        if fell or stalled:
            r -= self.fall_cost
        done = fell or stalled
        trunc = self.steps >= self.max_len
        self.ep_ret += r; self.ep_goal += goal
        info = dict(trunc=trunc and not done, costs=costs, goal_now=goal, speed=float(np.linalg.norm(pc.data.qvel[0:3])),
                    stalled=stalled)
        if done or trunc:
            info.update(ep_ret=self.ep_ret, ep_len=self.steps, fell=fell, goal=self.ep_goal / self.steps)
        return self.obs(), r, done, info

    def stride_pose(self, rng, speed):
        return {}
