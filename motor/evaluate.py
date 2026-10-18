"""The evaluation system: every consequence of a behaviour, measured in the physics, in comparable units.

A behaviour is never judged by how it looks. Every control step the simulation is measured and each term below
turns one consequence into a cost per second (a dimensionless rate; the body's own scale is divided out, so a child
and a horse are judged alike). The goal is the only reward; the policy trades it against the costs with the weights
of a *motivation profile*, which is an input of the policy too (an urgent body accepts strain, an injured one guards).

  term        what is measured                                                              unit before the weight
  ---------   ---------------------------------------------------------------------------   ------------------------------
  goal        progress toward the goal (``goals.py``)                                       reward per second, 1 = goal met
  injury      fracture risk per body region from contact force and loading rate             expected injuries (weighted by
              (``injury.risk``: tolerance per region, a fast load is more brittle); an      region) per event; a certain
              event, so one step carries its whole cost                                     leg fracture ~ 3 s of goal
  pain        load on sore regions (a sprain carries weight, a broken arm moves),            body weights on a sore part,
              joints pressed into the end of their range, and the load on the            plus (depth into the last 10 %)^2,
              joint-limit constraints (ligaments) wherever it comes from                   plus limit torque / capacity
  strain      muscle torque against its capacity, squared, mass-weighted mean over joints   (tau / tau_max)^2
  energy      metabolic power: positive work / 0.25 efficiency, negative work / 1.2,        m g sqrt(g l)  (l = leg length)
              holding torque (isometric cost)
  fatigue     drain of the energy reserve (refills at rest; a low reserve weakens the body)  (1 - reserve)^2
  balance     fall risk: the capture point (where the body must step to stop) out of a      leg lengths^2, rad^2
              step's reach of the feet; pelvis tilt
  gaze        head spin                                                                    (rad/s)^2 / 10
  smooth      change of the muscle commands between steps (motor economy)                   (normalised action change)^2
  heat        heat lost through the exposed surface beyond what effort produces (cold only)  stage 3, zero until then
  breath      oxygen spent with the mouth under water or in smoke                           stage 3, zero until then

Wind and water are not terms: they are forces in the simulation, and what they cost shows up as energy, strain and
fall risk. A fall ends the episode; its contact forces have already been scored as injury.

Calibration: ``DEFAULT_WEIGHTS`` are set so that a steady walk of a healthy adult at 1.3 m/s costs about a third of
the goal reward and no single comfort term dominates; ``probes.py`` checks that each term, made large, changes the
behaviour the way its consequence predicts.
"""
from __future__ import annotations
import math
import numpy as np
import mujoco

import injury

G = 9.81
TERMS = ("injury", "pain", "strain", "energy", "fatigue", "balance", "gaze", "smooth", "heat", "breath")
DEFAULT_WEIGHTS = dict(injury=6.0, pain=1.0, strain=0.6, energy=0.35, fatigue=0.5, balance=0.5, gaze=0.15,
                       smooth=0.1, heat=1.0, breath=1.0)
# Spread of the motivation profile in training: each weight is multiplied by exp(U(-s, s)).
PROFILE_SPREAD = dict(injury=1.0, pain=1.0, strain=0.8, energy=0.8, fatigue=0.8, balance=0.6, gaze=0.6, smooth=0.5,
                      heat=0.8, breath=0.6)

REGION_OF_GROUP = {"head": "head", "neck": "head", "trunk": "torso", "arm": "arms", "leg": "legs", "foot": "legs",
                   "wing": "arms", "tail": "torso"}
MUSCLE_EFF = 0.25          # positive work: chemical energy per joule of work
NEG_EFF = 1.2              # negative work costs about a fifth as much
ISO = 0.12                 # holding a torque costs energy too (per N m per rad/s of a reference speed)
RESERVE_TAU = 60.0         # s: how fast a reserve that is drained by twice the sustainable power empties
FLIGHT_UNIT = 6.0          # flight power unit in m g sqrt(g span): the cheapest searched level-flight crow wingbeat costs 1
# Local muscle endurance (Frey-Law and Avin 2010, a power model fitted across joints): a muscle can hold a share f of
# its strength for about ET = 20 f^-1.7 seconds (a quarter of the strength: some 3.5 minutes; a tenth: 17 minutes). The
# load that counts is the sustained one (|tau|/tau_max low-passed over ENDURE_TAU): a stride's legs rest between pushes,
# a held-out arm does not. The cost per second is ENDURE_K x sum of 1/ET: a walker holding its arms out paid about 0.4
# more strain per second than with them hanging (why people do not walk with their arms out, though energy hardly shows
# it), and a crouched gait pays more than an upright one (30 at first made the legs' share alone about 3 per second)
ENDURE_B0, ENDURE_B1, ENDURE_TAU, ENDURE_K = 20.0, -1.7, 1.0, 10.0
ELASTIC_RETURN = 0.6      # elastic (wing) joints: the share of the braking work that tendons and feathers store and
                           # give back (bird flight muscle: about half to two thirds); the rest is lost
SUSTAIN = 0.35             # sustainable metabolic power in m g sqrt(g l): about a brisk walk


def profile_vector(w: dict) -> np.ndarray:
    """The motivation profile as the policy sees it: log of each weight relative to the default."""
    return np.array([math.log(w[t] / DEFAULT_WEIGHTS[t]) for t in TERMS], np.float32)


def sample_profile(rng, spread=1.0) -> dict:
    return {t: DEFAULT_WEIGHTS[t] * math.exp(rng.uniform(-1, 1) * PROFILE_SPREAD[t] * spread) for t in TERMS}


class Evaluator:
    """Measures one physical body (a ``physics.PhysChar``) every control step."""

    def __init__(self, pc, sore: dict | None = None):
        self.pc = pc; m = pc.model; sk = pc.sk; c = pc.c
        self.mass = pc.total_mass
        self.leg = c.leg_len()
        self.p_unit = self.mass * G * math.sqrt(G * self.leg)          # power unit
        self.sustain = SUSTAIN; self.reserve_tau = RESERVE_TAU
        self.elastic = np.zeros(pc.nj, bool); self.elastic_tau = 0.25
        self.sore = dict(sore or {})                                    # {"leg_L": 0..1, "leg_R", "arm_L", "arm_R", "torso", "head"}
        # geom -> region, and which geoms are feet / hands
        self.region_of_geom = {}
        for i, b in enumerate(sk.bones):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_" + b.name)
            if gid >= 0:
                reg = REGION_OF_GROUP.get(b.group, "torso")
                if b.name.startswith(("pelvis", "lumbar")):
                    reg = "abdomen"
                self.region_of_geom[gid] = reg
        self.foot_geoms = set(pc.foot_geoms); self.hand_geoms = set(pc.hand_geoms)
        # every foot carries its own sore: "leg_L"/"leg_R" for a biped, "leg_HL" ... "leg_FR" for a quadruped
        key = lambda l: l.name if l.name.startswith("leg_") else "leg_" + l.name
        self.leg_side = {g: key(l) for g, l in zip(pc.foot_geoms, c.legs)}
        for l in c.legs:                                     # the toes are part of the foot
            extra = [l.toes] if l.toes is not None else []
            if c.kind == "quadruped":                        # a standing dog rests the metapodial near the paw on the ground
                extra.append(l.chain[-2])
            for b in extra:
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_" + sk.bones[b].name)
                if g >= 0:
                    self.foot_geoms.add(g); self.leg_side[g] = key(l)
        self.arm_side = {}
        for a in c.arms:
            for i in a.chain:
                self.arm_side[i] = a.side
        # joint tables: torque capacity and the mass each joint carries (mass-weighted strain)
        self.tau_max = np.maximum(pc.tau, 1e-3)
        self.endure = False                                   # local muscle endurance in the strain cost (per family)
        carried = []
        for (i, _a) in pc.dof_list:
            carried.append(float(m.body_subtreemass[pc.bid[i]]))
        w = np.array(carried); self.w_strain = w / w.sum()
        self.rng_lo, self.rng_hi = pc.lo, pc.hi
        self.dof_bone = pc.dof_bone
        self.reset()

    def flight(self, span, elastic, sustain=1.4, beat_s=0.25, reserve_tau=20.0):
        """A flier. Power unit: FLIGHT_UNIT x m g sqrt(g x wingspan), what the cheapest level-flight wingbeat found by
        search costs in this physics (with the wing's downwash: about 80 W metabolic for a crow against a real crow's 10
        or so; before, the unit was a beat searched for thrust alone, 170 to 200 W). Sustainable: 1.2 of
        those (1.4 now); a flight reserve drains three times as fast as a walker's (``reserve_tau``), so a wasteful beat tires the
        bird within a flight and a glide lets it recover (1.2 at first: level flight costs about 1.5, so even a rested bird
        sank and unlearned flying). The wing joints are elastic (``ELASTIC_RETURN``)."""
        self.p_unit = FLIGHT_UNIT * self.mass * G * math.sqrt(G * span)
        self.reserve_tau = reserve_tau
        self.sustain = sustain; self.elastic = np.asarray(elastic, bool); self.elastic_tau = beat_s
        self.w_pos = np.zeros(len(self.elastic)); self.w_neg = np.zeros(len(self.elastic))

    def reset(self):
        self.w_pos = np.zeros(len(self.elastic)); self.w_neg = np.zeros(len(self.elastic))
        self.peak = None
        self.reserve = 1.0
        self.prev_force = {r: 0.0 for r in injury.REGIONS}
        self.prev_action = None
        self.totals = {t: 0.0 for t in TERMS}
        self.sustained = np.zeros(len(self.tau_max))

    # ------------------------------------------------------------------ measurements
    def contact_forces(self):
        """Normal force per region (in body weights) and per foot, from all contacts with the ground."""
        pc = self.pc; m = pc.model; d = pc.data
        reg = {r: 0.0 for r in injury.REGIONS}; foot = {}; f6 = np.zeros(6)
        bad = False
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = con.geom1, con.geom2
            if m.geom_contype[g1] == 1 and m.geom_contype[g2] == 2:
                g = g2
            elif m.geom_contype[g2] == 1 and m.geom_contype[g1] == 2:
                g = g1
            else:
                continue
            mujoco.mj_contactForce(m, d, k, f6)
            f = abs(f6[0]) / (self.mass * G)
            r = self.region_of_geom.get(g, "torso")
            reg[r] += f
            if g in self.foot_geoms:
                foot[self.leg_side[g]] = foot.get(self.leg_side[g], 0.0) + f
            elif g not in self.hand_geoms:
                bad = True
        return reg, foot, bad

    def probe(self):
        """Call after every physics sub-step: keeps the peak contact force per region and foot (impacts last a few ms),
        and adds up the work of the elastic joints (a wingbeat is too fast to sample at the control rate)."""
        if self.elastic.any():
            pc = self.pc
            pw = pc.data.actuator_force[pc.aid] * pc.data.qvel[pc.vadr] * pc.model.opt.timestep
            self.w_pos += np.maximum(pw, 0.0); self.w_neg += np.maximum(-pw, 0.0)
        reg, foot, bad = self.contact_forces()
        if self.peak is None:
            self.peak = (reg, foot, bad)
            return
        pr, pf, pb = self.peak
        for r in reg:
            pr[r] = max(pr[r], reg[r])
        for k, v in foot.items():
            pf[k] = max(pf.get(k, 0.0), v)
        self.peak = (pr, pf, pb or bad)

    def step(self, action, dt, weights, goal_reward, strength_scale=1.0):
        """Score one control step. Returns (reward, {term: cost per second}, fell_contact)."""
        pc = self.pc; d = pc.data
        tau = d.actuator_force[pc.aid].copy()
        qd = pc.qdvec(); q = pc.qvec()
        # --- injury and pain from contact
        reg, foot, bad = self.peak if self.peak is not None else self.contact_forces()
        self.peak = None
        inj = 0.0
        for r in injury.REGIONS:
            rate = max(0.0, (reg[r] - self.prev_force[r]) / dt)
            inj += injury.WEIGHT[r] * injury.risk(reg[r], rate, r)
            self.prev_force[r] = reg[r]
        inj /= dt                                            # an event: its expected injuries are spent in this one step
        s = self.sore
        pain = (sum(s.get(k, 0.0) * f for k, f in foot.items())
                + s.get("torso", 0.0) * (reg["torso"] + reg["abdomen"]) + s.get("head", 0.0) * reg["head"])
        for j, i in enumerate(self.dof_bone):                # a sore arm hurts when it works
            side = self.arm_side.get(int(i))
            if side is not None:
                pain += 0.5 * s.get("arm_" + side, 0.0) * abs(tau[j]) / self.tau_max[j]
        # a joint driven into the end of its range (the last 5 %) by its own muscle: ligaments and capsule hurt
        span = self.rng_hi - self.rng_lo
        hi_d = np.maximum(0.0, q - (self.rng_hi - 0.05 * span)) / (0.05 * span + 1e-6)
        lo_d = np.maximum(0.0, (self.rng_lo + 0.05 * span) - q) / (0.05 * span + 1e-6)
        press = np.maximum(0.0, tau) * hi_d ** 2 + np.maximum(0.0, -tau) * lo_d ** 2
        pain += float(np.sum(press / self.tau_max))
        # the load on ligaments and the stretched opposing muscles at the end of a soft range, whoever puts it there
        # (the body's own muscle, gravity, momentum): a joint held at its stop hurts in proportion to that load. Only
        # soft-range joints: a straight knee or a locked hip resting on its ligaments is how one stands
        if pc.soft_mask.any():
            pain += float(np.sum(pc.limit_torque()[pc.soft_mask] / self.tau_max[pc.soft_mask]))
        # --- strain and energy
        u = tau / (self.tau_max * strength_scale)
        strain = float(np.sum(self.w_strain * u ** 2))
        if self.endure:
            self.sustained += (np.abs(u) - self.sustained) * min(1.0, dt / ENDURE_TAU)
            f = np.clip(self.sustained, 0.02, 1.0)
            strain += ENDURE_K * float(np.sum(1.0 / (ENDURE_B0 * f ** ENDURE_B1)))
        pw = tau * qd
        p_met = (np.sum(np.maximum(pw, 0.0) * ~self.elastic) / MUSCLE_EFF + np.sum(np.maximum(-pw, 0.0) * ~self.elastic) / NEG_EFF
                 + ISO * np.sum(np.abs(tau)) * math.sqrt(G / self.leg) * 0.1)
        if self.elastic.any():
            # elastic joints, work summed over every physics sub-step: tendons return ELASTIC_RETURN of the braking work
            # into the next stroke, the muscle does the rest. When all of it was counted as returned, beating twice as
            # fast as a crow does cost nothing, and the trained crow beat at the fastest rate allowed
            wp = self.w_pos[self.elastic].sum() / dt; wn = self.w_neg[self.elastic].sum() / dt
            p_met += max(wp - ELASTIC_RETURN * wn, 0.0) / MUSCLE_EFF + (1.0 - ELASTIC_RETURN) * wn / NEG_EFF
            self.w_pos[:] = 0.0; self.w_neg[:] = 0.0
        energy = float(p_met / self.p_unit)
        # --- fatigue: the reserve drains above the sustainable power and refills below it
        self.reserve = float(np.clip(self.reserve - (energy - self.sustain) / (self.sustain * self.reserve_tau) * dt, 0.0, 1.0))
        fatigue = (1.0 - self.reserve) ** 2
        # --- balance: fall risk. The capture point (centre of mass plus velocity over the pendulum frequency) is where the
        # body must step to stop; walking keeps it within a step of the feet, a fall puts it out of reach.
        com = pc.com(); fp = pc.foot_points().reshape(-1, 3)
        vcom = d.subtree_linvel[pc.bid[0]][:2]
        w0 = math.sqrt(G / max(com[2] - fp[:, 2].min(), 0.2))
        xcap = com[:2] + vcom / w0
        reach = float(np.min(np.linalg.norm(fp[:, :2] - xcap[None, :], axis=1)))
        _p, R = pc.root_pose()
        tilt = math.acos(max(-1.0, min(1.0, float(R[2, 2]))))
        balance = 4.0 * max(0.0, reach - 0.5 * self.leg) ** 2 / self.leg ** 2 + 0.5 * tilt ** 2
        # --- gaze: head spin
        head = pc.c.head
        gaze = 0.0
        if head >= 0:
            w = d.cvel[pc.bid[head]][:3]
            gaze = float(w @ w) / 10.0
        # --- smoothness
        smooth = 0.0 if self.prev_action is None else float(np.mean((action - self.prev_action) ** 2))
        self.prev_action = np.array(action, copy=True)
        costs = dict(injury=inj, pain=pain, strain=strain, energy=energy, fatigue=fatigue, balance=balance, gaze=gaze,
                     smooth=smooth, heat=0.0, breath=0.0)
        # a single violent step must not swamp training (an injury is an event and may be large)
        costs = {t: min(float(v), 20.0 if t != "injury" else 10.0 / dt) for t, v in costs.items()}
        total = sum(weights[t] * costs[t] for t in TERMS)
        for t in TERMS:
            self.totals[t] += costs[t] * dt
        return (goal_reward - total) * dt, costs, bad
