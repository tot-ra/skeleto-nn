"""Procedural locomotion planner for legged bodies (biped, quadruped, bird).

Nothing here is specific to stairs, slopes, obstacles or one species. Everything comes from a few
general mechanisms:

* gait parameters follow from the Froude number v^2/(g h) (dynamic similarity, Alexander 1976), so
  a dwarf, a horse and a cat pick their own cadence, duty factor and gait;
* every foot is planted in the world while in stance and swings between footholds that are searched
  on the terrain (flat, away from edges, reachable), so stairs and kerbs are just terrain;
* swing height comes from the terrain between lift-off and landing (step over, step up);
* the trunk height is the highest value the stance legs can reach, so the pelvis vaults over a
  planted leg and dips in double support by itself;
* a low ceiling lowers and tilts the trunk; the trunk pitch also moves the whole-body centre of
  mass over the feet, so a belly, a pack or armour change posture without extra code;
* flight (running, jumping) is ballistic between take-off and the next touchdown.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from skeleton import rot, rot_axis, align, euler_from_R
from ik import PlanarLeg, leg_ik, arm_ik, basis, clamp_lim
from terrain import Terrain

G = 9.81

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def smooth(x):
    x = min(1.0, max(0.0, x)); return x * x * x * (x * (6 * x - 15) + 10)

def interp_table(x, xs, ys):
    return float(np.interp(x, xs, ys))

# gait tables: phase of touchdown for legs in the order [HL, HR, FL, FR] (quadruped) or [L, R] (biped)
QUAD_GAITS = {
    "walk":   [0.00, 0.50, 0.25, 0.75],
    "trot":   [0.00, 0.50, 0.50, 1.00],
    "canter": [0.00, 0.33, 0.33, 0.66],
    "gallop": [0.00, 0.10, 0.50, 0.60],
}
DUTY_X = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 6.0]
DUTY_Y = [0.72, 0.68, 0.60, 0.50, 0.40, 0.32, 0.28, 0.25]
QUAD_DUTY_Y = [0.75, 0.70, 0.62, 0.55, 0.45, 0.36, 0.30, 0.25]

@dataclass
class Cmd:
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))      # desired world velocity
    heading: float | None = None                                    # desired body heading (None: along v)
    crouch: float = 0.0                                             # 0..1 deliberate lowering
    arms: dict | None = None                                        # {"L": world target or None, "R": ...}
    look: np.ndarray | None = None                                  # world point to look at
    carry: dict | None = None                                       # arm hold override
    pelvis: dict | None = None                                      # {"z": absolute pelvis height, "pitch": trunk pitch, "xy": world xy}
    torso_yaw: float = 0.0                                          # extra chest yaw (attacks)
    arm_pole: dict | None = None

class LegState:
    def __init__(self, leg, idx):
        self.leg = leg; self.idx = idx
        self.stance = True
        self.p = 0.0                 # phase (rhythmic mode)
        self.planted = np.zeros(3)   # foot centre in world while in stance
        self.yaw = 0.0
        self.lift = np.zeros(3); self.target = np.zeros(3); self.swing_t = 0.0; self.swing_T = 0.4
        self.lift_yaw = 0.0; self.apex = 0.0; self.target_yaw = 0.0; self.toe_bend = 0.0
        self.u = 0.5                 # stance progress 0..1
        self.stance_t = 0.0
        self.ground_n = np.array([0, 0, 1.0])
        self.td_event = False; self.lo_event = False
        self.hip_w = np.zeros(3); self.ankle_w = np.zeros(3); self.R_foot = np.eye(3); self.flat_pitch = 0.0
        self.flight_pitch = 0.0
        self.pl: PlanarLeg | None = None

class Walker:
    def __init__(self, creature, terrain: Terrain, pos=(0.0, 0.0), heading=0.0, dt=1 / 60.0, seed=0):
        self.c = creature; self.sk = creature.skel; self.terrain = terrain; self.dt = dt
        self.rng = np.random.default_rng(seed)
        sk = self.sk; c = creature
        self.kind = c.kind
        self.pos = np.array(pos[:2], float); self.heading = heading; self.v = np.zeros(2); self.yaw_rate = 0.0
        g0 = float(terrain.h(pos[0], pos[1]))
        self.z = g0 + c.z0 * c.params.get("stand_ratio", 1.0) if c.kind != "quadruped" else g0 + c.z0
        self.vz = 0.0; self.z_f = self.z
        self.t = 0.0; self.S = None; self.R_p = np.eye(3); self.root = np.zeros(3)
        self.q = sk.zeros()
        self.legs = [LegState(l, i) for i, l in enumerate(c.legs)]
        for ls in self.legs: ls.pl = PlanarLeg(sk, ls.leg)
        self.hip_h = c.z0
        self.phase = 0.0; self.freq = 0.0; self.duty = 0.6; self.gait_offsets = np.zeros(len(self.legs))
        self.pitch = 0.0; self.roll = 0.0; self.pitch_s = 0.0
        self.crouch = 0.0; self.lean_com = 0.0
        self.mode = "stand"
        self.flight = False; self.flight_t = 0.0; self.flight_T = 0.0; self.z_to = self.z; self.z_td = self.z
        self.speed = 0.0; self.Fr = 0.0
        self.com = np.zeros(3); self.support_c = np.zeros(2)
        self.extra_vel = np.zeros(2)               # impulse-driven COM velocity offset (push)
        self.trunk_wobble = np.zeros(3); self.wob_v = np.zeros(3)
        self.nwb = c.params.get("nwb"); self.aids = None; self.leg_gait = c.params.get("leg_gait", {}); self.leg_duty = {}
        self.leg_ovr = None; self.jumpplan = None; self.refused_jump = False; self.hit_flash = 0.0; self.auto_duck = True
        self.tail_state = np.zeros((max(1, len(c.tail)), 2)); self.tail_v = np.zeros_like(self.tail_state)
        self.arm_phase = 0.0
        self.foot_yaw_out = math.radians(6.0) if c.kind == "biped" else 0.0
        self.hip_lat = {}
        S, E, R = sk.fk(np.zeros(3), np.eye(3), sk.zeros())
        self.rest_hips = {}
        for ls in self.legs:
            h = ls.leg.chain[0]
            self.rest_hips[ls.idx] = S[h] - S[0]
        # neutral foot positions relative to the hip projection: standing pose
        self.neutral = {}
        for ls in self.legs:
            f = ls.leg.chain[-1]
            ctr = S[f] + R[f] @ (0.5 * (ls.leg.heel + ls.leg.ball))
            self.neutral[ls.idx] = np.array([ctr[0] - S[0][0], ctr[1] - S[0][1]])
        self.foot_len = {ls.idx: float(np.linalg.norm(ls.leg.ball - ls.leg.heel)) for ls in self.legs}
        self.mid_local = {ls.idx: 0.5 * (ls.leg.heel + ls.leg.ball) for ls in self.legs}
        self.reach = float(np.mean([l.reach for l in c.legs])) if c.legs else 1.0
        self.max_step = 1.15 * self.reach
        self.rest_com_off = None
        self.head_top = self._head_top_rest()
        self._init_stance()
        self.trace = []
        import hit as _hit; _hit.init(self)

    # ----------------------------------------------------------------------------------------
    def _head_top_rest(self):
        S, E, R = self.sk.fk(np.zeros(3), np.eye(3), self.sk.zeros())
        return float(E[:, 2].max())

    def _init_stance(self):
        cth, sth = math.cos(self.heading), math.sin(self.heading)
        Rh = np.array([[cth, -sth], [sth, cth]])
        for ls in self.legs:
            xy = self.pos + Rh @ self.neutral[ls.idx]
            ls.planted = np.array([xy[0], xy[1], float(self.terrain.h(xy[0], xy[1]))])
            ls.yaw = self.heading; ls.stance = True; ls.u = 0.5
            ls.ground_n = self.terrain.normal(xy[0], xy[1])
        self.z = float(np.mean([ls.planted[2] for ls in self.legs])) + self.c.z0 * (self.c.params.get("stand_ratio", 1.0) if self.kind != "quadruped" else 1.0)
        self.z_f = self.z

    # ---- gait selection ---------------------------------------------------------------------
    def gait_params(self, speed):
        h = max(self.hip_h, 0.05)
        Fr = speed * speed / (G * h)
        self.Fr = Fr
        if self.kind == "quadruped":
            duty = interp_table(Fr, DUTY_X, QUAD_DUTY_Y)
            lam = 2.3 * h * max(Fr, 1e-4) ** 0.3 * (1 + 0.6 * smooth((Fr - 1.5) / 2.5))
            stiff = self.c.params.get("spine_flex", 1.0)
            # offsets: blend gaits by Froude number (walk -> trot -> canter -> gallop)
            def mix(a, b, x): return [(1 - x) * p + x * q for p, q in zip(QUAD_GAITS[a], QUAD_GAITS[b])]
            if Fr < 0.25: off = QUAD_GAITS["walk"]
            elif Fr < 0.5: off = mix("walk", "trot", smooth((Fr - 0.25) / 0.25))
            elif Fr < 2.2: off = QUAD_GAITS["trot"]
            elif Fr < 3.2: off = mix("trot", "canter", smooth((Fr - 2.2) / 1.0))
            elif Fr < 5.0: off = QUAD_GAITS["canter"]
            else: off = mix("canter", "gallop", smooth((Fr - 5.0) / 3.0))
            # leg order in c.legs is [HL, HR, FL, FR]
            off = np.array(off)
        else:
            duty = interp_table(Fr, DUTY_X, DUTY_Y)
            lam = 2.3 * h * max(Fr, 1e-4) ** 0.3
            off = np.array([0.0, 0.5])
            if self.kind == "bird":
                duty = max(duty, 0.55) if Fr < 0.8 else duty
        lam *= self.c.params.get("stride_scale", 1.0)
        f = speed / max(lam, 1e-3)
        return f, float(np.clip(duty, 0.2, 0.8)), off

    # ---- foothold search --------------------------------------------------------------------
    def foothold(self, nominal_xy, cur_z, hip_xyz, yaw):
        tr = self.terrain
        best, bc = None, 1e9
        flen = self.foot_len[0] if self.legs else 0.2
        cy, sy = math.cos(yaw), math.sin(yaw)
        step = 0.045 * self.reach
        for ix in range(-3, 4):
            for iy in range(-2, 3):
                off = np.array([ix * step, iy * step * 0.8])
                p = nominal_xy + np.array([cy * off[0] - sy * off[1], sy * off[0] + cy * off[1]])
                h = float(tr.h(p[0], p[1]))
                if tr.is_pit(p[0], p[1]): continue
                # footprint: heel and toe heights must agree (edge avoidance)
                hf = [float(tr.h(p[0] + cy * a, p[1] + sy * a)) for a in (-0.5 * flen, 0.5 * flen)]
                edge = max(abs(hf[0] - h), abs(hf[1] - h))
                hw = [float(tr.h(p[0] - sy * a, p[1] + cy * a)) for a in (-0.35 * flen, 0.35 * flen)]
                edge = max(edge, max(abs(hw[0] - h), abs(hw[1] - h)))
                dz = h - cur_z
                reach_ok = hip_xyz[2] - h
                if reach_ok < 0.2 * self.reach: continue
                c = np.dot(off, off) * 4.0 + 30.0 * edge + 3.0 * max(0.0, abs(dz) - 0.55 * self.reach)
                if c < bc: bc, best = c, np.array([p[0], p[1], h])
        if best is None:
            best = np.array([nominal_xy[0], nominal_xy[1], float(tr.h(nominal_xy[0], nominal_xy[1]))])
        return best

    # ---- main step --------------------------------------------------------------------------
    def step(self, cmd: Cmd, dt=None):
        dt = dt or self.dt
        sk, c, tr = self.sk, self.c, self.terrain
        self.t += dt
        import hit as _hit
        if self.down is not None: return _hit.down_step(self, dt)
        cmd = _hit.modify_cmd(self, cmd, dt)
        if self.aids is not None: cmd = self.aids.apply(self, cmd, dt)
        vdes = np.asarray(cmd.v, float) if self.jumpplan is None else np.zeros(2)
        vcap = c.params.get("v_max")
        if vcap and np.linalg.norm(vdes) > vcap: vdes = vdes / np.linalg.norm(vdes) * vcap       # clothing sets how fast this body can move
        sp_des = float(np.linalg.norm(vdes))
        # --- body velocity and heading
        acc = 3.0 * math.sqrt(self.reach) + 1.0
        dv = vdes + self.extra_vel * 0 - self.v
        nv = np.linalg.norm(dv)
        if nv > acc * dt: dv = dv / nv * acc * dt
        self.v = self.v + dv
        self.speed = float(np.linalg.norm(self.v))
        if cmd.heading is not None:
            hd = cmd.heading
        elif sp_des > 0.15:
            hd = math.atan2(vdes[1], vdes[0])
        else:
            hd = self.heading
        dh = wrap(hd - self.heading)
        max_rate = 2.8 if self.kind != "quadruped" else 2.0
        # turning is limited when moving fast (curvature limit)
        max_rate = max_rate / (1.0 + 0.35 * self.speed)
        new_rate = float(np.clip(dh * 5.0, -max_rate, max_rate))
        self.yaw_rate += (new_rate - self.yaw_rate) * min(1.0, 8 * dt)
        self.heading = wrap(self.heading + self.yaw_rate * dt)
        fwd = np.array([math.cos(self.heading), math.sin(self.heading)])
        left = np.array([-fwd[1], fwd[0]])
        # world velocity follows heading (no sideways skating at speed)
        vf = float(self.v @ fwd)
        # --- gait
        f, duty, off = self.gait_params(max(self.speed, 1e-3))
        moving = self.speed > 0.12 or sp_des > 0.2
        mode_prev = self.mode
        self.mode = "go" if moving else "stand"
        if self.mode == "go" and mode_prev == "stand":
            self._start_gait(off, duty)
        if self.mode == "go":
            self.freq = f; self.duty = duty
            self.phase = (self.phase + f * dt) % 1.0
            # pull each leg's phase towards the pattern
            for ls in self.legs:
                target = (self.phase - off[ls.idx]) % 1.0
                err = wrap((target - ls.p) * 2 * math.pi) / (2 * math.pi)
                ls.p = (ls.p + (f + 3.0 * err) * dt) % 1.0
        # root xy integration
        if self.mode == "go" or self.speed > 0.02:
            self.pos = self.pos + self.v * dt
        # --- per leg state machine
        hips = self._hip_world_xy()
        in_flight = self.jumpplan is not None and self.jumpplan["stage"] == "flight"
        self.freeze_feet = cmd.pelvis is not None
        for ls in self.legs:
            ls.td_event = ls.lo_event = False
            if self.nwb is not None and ls.leg.name == self.nwb:          # a leg that carries no weight (held up, on crutches, or missing)
                ls.stance = False; ls.u = 0.5; continue
            sd = ls.leg.side; duty_i = float(np.clip(duty * self.leg_gait.get(ls.leg.name, self.leg_gait.get(sd, {})).get("duty", 1.0) * self.leg_duty.get(sd, 1.0), 0.15, 0.9))
            if in_flight:
                if not ls.stance:
                    ls.swing_t += dt
                    if ls.swing_t >= ls.swing_T: self._touchdown(ls)
            elif self.mode == "go":
                should_stance = ls.p < duty_i
                if ls.stance and not should_stance:
                    self._liftoff(ls, hips, f, duty_i, left, fwd)
                elif (not ls.stance) and should_stance:
                    self._touchdown(ls)
                if ls.stance:
                    ls.u = min(1.0, ls.p / max(duty_i, 1e-3))
                else:
                    ls.swing_T = (1 - duty_i) / max(f, 1e-3)
                    ls.swing_t = min(ls.swing_T, ((ls.p - duty_i) / max(1 - duty_i, 1e-3)) * ls.swing_T)
            else:
                self._reactive_step(ls, hips, fwd, left, dt)
        # --- spring-damper response of the trunk to pushes and hits
        k_w = 70.0 * c.params.get("balance", 1.0); c_w = 9.0 * c.params.get("balance", 1.0) ** 0.5
        self.wob_v += (-k_w * self.trunk_wobble - c_w * self.wob_v) * dt
        self.trunk_wobble += self.wob_v * dt
        self.hit_flash = max(0.0, self.hit_flash - dt)
        if self.jumpplan is not None:
            self._jump_step(cmd, dt, fwd, left)
        # --- trunk pose then legs
        self._pose(cmd, dt, fwd, left)
        return self.snapshot()

    # ---- gait events --------------------------------------------------------------------
    def _start_gait(self, off, duty):
        """Leave standing: choose phases so that the first leg of the pattern lifts now."""
        n = len(self.legs)
        # the leg that is furthest behind its neutral position steps first
        self.phase = duty % 1.0
        for ls in self.legs:
            ls.p = (self.phase - off[ls.idx]) % 1.0
        # legs currently in a reactive swing keep their swing; others start in stance with matching progress
        for ls in self.legs:
            if not ls.stance:
                ls.p = duty + 0.02

    def _hip_world_xy(self):
        cth, sth = math.cos(self.heading), math.sin(self.heading)
        Rh = np.array([[cth, -sth], [sth, cth]])
        out = {}
        for ls in self.legs:
            out[ls.idx] = self.pos + Rh @ self.rest_hips[ls.idx][:2]
        return out

    def _liftoff(self, ls, hips, f, duty, left, fwd):
        ls.stance = False; ls.lo_event = True
        ls.lift = ls.planted.copy(); ls.lift_yaw = ls.yaw
        self._plan_target(ls, hips, f, duty, fwd, left)

    def _plan_target(self, ls, hips, f, duty, fwd, left):
        t_sw = (1 - duty) / max(f, 1e-3)
        t_st = duty / max(f, 1e-3)
        # predict the hip at mid-stance: it moves with the body; turning curves the path
        ahead = t_sw + 0.5 * t_st if self.mode == "go" else 0.3
        hip_pred = hips[ls.idx] + self.v * ahead
        hd_pred = self.heading + self.yaw_rate * ahead
        cp, sp_ = math.cos(hd_pred), math.sin(hd_pred)
        nominal_rel = self.neutral[ls.idx] - self.rest_hips[ls.idx][:2]
        Rp = np.array([[cp, -sp_], [sp_, cp]])
        nominal = hip_pred + Rp @ nominal_rel
        # step-length scaling with speed is implicit (hip moves); add a touch of capture-point feedback
        # limit the step so the leg can reach
        hip_z = self.z
        hz = hip_z + self.rest_hips[ls.idx][2]
        cur_z = ls.lift[2]
        # limit by reach: keep nominal within a circle around the predicted hip
        dxy = nominal - hip_pred
        lim = 0.92 * self.reach
        if np.linalg.norm(dxy) > lim: nominal = hip_pred + dxy / np.linalg.norm(dxy) * lim
        # skirt / armour restricts the stride
        tgt = self.foothold(nominal, cur_z, np.array([hip_pred[0], hip_pred[1], hz]), hd_pred)
        ls.target = tgt; ls.target_yaw = hd_pred + (self.foot_yaw_out if ls.leg.side == "L" else -self.foot_yaw_out)
        ls.swing_T = max(t_sw, 0.12)
        ls.swing_t = 0.0
        # swing apex from terrain between lift and target
        mx = self.terrain.max_along(ls.lift, tgt, 24, half_width=0.05)
        base = max(ls.lift[2], tgt[2])
        clr = (0.05 + 0.10 * min(1.0, self.speed / 2.0)) * self.reach * (0.7 if self.kind == "quadruped" else 1.0)
        if self.kind == "quadruped": clr *= 0.7
        ls.apex = max(base, mx + 0.045 * self.reach) + clr
        ls.apex = max(ls.apex, base + clr)

    def _touchdown(self, ls):
        ls.stance = True; ls.td_event = True
        ls.planted = ls.target.copy(); ls.yaw = ls.target_yaw
        ls.ground_n = self.terrain.normal(ls.planted[0], ls.planted[1])
        ls.stance_t = 0.0

    def _reactive_step(self, ls, hips, fwd, left, dt):
        """Standing: take a step with a foot that has drifted from its neutral position, one leg at a time."""
        n_swing = sum(1 for l in self.legs if not l.stance)
        if ls.stance:
            ls.u = 0.5
            cth, sth = math.cos(self.heading), math.sin(self.heading)
            Rh = np.array([[cth, -sth], [sth, cth]])
            want = self.pos + Rh @ self.neutral[ls.idx]
            err = np.linalg.norm(ls.planted[:2] - want)
            thresh = 0.30 * self.reach if self.kind != "quadruped" else 0.22 * self.reach
            if len(self.legs) == 4: max_sw = 1
            else: max_sw = 1
            # a leg lifts only when its mirror/diagonal partner is down and it is the worst offender
            worst = max(np.linalg.norm(l.planted[:2] - (self.pos + Rh @ self.neutral[l.idx])) for l in self.legs if l.stance)
            if err > thresh and n_swing < max_sw and err >= worst - 1e-6 and not getattr(self, 'freeze_feet', False):
                ls.stance = False; ls.lo_event = True
                ls.lift = ls.planted.copy(); ls.lift_yaw = ls.yaw
                tgt = self.foothold(want, ls.lift[2], np.array([self.pos[0], self.pos[1], self.z]), self.heading)
                ls.target = tgt; ls.target_yaw = self.heading + (self.foot_yaw_out if ls.leg.side == "L" else -self.foot_yaw_out)
                ls.swing_T = 0.45 * math.sqrt(self.reach / 0.9); ls.swing_t = 0.0
                mx = self.terrain.max_along(ls.lift, tgt, 20, half_width=0.05)
                base = max(ls.lift[2], tgt[2])
                ls.apex = max(base, mx + 0.04 * self.reach) + 0.06 * self.reach
        else:
            ls.swing_t += dt
            if ls.swing_t >= ls.swing_T:
                self._touchdown(ls)

    # ---- pose -----------------------------------------------------------------------------
    def _swing_pose(self, ls):
        """Foot centre, yaw and toe-up pitch of a swinging foot."""
        tau = min(1.0, ls.swing_t / max(ls.swing_T, 1e-3))
        sh = smooth((tau - 0.06) / 0.94)
        xy = ls.lift[:2] + (ls.target[:2] - ls.lift[:2]) * sh
        z0, z1, za = ls.lift[2], ls.target[2], ls.apex
        if tau < 0.45:
            k = math.sin(0.5 * math.pi * tau / 0.45); z = z0 + (za - z0) * k
        else:
            k = 0.5 - 0.5 * math.cos(math.pi * (tau - 0.45) / 0.55); z = za + (z1 - za) * k
        yaw = ls.lift_yaw + wrap(ls.target_yaw - ls.lift_yaw) * smooth(tau)
        hike = self.leg_gait.get(ls.leg.name, self.leg_gait.get(ls.leg.side, {})).get("hike", 0.0)
        if hike:                                    # stiff leg: swung out and round (circumduction)
            lat = np.array([-math.sin(self.heading), math.cos(self.heading)]) * (1.0 if ls.leg.side == "L" else -1.0)
            xy = xy + lat * hike * math.sin(math.pi * tau); z = z + 0.5 * hike * math.sin(math.pi * tau)
        amp = self.c.params.get("toe_amp", 1.0)
        pitch_up = amp * (-0.55 * (1 - tau) ** 2 + 0.35 * smooth((tau - 0.55) / 0.45) * 1.0 - 0.1 * math.sin(math.pi * tau))
        return np.array([xy[0], xy[1], z]), yaw, pitch_up

    def _foot_flat_R(self, yaw, n):
        """World rotation of a foot lying flat on a surface with normal n and heading yaw."""
        Rz_ = rot("Z", yaw)
        Rn = align(np.array([0, 0, 1.0]), n)
        return Rn @ Rz_

    def _stance_foot(self, ls):
        """Ankle world position and foot rotation of a planted foot, with heel-strike and toe-off rolling."""
        leg = ls.leg; mid = self.mid_local[ls.idx]
        Rf = self._foot_flat_R(ls.yaw, ls.ground_n)
        ctr = ls.planted
        ankle_flat = ctr - Rf @ mid
        u = ls.u
        amp = self.c.params.get("toe_amp", 1.0)
        heel_up = amp * math.radians(15) * (1 - smooth(u / 0.18))      # heel strike: toes up, pivot on heel
        toe_off = amp * math.radians(32) * smooth((u - 0.55) / 0.45)   # push-off: heel up, pivot on ball
        alpha = self.c.params.get("heel_pitch", 0.0)
        if alpha > 0:                                    # high heels: the foot always stands pitched forward on its ball
            heel_up *= max(0.0, 1 - alpha / 0.3); toe_off = alpha + toe_off * (1 - alpha / 0.7)
        if self.mode != "go":
            heel_up = 0.0; toe_off = alpha
        ls.toe_bend = -toe_off
        ax = Rf[:, 1]
        if heel_up > 1e-4:
            P = ctr + Rf @ (leg.heel - mid)
            Rr = rot_axis(ax, -heel_up)
            return P + Rr @ (ankle_flat - P), Rr @ Rf
        if toe_off > 1e-4:
            P = ctr + Rf @ (leg.ball - mid)
            Rr = rot_axis(ax, toe_off)
            return P + Rr @ (ankle_flat - P), Rr @ Rf
        return ankle_flat, Rf

    def _targets(self):
        """Ankle targets and foot rotations for every leg at the current time."""
        out = {}
        for ls in self.legs:
            mid = self.mid_local[ls.idx]
            if self.nwb is not None and ls.leg.name == self.nwb and self.S is not None:
                hip = self.S[ls.leg.chain[0]]; f_ = np.array([math.cos(self.heading), math.sin(self.heading)])
                hang = 0.55 if self.kind == "quadruped" else 0.80
                pt = np.array([hip[0] + f_[0] * (0.05 if self.kind == "quadruped" else 0.10), hip[1] + f_[1] * 0.05, hip[2] - hang * ls.pl.reach])
                Rf = self._foot_flat_R(self.heading, np.array([0, 0, 1.0])); out[ls.idx] = (pt - Rf @ mid, Rf); continue
            ov = self.leg_ovr
            if ov is not None and ov["side"] == ls.leg.side and self.t < ov["until"]:       # a deliberate leg movement (lifting the knee to block)
                if ls.stance:
                    ls.stance = False; ls.lift = ls.planted.copy(); ls.lift_yaw = ls.yaw
                ls.target = ov["point"].copy(); ls.target_yaw = self.heading; ls.apex = float(ov["point"][2]); ls.swing_t = 0.0; ls.swing_T = 1.0
                Rf = self._foot_flat_R(self.heading, np.array([0, 0, 1.0]))
                out[ls.idx] = (ov["point"] - Rf @ mid, Rf); continue
            if ls.stance:
                out[ls.idx] = self._stance_foot(ls)
            elif self.jumpplan is not None and self.jumpplan["stage"] == "flight":
                jp = self.jumpplan; tau = min(1.0, jp["tf"] / jp["T"])
                ch, sh = math.cos(self.heading), math.sin(self.heading)
                Rh = np.array([[ch, -sh], [sh, ch]])
                if self.kind == "quadruped":
                    tuck = 0.35 + 0.25 * min(1.0, self.c.params.get("spine_flex", 1.0))     # supple bodies fold the legs further
                    bump = smooth((tau - 0.2) / 0.2) * (1 - smooth((tau - 0.62) / 0.2))
                    if ls.leg.fore:       # reach forward and up, fold at the apex, reach out to land first
                        e = 0.80 - (0.80 - tuck) * bump + 0.20 * smooth((tau - 0.7) / 0.3)
                        dx = 0.30 * (1 - smooth(tau / 0.3)) * 0.4 + 0.35 * smooth((tau - 0.6) / 0.4)
                    else:                 # trail straight behind while pushing, then swing under the belly
                        e = 1.0 - (1.0 - tuck) * smooth((tau - 0.15) / 0.25) + (0.95 - tuck) * smooth((tau - 0.72) / 0.28)
                        dx = -0.45 * (1 - smooth(tau / 0.3)) + 0.10 * smooth((tau - 0.5) / 0.5)
                    xy = self.pos + Rh @ self.neutral[ls.idx] + np.array([ch, sh]) * dx * self.reach
                    hip = self.S[ls.leg.chain[0]] if getattr(self, "S", None) is not None else None
                    zhip = (hip[2] + self.z - self.root[2]) if hip is not None else self.z
                    zc = max(float(self.terrain.h(xy[0], xy[1])), zhip - e * ls.pl.reach)
                else:
                    e = 1.0 - 0.50 * smooth(tau / 0.25) + 0.49 * smooth((tau - 0.62) / 0.38)
                    xy = self.pos + Rh @ self.neutral[ls.idx] + np.array([ch, sh]) * 0.12 * self.reach * math.sin(math.pi * min(1.0, tau * 1.1))
                    zc = max(float(self.terrain.h(xy[0], xy[1])), self.z - e * self.c.z0)
                pitch_up = -0.55 * (1 - smooth(tau / 0.3)) + 0.3 * smooth((tau - 0.7) / 0.3)
                Rf = self._foot_flat_R(self.heading, np.array([0, 0, 1.0]))
                Rf2 = rot_axis(Rf[:, 1], -pitch_up) @ Rf
                out[ls.idx] = (np.array([xy[0], xy[1], zc]) - Rf2 @ mid, Rf2)
            else:
                u_s = ls.swing_t / max(ls.swing_T, 1e-3)
                ls.toe_bend = 0.35 * (1 - smooth(u_s / 0.3)) - 0.15 * smooth((u_s - 0.6) / 0.4)
                ctr, yaw, pitch_up = self._swing_pose(ls)
                n = ls.ground_n
                Rf = self._foot_flat_R(yaw, n)
                Rr = rot_axis(Rf[:, 1], -pitch_up)
                Rf2 = Rr @ Rf
                out[ls.idx] = (ctr - Rf2 @ mid, Rf2)
        return out

    def _pose(self, cmd, dt, fwd, left):
        sk, c, tr = self.sk, self.c, self.terrain
        stance_legs = [ls for ls in self.legs if ls.stance]
        tg = self._targets()
        # --- trunk orientation (yaw, pitch, roll) ---
        speed = self.speed
        v_n = speed / math.sqrt(G * self.reach)
        lean_speed = (0.05 + 0.20 * min(1.0, v_n * 1.4)) if self.kind == "biped" else 0.0
        if self.kind == "quadruped":
            hind = [ls.planted[2] for ls in self.legs if not ls.leg.fore]; fore = [ls.planted[2] for ls in self.legs if ls.leg.fore]
            span = abs(self.rest_hips[[l.idx for l in self.legs if l.leg.fore][0]][0] - self.rest_hips[[l.idx for l in self.legs if not l.leg.fore][0]][0])
            slope_pitch = -math.atan2(np.mean(fore) - np.mean(hind), max(span, 0.2))
        else:
            ahead = self.pos + fwd * 0.6 * self.reach
            slope_pitch = -0.7 * math.atan2(float(tr.h(ahead[0], ahead[1])) - float(tr.h(self.pos[0], self.pos[1])), 0.6 * self.reach)
        # ceiling ahead lowers the trunk
        head_h = self._head_height_above_root()
        ceil = min(float(tr.ceiling(self.pos[0] + fwd[0] * d, self.pos[1] + fwd[1] * d)) for d in (0.0, 0.4, 0.9, 1.5))
        duck = 0.0
        if np.isfinite(ceil):
            ground = float(np.mean([ls.planted[2] for ls in self.legs]))
            free = ceil - ground
            need = head_h + 0.06
            duck = float(np.clip((need - free) / max(0.8 * self.c.z0, 1e-3), 0.0, 1.0))
        if not self.auto_duck: duck = 0.0
        self.duck_s = getattr(self, "duck_s", 0.0) + (duck - getattr(self, "duck_s", 0.0)) * min(1.0, 6 * dt)
        crouch = max(cmd.crouch, self.duck_s)
        self.crouch += (crouch - self.crouch) * min(1.0, 8 * dt)
        pitch_t = lean_speed + slope_pitch + 0.45 * self.crouch * (1 if self.kind == "biped" else 0.3)
        jpitch = self._jump_pitch()
        if jpitch is not None: pitch_t = jpitch
        if cmd.pelvis is not None and "pitch" in cmd.pelvis: pitch_t = cmd.pelvis["pitch"]
        # sway and gait-coupled yaw/roll from the actual foot excursions
        wL = sum((1 - (ls.u if ls.stance else 1)) * 0 + (1.0 if ls.stance else 0.0) for ls in self.legs if ls.leg.side == "L")
        wR = sum((1.0 if ls.stance else 0.0) for ls in self.legs if ls.leg.side == "R")
        nL = max(1, sum(1 for ls in self.legs if ls.leg.side == "L")); nR = max(1, sum(1 for ls in self.legs if ls.leg.side == "R"))
        sway_w = (wL / nL - wR / nR)
        self.sway = getattr(self, "sway", 0.0); self.sway += (sway_w - self.sway) * min(1.0, 7 * dt)
        turn_roll = float(np.clip(self.yaw_rate * speed / G, -0.35, 0.35))
        roll_t = -0.9 * turn_roll + (0.02 * self.sway if self.kind == "biped" else 0.0) + self.inj_roll
        self.pitch += (pitch_t - self.pitch) * min(1.0, 6 * dt)
        self.roll += (roll_t - self.roll) * min(1.0, 8 * dt)
        # gait-coupled pelvis yaw: rotate towards the leg that is going forward
        yaw_gait = 0.0
        if self.kind == "biped" and len(self.legs) == 2:
            fx = []
            for ls in self.legs:
                if ls.stance: pc = ls.planted
                else: pc = self._swing_pose(ls)[0]
                fx.append(float((pc[:2] - self.pos) @ fwd))
            yaw_gait = float(np.clip((fx[0] - fx[1]) / max(self.reach, 1e-3), -1, 1)) * 0.20 * min(1.0, 0.4 + speed / 2.0) * (1 if speed > 0.2 else 0)
        # body response to an external push (spring-damper wobble, set by apply_push)
        wob = self.trunk_wobble
        R_p = rot("Z", self.heading + yaw_gait * 0.5) @ rot("Y", self.pitch + wob[1]) @ rot("X", self.roll + wob[0])
        # --- trunk root height ---
        q = sk.zeros()
        self.q = q
        self._pose_spine(q, cmd, dt, yaw_gait, speed, wob)
        # FK of the trunk with root at the origin to learn the hip offsets (orientation only)
        S0, E0, R0 = sk.fk(np.zeros(3), R_p, q)
        hipoff = {ls.idx: S0[ls.leg.chain[0]] for ls in self.legs}
        sway_off = left * (0.028 * self.reach * self.sway) if self.kind == "biped" else np.zeros(2)
        root_xy = self.pos + sway_off
        if cmd.pelvis is not None and "xy" in cmd.pelvis:
            root_xy = np.asarray(cmd.pelvis["xy"], float)
        # height caps from the stance legs
        z_cap = 1e9
        for ls in self.legs:
            a = tg[ls.idx][0]
            h0 = hipoff[ls.idx]
            if ls.stance:
                dxy = a[:2] - (root_xy + h0[:2])
                dd = float(np.hypot(*dxy))
                rr = 0.992 * ls.pl.reach
                if dd < rr:
                    zc = a[2] - h0[2] + math.sqrt(max(rr * rr - dd * dd, 0.0))
                    z_cap = min(z_cap, zc)
                else:
                    z_cap = min(z_cap, a[2] - h0[2] + 0.2 * rr)
        ground = float(np.mean([ls.planted[2] for ls in self.legs if ls.stance])) if stance_legs else float(np.mean([ls.planted[2] for ls in self.legs]))
        ratio = self.c.params.get("stand_ratio", 1.0)
        nominal = self.c.z0 * ratio * (1 - 0.55 * self.crouch)
        # leg compression in running stance
        comp = 0.0
        if stance_legs and self.mode == "go":
            uu = float(np.mean([ls.u for ls in stance_legs]))
            comp = (0.025 + 0.06 * min(2.0, self.Fr)) * self.reach * math.sin(math.pi * uu) * (1.0 if self.duty < 0.55 else 0.3)
        z_des = ground + nominal - comp - self.inj_dip
        if self.kind == "quadruped":
            z_des = ground + nominal - 0.6 * comp
        jp = self.jumpplan
        if jp is not None and jp["stage"] == "flight":
            tt = min(jp["tf"], jp["T"])
            z = jp["apex"] - 0.5 * G * (tt - jp["Tu"]) ** 2
            self.z_f = z
        elif not stance_legs and not ((self.nwb is not None or self.aids is not None) and self.speed < 2.4):
            # flight: ballistic from the last stance height, landing at the next touchdown
            self._flight_update(dt, ground, nominal)
            z = self.z
        else:
            if self.flight:
                self.flight = False
            self.z_f += (z_des - self.z_f) * min(1.0, 14 * dt)
            z = min(self.z_f, z_cap - 0.004)
            self.vz = (z - self.z) / dt
        if cmd.pelvis is not None and "z" in cmd.pelvis:
            z = float(cmd.pelvis["z"]); self.z_f = z
        self.z = z
        root = np.array([root_xy[0], root_xy[1], z])
        # --- whole-body centre of mass lean ---
        S, E, R = sk.fk(root, R_p, q)
        # ---- legs
        for ls in self.legs:
            a, Rf = tg[ls.idx]
            leg = ls.leg; ch = leg.chain
            par = sk.parent[ch[0]]
            hip_w = S[ch[0]]
            yaw_rel = 0.0
            if self.kind == "biped":
                fy = math.atan2(Rf[1, 0], Rf[0, 0]); py = math.atan2(R[par][1, 0], R[par][0, 0])
                yaw_rel = float(np.clip(wrap(fy - py), -0.5, 0.5)) * 0.6
            rows = leg_ik(sk, leg, ls.pl, R[par], hip_w, a, yaw_rel)
            for k, rw in zip(ch[:-1], rows):
                q[k] = rw
            # recompute world rotation up to the last flex bone to get the foot's relative rotation
            Rcur = R[par]
            for k in ch[:-1]:
                o = sk.axes_order[k]; d = {"X": q[k, 0], "Y": q[k, 1], "Z": q[k, 2]}
                Rcur = Rcur @ rot(o[0], d[o[0]]) @ rot(o[1], d[o[1]]) @ rot(o[2], d[o[2]])
            rel = Rcur.T @ Rf
            e = euler_from_R(rel, sk.axes_order[ch[-1]])
            for ax, nm in enumerate("XYZ"):
                e[ax] = clamp_lim(sk, ch[-1], nm, e[ax]) if nm in sk.bones[ch[-1]].lim else 0.0
            q[ch[-1]] = e
            if leg.toes is not None: q[leg.toes] = [0.0, float(getattr(ls, "toe_bend", 0.0)), 0.0]
        # ---- arms, neck, tail
        self._pose_limbs(q, cmd, dt, fwd, left, speed, root, R_p)
        S, E, R = sk.fk(root, R_p, q)
        # balance: move trunk pitch so that the whole-body centre of mass sits over the hips as in the plain body
        self._com_lean(S, E, R, dt)
        self.S, self.E, self.R = S, E, R
        self.root = root; self.R_p = R_p
        self.com = sk.com(S, E, R)

    def _jump_step(self, cmd, dt, fwd, left):
        jp = self.jumpplan; tr = self.terrain
        jp["t"] += dt
        self.v = self.v * 0.0
        if jp["stage"] == "crouch":
            if not jp["run_up"]: cmd.crouch = max(cmd.crouch, 0.55 * smooth(jp["t"] / 0.32))
            hd = math.atan2(*(jp["target"] - self.pos)[::-1])
            self.heading = wrap(self.heading + wrap(hd - self.heading) * min(1.0, 8 * dt))
            if jp["t"] > 0.40:
                self._launch(jp)
        elif jp["stage"] == "flight":
            T = jp["T"]; tau = min(1.0, jp["tf"] / T); jp["tf"] += dt
            self.pos = jp["p0"] + (jp["target"] - jp["p0"]) * min(1.0, jp["tf"] / T)
            if jp["tf"] >= T:
                jp["stage"] = "land"; jp["t"] = 0.0
        elif jp["stage"] == "land":
            cmd.crouch = max(cmd.crouch, 0.5 * (1 - smooth(jp["t"] / 0.45)))
            if jp["t"] > 0.5 or (jp["run_up"] and jp["t"] > 0.12):
                if jp["run_up"]: self.v = jp["v0"] * 0.9
                self.jumpplan = None

    def _launch(self, jp):
        tr = self.terrain
        p0 = self.pos.copy(); tgt = jp["target"]
        mx = tr.max_along(p0, tgt, 40, half_width=0.25)
        info = self._can_jump(p0, tgt, jp["clear"]) if jp.get("plan") is None else jp["plan"]
        z_to = info["z_to"]; zl = info["zl"]; T = info["T"]
        vz0 = (zl - z_to) / T + 0.5 * G * T
        apex = z_to + vz0 * vz0 / (2 * G); Tu = vz0 / G
        jp.update(stage="flight", tf=0.0, T=T, p0=p0, z_to=z_to, z_l=zl, apex=apex, Tu=Tu)
        self.flight = True; self.flight_jump = True
        for ls in self.legs:
            ls.stance = False; ls.lift = ls.planted.copy(); ls.lift_yaw = ls.yaw
            cth, sth = math.cos(self.heading), math.sin(self.heading)
            Rh = np.array([[cth, -sth], [sth, cth]])
            xy = tgt + Rh @ self.neutral[ls.idx]
            ls.target = np.array([xy[0], xy[1], float(tr.h(xy[0], xy[1]))]); ls.target_yaw = self.heading
            ls.swing_T = T; ls.swing_t = 0.0
            ls.apex = max(mx + 0.07, max(ls.lift[2], ls.target[2]) + 0.25 * self.reach)
            ls.jump_leg = True

    def _head_height_above_root(self):
        sk = self.sk
        S, E, R = sk.fk(np.zeros(3), np.eye(3), sk.zeros())
        top = float(E[:, 2].max())
        if self.kind == "quadruped": top = float(E[sk.idx["head"]][2]) + 0.1
        return top + self.c.z0

    def _pose_spine(self, q, cmd, dt, yaw_gait, speed, wob):
        """Distribute trunk bending on the spine joints: counter rotation, gallop flexion, balance lean."""
        sk, c = self.sk, self.c
        sp = c.spine
        if self.kind == "biped":
            # the column is a chain (3 lumbar + lower thoracic + chest): every joint takes a share, the upper ones more of the counter-rotation
            mob = sp[1:]; n = len(mob)
            share = np.ones(n) / n
            wy = np.arange(1, n + 1, dtype=float); wy /= wy.sum()
            pitch_total = 2 * (0.5 * self.crouch * 0.7 + 0.35 * self.lean_com + wob[1] * 0.5)
            for k, j in enumerate(mob):
                q[j, 1] = pitch_total * share[k]
                q[j, 2] = -yaw_gait * 1.2 * wy[k] + (0.4 + 0.6 * (k >= n - 2)) * cmd.torso_yaw * wy[k] * 1.0
                q[j, 0] = (-0.3 * self.roll + 0.5 * wob[0]) * 2 * share[k]
        elif self.kind == "quadruped":
            fl = c.params.get("spine_flex", 1.0)
            mob = sp[1:]; n = len(mob)
            # per-joint share: the column bends as a whole, so a cat (8 joints) and a horse (3 stiff ones) get different curves
            wgt = np.ones(n) / n
            flex = 0.0
            if self.mode == "go":
                # sagittal flexion rhythm: coupled to gait phase, only strong in bounding gaits
                amp = np.radians(14) * fl * smooth((self.Fr - 1.5) / 3.0)
                flex = amp * math.sin(2 * math.pi * (self.phase - 0.05))
                lat = np.radians(5) * fl * min(1.0, speed / 1.5) * math.sin(2 * math.pi * self.phase)
                for k, j in enumerate(mob):
                    q[j, 2] = lat * (1 - 1.6 * k / max(n - 1, 1))      # travelling lateral wave along the column
            flex += self._jump_spine()
            lim = {j: (sk.bones[j].lim["Y"][0], sk.bones[j].lim["Y"][1]) for j in mob}
            for k, j in enumerate(mob):
                lo, hi = np.radians(lim[j][0]), np.radians(lim[j][1])
                q[j, 1] = float(np.clip(flex * wgt[k] + wob[1] * 0.3 / n * 2, lo, hi))
                q[j, 0] += 0.5 * wob[0] / n * 2
        elif self.kind == "bird":
            q[sp[1], 1] = 0.0

    def _jump_profile(self):
        """Phase of a quadruped jump as (stage, tau) or None."""
        jp = self.jumpplan
        if jp is None or self.kind != "quadruped": return None
        if jp["stage"] == "flight": return "flight", min(1.0, jp["tf"] / jp["T"])
        if jp["stage"] == "crouch": return "crouch", min(1.0, jp["t"] / 0.4)
        return "land", min(1.0, jp["t"] / 0.5)

    def _jump_spine(self):
        """Total trunk flexion (rad, + = hunch) over a jump: coil, stretch at take-off, tuck at the apex, stretch to land, absorb."""
        pr = self._jump_profile()
        if pr is None: return 0.0
        st, tau = pr
        E = self.c.params.get("ext_rom", 0.3); F = self.c.params.get("flex_rom", 0.5)
        if st == "crouch": return 0.85 * F * smooth(tau)
        if st == "land": return 0.6 * F * (1 - smooth(tau))
        ext = -0.85 * E * (1 - smooth(tau / 0.3))                            # drive: the back extends with the hind legs
        tuck = 0.55 * F * smooth((tau - 0.25) / 0.2) * (1 - smooth((tau - 0.6) / 0.2))
        brace = -0.4 * E * smooth((tau - 0.75) / 0.25)
        return ext + tuck + brace

    def _jump_pitch(self):
        """Trunk pitch during the jump (negative = nose up): the cat rears up almost vertically, the horse barely rotates."""
        pr = self._jump_profile()
        if pr is None: return None
        st, tau = pr
        jp = self.jumpplan; pmax = self.c.params.get("jump_pitch", 0.6)
        if st == "crouch": return 0.15 * smooth(tau)
        if st == "land": return 0.35 * pmax * (1 - smooth(tau))
        pk = jp.get("pitch_k", 0.5) * pmax
        return pk * (-1.0 + 1.7 * smooth(tau))

    def _can_jump(self, p0, tgt, clear):
        """Ballistic jump that the legs can actually power. Two limits from the muscles: the vertical take-off speed (jump_gain * sqrt(g L))
        and the total take-off speed (1.5x that), to which a run-up adds horizontal speed. The lowest apex that fits both is used."""
        tr = self.terrain
        mx = tr.max_along(p0, tgt, 40, half_width=0.25)
        z_to = self.z; zl = float(tr.h(tgt[0], tgt[1])) + self.c.z0 * 0.96
        k = 0.30 if self.kind == "quadruped" else 0.45
        apex_min = max(mx + k * self.reach + 0.08 + (clear or 0.0), z_to + 0.12, zl + 0.12)
        D = float(np.linalg.norm(tgt - p0))
        gain = self.c.params.get("jump_gain", 1.0)
        cap = gain * math.sqrt(G * self.reach); vtot = 1.5 * cap
        run = 0.9 * float(np.linalg.norm(self.v))
        best = None
        for apex in np.linspace(apex_min, z_to + cap * cap / (2 * G), 40):
            if apex < apex_min - 1e-9: continue
            Tu = math.sqrt(2 * (apex - z_to) / G); Td = math.sqrt(2 * max(apex - zl, 1e-6) / G); T = Tu + Td
            vz0 = G * Tu; vh = D / T
            if vz0 <= cap and vh <= math.sqrt(max(vtot * vtot - vz0 * vz0, 0.0)) + run:
                best = (apex, T, vz0, vh); break
        if best is None:
            Tu = math.sqrt(2 * (apex_min - z_to) / G); Td = math.sqrt(2 * max(apex_min - zl, 1e-6) / G)
            return dict(mx=mx, z_to=z_to, zl=zl, apex=apex_min, T=Tu + Td, vz0=G * Tu, ok=False, cap=cap, vh=D / (Tu + Td))
        apex, T, vz0, vh = best
        return dict(mx=mx, z_to=z_to, zl=zl, apex=apex, T=T, vz0=vz0, ok=True, cap=cap, vh=vh)

    def _flight_update(self, dt, ground, nominal):
        if not self.flight:
            self.flight = True; self.flight_t = 0.0
            nxt = [((1 - ls.p) / max(self.freq, 1e-3)) for ls in self.legs if not ls.stance]
            self.flight_T = max(0.06, min(nxt)) if nxt else 0.1
            self.z_to = self.z
            # landing height: where the next foot will touch down
            tds = [ls for ls in self.legs if not ls.stance]
            tl = min(tds, key=lambda l: (1 - l.p)) if tds else None
            self.z_td = (tl.target[2] if tl is not None else ground) + nominal * 0.96
            self.flight_T0 = self.flight_T
        self.flight_t += dt
        T = max(self.flight_T, 1e-3)
        tau = min(1.0, self.flight_t / T)
        # recompute remaining time from the leg that lands next (closed loop)
        nxt = [((1 - ls.p) / max(self.freq, 1e-3)) for ls in self.legs if not ls.stance]
        if nxt:
            rem = max(min(nxt), 1e-3); tot = self.flight_t + rem
            tau = self.flight_t / tot; T = tot
        z = self.z_to + (self.z_td - self.z_to) * tau + 0.5 * G * T * T * tau * (1 - tau)
        self.z = z; self.vz = 0.0

    def _com_lean(self, S, E, R, dt):
        """Shift the trunk lean so the centre of mass stays where it would be for a plain body."""
        if self.kind != "biped": return
        sk = self.sk
        com = sk.com(S, E, R)
        hips = 0.5 * (S[self.c.legs[0].chain[0]] + S[self.c.legs[1].chain[0]])
        fwdv = np.array([math.cos(self.heading), math.sin(self.heading), 0.0])
        x_c = float((com - hips) @ fwdv)
        if self.rest_com_off is None:
            plain = type(self.c)  # reference captured on first call for the same body at rest
            S0, E0, R0 = sk.fk(np.array([0, 0, self.c.z0]), np.eye(3), sk.zeros())
            base_extra = sk.extra; sk.extra = []
            c0 = sk.com(S0, E0, R0); sk.extra = base_extra
            h0 = 0.5 * (S0[self.c.legs[0].chain[0]] + S0[self.c.legs[1].chain[0]])
            self.rest_com_off = float((c0 - h0)[0])
        err = x_c - (self.rest_com_off + 0.02)
        target = float(np.clip(-2.0 * err, -0.35, 0.45))
        self.lean_com += (target - self.lean_com) * min(1.0, 4 * dt)

    def _pose_limbs(self, q, cmd, dt, fwd, left, speed, root, R_p):
        sk, c = self.sk, self.c
        sw = min(1.0, speed / 1.0)
        S, E, R = sk.fk(root, R_p, q)
        # shoulder girdle: the clavicle lifts and swings forward with the arm (scapulo-humeral rhythm), so reach grows and the shoulder stays comfortable
        if self.kind == "biped":
            for arm in c.arms:
                if arm.clav is None: continue
                tgt = None if not cmd.arms else cmd.arms.get(arm.side)
                sg = 1.0 if arm.side == "L" else -1.0
                if tgt is not None:
                    d = np.asarray(tgt, float) - S[arm.chain[0]]
                    L_ = float(sum(sk.length[k] for k in arm.chain[:2])) + 1e-6
                    elev = float(np.clip((d[2] / L_ - 0.3) * 0.6, -0.1, 0.65)); prot = float(np.clip((d @ np.array([fwd[0], fwd[1], 0.0]) / L_ - 0.3) * 0.5, -0.1, 0.5))
                else:
                    elev = 0.02; prot = 0.0
                    arm_s = getattr(self, "_arm_sy", {}).get(arm.side, 0.0); prot = -0.35 * arm_s
                q[arm.clav] = [sg * elev, 0.0, -sg * prot]
            S, E, R = sk.fk(root, R_p, q)
        # arms swing against the legs of the same side
        if self.kind == "biped":
            for arm in c.arms:
                opp = [ls for ls in self.legs if ls.leg.side != arm.side][0]
                pc = opp.planted if opp.stance else self._swing_pose(opp)[0]
                ex = float((pc[:2] - self.pos) @ fwd) / max(self.reach, 1e-3)
                amp = (0.15 + 0.55 * min(1.0, speed / 3.0)) * c.params.get("arm_swing", 1.0)
                u, f, h = arm.chain
                tgt = None if not cmd.arms else cmd.arms.get(arm.side)
                if tgt is not None:
                    par = sk.parent[u]
                    pole = np.array([-fwd[0], -fwd[1], 0.0]) * 0.6 + np.array([left[0], left[1], 0.0]) * (1 if arm.side == "L" else -1) * 0.6 + np.array([0, 0, -0.3])
                    qu, qf, Eb, Rua = arm_ik(sk, arm, R[par], S[u], np.asarray(tgt, float), pole)
                    q[u] = qu; q[f] = qf; q[h] = 0.0
                else:
                    sy = -ex * amp * 1.6 - 1.8 * self.trunk_wobble[1]
                    self.__dict__.setdefault("_arm_sy", {})[arm.side] = sy
                    q[u] = [(0.07 + 1.6 * abs(self.trunk_wobble[1]) + 1.2 * abs(self.trunk_wobble[0])) * (1 if arm.side == "L" else -1), sy, 0.0]
                    q[f] = [0.0, -(0.12 + 0.25 * sw + 0.9 * max(0.0, min(1.0, (self.Fr - 0.5) / 1.5))) - 0.2 * max(0.0, sy), 0.0]
                    q[h] = 0.0
        # head: keep looking where the body goes; neck takes the share
        if c.neck:
            nk, hd = c.neck, c.head
            pitch_comp = -(self.pitch + float(np.sum(q[c.spine[1:], 1])) * (1.0 if self.kind == "biped" else 0.0))
            if self.kind == "quadruped":
                pitch_comp = -0.3 * self.pitch
            for k in nk: q[k, 1] = pitch_comp * 0.5 / len(nk) * 1.0
            q[hd, 1] = pitch_comp * 0.5 - 1.2 * self.trunk_wobble[1] * (1 if self.kind == 'biped' else 0.3)
            if self.kind == "quadruped":
                nod = c.params.get("neck_nod", 0.06); drop = c.params.get("neck_speed_drop", 0.1)
                go = self.mode == "go"
                stretch = drop * smooth(self.Fr / 3.0) if go else 0.0
                n = len(nk)
                for j, k in enumerate(nk):          # the nod travels up the neck: each joint lags the one below it
                    q[k, 1] += (nod * math.sin(2 * math.pi * self.phase - 0.6 * j) * sw if go else 0.0) / n * 1.6 + stretch / n
                q[hd, 1] += (nod * 0.5 * math.sin(2 * math.pi * self.phase - 0.6 * n) * sw if go else 0.0) - 0.6 * stretch
                if not go:                          # grazing/looking about at rest, slow independent drift of head and neck
                    q[hd, 1] += 0.05 * math.sin(0.7 * self.t); q[hd, 2] += 0.12 * math.sin(0.45 * self.t + 1.0)
                    for j, k in enumerate(nk): q[k, 2] += 0.05 * math.sin(0.45 * self.t + 1.0 + 0.3 * j)
        # tail: a damped chain whose type decides the behaviour (species table below)
        if c.tail:
            self._pose_tail(q, dt, sw)
        if self.kind == "bird":
            fold = getattr(self, "wing_fold", 1.0)
            for wi, chain in enumerate(c.wings):
                sg = 1.0 if wi == 0 else -1.0
                a_, b_, c_ = chain
                q[a_] = [sg * 0.25 * fold, 0.0, sg * 1.4 * fold]; q[b_] = [0.0, 0.0, sg * -2.55 * fold]; q[c_] = [0.0, 0.0, sg * 2.2 * fold]
            if self.mode == "go":
                bob = 0.35 * math.sin(2 * math.pi * self.phase + 1.0) * min(1.0, speed / 0.5)
                nk = c.neck
                q[nk[0], 1] += -0.6 * bob; q[nk[2], 1] += 0.6 * bob; q[nk[3], 1] += 0.5 * bob
        # reapply limits on every bone
        for i, b in enumerate(sk.bones):
            for ax, nm in enumerate("XYZ"):
                if nm in b.lim:
                    q[i, ax] = clamp_lim(sk, i, nm, q[i, ax])
                elif i != 0 and b.group not in ("leg", "foot"):
                    q[i, ax] = 0.0


    # ---- tails ----------------------------------------------------------------------------------
    TAILS = {
        # k, c: spring and damper of each joint | base: upward bend per joint | curl: extra bend per joint further out | gain: counter-swing
        # to the yaw rate | wag: (amplitude rad, Hz) scaled by mood | lift: raise with speed | swish: flicks per second | sway: gait coupling
        "balance": dict(k=10.0, c=2.0, base=0.30, curl=0.14, gain=0.95, wag=(0.16, 0.5), lift=0.25, swish=0.0, sway=0.10),   # cat: balance organ, tip curls
        "wag":     dict(k=14.0, c=3.0, base=0.45, curl=0.05, gain=0.25, wag=(0.60, 3.2), lift=0.45, swish=0.0, sway=0.12),   # dog: signal
        "wag_low": dict(k=16.0, c=4.0, base=-0.15, curl=0.0, gain=0.15, wag=(0.10, 1.2), lift=0.35, swish=0.0, sway=0.08),  # wolf: carried low
        "hair":    dict(k=5.0, c=1.0, base=0.22, curl=-0.10, gain=0.40, wag=(0.0, 0.0), lift=0.95, swish=0.28, sway=0.15),   # horse: heavy hair, flags at speed, flicks flies
        "curl":    dict(k=40.0, c=6.0, base=0.70, curl=0.40, gain=0.02, wag=(0.06, 1.0), lift=0.0, swish=0.0, sway=0.05),    # pig: tight spiral
        "none":    dict(k=30.0, c=6.0, base=0.0, curl=0.0, gain=0.0, wag=(0.0, 0.0), lift=0.0, swish=0.0, sway=0.0),
    }

    def _pose_tail(self, q, dt, sw):
        c = self.c; tp = self.TAILS[c.params.get("tail_type", "wag")]; n = len(c.tail)
        mood = getattr(self, "mood", 0.3)
        speed_lift = tp["lift"] * smooth(self.Fr / 2.5) if self.mode == "go" else 0.0
        lat_drive = -self.yaw_rate * tp["gain"] * 0.6
        # random flicks (horse swishing flies): an impulse into the lateral velocity of the whole chain
        if tp["swish"] > 0 and self.rng.random() < tp["swish"] * dt * (1.0 if self.mode == "stand" else 0.4):
            self.tail_v[:, 0] += self.rng.choice([-1.0, 1.0]) * 5.0
        amp, hz = tp["wag"]
        for i, b in enumerate(c.tail):
            u = (i + 1) / n
            base_up = tp["base"] / n * 2.0 * (1.0 if i == 0 else 0.5) + tp["curl"] * u + speed_lift * (0.6 if i == 0 else 0.25) - 0.3 * self.pitch * u
            lat_t = lat_drive * (0.5 + 0.5 * u)
            lat_t += amp * mood * math.sin(2 * math.pi * hz * self.t - 0.5 * i) * (0.6 + 0.4 * u) if hz > 0 else 0.0
            if self.mode == "go": lat_t += tp["sway"] * math.sin(2 * math.pi * self.phase + 0.7 * i) * sw
            ks = tp["k"] * (1.0 - 0.45 * u); cs = tp["c"]
            self.tail_v[i, 0] += (-ks * (self.tail_state[i, 0] - lat_t) - cs * self.tail_v[i, 0]) * dt
            self.tail_v[i, 1] += (-ks * (self.tail_state[i, 1] - base_up) - cs * self.tail_v[i, 1]) * dt
            self.tail_state[i] += self.tail_v[i] * dt
            q[b, 2] = float(np.clip(self.tail_state[i, 0], -1.0, 1.0)); q[b, 1] = float(np.clip(self.tail_state[i, 1], -1.2, 1.2))

    # ---- external events ----------------------------------------------------------------
    def apply_push(self, impulse_dir, strength, chest=True):
        """strength = change of the centre-of-mass velocity (impulse / mass, m/s). The body steps to recover
        because foot placement follows the predicted hip position, and the trunk whips like a spring."""
        d = np.array(impulse_dir, float); d = d / (np.linalg.norm(d[:2]) + 1e-9)
        self.v = self.v + d[:2] * strength
        fwd = np.array([math.cos(self.heading), math.sin(self.heading)]); left = np.array([-fwd[1], fwd[0]])
        xl, yl = float(d[:2] @ fwd), float(d[:2] @ left)
        k = 2.6 * strength * (1.0 if chest else 0.5)
        self.wob_v += np.array([-k * yl, k * xl, 0.6 * k * (yl if chest else 0.0)])
        self.hit_flash = 0.25

    def arm_usable(self, side):
        import hit as _hit; return _hit.arm_usable(self, side)

    def apply_hit(self, region, direction, strength):
        import hit as _hit; _hit.apply_hit(self, region, direction, strength)

    def start_jump(self, target_xy, clearance=None):
        """Crouch, take off from both feet, fly a ballistic arc over whatever lies between, land at target_xy."""
        tgt = np.asarray(target_xy, float)
        info = self._can_jump(self.pos.copy(), tgt, clearance)
        self.refused_jump = not info["ok"]
        if self.refused_jump: return False                       # this body cannot push off hard enough: the caller must go around
        run_up = float(np.linalg.norm(self.v)) > 1.5
        self.jumpplan = dict(stage="crouch", t=0.40 if run_up else 0.0, target=tgt, clear=clearance, v0=self.v.copy(), run_up=run_up,
                             pitch_k=float(np.clip(math.atan2(info["vz0"], max(info["vh"], 0.3)) / 1.2, 0.2, 1.0)))
        return True

    def snapshot(self):
        return dict(S=self.S.copy(), E=self.E.copy(), R=self.R.copy(), root=self.root.copy(), q=self.q.copy(), t=self.t, com=self.com.copy(),
                    feet=[(ls.planted.copy() if ls.stance else self._swing_pose(ls)[0], ls.stance) for ls in self.legs], heading=self.heading, speed=self.speed)
