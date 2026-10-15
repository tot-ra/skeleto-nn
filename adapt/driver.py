"""Drive the planner with the style network: situation in, adapted motion out.

    drv = StyleDriver(walker, StyleNet("results/style_net.json"), env=lambda t, w: (["ice"], Exposure()))
    cmd = drv(cmd, dt)            # every planner step, before walker.step(cmd)

Each step the driver gathers channels (the conditions under the creature + its body state), asks the
network for a style (at ``rate`` Hz, smoothed), and applies it: gait knobs go to ``walker.style``, which the
planner reads (stride, duty, stance width, crouch, trunk lean, foot roll, clearance, head, shoulders,
tail, shiver); arm goals (held out, hugged, pressing a wound) go into ``Cmd.arms`` and are solved by the
planner's arm IK; speed and limp go into the command and the leg duty. The physiology then advances
with the style actually used, so hugging when cold or pressing a wound changes the future channels.

Calibration: what is applied is ``neutral + net(situation) - net(no situation)`` for this body and
speed, so in a neutral situation the planner's motion is exactly unchanged.
"""
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from .schema import STYLE, KN, neutral_channels, neutral_style, knob_active
from .conditions import channels_from
from .physiology import BodyState, Exposure
from .net import StyleNet

G = 9.81
ENV_KEYS = ("friction", "sink", "air_temp", "wind_fwd", "wind_lat", "dark", "smoke", "insulation", "load", "dizzy")


class BodyPoints:
    """Map a body-frame point (x front, y left, z height fraction) to the nearest bone of any skeleton, so
    a wound or a point to protect follows the body as it moves."""
    def __init__(self, creature):
        sk = creature.skel
        S, E, R = sk.fk(np.zeros(3), np.eye(3), sk.zeros())
        mid = 0.5 * (S + E)
        z0, z1 = float(min(S[:, 2].min(), E[:, 2].min())), float(max(S[:, 2].max(), E[:, 2].max()))
        self.height = max(z1 - z0, 1e-3); self.z0 = z0
        self.half_w = max(float(np.abs(mid[:, 1]).max()), 0.05)
        self.half_l = max(float(np.abs(mid[:, 0]).max()), 0.05)
        self.S, self.R, self.mid = S, R, mid
        self.groups = [b.group for b in sk.bones]

    def bind(self, x, y, z):
        """(bone index, offset in the bone frame) for a body-frame point."""
        p = np.array([0.6 * x * self.half_l + 0.0, y * self.half_w, self.z0 + z * self.height])
        d = np.linalg.norm(self.mid - p, axis=1)
        b = int(np.argmin(d))
        surf = p.copy(); surf[0] += 0.04 * self.height * np.sign(x if abs(x) > 0.1 else 1.0)   # on the skin, not inside the bone
        return b, self.R[b].T @ (surf - self.S[b])

    @staticmethod
    def world(walker, bound):
        b, off = bound
        return walker.S[b] + walker.R[b] @ off


class StyleDriver:
    def __init__(self, walker, net: StyleNet, env=None, state: BodyState | None = None, rate=10.0, smooth_t=0.35,
                 airborne=False, conditions=None, calibrate=True, physio_speed=1.0):
        self.w = walker; self.net = net; self.env = env; self.rate = rate; self.smooth_t = smooth_t
        c = walker.c
        self.kind = "biped" if c.kind == "biped" else ("bird" if c.kind == "bird" else "quadruped")
        self.mass = float(c.skel.total_mass()) if callable(getattr(c.skel, "total_mass", None)) else float(c.params.get("mass", 70.0))
        self.leg = float(c.z0)
        self.state = state or BodyState(mass=self.mass)
        self.airborne = airborne
        self.static = list(conditions or []); self.physio_speed = physio_speed
        self.points = BodyPoints(c)
        self.style = neutral_style(); self.raw = neutral_style(); self.ch = neutral_channels()
        self._t_next = 0.0; self._cal = {}; self.calibrate = calibrate
        self.t = 0.0; self._wound_bind = None; self._wound_key = None
        self.trace = []
        walker.style = dict(self.style); walker.style_driver = self

    # ---- channels ------------------------------------------------------------------------------
    def channels(self):
        conds, exp = (self.env(self.t, self.w) if self.env else ([], Exposure()))
        env = channels_from(self.static + list(conds))
        ch = dict(env); ch.update({k: max(env.get(k, 0.0), v) if k.startswith("pain") or k in ("wet", "fatigue", "core_cold") else v
                                   for k, v in self.state.channels().items()})
        # pain from blows taken by the planner's hit model (hit.py) joins the same channels
        pain = getattr(self.w, "pain", None)
        if pain:
            for src, dst in (("leg_L", "pain_leg_L"), ("leg_R", "pain_leg_R"), ("arm_L", "pain_arm_L"), ("arm_R", "pain_arm_R"),
                             ("torso", "pain_torso"), ("head", "pain_head"), ("groin", "pain_torso")):
                ch[dst] = max(ch.get(dst, 0.0), float(pain.get(src, 0.0)))
        return ch, env, exp

    def _neutral_for(self, fr):
        key = round(fr, 2)
        if key not in self._cal:
            self._cal[key] = self.net.style(self.kind, self.mass, self.leg, self.airborne, fr, neutral_channels())
        return self._cal[key]

    def query(self, fr, ch):
        st = self.net.style(self.kind, self.mass, self.leg, self.airborne, fr, ch)
        if not self.calibrate: return st
        base = self._neutral_for(fr)
        out = {}
        for k in STYLE:
            v = k.neutral + st[k.name] - base[k.name]
            out[k.name] = min(k.hi, max(k.lo, v)) if knob_active(k, self.kind, self.airborne) else k.neutral
        return out

    # ---- per step ------------------------------------------------------------------------------
    def __call__(self, cmd, dt):
        w = self.w; self.t += dt
        v_cmd = np.asarray(cmd.v, float); sp = float(np.linalg.norm(v_cmd))
        fr = sp * sp / (G * self.leg)
        ch, env, exp = self.channels(); self.ch = ch
        if self.t >= self._t_next:
            self._t_next = self.t + 1.0 / self.rate
            self.raw = self.query(min(fr, 3.0), ch)
        a = min(1.0, dt / self.smooth_t)
        for k, v in self.raw.items(): self.style[k] += (v - self.style[k]) * a
        st = self.style
        w.style = dict(st)
        # speed: the share of the command taken, caution, and what the body can still hold
        strength = (1 - 0.65 * ch["blood_lost"]) * (1 - 0.45 * ch["fatigue"]) * (1 - 0.35 * ch["core_cold"])
        fr_cap = 3.0 * strength ** 2 * (1 - 0.75 * ch["sink"]) ** 2 * (1 - 0.4 * ch["load"])
        v_cap = math.sqrt(max(fr_cap, 1e-4) * G * self.leg)
        v = v_cmd * st["speed_frac"] * (1 - 0.35 * st["care"])
        if np.linalg.norm(v) > v_cap: v = v / np.linalg.norm(v) * v_cap
        # limp: less time on the sore side (the planner's per-side duty)
        w.leg_duty["L"] = 1.0 - 0.35 * max(st["limp"], 0.0)
        w.leg_duty["R"] = 1.0 - 0.35 * max(-st["limp"], 0.0)
        arms = self._arms(cmd, ch, st) if self.kind == "biped" and w.S is not None else cmd.arms
        crouch = max(cmd.crouch, st["crouch"])
        out = replace(cmd, v=v, arms=arms, crouch=crouch)
        # physiology with the style in use
        vn = float(np.linalg.norm(v)) / math.sqrt(G * self.leg)
        power = (0.9 * vn * vn * (1 + ch["load"]) + 0.9 * st["crouch"] ** 2 * (0.4 + vn) + 0.7 * (st["width"] - 1) ** 2
                 + 0.25 * st["arm_out"] + 1.6 * ch["sink"] * vn)
        press = max(st["press_L"], st["press_R"]) if self.kind == "biped" else 0.0
        self.state.step(dt * self.physio_speed, {k: env[k] for k in env}, exp, st, self.kind, self.airborne, power, 0.9 * press)
        if int(self.t * 20) != int((self.t - dt) * 20):
            self.trace.append(dict(t=self.t, ch={k: round(float(ch[k]), 3) for k in ("friction", "wet", "core_cold", "bleed", "blood_lost", "fatigue", "pain_leg_L")},
                                   st={k: round(float(v), 3) for k, v in st.items()}))
        return out

    def at(self, t):
        """The traced channels and style nearest to time t (for captions drawn after the simulation)."""
        if not self.trace: return None
        return min(self.trace, key=lambda r: abs(r["t"] - t))

    def _arms(self, cmd, ch, st):
        w = self.w; c = w.c; sk = w.sk
        arms = dict(cmd.arms or {})
        fwd = np.array([math.cos(w.heading), math.sin(w.heading), 0.0]); left = np.array([-fwd[1], fwd[0], 0.0])
        key = (round(ch["wound_x"], 2), round(ch["wound_y"], 2), round(ch["wound_z"], 2))
        if key != self._wound_key:
            self._wound_key = key; self._wound_bind = self.points.bind(*key)
        wound = BodyPoints.world(w, self._wound_bind)
        for arm in c.arms:
            side = arm.side; sg = 1.0 if side == "L" else -1.0
            sh = w.S[arm.chain[0]]; L = float(sum(sk.length[k] for k in arm.chain[:2]))
            hang = sh + np.array([0, 0, -0.95 * L]) + fwd * 0.05 * L
            base = arms.get(side) if arms.get(side) is not None else hang
            press, hug, out_ = st["press_" + side], st["hug_" + side], st["arm_out"]
            # a hugged arm crosses to the other side of the chest; an arm held out reaches sideways and a little forward
            chest = w.S[c.spine[-1]] if c.spine else sh
            hug_t = chest + fwd * 0.18 * L - left * sg * 0.22 * L + np.array([0, 0, -0.18 * L])
            out_t = sh + left * sg * 0.8 * L + fwd * 0.2 * L + np.array([0, 0, -0.35 * L])
            wsum = press + hug + 0.8 * out_
            if wsum < 0.04: continue
            wt = min(1.0, wsum)
            tgt = (press * wound + hug * hug_t + 0.8 * out_ * out_t) / wsum
            arms[side] = base * (1 - wt) + tgt * wt
        return arms or None
