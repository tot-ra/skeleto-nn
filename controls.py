"""Joystick or key input to body commands, with the inertia of a body.

The stick says where to go and how fast. A body cannot do that at once: it moves along where it faces, it needs distance to brake, and it
has to turn before it can go the other way. The rules, all from the same few numbers (acceleration, braking, turning rate, friction):
  * speed follows the stick magnitude (a walk up to v_walk, a run up to v_run);
  * the allowed speed falls with the angle between where the body faces and where the stick points: full speed within 25 degrees, none beyond
    about 110 degrees (it has to stop first, then turn on the spot, then accelerate);
  * a turn at speed is limited by friction (lateral acceleration <= mu g): the faster, the wider the curve, so a sharp turn first slows the body;
  * braking is harder than speeding up, the trunk leans back while it brakes and forward while it accelerates, and it banks into a curve."""
from __future__ import annotations
import math
import numpy as np
from planner import Cmd, wrap, smooth

class Joystick:
    def __init__(self, stick, v_walk=1.5, v_run=5.0, mu=0.7, naive=False):
        self.stick = stick; self.v_walk = v_walk; self.v_run = v_run; self.mu = mu; self.naive = naive
        self.v_prev = np.zeros(2); self.psi_d = 0.0; self.history = []

    def __call__(self, t, w, ctx):
        ang, mag = self.stick(t); mag = float(np.clip(mag, 0, 1))
        self.psi_d = ang if mag > 0.05 else self.psi_d
        vmax = float(w.c.params.get("v_max") or 99.0)
        v_target = min(vmax, self.v_walk + max(0.0, mag - 0.5) * 2.0 * (self.v_run - self.v_walk)) * (mag / 0.5 if mag < 0.5 else 1.0) if mag > 0.05 else 0.0
        if mag <= 0.5: v_target = min(vmax, self.v_walk * mag / 0.5)
        if self.naive:                                                              # the body goes where the stick points, at once
            v = v_target * np.array([math.cos(ang), math.sin(ang)]) if mag > 0.05 else np.zeros(2)
            self.history.append((t, ang, mag, w.speed)); return Cmd(v=v, heading=None)
        e = wrap(self.psi_d - w.heading); ae = abs(e)
        f = 1.0 if ae < math.radians(25) else max(0.0, math.cos((ae - math.radians(25)) / math.radians(85) * math.pi / 2)) if ae < math.radians(110) else 0.0
        speed = float(w.speed)
        # friction limit on the turn: v * omega <= mu g
        omega_max = self.mu * 9.81 / max(speed, 1.0)
        v_cmd = v_target * f
        # when the body is faster than the corner allows, the command slows it first
        if ae > math.radians(25) and speed * min(abs(e) * 3.0, 3.0) > self.mu * 9.81 * 1.0: v_cmd = min(v_cmd, 0.5 * speed)
        fwd = np.array([math.cos(w.heading), math.sin(w.heading)])
        # heading: toward the stick, no faster than friction and the body's own turning allow
        w.brake_gain = 1.8
        a_long = float((w.v - self.v_prev) @ fwd) / (1 / 60.0); self.v_prev = w.v.copy()
        lean = 0.05 + 0.20 * min(1.0, speed / 6.0) - 0.035 * float(np.clip(-a_long, 0, 6.0)) + 0.02 * float(np.clip(a_long, 0, 4.0))
        self.history.append((t, ang, mag, speed))
        return Cmd(v=fwd * v_cmd, heading=self.psi_d if mag > 0.05 else None, pelvis={"pitch": lean})
