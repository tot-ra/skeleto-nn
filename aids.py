"""Walking aids and missing legs: crutches for a body whose one leg carries no weight, and the arm goals that go with them.

The walker already knows when its good leg is in stance or in swing. The crutch tips follow that rhythm: they are planted while the
good leg swings through (they carry the body), and they move ahead while the good foot is down. The hands hold the grips beside the
hips, so the arms are IK goals like any other and the pelvis rides between the planted points."""
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from planner import smooth

class Crutches:
    def __init__(self, w, length=0.95, reach=0.55, lateral=0.30):
        self.length, self.reach, self.lat = length, reach, lateral
        self.tips = {}; self.from_tip = {}; self.to_tip = {}; self.phase = "planted"; self.last_stance = True; self.t_move = 0.0
        self.good = next(ls for ls in w.legs if ls.leg.name != w.nwb)
        self.grips = {}; self.tops = {}
        self._place(w, init=True)

    def _fl(self, w):
        f = np.array([math.cos(w.heading), math.sin(w.heading)]); return f, np.array([-f[1], f[0]])

    def _target(self, w, side):
        f, l = self._fl(w); foot = self.good.planted[:2] if self.good.stance else self.good.target[:2]
        xy = foot + f * self.reach + l * (self.lat if side == "L" else -self.lat)
        return np.array([xy[0], xy[1], float(w.terrain.h(xy[0], xy[1]))])

    def _place(self, w, init=False):
        for sd in "LR":
            t = self._target(w, sd); self.tips[sd] = t.copy(); self.from_tip[sd] = t.copy(); self.to_tip[sd] = t.copy()

    def apply(self, w, cmd, dt):
        """Update the tips from the good leg's phase and return a command whose arms hold the grips."""
        st = self.good.stance
        if st and not self.last_stance:                      # good foot has just landed: lift the crutches and move them ahead
            self.phase = "moving"; self.t_move = 0.0
            for sd in "LR": self.from_tip[sd] = self.tips[sd].copy(); self.to_tip[sd] = self._target(w, sd)
        self.last_stance = st
        if self.phase == "moving":
            self.t_move += dt; T = max(0.25, 0.55 * (self.good.swing_T + 0.3)); u = smooth(self.t_move / T)
            for sd in "LR":
                lift = 0.10 * math.sin(math.pi * u); self.tips[sd] = self.from_tip[sd] * (1 - u) + self.to_tip[sd] * u + np.array([0, 0, lift])
            if self.t_move >= T: self.phase = "planted"
        if not st or self.phase == "planted":
            pass
        f, l = self._fl(w); pel = w.root if w.root is not None else np.zeros(3)
        arms = {}
        for sd in "LR":
            grip = np.array([w.pos[0], w.pos[1], pel[2] + 0.02]) + np.append(f * 0.06, 0.0) + np.append(l * (0.24 if sd == "L" else -0.24), 0.0)
            tip = self.tips[sd]; ax = grip - tip; ax = ax / max(np.linalg.norm(ax), 1e-6)
            self.grips[sd] = grip; self.tops[sd] = grip + ax * 0.22
            arms[sd] = grip
        return replace(cmd, arms=arms)

    def segments(self):
        """(tip, top) of each crutch for drawing."""
        return {sd: (self.tips[sd], self.tops[sd]) for sd in "LR" if sd in self.tops}

def use_crutches(w, nwb_leg):
    """Put the leg `nwb_leg` ('leg_L' ...) out of use and give the body two crutches."""
    w.nwb = nwb_leg; w.aids = Crutches(w); return w.aids
