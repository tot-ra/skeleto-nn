"""Kinematic cruising flight whose wingbeat comes from the style network (for demos and as the game's model).

The species sets a base wingbeat frequency (heavier birds beat slower, f ~ m^-0.28) and the network sets
``flap_rate`` and ``flap_amp`` relative to it and ``glide``, the share of each bout spent gliding with the
wings held out. Altitude follows the lift balance of the cost model: a style that flaps too little for
the load sinks, so a wrong style is visible as well as costly.
"""
from __future__ import annotations
import math
import numpy as np
from skeleton import rot
from .conditions import channels_from
from .schema import neutral_channels


class FlightActor:
    def __init__(self, creature, net, conditions=(), y=0.0, z=1.5, speed=2.2, bout=1.1, calibrate=True):
        self.c = creature; self.net = net; self.y = y; self.z = z; self.speed = speed; self.bout = bout
        self.ch = channels_from(list(conditions))
        m = float(creature.params.get("mass", 0.45)); self.mass = m
        self.f_beat = 3.2 * (m / 0.45) ** -0.28
        st = net.style("bird", m, float(creature.z0), True, 0.5, self.ch)
        if calibrate:
            base = net.style("bird", m, float(creature.z0), True, 0.5, neutral_channels())
            for k in ("flap_rate", "flap_amp", "glide", "fluff", "tremor"):
                neutral = {"flap_rate": 1.0, "flap_amp": 1.0}.get(k, 0.0)
                st[k] = neutral + st[k] - base[k]
        self.st = st
        self.phase = 0.0; self.t = 0.0; self.alt = 0.0; self.vz = 0.0

    def lift_balance(self):
        ch, st = self.ch, self.st
        need = (1 + 0.25 * ch["wet"]) * (1 + ch["load"]) * (1 + 0.35 * max(ch["wind_fwd"], 0.0))
        lift = (st["flap_rate"] * st["flap_amp"]) ** 2 * (1 - st["glide"]) + 0.9 * st["glide"] * max(-ch["wind_fwd"], 0.0)
        return lift - need

    def pose(self, t, ctx=None):
        dt = t - self.t; self.t = t
        st = self.st; g = min(max(st["glide"], 0.0), 0.8)
        local = t % self.bout
        flapping = local < (1 - g) * self.bout
        if flapping: self.phase += 2 * math.pi * self.f_beat * st["flap_rate"] * dt
        self.alt += 0.25 * self.lift_balance() * dt
        sk = self.c.skel; q = sk.zeros()
        A = 0.75 * st["flap_amp"] if flapping else 0.0
        ph = self.phase
        for wi, (a, b, cc) in enumerate(self.c.wings):
            sg = 1.0 if wi == 0 else -1.0
            el = 0.5 + 0.5 * math.cos(ph + 0.5); wr = 0.5 + 0.5 * math.cos(ph + 1.2)
            q[a] = [sg * (A * 0.95 * math.cos(ph) + 0.12), 0.0, sg * 0.18]
            q[b] = [0.0, 0.0, sg * (-(0.15 + 0.55 * el * A))]
            q[cc] = [0.0, 0.0, sg * (0.1 + 0.5 * wr * A)]
        for l in self.c.legs:
            th, sh, me, to = l.chain
            q[th] = [0, 0.7, 0]; q[sh] = [0, 1.2, 0]; q[me] = [0, -0.9, 0]; q[to] = [0, 0.9, 0]
        nk = self.c.neck
        q[nk[0], 1] = -0.5; q[nk[1], 1] = 0.2; q[nk[2], 1] = 0.8; q[nk[3], 1] = 0.5
        for kk, bn in enumerate(sk.bones):
            for ax, nm in enumerate("XYZ"):
                if nm in bn.lim:
                    lo, hi = bn.lim[nm]; q[kk, ax] = float(np.clip(q[kk, ax], math.radians(lo), math.radians(hi)))
        heave = 0.02 * math.cos(ph) * A
        root = np.array([self.speed * t, self.y, self.z + self.alt + heave])
        return sk.fk(root, rot("Y", -math.radians(8)), q)
