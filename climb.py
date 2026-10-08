"""Climbing a rock face or a tree where only certain points can be held.

A climb is a sequence of reaches. The body keeps three points of contact (three of hand L, hand R, foot L, foot R stay on holds); the
fourth limb moves to a new hold that is inside its reach, free, usable by that limb, and moves the body towards the goal. The pelvis
settles where the supports allow: close to the wall, over the feet, within arm and leg reach of every hold it uses. If no hold is
reachable the climber shifts its weight and tries again; if nothing helps it reports that it is stuck (no scripted route)."""
from __future__ import annotations
import math
import numpy as np
from goals import minjerk, R_ypr, spine_split
from posing import Poser, foot_rot, ankle_for

class Hold:
    def __init__(self, p, kind="both", size=0.04):
        self.p = np.asarray(p, float); self.kind = kind; self.size = size      # kind: hand | foot | both

class Climber:
    def __init__(self, creature, holds, wall_normal, start_pelvis_xy, goal_z, t_reach=0.85, seed=0):
        self.c = creature; self.sk = creature.skel; self.poser = Poser(creature)
        self.holds = holds; self.n = np.asarray(wall_normal, float) / np.linalg.norm(wall_normal)       # points from the wall to the climber
        self.goal_z = goal_z; self.t_reach = t_reach
        self.rng = np.random.default_rng(seed)
        self.Larm = float(sum(self.sk.length[k] for k in creature.arms[0].chain[:2])) + 0.05
        self.Lleg = creature.legs[0].reach
        self.lat = np.array([-self.n[1], self.n[0], 0.0])        # horizontal along the wall
        self.yaw = math.atan2(-self.n[1], -self.n[0])             # facing the wall
        self.limbs = {}                                           # name -> hold index
        self.moving = None; self.t_move0 = 0.0; self.from_p = None; self.to_idx = None
        self.pelvis = np.append(np.asarray(start_pelvis_xy, float), creature.z0); self.stuck = False
        self.t = 0.0; self.dt = 1 / 60; self.log = []; self.done = False; self.relax = False; self.stuck_t = 0.0
        self.pending_idle = 0.0
        self._init_on_ground(start_pelvis_xy)

    # ---- geometry ---------------------------------------------------------------------------
    def _init_on_ground(self, xy):
        base = np.asarray(xy, float)
        self.ground_pt = {"fL": np.append(base + self.lat[:2] * 0.10, 0.0), "fR": np.append(base - self.lat[:2] * 0.10, 0.0)}
        self.foot_ground = {"fL": True, "fR": True}
        self.hold_of = {}
        self.on_ground = True
        self.pelvis = np.append(base, self.c.z0)

    def supports(self, exclude=None):
        """Positions of every limb that is planted right now (on a hold, or a foot still on the ground)."""
        out = {}
        for nm, i in self.hold_of.items():
            if nm != exclude: out[nm] = self.holds[i].p
        for nm in ("fL", "fR"):
            if self.foot_ground[nm] and nm != exclude and (self.moving is None or self.moving[0] != nm): out[nm] = self.ground_pt[nm]
        return out

    def _pelvis_target(self, sup):
        hands = [p for k, p in sup.items() if k[0] == "h"]; feet = [p for k, p in sup.items() if k[0] == "f"]
        pts = hands + feet
        wallpt = np.mean(pts, axis=0) if pts else self.pelvis
        xy = wallpt[:2] + self.n[:2] * 0.34
        if feet: xy = np.array([np.mean([f[0] for f in feet]), np.mean([f[1] for f in feet])]) * 0.5 + xy * 0.5
        if feet: z = np.mean([f[2] for f in feet]) + 0.62 * self.Lleg * 1.15 + 0.12
        else: z = max(h[2] for h in hands) - 0.5 - 0.85 * self.Larm       # hanging by the hands
        for h in hands: z = max(z, h[2] - 0.5 - 0.9 * self.Larm + 0.05)
        for f in feet: z = min(z, f[2] + 0.95 * self.Lleg + 0.05)
        return np.array([xy[0], xy[1], z])

    def _reachable(self, name, hold, pelvis):
        p = hold.p
        if name[0] == "h":
            if hold.kind == "foot" and not self.relax: return False        # only when desperate: a foot hold used as a poor handhold
            sh = pelvis + np.array([0, 0, 0.52]) + self.lat * (0.18 if name == "hL" else -0.18)
            return np.linalg.norm(p - sh) <= 0.95 * self.Larm + 0.05
        if hold.kind == "hand" and not self.relax: return False
        hip = pelvis + self.lat * (0.09 if name == "fL" else -0.09)
        return np.linalg.norm(p - hip) <= 0.92 * self.Lleg

    def _current(self, nm):
        if nm in self.hold_of: return self.holds[self.hold_of[nm]].p
        if nm[0] == "f" and self.foot_ground[nm]: return self.ground_pt[nm]
        return None

    def _choose_move(self):
        """Pick the limb to move and its next hold. Three points stay; the cost prefers balance, no overreach and real progress."""
        occupied = set(self.hold_of.values()); best = None
        if self.moving is not None: return None
        for nm in ("hL", "hR", "fL", "fR"):
            sup = self.supports(exclude=nm)
            n_grounded = sum(1 for k in sup if k[0] == "f" and self.foot_ground[k])
            if not (len(sup) >= 3 or (len(sup) == 2 and n_grounded == 2)): continue
            pel = self._pelvis_target(sup)
            cur = self._current(nm)
            hands_z = [p[2] for k, p in sup.items() if k[0] == "h"]
            for idx, h in enumerate(self.holds):
                if idx in occupied: continue
                if not self._reachable(nm, h, pel): continue
                gain = h.p[2] - (cur[2] if cur is not None else 0.6)
                if nm[0] == "h" and gain < 0.05: continue
                if nm[0] == "f" and gain < 0.02 and cur is not None and not self.foot_ground[nm]: continue
                if nm[0] == "f" and h.p[2] > pel[2] - 0.15: continue
                if nm[0] == "f" and hands_z and h.p[2] > max(hands_z) - 0.35: continue          # feet stay below the hands
                lat = abs(float((h.p - pel) @ self.lat))
                reach = np.linalg.norm(h.p - (pel + np.array([0, 0, 0.5]))) / self.Larm if nm[0] == "h" else np.linalg.norm(h.p - pel) / self.Lleg
                lowness = (cur[2] if cur is not None else 0.0)
                cost = 1.2 * lat + 0.8 * reach + 0.5 * lowness - 0.8 * gain
                if best is None or cost < best[0]: best = (cost, nm, idx)
        return best

    # ---- time stepping -----------------------------------------------------------------------
    def step(self, dt):
        self.t += dt; self.dt = dt
        if self.moving is None and self.pending_idle <= 0:
            ch = self._choose_move()
            if ch is not None: self._start(ch[1], ch[2])
            else:
                top = [p for p in self.supports().values() if p[2] >= self.goal_z - 0.3]
                if top: self.done = True
                else:
                    self.pending_idle = 0.3; self.stuck = True; self.stuck_t += 0.3
                    if self.stuck_t > 1.2: self.relax = True
        elif self.pending_idle > 0: self.pending_idle -= dt
        if self.moving is not None and self.t - self.t_move0 >= self.t_reach:
            nm, idx = self.moving
            self.hold_of[nm] = idx; self.moving = None; self.stuck = False; self.stuck_t = 0.0; self.relax = False
            if nm[0] == "f": self.foot_ground[nm] = False
        return self.pose()

    def _start(self, nm, idx):
        cur = self._current(nm)
        self.from_p = np.asarray(cur if cur is not None else self.pelvis + np.array([0, 0, -0.1]) + self.lat * (0.3 if nm == "hL" else -0.3), float)
        self.moving = (nm, idx); self.t_move0 = self.t
        if nm in self.hold_of: del self.hold_of[nm]
        self.log.append((self.t, nm, idx))

    # ---- pose -------------------------------------------------------------------------------
    def _contact(self, nm):
        if self.moving is not None and self.moving[0] == nm:
            u = minjerk((self.t - self.t_move0) / self.t_reach)
            p = self.from_p * (1 - u) + self.holds[self.moving[1]].p * u
            return p + self.n * (0.12 * math.sin(math.pi * u))
        cur = self._current(nm)
        if cur is not None: return cur
        return self.pelvis + np.array([0, 0, 0.05]) + self.lat * (0.28 if nm == "hL" else -0.28) + self.n * 0.1            # a free hand hangs by the side

    def pose(self):
        sup = self.supports()
        tgt = self._pelvis_target(sup) if sup else self.pelvis
        self.pelvis = self.pelvis + (tgt - self.pelvis) * (1 - math.exp(-4.0 * self.dt))
        R_p = R_ypr(self.yaw, 0.10, 0.0)
        feet = {}
        for nm, side in (("fL", "L"), ("fR", "R")):
            p = self._contact(nm)
            Rf = foot_rot(self.yaw, 0.30 if not (nm[0] == "f" and self.foot_ground[nm]) else 0.0)
            off = self.n * 0.06 + np.array([0, 0, 0.02]) if not self.foot_ground[nm] else np.zeros(3)
            feet[side] = (ankle_for(self.c, side, p + off, Rf), Rf)
        hands = {"L": self._contact("hL") + self.n * 0.02, "R": self._contact("hR") + self.n * 0.02}
        spq = spine_split(self.c, (0, 0.08, 0), (0, 0.06, 0))
        S, E, R, q = self.poser.pose(self.pelvis, R_p, spq, feet, hands)
        return S, E, R
