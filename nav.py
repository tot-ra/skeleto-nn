"""Navigation on the same terrain the planner walks on: A* over a coarse grid whose edge cost comes from the
height change (step up costs effort, too high is blocked) and the headroom (ducking costs, too low is blocked),
then a path follower that outputs velocity commands. Other agents are avoided with a social-force term."""
from __future__ import annotations
import heapq, math
import numpy as np
from planner import Cmd

class Navigator:
    def __init__(self, terrain, creature, res=0.12, radius=0.28, max_up=None, max_down=None, jump_ok=False, headroom_ok=0.55):
        self.t = terrain; self.res = res; self.radius = radius
        reach = creature.leg_len()
        self.max_up = max_up if max_up is not None else 0.5 * reach
        self.max_down = max_down if max_down is not None else 0.9 * reach
        self.body_h = creature.z0 + float(creature.skel.fk(np.zeros(3), np.eye(3), creature.skel.zeros())[1][:, 2].max()) if creature.kind == "biped" else creature.z0 * 1.4
        self.headroom_ok = headroom_ok
        self.jump_ok = jump_ok
        self.path = []
        self.i = 0

    def _cell(self, p):
        return (int(round((p[0] - self.t.x0) / self.res)), int(round((p[1] - self.t.y0) / self.res)))

    def _xy(self, c):
        return np.array([self.t.x0 + c[0] * self.res, self.t.y0 + c[1] * self.res])

    def plan(self, start, goal, avoid=()):
        t = self.t; r = self.res
        s = self._cell(start); g = self._cell(goal)
        nx = int((t.nx - 1) * t.res / r); ny = int((t.ny - 1) * t.res / r)
        # blocked cells: pits and headroom, inflated by the body radius
        def h(c): return float(t.h(*self._xy(c)))
        rad = int(math.ceil(self.radius / r))
        open_ = [(0.0, s)]; came = {s: None}; cost = {s: 0.0}
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        hc = {}
        def H(c):
            if c not in hc: hc[c] = h(c)
            return hc[c]
        def blocked(c):
            if not (0 <= c[0] < nx and 0 <= c[1] < ny): return True
            p = self._xy(c)
            if t.is_pit(*p): return True
            for k in range(0, 8):
                a = k * math.pi / 4
                q = p + self.radius * np.array([math.cos(a), math.sin(a)])
                if t.is_pit(*q): return True
                if H(c) - float(t.h(*q)) < -self.max_up * 1.02 and float(t.h(*q)) - H(c) > self.max_up * 1.2:
                    return True       # wall right next to the cell
            if t.ceiling(*p) - H(c) < self.headroom_ok * self.body_h: return True
            for a in avoid:
                if np.linalg.norm(p - a[:2]) < a[2] + self.radius: return True
            return False
        bl = {}
        def B(c):
            if c not in bl: bl[c] = blocked(c)
            return bl[c]
        best = None
        while open_:
            _, c = heapq.heappop(open_)
            if c == g or (np.linalg.norm(self._xy(c) - goal) < 0.25):
                best = c; break
            for d in nb:
                n = (c[0] + d[0], c[1] + d[1])
                if B(n): continue
                dh = H(n) - H(c)
                if dh > self.max_up or -dh > self.max_down: continue
                step = math.hypot(*d) * r
                head = t.ceiling(*self._xy(n)) - H(n)
                duck = 0.0
                if np.isfinite(head): duck = 4.0 * max(0.0, (self.body_h + 0.1 - head) / self.body_h)
                nc = cost[c] + step * (1 + 5.0 * max(0.0, dh) / max(self.max_up, 1e-3) + 1.5 * max(0.0, -dh) / self.max_down + duck)
                if n not in cost or nc < cost[n]:
                    cost[n] = nc; came[n] = c
                    hh = math.hypot(n[0] - g[0], n[1] - g[1]) * r
                    heapq.heappush(open_, (nc + hh, n))
        if best is None:
            self.path = [np.asarray(goal, float)]; self.i = 0; return self.path
        pts = []
        c = best
        while c is not None:
            pts.append(self._xy(c)); c = came[c]
        pts = pts[::-1]
        self.path = self._smooth(pts, B); self.path.append(np.asarray(goal, float)); self.i = 0
        return self.path

    def _smooth(self, pts, B):
        """String pulling: keep only the corners that the straight line cannot skip."""
        if len(pts) < 3: return pts
        out = [pts[0]]; i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                if self._clear(pts[i], pts[j], B): break
                j -= 1
            out.append(pts[j]); i = j
        return out

    def _clear(self, a, b, B):
        n = int(np.linalg.norm(b - a) / (self.res * 0.5)) + 1
        for k in range(1, n):
            p = a + (b - a) * k / n
            c = self._cell(p)
            if B(c): return False
            # no surprise height changes: the line should cost about the same as walking along it
            if abs(float(self.t.h(*p)) - float(self.t.h(*(a + (b - a) * (k - 1) / n)))) > 0.3 * self.max_up: return False
        return True

    def cmd(self, pos, speed, others=(), look=0.9, arrive=0.35):
        if not self.path: return Cmd()
        pos = np.asarray(pos[:2], float)
        while self.i < len(self.path) - 1 and np.linalg.norm(self.path[self.i] - pos) < 0.35:
            self.i += 1
        tgt = self.path[self.i]
        d = tgt - pos; dist = np.linalg.norm(d)
        final = self.i >= len(self.path) - 1
        if final and dist < 0.12: return Cmd(v=np.zeros(2))
        dirv = d / max(dist, 1e-6)
        v = dirv * speed
        if final: v *= float(np.clip((dist) / (arrive * 3), 0.0, 1.0)) ** 0.8 if dist < arrive * 3 else 1.0
        # slow down for height changes ahead
        ahead = pos + dirv * 0.7
        dh = abs(float(self.t.h(*ahead)) - float(self.t.h(*pos)))
        v *= 1.0 / (1.0 + 2.0 * dh)
        # social avoidance
        for o in others:
            rel = pos - o[:2]; dd = np.linalg.norm(rel)
            if 1e-6 < dd < 1.6:
                side = np.array([-dirv[1], dirv[0]])
                v += (rel / dd) * speed * 0.9 * (1.6 - dd) / 1.6 + side * 0.25 * speed * (1.6 - dd) / 1.6 * (1 if np.cross(dirv, -rel) >= 0 else -1) * 0
        n = np.linalg.norm(v)
        if n > speed: v = v / n * speed
        return Cmd(v=v)
