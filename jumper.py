"""Knowing what you can jump: a self-model that is learned by practising, a decision that weighs the pain of a failed jump, and a
controller that runs up, jumps - or stops at the edge and is afraid.

* The body's true jump range depends on the run-up speed (planner.jump_range). The agent does not know it exactly: it keeps a belief
  over its own capacity, learns it from practice jumps over marked distances on flat ground (a short failure only costs a stumble), and
  starts from a prior that depends on character (a child over-estimates, an old person under-estimates).
* At a gap it asks: with which run-up speed is the jump believed possible, and how likely is it to fail? A failure costs pain that grows
  with the depth of what is below; not jumping costs a detour. It jumps when the expected pain is lower than the detour.
* If it decides not to jump it brakes before the edge and shows fear. If it decided to jump but the real body cannot make it (an
  over-confident belief), the take-off is refused at the edge: it skids to a stop and flails its arms - and its belief is corrected."""
from __future__ import annotations
import math
import numpy as np
from planner import Cmd, smooth

def _phi(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

class SelfModel:
    def __init__(self, w, prior_mean=1.0, prior_sd=0.3, rng=None, v_ref_frac=0.8):
        self.w = w; self.rng = rng or np.random.default_rng(0)
        self.v_sprint = float(w.c.params.get("v_sprint", 6.2)); self.v_ref = v_ref_frac * self.v_sprint
        self.speeds = [float(v) for v in np.arange(0.0, self.v_ref - 1e-6, 0.5)] + [float(self.v_ref)]
        self.table = {v: w.jump_range(np.array([1.0, 0.0]), v) for v in self.speeds}                 # the true mechanics
        self.C_true = max(self.table[self.v_ref], 0.1)
        self.g = {v: max(self.table[v], 0.05) / self.C_true for v in self.speeds}                    # how the range grows with the run-up (known)
        self.grid = np.linspace(0.2 * self.C_true, 2.2 * self.C_true, 400)
        pm, ps = prior_mean * self.C_true, prior_sd * self.C_true
        self.post = np.exp(-0.5 * ((self.grid - pm) / ps) ** 2); self.post /= self.post.sum()
        self.history = []

    # ---- belief ---------------------------------------------------------------------------------
    def mean_sd(self):
        m = float((self.grid * self.post).sum()); sd = float(math.sqrt(((self.grid - m) ** 2 * self.post).sum())); return m, sd

    def p_can(self, d_ref):
        """Believed probability that the true capacity is at least d_ref (in reference-run-up metres)."""
        return float(self.post[self.grid >= d_ref].sum())

    def observe(self, d, success):
        """Update from one attempt at reference distance d: success means capacity >= d (up to execution noise)."""
        s = 0.05 * d + 0.04
        like = np.array([_phi((c - d) / s) for c in self.grid]); like = like if success else 1.0 - like
        self.post = self.post * like + 1e-12; self.post /= self.post.sum()

    def practise(self, n):
        """Jump over distances marked on flat ground, each a little beyond what is currently believed to be safe; a failure is a stumble."""
        for k in range(n):
            m, sd = self.mean_sd(); d = float(np.clip(m + 0.4 * sd, 0.3, 2.2 * self.C_true))
            success = d <= self.C_true * (1.0 + float(self.rng.normal(0, 0.03)))
            self.observe(d, success); m2, sd2 = self.mean_sd(); self.history.append((k + 1, d, success, m2, sd2))
        return self.history

    # ---- decisions --------------------------------------------------------------------------------
    def plan(self, D_jump, depth, detour=0.25):
        """Choose the lowest run-up speed for a jump of D_jump metres over something `depth` metres deep, or decide not to jump.
        Returns (speed or None, believed probability of success at the best speed, expected pain)."""
        pain_fail = min(3.0, 1.2 * depth + 0.15)
        last = None
        for v in self.speeds:
            p = self.p_can(1.05 * D_jump / self.g[v]); ep = (1 - p) * pain_fail; last = (p, ep)
            if ep < detour: return v, p, ep
        return None, last[0], last[1]

    def correct_after_balk(self, D_jump, v):
        """The take-off was refused: the real capacity at this speed is below D_jump."""
        key = min(self.speeds, key=lambda s: abs(s - v)); self.observe(D_jump / self.g[key], False)

# ---- the controller -----------------------------------------------------------------------------------
class GapJumper:
    """Cmd function for an actor that has to cross a gap along +x. Phases: walk, run-up, jump | brake, fear, turn away."""
    def __init__(self, model: SelfModel, y, x_edge, gap, depth, label="", acc=None, detour=0.25):
        self.m = model; self.y = y; self.x_edge = x_edge; self.gap = gap; self.depth = depth
        self.D = gap + 0.9; self.x_take = x_edge - 0.35; self.x_land = x_edge + gap + 0.55
        self.v_run, self.p, self.exp_pain = model.plan(self.D, depth, detour)
        self.decision = "jump" if self.v_run is not None else "refuse"
        self.state = "approach"; self.t_state = 0.0; self.outcome = ""; self.label = label; self.acc = acc

    def _set(self, st, t): self.state = st; self.t_state = t

    def __call__(self, t, w, ctx):
        acc = self.acc or ((3.0 * math.sqrt(w.reach) + 1.0) * w.c.params.get("acc_scale", 1.0))
        x = float(w.pos[0]); f = np.array([math.cos(w.heading), math.sin(w.heading), 0.0])
        sh = lambda sd: w.S[w.sk.idx["uarm_" + sd]] if w.S is not None else np.zeros(3)
        if self.state == "approach":
            if self.decision == "jump":
                d_run = (self.v_run ** 2) / (2 * acc) + 0.6                       # distance needed to reach the run-up speed
                x_go = self.x_take - d_run
                if x < x_go - 0.05: return Cmd(v=np.array([1.2, 0.0]), heading=0.0)
                self._set("run", t); return Cmd()
            if x < self.x_edge - 1.9: return Cmd(v=np.array([1.2, 0.0]), heading=0.0)      # refusing: walk up and stop short of the edge
            if x < self.x_edge - 0.9: return Cmd(v=np.array([0.6, 0.0]), heading=0.0)
            self._set("fear", t); return Cmd()
        if self.state == "run":
            if t - self.t_state < 0.35: return Cmd(heading=0.0)                   # a breath, then go
            if x < self.x_take: return Cmd(v=np.array([max(self.v_run, 0.8), 0.0]), heading=0.0)
            if w.start_jump((self.x_land, self.y)): self._set("flight", t); self.outcome = "jumped"; return Cmd()
            self.outcome = "balked"; self.m.correct_after_balk(self.D, self.v_run); self._set("balk", t); return Cmd()
        if self.state == "flight":
            if w.jumpplan is None: self._set("done", t)
            return Cmd(v=np.zeros(2), heading=0.0)
        if self.state == "done":
            return Cmd(v=np.zeros(2), heading=0.0)
        if self.state == "balk":
            u = t - self.t_state; w.v = w.v * 0.93                                    # skid to a stop
            wind = math.sin(2 * math.pi * 2.0 * u)
            arms = {"L": sh("L") + np.array([0, 0.25, 0.45 + 0.25 * wind]) + f * 0.15, "R": sh("R") + np.array([0, -0.25, 0.45 - 0.25 * wind]) + f * 0.15}
            if u > 2.4: self._set("turn", t)
            return Cmd(v=np.zeros(2), heading=0.0, arms=arms, crouch=0.15, pelvis={"pitch": -0.10})
        if self.state == "fear":
            u = t - self.t_state
            look = np.array([self.x_edge + self.gap * 0.5, self.y, -self.depth * 0.6])
            tremble = 0.03 * math.sin(2 * math.pi * 7.0 * u)
            arms = {"L": sh("L") + np.array([0.20, 0.07, -0.30 + tremble]), "R": sh("R") + np.array([0.20, -0.07, -0.30 - tremble])}      # hands drawn up to the chest
            v = np.array([-0.45 if 0.8 < u < 1.5 else 0.0, 0.0])                      # a step back
            if u > 3.2: self._set("turn", t)
            return Cmd(v=v, heading=0.0, arms=arms, look=look, crouch=0.10, pelvis={"pitch": -0.12})
        if self.state == "turn":
            u = t - self.t_state
            return Cmd(v=np.array([-1.0 if u > 0.8 else 0.0, 0.0]), heading=math.pi)
        return Cmd()

class Playful:
    """Surplus energy: a body with more energy than it needs (energy above 1.3) spends it in hops while it walks, like a kid or a young goat; weak muscles make the hops low."""
    def __init__(self, w, rng, rate=0.5, hop=0.45):
        self.rng = rng; self.e = float(rng.uniform(0, 0.5)); self.rate = rate; self.hop = hop
        self.energy = float(w.c.params.get("energy", 1.0)); self.vigor = float(w.c.params.get("muscle", 1.0))

    def step(self, w, dt):
        if self.energy <= 1.3 or w.jumpplan is not None or w.down is not None or getattr(w, "stamina", 1.0) < 0.4: return
        self.e += (self.energy - 1.0) * self.rate * dt
        if self.e >= 1.0 and w.speed > 0.5:
            f = np.array([math.cos(w.heading), math.sin(w.heading)])
            if w.start_jump(w.pos + f * (self.hop * max(0.5, self.vigor))):
                if w.jumpplan is not None: w.jumpplan["t"] = max(w.jumpplan["t"], 0.28)
                self.e = float(self.rng.uniform(0.0, 0.3))

class ObstacleRunner:
    """Run along +x over rough ground and decide for each thing ahead what to do, with the body's own limits.
    Low things (a log) are stepped over by the planner; a pit is jumped with the run-up the self-model asks for or the way is abandoned;
    an obstacle that is too high to jump but low enough to put the hands on (up to about shoulder height) is vaulted: the hands go to the
    top, the body passes close over it; anything higher is gone round."""
    def __init__(self, model: SelfModel, y, v_run=4.0, detour_y=3.5, label=""):
        self.m = model; self.y = y; self.v = v_run; self.detour_y = detour_y; self.label = label
        self.action = None; self.log = []; self.handled = set(); self.vault = None; self.detour = None; self.t_flight = 0.0; self.stop_t = None

    def _scan(self, w, look=7.0):
        tr = w.terrain; x0 = float(w.pos[0]); g0 = float(tr.h(x0, self.y))
        prev = g0; i = 0
        for d in np.arange(0.3, look, 0.1):
            x = x0 + d; h = float(tr.h(x, self.y)); pit = bool(tr.is_pit(x, self.y))
            if pit:
                e = d
                while e < look + 6 and tr.is_pit(x0 + e, self.y): e += 0.1
                return dict(kind="pit", d=d, width=e - d, x_start=x0 + d, depth=1.5)
            if h - g0 > 0.28:
                e = d
                while e < look + 3 and float(tr.h(x0 + e, self.y)) - g0 > 0.2: e += 0.1
                return dict(kind="obstacle", d=d, height=float(tr.h(x0 + d + 0.05, self.y)) - g0, thick=e - d, x_start=x0 + d, x_end=x0 + e)
        return None

    def __call__(self, t, w, ctx):
        f = np.array([math.cos(w.heading), math.sin(w.heading), 0.0])
        sh = lambda sd: w.S[w.sk.idx["uarm_" + sd]] if w.S is not None else np.zeros(3)
        if w.jumpplan is not None:
            jp = w.jumpplan; tau = jp.get("tf", 0.0) / max(jp.get("T", 1.0), 1e-3) if jp["stage"] == "flight" else 0.0
            if self.vault is not None and jp["stage"] in ("flight", "crouch"):
                xo = self.vault["x_mid"]; top = self.vault["top"]; hold = 0.12 < tau < 0.72
                arms = {"L": np.array([xo, self.y + 0.22, top + 0.02]), "R": np.array([xo, self.y - 0.22, top + 0.02])} if hold else None
                return Cmd(v=np.zeros(2), heading=0.0, arms=arms)
            return Cmd(v=np.zeros(2), heading=0.0)
        self.vault = None
        if self.stop_t is not None:                                              # gave up in front of something it cannot cross: afraid, then turns away
            u = t - self.stop_t; tremble = 0.03 * math.sin(2 * math.pi * 7.0 * u)
            if u < 3.0:
                arms = {"L": sh("L") + np.array([0.20, 0.07, -0.30 + tremble]), "R": sh("R") + np.array([0.20, -0.07, -0.30 - tremble])}
                return Cmd(v=np.array([-0.45 if 0.8 < u < 1.5 else 0.0, 0.0]), heading=0.0, arms=arms, crouch=0.10, pelvis={"pitch": -0.12})
            return Cmd(v=np.array([-1.0 if u > 3.8 else 0.0, 0.0]), heading=math.pi)
        if self.detour is not None:
            tx, ty = self.detour
            if abs(w.pos[1] - self.y) > 0.0 or True:
                v = np.array([tx - w.pos[0], ty - w.pos[1]]); n = np.linalg.norm(v)
                if n < 0.6:
                    if ty != self.y: self.detour = (tx + 3.0, self.y)
                    else: self.detour = None
                return Cmd(v=v / max(n, 1e-6) * self.v, heading=None)
        ev = self._scan(w)
        if ev is not None and round(ev["x_start"] * 2) / 2 not in self.handled:
            take = 0.55 + 0.17 * self.v                                            # where to leave the ground
            if ev["kind"] == "pit":
                D = ev["width"] + 1.15; v_need, p, ep = self.m.plan(D, ev["depth"])
                if v_need is None:
                    if ev["d"] > 1.6: return Cmd(v=np.array([min(self.v, 1.4), 0.0]), heading=0.0)                 # walk up to the edge, then stop
                    self.handled.add(round(ev["x_start"] * 2) / 2); self.log.append((t, "pit", ev["width"], "afraid")); self.stop_t = t; return Cmd(v=np.zeros(2), heading=0.0)
                if ev["d"] <= 0.60 and w.speed >= min(v_need, self.v) * 0.9:
                    tgt = (ev["x_start"] + ev["width"] + 0.55, self.y)
                    ok = w.start_jump(tgt); self.handled.add(round(ev["x_start"] * 2) / 2); self.log.append((t, "pit", ev["width"], "jumped" if ok else "balked")); return Cmd(v=np.zeros(2), heading=0.0)
            else:
                h = ev["height"]
                if h <= 0.34: return Cmd(v=np.array([self.v, 0.0]), heading=0.0)             # step over
                if h > 1.15:
                    self.handled.add(round(ev["x_start"] * 2) / 2); self.log.append((t, "wall", h, "round")); self.detour = (ev["x_start"] - 0.5, self.y + self.detour_y); return Cmd(v=np.array([self.v * 0.5, 0.0]), heading=0.0)
                if ev["d"] <= take + (0.15 if h > 0.7 else 0.0):
                    vault = h > 0.7
                    tgt = (ev["x_end"] + 0.7, self.y)
                    ok = w.start_jump(tgt, clearance=(-0.28 if vault else 0.0))
                    if ok and vault: self.vault = dict(x_mid=0.5 * (ev["x_start"] + ev["x_end"]), top=float(w.terrain.h(ev["x_start"] + 0.05, self.y)))
                    self.handled.add(round(ev["x_start"] * 2) / 2); self.log.append((t, "vault" if vault else "hop", h, "ok" if ok else "balked"))
                    if not ok: self.detour = (ev["x_start"] - 0.5, self.y + self.detour_y); return Cmd(v=np.array([self.v * 0.4, 0.0]), heading=0.0)        # cannot do it: go round
                    return Cmd(v=np.zeros(2), heading=0.0)
        return Cmd(v=np.array([self.v, 0.0]), heading=0.0)

class Starter:
    """From a squat to a run. How it is done depends on how urgent it is and how strong the legs are:
      urgency low   : stand up first, then walk and speed up
      urgency high  : push out of the squat with the trunk pitched forward and the head down, the height coming up as the speed does
      urgency full  : a sprinter's start - hands on the ground, a low drive, then the hands leave the ground."""
    def __init__(self, urgency, v_target, t_go=1.2, hands_down=False):
        self.u = urgency; self.v_t = v_target; self.t_go = t_go; self.hands_down = hands_down

    def __call__(self, t, w, ctx):
        f = np.array([math.cos(w.heading), math.sin(w.heading), 0.0]); l = np.array([-f[1], f[0], 0.0])
        sh = lambda sd: w.S[w.sk.idx["uarm_" + sd]] if w.S is not None else np.zeros(3)
        if t < self.t_go:
            arms = None
            if self.hands_down and w.S is not None:
                base = np.array([w.pos[0], w.pos[1], 0.02])
                arms = {"L": base + f * 0.35 + l * 0.20, "R": base + f * 0.35 - l * 0.20}
            return Cmd(v=np.zeros(2), heading=0.0, crouch=1.0, arms=arms, pelvis={"pitch": 0.25 if self.hands_down else 0.0})
        u = t - self.t_go; sp = float(w.speed)
        muscle = float(w.c.params.get("muscle", 1.0))
        if self.u < 0.4:                                                          # stand, then go
            rise = min(1.0, u / 1.0)
            v = self.v_t * smooth((u - 0.9) / 2.5) if u > 0.9 else 0.0
            return Cmd(v=np.array([v, 0.0]), heading=0.0, crouch=1.0 - rise)
        w.acc_boost = 1.0 + 0.9 * self.u * min(1.6, muscle)                          # a harder push off the ground
        v = self.v_t if u > 0.15 else 0.4 * self.v_t
        frac = float(np.clip(sp / max(self.v_t, 1e-3), 0.0, 1.0))
        crouch = max(0.0, 1.0 - 1.6 * frac) if frac < 0.65 else 0.0                 # the body comes up as the speed does
        lean = 0.5 * self.u * (1.0 - frac)                                           # trunk pitched forward, head down
        arms = None
        if self.hands_down and frac < 0.18 and w.S is not None:
            base = np.array([w.pos[0], w.pos[1], 0.02]); arms = {"L": base + f * 0.35 + l * 0.20, "R": base + f * 0.35 - l * 0.20}
        return Cmd(v=np.array([v, 0.0]), heading=0.0, crouch=crouch, arms=arms, pelvis={"pitch": lean}, look=None)

class FenceJumper:
    """A horse (or any quadruped) approaching a fence along +x: it asks its own body whether it can make the height at the speed it has, and either
    takes off at a distance that grows with the height, or refuses: brakes, swerves aside and stops (a refusal)."""
    def __init__(self, y, x_fence, height, thick=0.25, v_approach=6.0, label=""):
        self.y = y; self.xf = x_fence; self.h = height; self.thick = thick; self.v = v_approach; self.label = label
        self.state = "approach"; self.t0 = 0.0; self.outcome = ""

    def __call__(self, t, w, ctx):
        x = float(w.pos[0]); d = self.xf - x
        if self.state == "approach":
            take = 0.9 + 1.1 * self.h * (w.reach / 1.0) ** 0.5 * 0.9              # take off earlier for a higher fence
            if d <= take + 3.0 and self.outcome == "":
                tgt = (self.xf + self.thick + take * 0.9, self.y)
                info = w._can_jump(w.pos.copy(), np.asarray(tgt), None, run_speed=w.speed)
                if d <= take:
                    if w.start_jump(tgt): self.state = "flight"; self.outcome = "jumped"; return Cmd(v=np.zeros(2), heading=0.0)
                    self.state = "refuse"; self.t0 = t; self.outcome = "refused"; return Cmd()
            return Cmd(v=np.array([self.v, 0.0]), heading=0.0)
        if self.state == "flight":
            if w.jumpplan is None: self.state = "after"
            return Cmd(v=np.zeros(2), heading=0.0)
        if self.state == "after":
            return Cmd(v=np.array([3.0, 0.0]), heading=0.0)
        if self.state == "refuse":                                                  # brake and swerve aside
            u = t - self.t0
            return Cmd(v=np.array([max(0.0, 2.5 - 2.5 * u), 1.8 if u < 1.5 else 0.0]), heading=None if u < 1.5 else 1.2)
        return Cmd()
