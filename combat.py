"""A fight as a set of reactions, with no scripted duel.

A Strike is a swinging stick aimed at a region of the opponent's body (hand goals + IK on the attacker). The defender notices it
after a reaction delay and picks a reaction by simulating each candidate against the same geometry that will decide the hit:
duck, side step, step back, hop back, hop up, block - or none (flinch and take it). At the moment of impact the hit is resolved from
the real positions of the tip and the body: a miss if the body is no longer there, a block if the forearm is on the tip, a hit otherwise
(and then hit.apply_hit does pain, stagger or knock-down). Nothing here is specific to the number of fighters or to a body part."""
from __future__ import annotations
import math
import numpy as np
from planner import Cmd, smooth, wrap
import skills as SK
import hit as H
import injury as INJ

STICK = 0.85
REACT_COST = dict(duck=0.8, side=1.0, back=1.2, hop_back=1.8, hop_up=1.6, block=2.5, none=9.0)
EFFORT = dict(duck=0.05, side=0.08, back=0.10, hop_back=0.20, hop_up=0.15, block=0.10, none=0.0)       # in units of weighted pain
REACT_TIME = dict(duck=0.12, side=0.14, back=0.16, hop_back=0.22, hop_up=0.22, block=0.12)

def seg_dist(p, a, b):
    ab = b - a; u = float(np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0.0, 1.0)); return float(np.linalg.norm(p - (a + u * ab)))

def expected_pain(region, strength):
    """Weighted pain of a blow (the same weights as the injury score: the head matters three times as much as an arm)."""
    grp = {"head": "head", "torso": "torso", "groin": "torso"}.get(region, "arms" if region.startswith("arm") else "legs")
    return strength * H._spec(region)["pain"] * INJ.WEIGHT[grp]

def hit_test(tip, segs, travel=None):
    """Smallest clearance (negative = touching) between the stick tip, and the path it keeps sweeping along, and the region's capsules."""
    pts = [tip] if travel is None else [tip + travel * s for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return min(seg_dist(p, a, b) - r for p in pts for a, b, r in segs)

def travel_of(st):
    """The swing does not stop at the aim point: a chop continues down, a slash continues across, an arrow keeps flying."""
    if st.projectile: return st.bdir * 0.30
    if st.kind == "thrust": f, l = H._fl(st.atk); return np.append(f, 0.0) * 0.25
    if st.kind == "chop": return np.array([0.0, 0.0, -0.40])
    f, l = H._fl(st.atk); return np.append(l, 0.0) * 0.40

WEAPONS = {
    "stick": dict(len=0.85, speed=1.00, dmg=1.0, r=0.00),
    "axe":   dict(len=0.90, speed=0.72, dmg=1.7, r=0.07),       # slower, heavier head, wider hit
    "bow":   dict(len=0.70, speed=1.00, dmg=1.1, r=0.02, v=30.0),
    "sword": dict(len=1.00, speed=0.90, dmg=1.5, r=0.03),       # light, fast, cuts (chop or slash)
    "spear": dict(len=2.00, speed=0.85, dmg=1.4, r=0.03),       # long reach, a straight thrust
    "fist":  dict(len=0.12, speed=1.30, dmg=0.7, r=0.04),       # short and fast, a straight punch
}
THRUST = ("spear", "fist")

class Strike:
    def __init__(self, atk, vic, region, kind, t0, strength, side="R", weapon="stick"):
        self.atk, self.vic, self.region, self.kind, self.t0, self.strength, self.side = atk, vic, region, kind, t0, strength, side
        self.weapon = weapon; self.wp = WEAPONS[weapon]; self.scale = self.wp["speed"]
        W, S = SK.STRIKE["windup"] / self.scale, SK.STRIKE["strike"] / self.scale
        self.t_launch = t0 + W; self.t_imp = t0 + W + S; self.t_end = t0 + (SK.STRIKE["windup"] + SK.STRIKE["strike"] + SK.STRIKE["hold"] + SK.STRIKE["recover"]) / self.scale
        self.aim = None; self.resolved = False; self.outcome = None
        self.react = None; self.react_t = None; self.decided = False
        self.hand = None; self.bdir = None; self.projectile = False

    def aim_point(self):
        return H.region_point(self.vic, self.region)

    def update(self, t):
        w = self.atk
        if t < self.t_launch or self.aim is None: self.aim = self.aim_point()
        tt = self.t0 + (t - self.t0) * self.scale
        self.hand, ph, p = SK.strike_hand(w, tt, self.t0, self.aim, self.side, self.kind if self.kind in ("chop", "thrust") else "slash")
        self.bdir = SK.blade_dir(w, self.hand, ph, p, self.aim, self.side)
        self.phase, self.p = ph, p
        return self.hand, ph, p

    def arms(self, t):
        """Hand goals of the attacker and the body twist for this moment, or None while on guard."""
        hand, ph, p = self.update(t)
        if ph == "guard": return None, 0.0, ph
        ty = -0.45 * (SK.minjerk(p) if ph == "windup" else 1.0 if ph in ("strike", "hold") else 1 - SK.minjerk(p)) * (1 if ph != "strike" else -1.4)
        return {"R": hand}, ty, ph

    def tips(self):
        L = self.wp["len"]
        return [self.hand + self.bdir * L * s for s in (0.55, 0.7, 0.85, 1.0)]

class ArrowStrike(Strike):
    """An arrow: drawn for 0.6 s, released with the aim fixed, flies at 30 m/s on a gentle arc; the defender can only react to the aim."""
    def __init__(self, atk, vic, region, t0, strength, side="L"):
        super().__init__(atk, vic, region, "arrow", t0, strength, side, "bow")
        self.projectile = True; self.t_draw = 0.6
        self.t_launch = t0 + self.t_draw + 0.12
        D = float(np.linalg.norm(vic.pos - atk.pos)); self.T_f = max(0.12, D / self.wp["v"])
        self.t_imp = self.t_launch + self.T_f; self.t_end = self.t_imp + 0.5
        self.origin = None; self.pos_tail = None

    def update(self, t):
        w = self.atk
        if t < self.t_launch or self.aim is None:
            self.aim = self.aim_point(); self.origin = w.S[w.sk.idx["uarm_L"]] + np.array([math.cos(w.heading), math.sin(w.heading), 0.0]) * 0.5
        u = float(np.clip((t - self.t_launch) / self.T_f, 0.0, 1.0)) if t >= self.t_launch else 0.0
        d = self.aim - self.origin
        hump = 0.5 * G_ * self.T_f * self.T_f * u * (1 - u)                   # the aim point is where the arc ends
        tip = self.origin + d * u + np.array([0, 0, hump]) + (d / max(np.linalg.norm(d), 1e-6)) * 0.0
        dirn = d / max(np.linalg.norm(d), 1e-6)
        if t >= self.t_launch and u >= 1.0: tip = self.aim.copy()
        self.bdir = dirn; self.hand = tip - dirn * self.wp["len"]
        self.phase = "draw" if t < self.t_launch else "fly" if u < 1.0 else "stuck"; self.p = u
        return self.hand, self.phase, u

    def arms(self, t):
        self.update(t)
        w = self.atk; f = np.array([math.cos(w.heading), math.sin(w.heading), 0.0])
        d = self.aim - w.S[w.sk.idx["uarm_L"]]; d = d / max(np.linalg.norm(d), 1e-6)
        sh_l = w.S[w.sk.idx["uarm_L"]]; sh_r = w.S[w.sk.idx["uarm_R"]]
        draw = SK.minjerk((t - self.t0) / self.t_draw) if t < self.t_launch else max(0.0, 1.0 - (t - self.t_launch) / 0.25)
        L = float(sum(w.sk.length[k] for k in w.c.arms[0].chain[:2]))
        left = sh_l + d * 0.95 * L
        right = (sh_r + d * (0.95 * L - 0.55 * L * draw)) + np.array([0, 0, 0.05])
        return {"L": left, "R": right}, 0.35, self.phase if t < self.t_end else "guard"

    def tips(self):
        return [self.hand + self.bdir * self.wp["len"] * s for s in (0.55, 0.8, 1.0)]

G_ = 9.81

# ---- the defender's decision --------------------------------------------------------------------------
def decide(strike, t_now, forced=None):
    """Choose the cheapest reaction that is predicted to avoid the blow in the time that is left."""
    v = strike.vic; f, l = H._fl(v)
    ta = strike.t_imp - t_now
    segs = [sg for r in H.HIT_REGIONS for sg in H.region_segments(v, r)]          # moving must not expose another part of the body
    atk_xy = strike.atk.pos
    away = v.pos - atk_xy; away = away / (np.linalg.norm(away) + 1e-9)
    side_l = np.append(np.array([-away[1], away[0]]), 0.0)
    shift = lambda off: [(a + off, b + off, r) for a, b, r in segs]
    tv = travel_of(strike)
    clear = lambda segs_: hit_test(strike.aim, segs_, tv)
    drop = {"head": 0.45, "torso": 0.30}.get(strike.region, 0.0)
    cands = {}
    t_ok = lambda k: ta >= REACT_TIME[k]
    disp = lambda dmax, tmove: dmax * min(1.0, max(0.0, ta - 0.05) / tmove)
    for name, off in (("side_L", side_l * disp(0.60, 0.30)), ("side_R", -side_l * disp(0.60, 0.30)), ("back", np.append(away, 0.0) * max(0.0, disp(0.80, 0.34) - 0.15))):
        k = name.split("_")[0]
        if t_ok(k): cands[name] = (k, clear(shift(off)))
    if t_ok("duck"): cands["duck"] = ("duck", clear(shift(np.array([0, 0, -drop * min(1.0, ta / 0.30)]))))
    if t_ok("hop_back"): cands["hop_back"] = ("hop_back", clear(shift(np.append(away, 0.2) * min(1.0, max(0.0, ta - 0.10) / 0.45) * np.array([0.9, 0.9, 1.0]))))
    if t_ok("hop_up") and strike.region.startswith("leg"): cands["hop_up"] = ("hop_up", clear(shift(np.array([0, 0, 0.30 * min(1.0, max(0.0, ta - 0.10) / 0.30)]))))
    # Choose by expected pain: effort of the move plus the weighted pain of the blow if the move fails. Strong blows to the head are worth
    # a big move, a weak jab to the arm is taken.
    p_hit = expected_pain(strike.region, strike.strength)
    opts = {"none": p_hit}
    for name, (kind, cl) in cands.items():
        opts[name] = EFFORT[kind] + (0.0 if cl > 0.10 else p_hit)
    vf = strike.vf
    arm_free = v.pain["arm_L"] < 0.3 or v.pain["arm_R"] < 0.3
    if vf.shield and v.pain["arm_L"] < 0.3 and t_ok("block") and not strike.region.startswith("leg"):
        opts["shield"] = 0.06 + expected_pain("arm_L", strike.strength * 0.1)           # a shield takes almost all of it
    elif t_ok("block") and not strike.region.startswith("leg") and arm_free and not strike.projectile:
        opts["block"] = EFFORT["block"] + expected_pain("arm_R", strike.strength * 0.35)     # the forearm takes a third and hurts
    if vf.leg_block and strike.region.startswith("leg") and ta >= 0.16:
        opts["block_leg"] = 0.10 + expected_pain(strike.region, strike.strength * 0.4)      # lift the knee: the shin takes it
    if forced and forced in opts: return forced
    return min(opts, key=lambda k: (opts[k], 0 if k == "none" else 1))

def start_reaction(strike, name, t):
    v = strike.vic; f, l = H._fl(v)
    away = v.pos - strike.atk.pos; away = away / (np.linalg.norm(away) + 1e-9)
    acc = 3.0 * math.sqrt(v.reach) + 1.0
    if name.startswith("side"):
        sg = 1.0 if name.endswith("L") else -1.0
        d = np.array([-away[1], away[0]]) * sg
        v.v = d * math.sqrt(2 * acc * 0.62)
    elif name == "back":
        v.v = away * math.sqrt(2 * acc * 0.85)
    elif name == "hop_back":
        v.start_jump(v.pos + away * 0.95); 
        if v.jumpplan is not None: v.jumpplan["t"] = 0.30
    elif name == "block_leg":
        side = strike.region[-1]; ls = next(l for l in v.legs if l.leg.side == side)
        f, l = H._fl(v); hip = v.S[ls.leg.chain[0]]
        tgt = np.array([hip[0] + f[0] * 0.40, hip[1] + f[1] * 0.40, hip[2] - 0.42])
        v.leg_ovrs.append(dict(name=ls.leg.name, point=tgt, until=v.t + 0.65))
    elif name == "hop_up":
        mx = 0.0; need = (v.z + 0.28) - (mx + 0.62 * v.reach + 0.08)
        v.start_jump(v.pos + 0.05 * away, clearance=need)
        if v.jumpplan is not None: v.jumpplan["t"] = 0.30

class Fighter:
    """Policy of one body: how it closes in, whom it hits where, and how fast it notices."""
    def __init__(self, idx, opp, rng, attacks=True, react=True, delay=(0.15, 0.35), strength=(0.8, 2.0), regions=None, cooldown=(1.4, 2.6), dist=1.30, forced=None, first=1.2, weapon="stick", shield=False, leg_block=False):
        self.idx, self.opp, self.rng = idx, opp, rng
        self.attacks, self.react, self.delay, self.strength, self.cool, self.dist, self.forced = attacks, react, delay, strength, cooldown, dist, forced
        self.regions = regions or {"head": 2, "torso": 3, "arm_L": 1, "arm_R": 1, "leg_L": 1, "leg_R": 1}
        self.next_attack = first; self.strike = None; self.cur_react = None; self.react_until = 0.0; self.weapon_dropped = None
        self.block_side = "L"; self.weapon = weapon; self.shield = shield; self.leg_block = leg_block

class Arena:
    def __init__(self, fighters):
        self.f = fighters; self.strikes = []; self.log = []; self.t_last = -1.0; self.cmds = {}; self.hist = {}; self.arrow_hist = {}; self.shield_hist = {}

    def tick(self, t, ctx):
        if t == self.t_last: return
        self.t_last = t; W = ctx.walkers
        if any(w.S is None for w in W):
            for fi in self.f: self.cmds[fi.idx] = Cmd()
            return
        for fi in self.f:
            w = W[fi.idx]; o = W[fi.opp]
            if w.down is not None or o.down is not None: fi.next_attack = max(fi.next_attack, t + 1.0)
        # --- new strikes
        for fi in self.f:
            w = W[fi.idx]; o = W[fi.opp]
            if fi.attacks and fi.strike is None and t >= fi.next_attack and w.down is None and o.down is None and w.arm_usable("R"):
                if np.linalg.norm(o.pos - w.pos) < fi.dist + 0.35:
                    regs = list(fi.regions); wts = np.array([fi.regions[r] for r in regs], float)
                    region = str(fi.rng.choice(regs, p=wts / wts.sum()))
                    kind = "chop" if (region in ("head", "arm_L", "arm_R") and fi.rng.random() < 0.6) else "slash"
                    if region.startswith("leg"): kind = "slash"
                    if fi.weapon in THRUST: kind = "thrust"
                    wp = WEAPONS[fi.weapon]
                    if fi.weapon == "bow": st = ArrowStrike(w, o, region, t, float(fi.rng.uniform(*fi.strength)) * wp["dmg"])
                    else: st = Strike(w, o, region, kind, t, float(fi.rng.uniform(*fi.strength)) * wp["dmg"], weapon=fi.weapon)
                    st.vf = next(x for x in self.f if x.idx == fi.opp)
                    st.delay = float(fi.rng.uniform(*W_delay(self, fi.opp)))
                    st.label = f"{region} {fi.weapon if fi.weapon != 'stick' else kind}"
                    fi.strike = st; self.strikes.append(st); st.fi = fi
        # --- update strikes and reactions
        for st in self.strikes:
            if st.resolved:
                if t > st.t_end and st.fi.strike is st: st.fi.strike = None; st.fi.next_attack = t + float(st.fi.rng.uniform(*st.fi.cool))
                continue
            st.update(t)
            vf = next(x for x in self.f if x.idx == W.index(st.vic))
            if vf.react and not st.decided and t >= st.t0 + st.delay and st.vic.down is None and st.vic.jumpplan is None:
                name = decide_wait(st, t, vf.forced)
                if name is not None:
                    st.decided = True; st.react = name
                    start_reaction(st, name, t)
                    vf.cur_react = (name, t); vf.react_until = st.t_imp + 0.45
                    if name == "shield": vf.block_side = "L"
                    if name == "block": vf.block_side = "L" if np.linalg.norm(st.vic.S[st.vic.sk.idx["uarm_L"]] - st.aim) < np.linalg.norm(st.vic.S[st.vic.sk.idx["uarm_R"]] - st.aim) else "R"
            if t >= st.t_imp: self.resolve(st, t, ctx)
        # --- commands
        for fi in self.f:
            w = W[fi.idx]; o = W[fi.opp]
            rel = o.pos - w.pos; dist = float(np.linalg.norm(rel)); head = math.atan2(rel[1], rel[0])
            v = np.zeros(2); arms = None; ty = 0.0; crouch = 0.0
            if getattr(fi, "walk", None) is not None and t >= fi.walk[0]: v = np.asarray(fi.walk[1], float); head = None
            elif dist > fi.dist + 0.05 and w.down is None and o.down is None: v = rel / dist * 1.4
            elif dist < fi.dist - 0.35 and w.down is None: v = -rel / dist * 0.9
            st = fi.strike
            if st is not None and (not st.resolved or t <= st.t_end):
                arms, ty, ph = st.arms(t)
                if arms is not None:
                    if ph == "strike" and st.region.startswith("leg"): crouch = 0.5
                    if ph == "strike": v = v + rel / max(dist, 1e-6) * 0.9
                    if st.projectile: self.arrow_hist[(fi.idx, round(t * 60))] = (st.hand + st.bdir * st.wp["len"] * 0.5, st.bdir, st.phase)
                    else: self.hist[(fi.idx, round(t * 60))] = (st.hand, st.bdir)
            if arms is None:
                g = SK.strike_hand(w, t, 1e9, o.com, "R")[0]; arms = {"R": g}
                self.hist[(fi.idx, round(t * 60))] = (g, SK.blade_dir(w, g, "guard", 0, o.com))
                if fi.shield:
                    sp_, nrm_ = SK.shield_pose(w, o.com, "L"); arms["L"] = sp_
            if fi.shield: self.shield_hist[(fi.idx, round(t * 60))] = None
            if fi.cur_react is not None and t < fi.react_until:
                name, t_r = fi.cur_react
                if name == "duck": crouch = 1.0
                if name in ("block", "shield"):
                    tip = None
                    for s2 in self.strikes:
                        if s2.vic is w and not s2.resolved and s2.hand is not None: tip = s2.hand + s2.bdir * s2.wp["len"] * 0.8
                    if tip is not None:
                        pos, _ = SK.shield_pose(w, tip, fi.block_side, dist=0.50 if name == "shield" else 0.52)
                        arms = dict(arms); arms[fi.block_side] = pos
            if fi.weapon_dropped is not None: arms = None if False else arms
            if fi.weapon_dropped is None and w.pain["arm_R"] > 0.5:
                fi.weapon_dropped = (t, np.array([w.pos[0] + 0.3, w.pos[1] - 0.2, 0.03]))
            if fi.weapon_dropped is not None:
                self.hist[(fi.idx, round(t * 60))] = (fi.weapon_dropped[1], np.array([0.9, 0.3, 0.02]))
            if fi.shield and arms is not None and "L" in arms:
                nrm = (o.com - arms["L"]); nrm = nrm / max(np.linalg.norm(nrm), 1e-6)
                self.shield_hist[(fi.idx, round(t * 60))] = (arms["L"] + nrm * 0.04, nrm)
            self.cmds[fi.idx] = Cmd(v=v, heading=head, arms=arms, torso_yaw=ty, crouch=crouch)

    def resolve(self, st, t, ctx):
        st.resolved = True
        v = st.vic; a = st.atk
        tips = st.tips(); rr = st.wp["r"]
        segs = H.region_segments(v, st.region); tv = travel_of(st)
        miss = min(hit_test(p, segs, tv) for p in tips) - rr; st.clear = miss; st.tip = tips[-1].copy()
        outcome = "miss"; dmg = 0.0
        vf = next(x for x in self.f if x.idx == ctx.walkers.index(v))
        sk = v.sk; i = sk.idx
        blocked = None
        if st.react == "shield":
            sh = v.S[i["farm_L"]] if False else v.E[i["hand_L"]]
            if min(float(np.linalg.norm((p + tv * u) - sh)) for p in tips for u in (0, .5, 1)) < 0.34 + rr: blocked = ("arm_L", 0.1)
        elif st.react == "block" and v.pain["arm_" + vf.block_side] < 0.3 and not st.projectile:
            sd = vf.block_side; fa = (v.S[i["farm_" + sd]], v.E[i["hand_" + sd]])
            if min(seg_dist(p + tv * u, fa[0], fa[1]) for p in tips for u in (0, .5, 1)) < 0.22 + rr: blocked = ("arm_" + sd, 0.35)
        elif st.react == "block_leg":
            sd = st.region[-1]; sh_ = (v.S[i["shank_" + sd]], v.E[i["shank_" + sd]])
            if min(seg_dist(p + tv * u, sh_[0], sh_[1]) for p in tips for u in (0, .5, 1)) < 0.20 + rr: blocked = ("leg_" + sd, 0.4)
        if blocked: outcome = "blocked"; dmg = blocked[1]
        elif miss < 0.05: outcome = "hit"; dmg = 1.0
        else:
            for reg in H.HIT_REGIONS:
                if reg != st.region and min(hit_test(p, H.region_segments(v, reg), tv) for p in tips) - rr < 0.0:
                    st.region = reg; outcome = "hit"; dmg = 1.0; break
        d = np.array([v.pos[0] - a.pos[0], v.pos[1] - a.pos[1]]); d = d / (np.linalg.norm(d) + 1e-9)
        sw = st.bdir[:2] if np.linalg.norm(st.bdir[:2]) > 0.05 else d
        if kind_horizontal(st) and not st.projectile: d = d * 0.7 + (sw / (np.linalg.norm(sw) + 1e-9)) * 0.3
        push_scale = 0.5 if st.projectile else 1.0
        if outcome == "hit": v.apply_hit(st.region, d, st.strength * dmg * (1.0 if not st.projectile else 0.9))
        elif outcome == "blocked":
            v.apply_hit(blocked[0], d, st.strength * dmg)
        st.outcome = outcome
        self.log.append(dict(t=t, atk=a, vic=v, region=st.region, outcome=outcome, react=st.react or "none", strength=st.strength, label=st.label))

def kind_horizontal(st): return st.kind != "chop"

def W_delay(arena, opp_idx):
    f = next(x for x in arena.f if x.idx == opp_idx); return f.delay

def decide_wait(st, t, forced):
    """Commit as late as the movement allows: the attacker keeps re-aiming until the blow is launched, so an early dodge is followed."""
    v = st.vic
    if t < st.t_launch - 0.08 and st.aim is not None:
        return None if (st.t_launch - t) > 0.0 else None
    return decide(st, t, forced)

def caption_text(arena, t):
    rows = [r for r in arena.log if r["t"] <= t and t - r["t"] < 2.2]
    if not rows: return ""
    return " | ".join(f"{r['region']} {r['outcome']}" + (f" ({r['react']})" if r["react"] != "none" else "") for r in rows[-5:])
