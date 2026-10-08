"""Blows with different weapons and limbs, and damage from different attackers (a wolf's teeth, a bear's paw, a horse's hoof).

Humans hit with a sword, an axe, a spear or a fist through the combat arena (the same reactions and pain as in catalog_combat); a kick
and every animal attack are one `LimbAttack`: a limb (or the head) reaches for a point of the victim in time, and if it gets there the blow
is applied to the victim's pain model (hit.apply_hit) with a strength that follows the attacker's mass and the speed of the limb."""
from __future__ import annotations
import math
import numpy as np
from bodies import *
from planner import Cmd
from shots import *
from catalog import shot
import hit as H
from catalog_combat import _build, SWORD_GEOM, SPEAR_GEOM, FIST_GEOM

@shot
def weapons_human():
    lanes = [dict(region="torso", weapon="sword", react=False, strength=1.5, range=1.25), dict(region="head", weapon="axe", kind="chop", react=False, strength=1.5, range=1.25),
             dict(region="torso", weapon="spear", react=False, strength=1.5, range=2.0), dict(region="head", weapon="fist", react=False, strength=1.3, range=0.95),
             dict(region="torso", weapon="fist", react=False, strength=1.3, range=0.95), dict(region="head", weapon="sword", react=True, delay=0.18, strength=1.5, range=1.25)]
    return _build(lanes, 9.0, "weapons_human", "sword, axe, spear and fist on the same body: the blade swings, the spear and the fist go straight; the last one sees the sword coming and dodges",
                  dict(dist=9.0, azimuth=90, elevation=-6, follow="all", look_z=0.9), spacing=3.0, size=(760, 340))

class LimbAttack:
    """One attacker, one victim: approach, reach with a limb or the head, and on contact apply the blow to the victim.

    limb: ("leg", name) | ("head",); target: a bone name of the victim; region: pain region for apply_hit; strength: velocity change in m/s."""
    def __init__(self, atk, vic, limb, bone, region, strength, t0, reach_t=0.35, stand=1.0, label="", away=False):
        self.atk, self.vic, self.limb, self.bone, self.region, self.strength, self.t0, self.reach_t = atk, vic, limb, bone, region, strength, t0, reach_t
        self.t_start = None; self.stand = stand; self.away = away; self.done = False; self.landed = None; self.label = label; self.p0 = None

    def target(self):
        s = self.vic.sk; return self.vic.S[s.idx[self.bone]] * 0.5 + self.vic.E[s.idx[self.bone]] * 0.5

    def cmd(self, t, w, ctx):
        if w.S is None or self.vic.S is None: return Cmd()
        tg = self.target(); rel = tg[:2] - w.pos; dist = float(np.linalg.norm(rel)); head = math.atan2(rel[1], rel[0])
        v = np.zeros(2)
        if t < self.t0:
            if dist > self.stand: v = rel / max(dist, 1e-6) * 1.6
            return Cmd(v=v, heading=head + (math.pi if self.away else 0.0))
        if self.limb[0] == "leg" and self.t_start is None:                  # a kick needs the other leg planted: wait until it is
            if self.atk is not None and all(l.stance for l in w.legs if l.leg.name != self.limb[1]) and w.speed < 0.3: self.t_start = t
            else: return Cmd(heading=head + (math.pi if self.away else 0.0))
        if self.t_start is None: self.t_start = t
        u = (t - self.t_start) / self.reach_t
        if self.limb[0] == "head":
            base = w.S[w.c.neck[0]]
            k = min(1.0, u) if u < 1.0 else max(0.0, 1.0 - (u - 1.0) / 1.2)          # reach, then draw back
            w.head_goal = base + (tg - base) * k if u < 2.2 else None
            if dist > self.stand * 0.95 and u < 1.0: v = rel / max(dist, 1e-6) * 3.2          # a lunge
        else:
            name = self.limb[1]; ls = next(l for l in w.legs if l.leg.name == name)
            if self.p0 is None: self.p0 = ls.planted.copy() if ls.stance else ls.target.copy()
            k = min(1.0, u) if u < 1.0 else max(0.0, 1.5 - 0.5 * u)
            ease = k * k * (3 - 2 * k); pt = self.p0 + (tg - self.p0) * ease; pt[2] = max(pt[2], 0.04 + 0.9 * math.sin(math.pi * min(1.0, ease)) * 0.0 + (tg[2] - self.p0[2]) * ease)
            if u < 2.0: w.leg_ovrs = [o for o in w.leg_ovrs if o.get("tag") != id(self)] + [dict(name=name, point=pt, until=t + 0.1, tag=id(self))]
        if not self.done and u >= 1.0:
            reach = self.limb[0] == "head" and np.linalg.norm(w.S[w.c.head] * 0.5 + w.E[w.c.head] * 0.5 - tg) < 0.35 or \
                    self.limb[0] == "leg" and np.linalg.norm(w.leg_ovrs[-1]["point"] - tg) < 0.3 if w.leg_ovrs else False
            if self.limb[0] == "head": reach = bool(np.linalg.norm(w.pos - self.vic.pos) < self.stand * 0.9 + 0.5)
            self.done = True; self.landed = bool(reach)
            if reach:
                d = np.append(self.vic.pos - w.pos, 0.0); H.apply_hit(self.vic, self.region, d, self.strength)
        return Cmd(v=v, heading=head + (math.pi if self.away else 0.0))

def _attack_shot(name, title, atk_c, limb, bone, region, strength, T=6.0, start=(1.8, 0.0), stand=0.9, t0=1.4, reach_t=0.35, victim=None, cam=None, victim_heading=None, away=False):
    tr = Terrain(-6, 12, -8, 8)
    V = victim or human(HumanBody(height=1.75, mass=76)); A = atk_c
    atk = LimbAttack(None, None, limb, bone, region, strength, t0, reach_t, stand, away=away)
    def cmd_v(t, w, ctx): atk.vic = w; return Cmd()
    def cmd_a(t, w, ctx):
        atk.atk = w
        if atk.vic is None and len(ctx.walkers) > 0: atk.vic = ctx.walkers[0]
        return atk.cmd(t, w, ctx)
    acts = [ActorSpec(V, pos=(0.0, 0.0), heading=math.pi if victim_heading is None else victim_heading, cmd=cmd_v, base=(0.35, 0.5, 0.7), label="victim"),
            ActorSpec(A, pos=start, heading=math.pi, cmd=cmd_a, base=(0.65, 0.45, 0.35), label="attacker")]
    def cap(t, ctx):
        v = ctx.walkers[0]
        if getattr(v, "pain", None) is None: return ""
        top = sorted(v.pain.items(), key=lambda kv: -kv[1])[:2]
        return ("hit: " if atk.landed else "") + ", ".join(f"{k} pain {x:.2f}" for k, x in top if x > 0.02) + (" (down)" if v.down is not None else "")
    return Shot(name, title, tr, acts, T, cam or dict(dist=5.5, azimuth=90, elevation=-8, follow="all", look_z=0.8), size=(600, 320), caption_fn=cap)

@shot
def kick_human():
    A = human(HumanBody(height=1.82, mass=86))
    return _attack_shot("kick_human", "a kick to the thigh: the attacker lifts the leg and drives the foot through the target; the victim takes the pain in the leg and limps", A, ("leg", "leg_R"), "thigh_L", "leg_L", 1.6, stand=1.0, T=7.0)

@shot
def wolf_bite():
    A = quadruped("wolf")
    return _attack_shot("wolf_bite", "a wolf lunges and bites the forearm: the neck and head reach the arm while the body drives forward; the arm is hurt and cannot be used", A, ("head",), "farm_R", "arm_R", 1.4, stand=1.3, T=7.0,
                        start=(3.5, 0.3), reach_t=0.4)

@shot
def bear_swipe():
    A = quadruped("bear")
    return _attack_shot("bear_swipe", "a bear stands over its front paws and swipes with one: a heavy blow to the head and chest that knocks the man down", A, ("leg", "FR"), "head", "head", 2.8, stand=1.5, T=8.0,
                        start=(3.5, 0.0), reach_t=0.5)

@shot
def horse_kick():
    A = quadruped("horse")
    return _attack_shot("horse_kick", "a horse kicks back with a hind hoof at a man standing behind it: a hard blow to the torso that throws him", A, ("leg", "HL"), "chest", "torso", 3.0, stand=1.2, T=8.0,
                        start=(-2.2, 0.0), t0=2.0, reach_t=0.4, victim_heading=math.pi, away=True)
