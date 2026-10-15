"""Condition-adaptive motion shots: the same planner and one network, different situations.

Every actor walks with a StyleDriver (adapt/driver.py); the only difference between lanes is the
situation given as data (conditions and exposures). No lane has its own motion code.
    python catalog.py gif adapt_ground out.gif      (catalog.py imports this module)
"""
from __future__ import annotations
import math, os
import numpy as np
from bodies import *
from planner import Cmd
from shots import *
from catalog import shot, straight_cmd
from adapt.net import StyleNet
from adapt.driver import StyleDriver
from adapt.physiology import BodyState, Exposure
from adapt.flight import FlightActor

NET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "style_net.json")
_NET = None


def net():
    global _NET
    if _NET is None: _NET = StyleNet(NET_PATH)
    return _NET


GROUND_RGBA = {"dry_ground": "0.62 0.55 0.45 1", "wet_cobbles": "0.38 0.40 0.44 1", "ice": "0.80 0.90 0.98 1", "wet_ice": "0.70 0.85 0.95 1",
               "deep_snow": "0.97 0.97 1.0 1", "mud": "0.36 0.27 0.18 1", "spilled_oil": "0.20 0.18 0.12 1", "packed_snow": "0.90 0.92 0.95 1",
               "wet_grass": "0.30 0.50 0.25 1", "fish_slime": "0.55 0.60 0.50 1"}


def lane_props(lanes, y0, dy, x0=-1.0, x1=14.0):
    out = []
    for i, conds in enumerate(lanes):
        g = next((c if isinstance(c, str) else c[0] for c in conds if (c if isinstance(c, str) else c[0]) in GROUND_RGBA), None)
        if g is None: continue
        y = y0 + i * dy
        out.append(dict(geom=f'<geom type="box" size="{(x1 - x0) / 2:.2f} {dy * 0.45:.2f} 0.004" pos="{(x0 + x1) / 2:.2f} {y:.2f} 0.0" rgba="{GROUND_RGBA[g]}" contype="0" conaffinity="0"/>'))
    return out


def _lanes_shot(name, title, lanes, labels, make, speed, T, dy=1.1, states=None, exposures=None, cam=None, size=(560, 320), events=None, physio_speed=1.0):
    tr = Terrain(); acts = []
    y0 = -dy * (len(lanes) - 1) / 2
    for i in range(len(lanes)):
        acts.append(ActorSpec(make(i), pos=(0.0, y0 + i * dy), cmd=straight_cmd(speed, T), label=labels[i], events=(events or {}).get(i, [])))
    sh = Shot(name, title + ": " + ", ".join(labels) + " (near to far)", tr, acts, T,
              cam or dict(dist=7.0, azimuth=58, elevation=-9, follow="all", look_z=0.8), props=lane_props(lanes, y0, dy), size=size)

    def post_init(walkers):
        sh.drivers = []
        for i, w in enumerate(walkers):
            st = states[i]() if states and states[i] else None
            ex = exposures[i] if exposures and exposures[i] else None
            env = (lambda ex: (lambda t, w: ([], ex(t)) if ex else ([], Exposure())))(ex)
            sh.drivers.append(StyleDriver(w, net(), env=env, conditions=lanes[i], state=st, physio_speed=physio_speed))
    sh.post_init = post_init
    sh.caption_fn = None
    return sh


@shot
def adapt_ground():
    lanes = [["dry_ground"], ["wet_cobbles"], ["mud"], ["deep_snow"], ["ice"], ["wet_ice"]]
    labels = ["dry", "wet cobbles", "mud", "deep snow", "ice", "wet ice"]
    return _lanes_shot("adapt_ground", "one network, six grounds", lanes, labels, lambda i: human(), 1.35, 10.0)


@shot
def adapt_injury():
    W = lambda r, s: [("bleeding", 1.0, dict(region=r, side=s))]
    lanes = [[], W("thigh", "L"), W("chest", "R"), [("sprain", 1.0, dict(side="L"))], [("broken_arm", 1.0, dict(side="R"))], ["bruised_ribs"], ["drunk"], ["exhausted"]]
    labels = ["healthy", "bleeding L thigh", "bleeding R chest", "sprained L ankle", "broken R arm", "bruised ribs", "drunk", "exhausted"]
    return _lanes_shot("adapt_injury", "injuries and states as channels", lanes, labels, lambda i: human(), 1.3, 10.0, dy=1.0, size=(600, 340))


@shot
def adapt_weather():
    lanes = [[], ["frost"], [("gale", 1.0, dict(dir=math.pi / 2))], [("gale", 0.9, dict(dir=0.0))], ["night"], ["smoke"]]
    labels = ["mild", "hard frost", "gale from the left", "headwind", "dark night", "smoke"]
    return _lanes_shot("adapt_weather", "weather and air", lanes, labels, lambda i: human(), 1.3, 10.0)


@shot
def adapt_rain_cold():
    """Exposure over time: rain soaks the body, wet clothes lose their warmth, the core cools; the network
    answers the growing channels (hug, hunch, shiver), which slows the cooling."""
    rain = lambda t: Exposure(rain=1.0 if t > 1.0 else 0.0)
    lanes = [[("frost", 0.55)], [("frost", 0.55)], [("frost", 0.55)]]
    states = [lambda: BodyState(), lambda: BodyState(), lambda: BodyState(core_cold=0.0)]
    exps = [None, rain, lambda t: Exposure(rain=1.0 if t > 1.0 else 0.0, immersion=0.6 if 4.0 < t < 5.0 else 0.0)]
    sh = _lanes_shot("adapt_rain_cold", "about -5 C, body time 40x: one stays dry, one walks into rain, one also falls into a ditch at 4 s", lanes,
                     ["dry", "rain from 1 s", "rain + ditch"], lambda i: human(), 1.2, 14.0, dy=1.3, states=states, exposures=exps, physio_speed=40.0,
                     cam=dict(dist=5.0, azimuth=35, elevation=-10, follow="all", look_z=0.9))
    sh.caption_fn = lambda t, ctx: "   ".join("wet {wet:.2f} cold {core_cold:.2f}".format(**d.at(t)["ch"]) for d in getattr(ctx.shot, "drivers", []) if d.at(t))
    return sh


@shot
def adapt_bleed_time():
    """A wound at 1 s: a hand goes to it, the blood lost grows and the body slows; pressure slows the loss."""
    def mk_state(): return BodyState()
    def wound(w, ctx):
        for d in ctx.shot.drivers:
            if d.w is w: d.state.wound("thigh", "L", rate=0.9, pain=0.5)
    sh = _lanes_shot("adapt_bleed_time", "a cut to the left thigh at 1 s, body time 20x", [[], []], ["healthy", "wounded at 1 s"], lambda i: human(), 1.4, 14.0, dy=1.4,
                     states=[mk_state, mk_state], events={1: [(1.0, wound)]}, physio_speed=20.0, cam=dict(dist=4.2, azimuth=35, elevation=-10, follow=1, look_z=0.8))
    sh.caption_fn = lambda t, ctx: "wounded: bleed {bleed:.2f} lost {blood_lost:.2f}".format(**ctx.shot.drivers[1].at(t)["ch"]) if ctx.shot.drivers[1].at(t) else ""
    return sh


@shot
def adapt_animals():
    sp = ["dog", "dog", "dog", "cat", "horse", "horse"]
    lanes = [[], ["ice"], ["deep_snow"], ["drenched", ("frost", 0.6)], ["mud"], [("sprain", 1.0, dict(side="L"))]]
    labels = ["dog dry", "dog on ice", "dog in deep snow", "cat soaked in the cold", "horse in mud", "horse lame on the left"]
    return _lanes_shot("adapt_animals", "the same network for quadrupeds", lanes, labels, lambda i: quadruped(sp[i]), 1.4, 10.0, dy=1.6,
                       cam=dict(dist=8.5, azimuth=58, elevation=-10, follow="all", look_z=0.5), size=(600, 340))


@shot
def adapt_birds_ground():
    lanes = [[], ["ice"], ["drenched", ("frost", 0.6)], [("sprain", 1.0, dict(side="L"))]]
    labels = ["crow", "crow on ice", "crow soaked in the cold", "crow with a hurt left foot"]
    return _lanes_shot("adapt_birds_ground", "birds walk with the same network", lanes, labels, lambda i: bird("crow"), 0.45, 8.0, dy=0.45,
                       cam=dict(dist=2.4, azimuth=58, elevation=-12, follow="all", look_z=0.15), size=(520, 300))


@shot
def adapt_birds_flight():
    cases = [([], "calm"), (["drenched"], "wet plumage"), ([("gale", 0.7, dict(dir=0.0))], "headwind"), ([("gale", 0.7, dict(dir=math.pi))], "tailwind"),
             (["heavy_load"], "carrying prey"), ([("broken_arm", 1.0, dict(side="L"))], "hurt left wing")]
    tr = Terrain(); acts = []
    for i, (conds, lab) in enumerate(cases):
        fa = FlightActor(bird("crow"), net(), conds, y=-0.9 * i, z=0.9 + (len(cases) - 1 - i) * 0.45)
        acts.append(ActorSpec(fa.c, kinematic=fa.pose, label=lab))
    return Shot("adapt_birds_flight", "crows in flight: wingbeat rate, amplitude and gliding from the network: " + ", ".join(l for _, l in cases) + " (far to near)",
                tr, acts, 6.0, dict(dist=6.0, azimuth=40, elevation=-14, follow="all", look_z=1.9), size=(560, 360))


# conditions added after the network was trained: only their channels are given
LATE_CONDITIONS = {
    "greasy_floor": dict(doc="a kitchen floor slick with fat", effects={"friction": ("min", 0.14)}),
    "slush": dict(doc="half-melted snow over cobbles", effects={"friction": ("min", 0.2), "sink": ("max", 0.15), "air_temp": ("set", 0.5)}),
    "tar_pitch": dict(doc="fresh pitch on the shipyard planks: sticky", effects={"friction": ("min", 1.1), "sink": ("max", 0.05)}),
}


@shot
def adapt_unseen():
    """Conditions written after training, described only by their channels, and combinations of known ones:
    they work without retraining because the network never sees condition names."""
    from adapt.conditions import CONDITIONS
    CONDITIONS.update(LATE_CONDITIONS)
    lanes = [["greasy_floor"], ["slush"], ["tar_pitch"], [("packed_snow", 1.0), ("gale", 0.8, dict(dir=-math.pi / 2))], [("drunk", 1.0), ("wet_planks", 1.0)]]
    labels = ["greasy floor (new)", "slush (new)", "sticky pitch (new)", "packed snow in a gale from the right", "drunk on wet planks"]
    return _lanes_shot("adapt_unseen", "conditions added after training and combinations, no retraining", lanes, labels, lambda i: human(), 1.3, 10.0, dy=1.2)
