"""Export everything an engine needs to run the style network, plus golden vectors to test a port.

    python -m adapt.export_engine OUT_DIR

Writes
  style_net.json          the weights with the input and output schema (same file as results/style_net.json)
  motion_conditions.json  the condition registry, body regions and region -> pain channel map
  style_net_golden.json   cases (body, conditions, channels) with the expected raw and calibrated style,
                          and a physiology run, so an engine port can assert it computes the same numbers
  leap_net.json           the jump network (``adapt/leap.py``)
  leap_net_golden.json    scenes (profiles in metres) with the expected decision, knobs and arc
"""
from __future__ import annotations
import json, math, os, shutil, sys
from .net import StyleNet
from .conditions import CONDITIONS, REGIONS, REGION_PAIN, channels_from
from .schema import neutral_channels, STYLE, knob_active
from .physiology import BodyState, Exposure

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(HERE, "results", "style_net.json")
LEAP = os.path.join(HERE, "results", "leap_net.json")

GOLDEN = [
    ("biped", 70.0, 0.9, False, 0.25, []),
    ("biped", 70.0, 0.9, False, 0.25, ["ice"]),
    ("biped", 70.0, 0.9, False, 0.25, [["bleeding", 1.0, {"region": "thigh", "side": "L"}]]),
    ("biped", 70.0, 0.9, False, 0.25, [["broken_arm", 1.0, {"side": "R"}]]),
    ("biped", 70.0, 0.9, False, 0.25, ["drenched", ["frost", 0.5]]),
    ("biped", 30.0, 0.6, False, 0.6, [["gale", 1.0, {"dir": 1.5707963}], "wet_cobbles"]),
    ("quadruped", 25.0, 0.5, False, 0.4, ["ice"]),
    ("quadruped", 500.0, 1.0, False, 0.3, ["mud", ["sprain", 1.0, {"side": "L"}]]),
    ("bird", 0.45, 0.1, True, 0.5, ["drenched"]),
    ("bird", 0.45, 0.1, False, 0.4, ["ice"]),
]


def _conds(lst):
    out = []
    for c in lst:
        if isinstance(c, str): out.append(c)
        elif len(c) == 2: out.append((c[0], c[1]))
        else: out.append((c[0], c[1], c[2]))
    return out


def golden(net: StyleNet):
    cases = []
    for kind, mass, leg, air, fr, conds in GOLDEN:
        ch = channels_from(_conds(conds))
        raw = net.style(kind, mass, leg, air, fr, ch)
        base = net.style(kind, mass, leg, air, fr, neutral_channels())
        cal = {k.name: (min(k.hi, max(k.lo, k.neutral + raw[k.name] - base[k.name])) if knob_active(k, kind, air) else k.neutral) for k in STYLE}
        cases.append(dict(kind=kind, mass=mass, leg=leg, airborne=air, froude=fr, conditions=conds, channels=ch, raw=raw, calibrated=cal))
    # physiology: 10 minutes of rain at -7 C with a fixed style, and a thigh wound pressed half the time
    env = channels_from([("frost", 0.5)])
    b = BodyState(); b.wound("thigh", "L", rate=0.8, pain=0.5)
    st = dict(hug_L=0.5, hug_R=0.5, tremor=0.6)
    trace = []
    for i in range(600):
        b.step(1.0, env, Exposure(rain=1.0), st, "biped", False, 0.3, 0.9 if i % 2 == 0 else 0.0)
        if i % 100 == 99:
            trace.append(dict(t=i + 1, wet=b.wet, core_cold=b.core_cold, blood_lost=b.blood_lost, fatigue=b.fatigue,
                              bleed=b.wound_channels()["bleed"]))
    return dict(cases=cases, physiology=dict(env=env, style=st, power=0.3, rain=1.0, press_even_steps=0.9, dt=1.0, wound=dict(region="thigh", side="L", rate=0.8, pain=0.5), trace=trace))


def leap_golden():
    from . import leap as L
    net = L.LeapNet(LEAP); cases = []
    sits = [[], ["ice"], [["sprain", 1.0, {"side": "L"}]], ["heavy_load"]]
    for conds in sits:
        ch = channels_from(_conds(conds))
        for name, feats, _ in L.SCENES:
            z, w, thin = L.profile(0.86, *feats)
            d = net.decide("biped", 58.0, 0.86, 0.0, 1.0, 0.9, ch, z, w, ~thin)
            cases.append(dict(scene=name, conditions=conds, channels=ch, kind="biped", mass=58.0, leg=0.86, speed=0.0, gain=1.0,
                              urgency=0.9, z=z.tolist(), w=w.tolist(), standable=(~thin).tolist(),
                              expect={k: v for k, v in d.items() if k != "logits"}))
    z, w, thin = L.profile(0.5, ("wall", 0.4, 0.6, 0.2, 0.0))
    ch = channels_from([])
    d = net.decide("quadruped", 25.0, 0.5, 2.0, 2.5, 0.7, ch, z, w, ~thin)
    cases.append(dict(scene="dog at a 0.6 m fence", conditions=[], channels=ch, kind="quadruped", mass=25.0, leg=0.5, speed=2.0, gain=2.5,
                      urgency=0.7, z=z.tolist(), w=w.tolist(), standable=(~thin).tolist(), expect={k: v for k, v in d.items() if k != "logits"}))
    return dict(cases=cases)


def main(out):
    os.makedirs(out, exist_ok=True)
    shutil.copy(NET, os.path.join(out, "style_net.json"))
    json.dump(dict(conditions=CONDITIONS, regions={k: list(v) for k, v in REGIONS.items()}, region_pain=REGION_PAIN),
              open(os.path.join(out, "motion_conditions.json"), "w"), indent=1)
    json.dump(golden(StyleNet(NET)), open(os.path.join(out, "style_net_golden.json"), "w"), indent=1)
    if os.path.exists(LEAP):
        shutil.copy(LEAP, os.path.join(out, "leap_net.json"))
        json.dump(leap_golden(), open(os.path.join(out, "leap_net_golden.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
