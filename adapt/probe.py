"""Print the network's style for named situations and bodies (a readable check of what it learned).

    python -m adapt.probe                 # default table
    python -m adapt.probe --md            # Markdown table for the docs
"""
from __future__ import annotations
import argparse, math, os
from .net import StyleNet
from .conditions import channels_from
from .schema import STYLE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BODIES = {
    "person": ("biped", 70.0, 0.9, False), "child": ("biped", 25.0, 0.6, False),
    "dog": ("quadruped", 25.0, 0.5, False), "cat": ("quadruped", 4.0, 0.25, False), "horse": ("quadruped", 500.0, 1.0, False),
    "crow_walk": ("bird", 0.45, 0.1, False), "crow_fly": ("bird", 0.45, 0.1, True), "gull_fly": ("bird", 0.9, 0.12, True),
}
CASES = [
    ("neutral", []), ("wet cobbles", ["wet_cobbles"]), ("ice", ["ice"]), ("wet ice", ["wet_ice"]),
    ("deep snow", ["deep_snow"]), ("mud", ["mud"]), ("frost", ["frost"]), ("drenched, 5 C", ["drenched", ("frost", 0.5)]),
    ("hypothermic", ["hypothermic", "frost"]), ("heat", ["heat"]),
    ("bleeding L thigh", [("bleeding", 1.0, dict(region="thigh", side="L"))]),
    ("bleeding R chest", [("bleeding", 1.0, dict(region="chest", side="R"))]),
    ("bleeding back", [("bleeding", 1.0, dict(region="back"))]),
    ("sprained L ankle", [("sprain", 1.0, dict(side="L"))]), ("broken R arm", [("broken_arm", 1.0, dict(side="R"))]),
    ("bruised ribs", ["bruised_ribs"]), ("drunk", ["drunk"]), ("exhausted", ["exhausted"]),
    ("gale from left", [("gale", 1.0, dict(dir=math.pi / 2))]), ("headwind", [("gale", 0.8, dict(dir=0.0))]),
    ("night", ["night"]), ("smoke", ["smoke"]), ("heavy load", ["heavy_load"]), ("drunk on ice", ["drunk", "ice"]),
]
SHOW = ["speed_frac", "stride", "duty_add", "width", "crouch", "trunk_pitch", "trunk_roll", "arm_out", "hug_L", "hug_R",
        "press_L", "press_R", "head_down", "shrug", "tremor", "foot_flat", "clearance", "limp", "care", "tail_tuck", "fluff",
        "flap_rate", "flap_amp", "glide"]


def table(net, body="person", froude=0.25, cases=CASES, md=False):
    kind, mass, leg, air = BODIES[body]
    rows = []
    for name, conds in cases:
        st = net.style(kind, mass, leg, air, froude, channels_from(conds))
        rows.append((name, st))
    keys = [k for k in SHOW if any(abs(st[k] - next(x.neutral for x in STYLE if x.name == k)) > 0.04 for _, st in rows)]
    if md:
        out = [f"| {body} | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for name, st in rows: out.append(f"| {name} | " + " | ".join(f"{st[k]:.2f}" for k in keys) + " |")
    else:
        out = [f"{body:18s}" + "".join(f"{k[:9]:>10s}" for k in keys)]
        for name, st in rows: out.append(f"{name:18s}" + "".join(f"{st[k]:10.2f}" for k in keys))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--net", default=os.path.join(HERE, "results", "style_net.json"))
    ap.add_argument("--body", default="person"); ap.add_argument("--froude", type=float, default=0.25); ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    print(table(StyleNet(a.net), a.body, a.froude, md=a.md))
