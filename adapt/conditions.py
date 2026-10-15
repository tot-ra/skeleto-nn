"""Situations as data: each condition only says how it changes physical channels.

A condition never names a pose or an animation. "Ice" lowers the friction, "deep snow" makes the
feet sink and the air cold, "bleeding thigh" opens a wound at a place on the body with a rate and
some pain. The network turns channels into motion; a new kind of ground, weather or injury is a new
entry here and, if it only uses existing channels, works without retraining.

Effects are ``channel: (op, value)`` with ops:

* ``"min"``: the channel becomes at most ``value`` (friction: the most slippery surface wins);
* ``"max"``: at least ``value`` (pain, wetness, sink depth, darkness);
* ``"add"``: added (wind, load, temperature offsets);
* ``"set"``: replaced (temperature, wound position).

Intensity 0..1 blends from the channel's neutral value to the effect. ``side`` ("L"/"R") picks a
body side for regional conditions; ``region`` picks the place of a wound; ``dir`` (radians, 0 = from
ahead, +pi/2 = from the left) turns wind into its forward and lateral parts.

Exposures (rain, immersion, a fire's warmth) change the body over time rather than at once; they are
inputs to ``physiology.py``, which keeps wetness, core temperature, blood loss and fatigue as state.
"""
from __future__ import annotations
import math
from .schema import CHANNELS, CH, neutral_channels

# Body-frame points (x front, y left, z height 0..1 of standing height); "_S" is replaced by the side.
REGIONS = {
    "head": (0.3, 0.0, 0.95), "face": (0.9, 0.0, 0.93), "neck": (0.4, 0.0, 0.86),
    "chest_S": (0.9, 0.45, 0.72), "belly": (0.9, 0.0, 0.58), "flank_S": (0.2, 0.9, 0.6),
    "back": (-0.9, 0.0, 0.68), "shoulder_S": (0.3, 0.9, 0.8),
    "upper_arm_S": (0.0, 1.0, 0.7), "forearm_S": (0.3, 1.0, 0.55), "hand_S": (0.5, 1.0, 0.45),
    "thigh_S": (0.6, 0.5, 0.42), "knee_S": (0.8, 0.5, 0.28), "shin_S": (0.6, 0.5, 0.17),
    "foot_S": (0.7, 0.5, 0.03),
}
# region -> pain channel it hurts
REGION_PAIN = {
    "head": "pain_head", "face": "pain_head", "neck": "pain_head", "chest": "pain_torso", "belly": "pain_torso",
    "flank": "pain_torso", "back": "pain_torso", "shoulder": "pain_arm", "upper_arm": "pain_arm", "forearm": "pain_arm",
    "hand": "pain_arm", "thigh": "pain_leg", "knee": "pain_leg", "shin": "pain_leg", "foot": "pain_leg",
}

CONDITIONS: dict[str, dict] = {
    # ---- ground
    "dry_ground": dict(doc="dry packed earth or cobbles", effects={"friction": ("min", 0.8)}),
    "wet_cobbles": dict(doc="rain-wet stone setts", effects={"friction": ("min", 0.42)}),
    "wet_grass": dict(doc="dew or rain on grass", effects={"friction": ("min", 0.35)}),
    "wet_planks": dict(doc="wet quay or ship planking", effects={"friction": ("min", 0.3)}),
    "mud": dict(doc="deep mud", effects={"friction": ("min", 0.38), "sink": ("max", 0.22)}),
    "sand": dict(doc="loose dry sand", effects={"friction": ("min", 0.6), "sink": ("max", 0.12)}),
    "packed_snow": dict(doc="trodden snow", effects={"friction": ("min", 0.26), "air_temp": ("set", -4.0)}),
    "deep_snow": dict(doc="fresh snow above the ankle", effects={"friction": ("min", 0.32), "sink": ("max", 0.45), "air_temp": ("set", -6.0)}),
    "ice": dict(doc="frozen puddle, iced street", effects={"friction": ("min", 0.09), "air_temp": ("set", -5.0)}),
    "wet_ice": dict(doc="thawing ice with a film of water", effects={"friction": ("min", 0.05), "air_temp": ("set", 0.0)}),
    "fish_slime": dict(doc="fish guts and slime on the landing", effects={"friction": ("min", 0.15)}),
    "spilled_oil": dict(doc="oil or tallow spilled on stone", effects={"friction": ("min", 0.1)}),
    # ---- air and weather
    "frost": dict(doc="hard frost", effects={"air_temp": ("set", -15.0)}),
    "deep_frost": dict(doc="midwinter cold snap", effects={"air_temp": ("set", -28.0)}),
    "heat": dict(doc="a hot summer noon", effects={"air_temp": ("set", 33.0)}),
    "drenched": dict(doc="soaked through (rain, a fall into water)", effects={"wet": ("max", 1.0)}),
    "damp": dict(doc="damp clothes or coat", effects={"wet": ("max", 0.45)}),
    "icy_water": dict(doc="wading or swimming in the winter sea", effects={"wet": ("max", 1.0), "air_temp": ("set", 1.0)}),
    "wind": dict(doc="strong wind; dir = where it comes from", effects={"wind": ("add", 0.5)}),
    "gale": dict(doc="a storm gale", effects={"wind": ("add", 1.0)}),
    "night": dict(doc="moonless night, no torch", effects={"dark": ("max", 0.85)}),
    "dusk": dict(doc="dusk or a dim lane", effects={"dark": ("max", 0.4)}),
    "smoke": dict(doc="smoke from a fire", effects={"smoke": ("max", 0.8)}),
    # ---- body
    "bleeding": dict(doc="open wound; region and side", region=True, effects={"bleed": ("max", 0.7), "pain@": ("max", 0.45)}),
    "graze": dict(doc="shallow cut", region=True, effects={"bleed": ("max", 0.2), "pain@": ("max", 0.2)}),
    "sprain": dict(doc="sprained ankle or knee; side", effects={"pain_leg_S": ("max", 0.6)}),
    "broken_leg": dict(doc="broken lower leg; side", effects={"pain_leg_S": ("max", 0.95)}),
    "broken_arm": dict(doc="broken forearm; side", effects={"pain_arm_S": ("max", 0.9)}),
    "bruised_ribs": dict(doc="bruised or cracked ribs", effects={"pain_torso": ("max", 0.6)}),
    "headache": dict(doc="a blow to the head", effects={"pain_head": ("max", 0.6)}),
    "concussion": dict(doc="concussion", effects={"pain_head": ("max", 0.6), "dizzy": ("max", 0.55)}),
    "drunk": dict(doc="drunk on ale", effects={"dizzy": ("max", 0.6)}),
    "faint": dict(doc="faint from blood loss or hunger", effects={"dizzy": ("max", 0.4), "fatigue": ("max", 0.6)}),
    "exhausted": dict(doc="spent after work or a run", effects={"fatigue": ("max", 0.9)}),
    "hypothermic": dict(doc="chilled to the core", effects={"core_cold": ("max", 0.7)}),
    "heavy_load": dict(doc="a sack, a yoke of buckets, a pack", effects={"load": ("add", 0.4)}),
    "rider": dict(doc="a rider on a horse", effects={"load": ("add", 0.18)}),
}


def _blend(neutral, target, k):
    return neutral + (target - neutral) * k


def channels_from(active, base: dict | None = None) -> dict:
    """``active`` is a list of ``name``, ``(name, intensity)``, ``(name, intensity, options)`` or a dict of
    measured channel values (``{"air_temp": -6.0}``, applied in order like a condition).

    Returns a full channel dict (unknown channels keep their neutral value)."""
    ch = dict(base) if base is not None else neutral_channels()
    for item in active:
        if isinstance(item, dict):                      # measured values straight from the world: {"air_temp": -6.0}
            for key, val in item.items():
                if key in CH: ch[key] = float(val)
            continue
        if isinstance(item, str):
            name, k, opt = item, 1.0, {}
        elif len(item) == 2:
            (name, k), opt = item, {}
        else:
            name, k, opt = item
        spec = CONDITIONS[name]
        side = opt.get("side", "L")
        for key, (op, val) in spec["effects"].items():
            if key == "wind":
                d = opt.get("dir", 0.0)
                ch["wind_fwd"] = ch["wind_fwd"] + val * k * math.cos(d)
                ch["wind_lat"] = ch["wind_lat"] + val * k * math.sin(d)
                continue
            if key == "pain@":
                region = opt.get("region", "thigh")
                key = REGION_PAIN[region]
                if key in ("pain_leg", "pain_arm"): key += "_" + side
            key = key.replace("_S", "_" + side)
            c = CHANNELS[CH[key]]
            cur = ch[key]
            tgt = _blend(c.neutral, val, k) if op != "add" else val * k
            if op == "min": ch[key] = min(cur, tgt)
            elif op == "max": ch[key] = max(cur, tgt)
            elif op == "add": ch[key] = cur + tgt
            else: ch[key] = tgt
        if spec.get("region"):
            region = opt.get("region", "thigh")
            rk = region + "_S" if region + "_S" in REGIONS else region
            x, y, z = REGIONS[rk]
            if rk.endswith("_S") and side == "R": y = -y
            ch["wound_x"], ch["wound_y"], ch["wound_z"] = x, y, z
    for c in CHANNELS:
        ch[c.name] = min(c.hi, max(c.lo, ch[c.name]))
    return ch


def describe() -> str:
    return "\n".join(f"{n:14s} {d['doc']}" for n, d in CONDITIONS.items())
