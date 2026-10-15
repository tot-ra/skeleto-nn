"""The vocabulary of the condition-adaptive motion layer.

Three lists define everything the network sees and says:

* ``BODY``: what kind of body it is (biped, quadruped, bird; mass; leg length; flying or not) and the
  commanded speed as a Froude number, so one network covers a sparrow, a cat, a person and a horse;
* ``CHANNELS``: physical and physiological quantities that a situation changes (ground friction, how
  deep the feet sink, how wet and cold the body is, blood loss and where the wound is, pain per region,
  wind, fatigue, dizziness, load, darkness, ...). A *condition* such as "ice" or "bleeding thigh" is
  never an input: it is data that sets channels (``conditions.py``). This is what makes the layer
  universal: an unseen condition that maps onto known channels needs no retraining at all;
* ``STYLE``: generic motor-style controls that every body plan understands (stride, speed taken,
  stance width, crouch, trunk lean, arms out / hugged / pressing a point, head down, shiver, foot
  placement, swing clearance, limp, tail tuck, feather or fur fluff, wingbeat rate, amplitude and
  glide share). The planner and the engine apply them; nothing names a situation.

Adding a channel appends to ``CHANNELS``; old weight files are widened with zero input columns
(``net.py``), so a warm-started retrain only has to learn the new column.
"""
from __future__ import annotations
from dataclasses import dataclass

KINDS = ("biped", "quadruped", "bird")


@dataclass(frozen=True)
class Channel:
    name: str
    lo: float
    hi: float
    neutral: float
    scale: float          # encoding: (value - neutral) / scale, then clipped to [-3, 3]
    doc: str
    log: bool = False     # encode log(value / neutral) / scale instead


@dataclass(frozen=True)
class Knob:
    name: str
    lo: float
    hi: float
    neutral: float
    kinds: tuple          # body kinds that can express it; others are pinned to neutral
    doc: str
    air: str = "ground"   # "ground": only while on the ground, "air": only in flight, "both"


CHANNELS: list[Channel] = [
    Channel("friction", 0.03, 1.2, 0.8, 1.0, "available ground friction coefficient", log=True),
    Channel("sink", 0.0, 1.0, 0.0, 0.3, "how deep a foot sinks, in leg lengths (mud, snow, sand)"),
    Channel("wet", 0.0, 1.0, 0.0, 0.5, "wetness of skin, fur or feathers"),
    Channel("air_temp", -35.0, 45.0, 15.0, 20.0, "air (or water) temperature around the body, deg C"),
    Channel("insulation", 0.0, 1.0, 0.3, 0.3, "clothing, fur or plumage insulation (0 bare, 0.3 light clothes, 0.8 winter furs)"),
    Channel("core_cold", 0.0, 1.0, 0.0, 0.4, "hypothermia: how far the core has cooled (state)"),
    Channel("wind_fwd", -1.0, 1.0, 0.0, 0.5, "wind from ahead (+) or behind (-), in units of 20 m/s"),
    Channel("wind_lat", -1.0, 1.0, 0.0, 0.5, "wind from the left (+) or right (-), in units of 20 m/s"),
    Channel("bleed", 0.0, 1.0, 0.0, 0.4, "bleeding rate of the worst open wound"),
    Channel("blood_lost", 0.0, 1.0, 0.0, 0.4, "fraction of the lethal blood loss already lost (state)"),
    Channel("wound_x", -1.0, 1.0, 0.0, 0.6, "wound (or point to protect): front (+) / back (-) of the body"),
    Channel("wound_y", -1.0, 1.0, 0.0, 0.6, "wound: left (+) / right (-) of the midline"),
    Channel("wound_z", 0.0, 1.0, 0.5, 0.3, "wound: height on the body, 0 = feet, 1 = top of the head"),
    Channel("pain_leg_L", 0.0, 1.0, 0.0, 0.4, "pain in the left leg (left hind and fore for animals)"),
    Channel("pain_leg_R", 0.0, 1.0, 0.0, 0.4, "pain in the right leg"),
    Channel("pain_arm_L", 0.0, 1.0, 0.0, 0.4, "pain in the left arm or wing"),
    Channel("pain_arm_R", 0.0, 1.0, 0.0, 0.4, "pain in the right arm or wing"),
    Channel("pain_torso", 0.0, 1.0, 0.0, 0.4, "pain in the trunk (ribs, belly)"),
    Channel("pain_head", 0.0, 1.0, 0.0, 0.4, "pain in the head"),
    Channel("fatigue", 0.0, 1.0, 0.0, 0.4, "spent energy reserve"),
    Channel("dizzy", 0.0, 1.0, 0.0, 0.4, "vestibular disturbance (drink, concussion, faintness)"),
    Channel("load", 0.0, 1.0, 0.0, 0.3, "carried load as a fraction of body mass"),
    Channel("dark", 0.0, 1.0, 0.0, 0.5, "how little of the ground can be seen"),
    Channel("smoke", 0.0, 1.0, 0.0, 0.5, "smoke or foul air at head height"),
]

BODY = ["kind_biped", "kind_quadruped", "kind_bird", "log_mass", "log_leg", "airborne", "froude"]

B, Q, W = ("biped",), ("quadruped",), ("bird",)
ALL = KINDS
STYLE: list[Knob] = [
    Knob("speed_frac", 0.0, 1.15, 1.0, ALL, "fraction of the commanded speed actually taken", "both"),
    Knob("stride", 0.45, 1.35, 1.0, ALL, "stride length relative to the Froude law (cadence follows)"),
    Knob("duty_add", -0.05, 0.18, 0.0, ALL, "extra share of the cycle with the feet on the ground"),
    Knob("width", 0.8, 2.2, 1.0, ALL, "stance width relative to normal"),
    Knob("crouch", 0.0, 0.6, 0.0, ALL, "lowering of the body by bending the legs"),
    Knob("trunk_pitch", -0.15, 0.7, 0.0, ALL, "forward lean or hunch of the trunk, rad"),
    Knob("trunk_roll", -0.35, 0.35, 0.0, ALL, "sideways lean, + to the left, rad"),
    Knob("arm_out", 0.0, 1.0, 0.0, B + W, "arms (or wings) held out for balance", "ground"),
    Knob("hug_L", 0.0, 1.0, 0.0, B + W, "left arm held to the body (wing folded tight)", "ground"),
    Knob("hug_R", 0.0, 1.0, 0.0, B + W, "right arm held to the body", "ground"),
    Knob("press_L", 0.0, 1.0, 0.0, B, "left hand pressed on the wound point", "ground"),
    Knob("press_R", 0.0, 1.0, 0.0, B, "right hand pressed on the wound point", "ground"),
    Knob("head_down", 0.0, 1.0, 0.0, ALL, "head lowered (look at the feet, tuck the face)", "ground"),
    Knob("shrug", 0.0, 1.0, 0.0, ALL, "shoulders raised and neck drawn in", "ground"),
    Knob("tremor", 0.0, 1.0, 0.0, ALL, "shivering amplitude", "both"),
    Knob("foot_flat", 0.0, 1.0, 0.0, ALL, "plant the whole foot: no heel strike, little toe-off"),
    Knob("clearance", 0.0, 1.0, 0.0, ALL, "extra swing height of the feet"),
    Knob("limp", -1.0, 1.0, 0.0, ALL, "+ less time and load on the left leg(s), - on the right"),
    Knob("care", 0.0, 1.0, 0.0, ALL, "caution: slower steps placed by looking"),
    Knob("tail_tuck", 0.0, 1.0, 0.0, Q, "tail pulled down and in", "ground"),
    Knob("fluff", 0.0, 1.0, 0.0, Q + W, "fur or feathers raised for insulation", "both"),
    Knob("flap_rate", 0.7, 1.5, 1.0, W, "wingbeat frequency relative to the species", "air"),
    Knob("flap_amp", 0.7, 1.3, 1.0, W, "wingbeat amplitude relative to the species", "air"),
    Knob("glide", 0.0, 0.8, 0.0, W, "share of time gliding between bouts of flapping", "air"),
]

CH = {c.name: i for i, c in enumerate(CHANNELS)}
KN = {k.name: i for i, k in enumerate(STYLE)}


def neutral_channels() -> dict:
    return {c.name: c.neutral for c in CHANNELS}


def neutral_style() -> dict:
    return {k.name: k.neutral for k in STYLE}


def knob_active(knob: Knob, kind: str, airborne: bool) -> bool:
    if kind not in knob.kinds:
        return False
    if knob.air == "both":
        return True
    return (knob.air == "air") == bool(airborne)
