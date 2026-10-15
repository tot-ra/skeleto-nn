"""Body state over time: what an interactive environment does to a creature, step by step.

Conditions in ``conditions.py`` describe the surroundings at an instant (the ground under the feet, the
air). Some channels are not surroundings but the body's own state, and they change with exposure:

* ``wet``        rises in rain (unless sheltered) and in water, dries with warmth, wind and a fire;
* ``core_cold``  follows the heat balance of the current style (the same formula the network was trained
                 on): a soaked person in a frosty wind cools within minutes, and hugging, hunching and
                 shivering slow it, which the network chooses because they slow it;
* ``blood_lost`` grows with the bleeding rate, less when a hand presses the wound and more with effort;
                 wounds clot over minutes, faster under pressure;
* ``fatigue``    grows with effort above a sustainable level and recovers at rest;
* ``pain_*``     wounds and blows add pain that fades.

So the loop is: environment -> channels -> network -> style -> (heat, blood, effort) -> state -> channels.
Times are seconds; constants are plausible orders of magnitude (cooling in minutes, a serious wound
lethal in about a quarter of an hour unattended), tuned for gameplay readability, not for medicine.
The game runs the same equations in GDScript (``motion_physiology.gd``).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from .conditions import REGIONS, REGION_PAIN

PAIN_CH = ("pain_leg_L", "pain_leg_R", "pain_arm_L", "pain_arm_R", "pain_torso", "pain_head")


def heat_balance(ch: dict, st: dict, kind: str, airborne: bool, power: float) -> tuple[float, float]:
    """(loss, production) in the units of ``model.py`` (same formula, numpy)."""
    fur = 1.0 if kind in ("quadruped", "bird") else 0.0
    gnd = 0.0 if airborne else 1.0
    dT = max(0.0, 37.0 - ch["air_temp"]) / 40.0
    wind = math.hypot(ch["wind_fwd"], ch["wind_lat"]) + (0.5 if airborne else 0.0)
    insul_fluff = st.get("fluff", 0.0) * fur * (0.4 - 0.3 * ch["wet"])
    pitch = min(max(st.get("trunk_pitch", 0.0), 0.0), 0.7) / 0.7
    exposed = 1 - 0.16 * (st.get("hug_L", 0) + st.get("hug_R", 0)) * gnd - 0.12 * pitch - 0.07 * st.get("head_down", 0) \
        - 0.08 * st.get("shrug", 0) - 0.1 * st.get("crouch", 0) - 0.05 * st.get("tail_tuck", 0) - insul_fluff + 0.12 * st.get("arm_out", 0)
    exposed = min(1.3, max(0.2, exposed))
    insulation = ch["insulation"] * (1 - 0.6 * ch["wet"])
    loss = dT * (1 + 2.2 * ch["wet"]) * (1 + 0.6 * math.sqrt(wind + 1e-6)) * exposed * (1.3 - insulation)
    prod = 0.38 + 0.5 * power + 1.3 * st.get("tremor", 0.0) * (1 - 0.4 * ch["fatigue"])
    return loss, prod


@dataclass
class Wound:
    region: str            # a key of conditions.REGIONS without the "_S"
    side: str = "L"
    rate: float = 0.6      # bleeding rate 0..1
    pain: float = 0.5

    def point(self):
        key = self.region + "_S" if self.region + "_S" in REGIONS else self.region
        x, y, z = REGIONS[key]
        return (x, -y if (key.endswith("_S") and self.side == "R") else y, z)

    def pain_channel(self):
        ch = REGION_PAIN[self.region]
        return ch + "_" + self.side if ch in ("pain_leg", "pain_arm") else ch


@dataclass
class Exposure:
    """What the surroundings do right now (the game fills this from weather, water and fire)."""
    rain: float = 0.0          # 0..1 rain or wet snow intensity
    shelter: float = 0.0       # 0..1 roof, cloak hood
    immersion: float = 0.0     # 0..1 how much of the body is in water
    fire: float = 0.0          # 0..1 close to a hearth or a fire


@dataclass
class BodyState:
    mass: float = 70.0
    wet: float = 0.0
    core_cold: float = 0.0
    blood_lost: float = 0.0
    fatigue: float = 0.0
    pain: dict = field(default_factory=lambda: {k: 0.0 for k in PAIN_CH})
    wounds: list = field(default_factory=list)

    def wound_channels(self) -> dict:
        out = {"bleed": 0.0}
        if self.wounds:
            w = max(self.wounds, key=lambda w: w.rate)
            out["bleed"] = w.rate
            out["wound_x"], out["wound_y"], out["wound_z"] = w.point()
        return out

    def channels(self) -> dict:
        ch = dict(wet=self.wet, core_cold=self.core_cold, blood_lost=self.blood_lost, fatigue=self.fatigue)
        ch.update(self.pain)
        for w in self.wounds:
            k = w.pain_channel(); ch[k] = max(ch.get(k, 0.0), w.pain)
        ch.update(self.wound_channels())
        return ch

    def step(self, dt: float, env: dict, exp: Exposure, st: dict, kind: str, airborne: bool, power: float, press_eff: float = 0.0):
        """Advance by dt seconds. ``env`` holds the surroundings channels (air_temp, wind, insulation),
        ``st`` the style in use, ``power`` the normalised effort, ``press_eff`` how well a hand presses."""
        size = (self.mass / 70.0) ** (1 / 3)
        # wetness
        dry = 0.0012 * (1 + max(env["air_temp"], 0.0) / 12.0) * (1 + 2 * math.hypot(env["wind_fwd"], env["wind_lat"])) * (1 + 4 * exp.fire)
        self.wet += (exp.rain * (1 - exp.shelter) * 0.012 + exp.immersion * 0.5) * (1 - self.wet) * dt - dry * self.wet * dt
        self.wet = min(1.0, max(0.0, self.wet))
        # heat
        ch = dict(env); ch.update(wet=self.wet, fatigue=self.fatigue)
        if exp.fire > 0: ch["air_temp"] = ch["air_temp"] + 30.0 * exp.fire
        loss, prod = heat_balance(ch, st, kind, airborne, power)
        tau = 420.0 * size
        self.core_cold = min(1.0, max(0.0, self.core_cold + (loss - prod - 0.12) / tau * dt))
        # blood
        heart = 0.55 + 0.45 * power / 0.3
        for w in self.wounds:
            self.blood_lost += w.rate * (1 - 0.8 * press_eff) * heart * dt / 900.0
            w.rate *= math.exp(-dt / (240.0 / (1 + 2.0 * press_eff)))
            w.pain *= math.exp(-dt / 1800.0)
        self.wounds = [w for w in self.wounds if w.rate > 0.02 or w.pain > 0.05]
        self.blood_lost = min(1.0, self.blood_lost)
        # effort
        self.fatigue = min(1.0, max(0.0, self.fatigue + (power - 0.45) / 300.0 * dt if power > 0.45 else self.fatigue - 0.004 * dt))
        for k in self.pain: self.pain[k] *= math.exp(-dt / 600.0)

    def wound(self, region, side="L", rate=0.6, pain=0.5):
        self.wounds.append(Wound(region, side, rate, pain))
