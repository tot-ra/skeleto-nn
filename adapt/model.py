"""Reduced-order biomechanics and physiology that score a motor style in a situation.

This is the teacher of the network. It is differentiable (torch) and vectorised, so a batch of tens of
thousands of (body, situation, style) cases costs milliseconds and the network trains in about a
minute on a laptop CPU. Every term is a physical or physiological consequence, never a picture of the
wanted motion: nothing here says "hug yourself when cold"; it says that heat is lost through the
exposed surface and that arms against the trunk expose less of it. The behaviour is what minimises the
sum.

Units are normalised: lengths by leg length h, times by sqrt(h/g), power by m g sqrt(g h). The constants
are order-of-magnitude values chosen so that the neutral situation reproduces the planner's normal gait
(stride and speed from the Froude law, see ``planner.Walker.gait_params``); they are plausible, not
measured. ``TERMS`` is the registry; a new channel brings a new term (see README "Adding a situation").

Terms (cost per second of motion, comparable scale):
  task     not going as fast as commanded
  energy   metabolic power: speed, stride away from the natural one, crouch, wide stance, held arms
  fall     slipping (required vs available friction) and tripping (sinking feet, darkness), times the
           injury of a fall from the height of the centre of mass
  balance  sideways perturbations (dizziness, blood loss, gusts, shivering) against the base of support
  wind     uncompensated wind moment on the trunk
  cold     heat loss (temperature difference, wetness, wind, exposed surface) above heat production
           (basal, motion, shivering); worse when the core is already cold
  heat     heat production above what can be lost
  bleed    blood lost per second: flow, minus the pressure of a hand on the wound, times heart rate
  pain     pain from loading a sore leg, moving a sore arm, jolting the trunk or the head
  smoke    breathing smoke: head height times breathing rate
  flight   altitude lost (lift below weight) and the power of flapping
  prior    staying close to the normal style; knobs a body cannot express are pinned
"""
from __future__ import annotations
import torch
from .schema import CHANNELS, STYLE, CH, KN, KINDS, knob_active

G = 9.81
TERMS: list = []
WEIGHTS = dict(task=3.0, energy=1.0, fall=6.0, balance=1.0, wind=1.0, cold=1.0, heat=1.0, bleed=1.0, pain=1.0,
               smoke=1.0, flight=1.0, prior=1.0)


def term(fn):
    TERMS.append((fn.__name__.replace("t_", ""), fn))
    return fn


def relu(x): return torch.clamp(x, min=0.0)
def sig(x): return torch.sigmoid(x)


class Ctx:
    """Everything a term needs; channels and knobs are attributes by name, each a (N,) tensor."""
    def __init__(self, body: dict, ch: torch.Tensor, st: torch.Tensor):
        self.ch = ch; self.st = st
        for c in CHANNELS: setattr(self, "c_" + c.name, ch[:, CH[c.name]])
        for k in STYLE: setattr(self, "k_" + k.name, st[:, KN[k.name]])
        self.Bp, self.Qd, self.Wb = body["kind_biped"], body["kind_quadruped"], body["kind_bird"]
        self.air = body["airborne"]; self.gnd = 1.0 - self.air
        self.mass = body["mass"]; self.h = body["leg"]
        self.Fr_cmd = body["froude"]
        derive(self)


def derive(x: Ctx):
    # --- capacity: blood loss, fatigue, cold and sinking feet limit the speed that can be held
    strength = (1 - 0.65 * x.c_blood_lost) * (1 - 0.45 * x.c_fatigue) * (1 - 0.35 * x.c_core_cold)
    x.strength = strength
    fr_cap = (3.0 * strength ** 2 * (1 - 0.75 * x.c_sink) ** 2 * (1 - 0.4 * x.c_load)) * x.gnd + 4.0 * x.air
    v_cmd = torch.sqrt(x.Fr_cmd + 1e-6)                                  # normalised speed v / sqrt(g h)
    v_want = x.k_speed_frac * v_cmd * (1 - 0.35 * x.k_care)
    v_cap = torch.sqrt(fr_cap + 1e-6)
    x.v_cmd = v_cmd
    x.v = (v_want - torch.nn.functional.softplus(v_want - v_cap, beta=12.0)).clamp(min=0.0)  # soft min
    x.Fr = x.v ** 2
    # --- gait from the Froude law, scaled by the stride knob (cadence follows: f = v / stride)
    lam0 = 2.3 * torch.clamp(x.Fr, min=1e-4) ** 0.3
    x.lam = lam0 * x.k_stride
    x.step = 0.5 * x.lam
    x.f = x.v / torch.clamp(x.lam, min=0.05)
    x.h_eff = 1 - 0.32 * x.k_crouch
    x.legs4 = x.Qd
    # --- arms: a hand cannot both press a wound, hug the body and be held out
    x.armL = x.k_hug_L + x.k_press_L + 0.6 * x.k_arm_out
    x.armR = x.k_hug_R + x.k_press_R + 0.6 * x.k_arm_out
    x.arm_free = 1 - 0.5 * torch.clamp((x.k_hug_L + x.k_press_L + x.k_hug_R + x.k_press_R) * 0.5, 0, 1)
    # --- metabolic power (normalised)
    stride_cost = 4.0 * torch.log(x.k_stride) ** 2 * (0.25 + x.Fr)
    load = 1 + x.c_load
    x.P_move = (0.9 * x.Fr * load + stride_cost + 0.9 * x.k_crouch ** 2 * load * (0.4 + x.v) + 0.7 * (x.k_width - 1) ** 2
                + 0.25 * x.k_arm_out + 0.04 * (x.k_hug_L + x.k_hug_R) + 0.1 * (x.k_press_L + x.k_press_R) + 1.6 * x.c_sink * (1 + 1.5 * x.k_clearance) * x.v
                + 0.35 * x.k_clearance ** 2 * x.v + 0.3 * x.k_duty_add * x.v + 0.25 * x.k_limp ** 2 * x.v
                + 0.12 * x.k_shrug + 0.1 * x.k_head_down + 0.3 * x.k_trunk_pitch ** 2) * x.gnd
    # flight relative to the species' own tuned wingbeat (rate = amp = 1, no extra gliding): the work of
    # flapping is what keeps the bird up; a tailwind lets it coast; wet plumage and a load make it heavier
    x.lift_need = (1 + 0.25 * x.c_wet) * (1 + x.c_load) * (1 + 0.35 * relu(x.c_wind_fwd))
    flap = (x.k_flap_rate * x.k_flap_amp) ** 2 * (1 - x.k_glide)
    x.lift = flap + 0.9 * x.k_glide * relu(-x.c_wind_fwd)
    x.P_fly = (0.9 * flap + 0.8 * (x.k_flap_rate - 1) ** 2 + 0.8 * (x.k_flap_amp - 1) ** 2 + 0.5 * x.k_glide ** 2
               + 0.12 + 0.35 * x.k_fluff) * x.air
    x.P = x.P_move + x.P_fly
    x.P0 = 0.25 + 0.9 * x.Fr_cmd * 0.25                                  # reference effort for heart rate
    # --- heat balance: loss through the exposed surface, production from the basal rate, motion, shivering
    dT = relu(37.0 - x.c_air_temp) / 40.0
    wind = torch.sqrt(x.c_wind_fwd ** 2 + x.c_wind_lat ** 2) + 0.5 * x.air
    fur = x.Qd + x.Wb
    insul_fluff = x.k_fluff * fur * (0.4 - 0.3 * x.c_wet)               # wet fur and feathers insulate badly
    x.exposed = torch.clamp(1 - 0.16 * (x.k_hug_L + x.k_hug_R) * x.gnd - 0.12 * x.k_trunk_pitch.clamp(0, 0.7) / 0.7 - 0.07 * x.k_head_down
                            - 0.08 * x.k_shrug - 0.1 * x.k_crouch - 0.05 * x.k_tail_tuck - insul_fluff + 0.12 * x.k_arm_out, 0.2, 1.3)
    insulation = x.c_insulation * (1 - 0.6 * x.c_wet)                     # soaked clothing or fur keeps little warmth
    x.loss = dT * (1 + 2.2 * x.c_wet) * (1 + 0.6 * torch.sqrt(wind + 1e-6)) * x.exposed * (1.3 - insulation)
    x.prod = 0.38 + 0.5 * x.P + 1.3 * x.k_tremor * (1 - 0.4 * x.c_fatigue)
    x.hot_loss = x.loss + 0.12 * x.k_arm_out + 0.25 * x.c_wet + 0.05


@term
def t_task(x):
    return (x.v_cmd - x.v) ** 2 / (x.v_cmd ** 2 + 0.05) * torch.clamp(x.Fr_cmd * 8, 0, 1)


@term
def t_energy(x):
    return 0.6 * x.P * (1 + 2.0 * x.c_fatigue) * (1 + 1.5 * x.c_blood_lost)


@term
def t_fall(x):
    # required friction: horizontal over vertical ground force at contact ~ half step over hip height
    mu_req = 0.5 * x.step / x.h_eff * (1 - 0.4 * x.k_foot_flat) * (1 + 0.25 * x.Fr) * (1 - 0.15 * x.legs4) * (1 - 0.4 * x.k_care)
    mu_req = mu_req + 0.05 * relu(x.k_width - 1) + 0.06 * torch.abs(x.k_trunk_roll - 0.3 * x.c_wind_lat) + 0.08 * x.Fr * x.k_tremor
    mu = x.c_friction
    p_slip = sig((mu_req - mu) / 0.02)
    recover = torch.clamp(0.25 + 0.3 * x.k_arm_out * (x.Bp + x.Wb) + 0.35 * (x.k_width - 1) + 1.8 * x.k_duty_add + 0.3 * x.k_crouch
                          + 0.4 * x.legs4 - 0.4 * x.c_dizzy, 0.0, 0.97)
    # tripping: the foot must clear the ground it sinks into, and in the dark the unseen kerbs
    c_req = 0.55 * x.c_sink + 0.12 * x.c_dark * (1 - 0.6 * x.k_care) * (1 - 0.4 * x.k_head_down)
    clr = 0.08 + 0.35 * x.k_clearance
    p_trip = sig((c_req - clr) / 0.02) * torch.clamp(0.3 + x.v, 0, 2) * x.k_stride
    steps = 2 * x.f + 0.2
    h_com = x.h_eff * (1 + 0.8 * x.Bp)                                    # a biped falls from higher up
    injury = h_com ** 1.5 * (1 + x.c_load) * (1 - 0.35 * x.k_arm_out * x.Bp) * (1 + 0.5 * x.c_blood_lost)
    x.p_slip, x.p_trip, x.mu_req = p_slip, p_trip, mu_req
    return steps * (p_slip * (1 - recover) + 0.6 * p_trip) * injury * x.gnd


@term
def t_balance(x):
    sigma = (0.55 * x.c_dizzy + 0.5 * x.c_blood_lost + 0.25 * x.k_tremor + 0.25 * torch.abs(x.c_wind_lat) + 0.6 * x.c_friction.clamp(max=0.3).neg().add(0.3)
             + 0.08 * x.Fr * (1 - x.arm_free) * x.Bp + 0.12 * torch.abs(x.k_limp) + 0.1 * x.c_load + 0.05 * x.v)
    margin = x.k_width * (1 + 0.5 * x.k_arm_out * (x.Bp + x.Wb)) * (1 + 1.5 * x.k_duty_add) * (1 + 0.25 * x.k_crouch) * (1 + 1.5 * x.Qd)
    return 2.0 * sigma ** 2 / margin ** 2 * x.gnd


@term
def t_wind(x):
    k = 0.5 * (1.0 / (x.mass / 70.0) ** (1 / 3)).clamp(0.4, 3.0) * x.h_eff
    lat = x.c_wind_lat * torch.abs(x.c_wind_lat) * k - x.k_trunk_roll
    fwd = x.c_wind_fwd * torch.abs(x.c_wind_fwd) * k - (x.k_trunk_pitch + 0.25 * x.k_crouch)
    return (6.0 * lat ** 2 + 1.5 * fwd ** 2 * (x.Bp + 0.3)) * x.gnd


@term
def t_cold(x):
    net = relu(x.loss - x.prod - 0.12)
    return 6.0 * net ** 2 * (1 + 3.0 * x.c_core_cold) + 2.5 * x.c_core_cold * relu(1 - x.prod / 0.9) * 0.3


@term
def t_heat(x):
    return 4.0 * relu(x.prod - x.hot_loss - 0.75) ** 2


def _reach(x, side: float):
    """How well a hand reaches the wound point: the front is easy, the back and the far side are hard,
    a low wound needs the trunk bent (the hunch costs energy elsewhere)."""
    wx, wy, wz = x.c_wound_x, x.c_wound_y, x.c_wound_z
    far = relu(-side * wy)                                                 # wound on the other side
    r = 1 - 0.7 * relu(-wx) * (1 + far) - 0.35 * far * relu(wx) - 1.2 * relu(0.95 - wz - 0.0) * 0 - 1.5 * relu(wz - 0.97)
    low = relu(0.5 - wz)                                                   # below the hip: bend to reach it
    r = r * torch.clamp(1 - 2.2 * relu(low - 0.9 * (x.k_trunk_pitch + 0.6 * x.k_crouch)), 0, 1)
    return torch.clamp(r, 0, 1)


@term
def t_bleed(x):
    useL = (1 - x.c_pain_arm_L) * (1 - 0.5 * x.k_tremor); useR = (1 - x.c_pain_arm_R) * (1 - 0.5 * x.k_tremor)
    a = x.k_press_L * _reach(x, 1.0) * useL; b = x.k_press_R * _reach(x, -1.0) * useR
    eff = torch.maximum(a, b) + 0.1 * torch.minimum(a, b)                 # a second hand adds little pressure
    x.press_eff = eff * x.Bp
    heart = 0.55 + 0.45 * x.P / x.P0.clamp(min=0.2)
    rate = x.c_bleed * (1 - 0.8 * x.press_eff) * heart
    return 4.0 * rate * (1 + 3.0 * x.c_blood_lost)


@term
def t_pain(x):
    shareL = 0.5 - 0.3 * x.k_limp
    impact = (0.25 + 2.0 * x.v ** 1.5) * (1 - 0.3 * x.k_foot_flat) * (1 - 0.3 * x.k_crouch) * x.k_stride
    legs = 2 * (x.c_pain_leg_L * shareL + x.c_pain_leg_R * (1 - shareL)) * impact
    swing = (0.25 + x.Fr) * x.Bp
    armL = x.c_pain_arm_L * (swing * relu(1 - x.k_hug_L - x.k_press_L) + 0.8 * x.k_arm_out + 0.6 * x.k_press_L)
    armR = x.c_pain_arm_R * (swing * relu(1 - x.k_hug_R - x.k_press_R) + 0.8 * x.k_arm_out + 0.6 * x.k_press_R)
    guard = 1 - 0.25 * torch.clamp(x.k_hug_L + x.k_hug_R, 0, 1.5)
    torso = x.c_pain_torso * (0.3 + 2.0 * x.Fr) * (1.1 - 0.5 * x.k_trunk_pitch.clamp(0, 0.6) / 0.6) * guard
    head = x.c_pain_head * (impact + 0.6 * x.k_tremor) * (1 - 0.2 * x.k_head_down)
    arms_air = (x.c_pain_arm_L + x.c_pain_arm_R) * x.k_flap_rate ** 2 * x.k_flap_amp ** 2 * (1 - x.k_glide) * x.air
    budget = 3.0 * (relu(x.armL - 1) ** 2 + relu(x.armR - 1) ** 2)
    return 2.5 * (legs + armL + armR + torso + head) * x.gnd + 2.0 * arms_air + budget


@term
def t_smoke(x):
    head_h = 1 - 0.6 * x.k_crouch - 0.25 * x.k_trunk_pitch - 0.1 * x.k_head_down
    breath = 0.5 + x.P
    return 3.0 * x.c_smoke * head_h * breath


@term
def t_flight(x):
    short = relu(x.lift_need * (1 + 0.3 * x.c_blood_lost) - x.lift * x.strength.clamp(min=0.3))
    over = relu(x.lift - 1.6 * x.strength)
    return (12.0 * short ** 2 + 3.0 * over ** 2) * x.air


PRIOR = torch.tensor([1.0] * len(STYLE))
PRIOR_W = 0.05


def prior(x, body):
    lo = torch.tensor([k.lo for k in STYLE]); hi = torch.tensor([k.hi for k in STYLE]); neu = torch.tensor([k.neutral for k in STYLE])
    d = (x.st - neu) / (hi - lo)
    act = active_mask(body)
    return (PRIOR_W * (d ** 2 * act).sum(1) + 4.0 * (d ** 2 * (1 - act)).sum(1))


def active_mask(body) -> torch.Tensor:
    n = body["kind_biped"].shape[0]
    m = torch.zeros(n, len(STYLE))
    for j, k in enumerate(STYLE):
        for kind in KINDS:
            for air in (0.0, 1.0):
                if knob_active(k, kind, air > 0.5):
                    m[:, j] += body["kind_" + kind] * (body["airborne"] if air > 0.5 else 1 - body["airborne"])
    return m.clamp(0, 1)


def evaluate(body: dict, ch: torch.Tensor, st: torch.Tensor, weights: dict | None = None):
    """Returns (total cost (N,), {term: (N,)}, ctx)."""
    w = dict(WEIGHTS); w.update(weights or {})
    x = Ctx(body, ch, st)
    parts = {name: fn(x) for name, fn in TERMS}
    parts["prior"] = prior(x, body)
    total = sum(w.get(k, 1.0) * v for k, v in parts.items())
    return total, parts, x
