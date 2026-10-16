"""Context-aware jumping: one small network decides whether and how a body crosses what lies ahead.

A wall to hop or vault, a higher surface to jump up onto or pull up onto, a drop to jump down or to lower
oneself from, a gap, a quay with water below: none of these is a case in code. The network sees

* the body (biped or quadruped, mass, leg length, approach speed, leg power as a jump gain, how much it
  wants to get across),
* the same 25 situation channels as the style network (``schema.CHANNELS``): ice lowers the push the feet
  can give and makes landings slip, a sore leg or blood loss weakens the take-off, a sore arm cannot pull,
  darkness and dizziness make the landing uncertain, a load is heavy in the air and deadly in deep water,
  cold water matters less to a body that is already soaked,
* a profile of the ground ahead: ``K`` samples every quarter leg length, each the height of the top of
  whatever is solid there and the depth of water standing on it,

and answers with a landing place (one of the samples, or "refuse": stay and do not jump) and a jump
program: where to take off, how high the arc goes, hands on the obstacle (vault or pull up), legs tucked,
how deep the landing gives, a landing roll, lowering oneself from the edge before letting go, arms out,
trunk lean. Hop, vault, mantle, jumping down, a hang-drop, a jump into the water and a refusal are what
comes out; the words are only labels for the result (``label``).

It is trained like the style network: by gradient descent through a differentiable model of the
consequences (``cost``): can the legs (and arms) power that arc, does the body clear what is in between,
does the take-off slip, what force does the landing put through the legs (the drop height over the
stroke the knees, a roll or the water give), does the body slip or run off a narrow top after landing,
what does cold or deep water do to it, what does it cost in effort and time, and how much ground it gains.
Lengths are in leg lengths, g = 1, speeds in sqrt(g * leg).

    python -m adapt.leap                      # train (CPU), writes results/leap_net.json
    python -m adapt.leap --check              # compare with a per-case optimisation
    python -m adapt.leap --probe              # what it does in named scenes
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
from .schema import CHANNELS, CH

K = 20                          # profile samples
DX = 0.25                       # spacing in leg lengths: the profile reaches 5 leg lengths ahead
XS = [DX * (i + 1) for i in range(K)]
LEAP_KINDS = ("biped", "quadruped")
BODY_IN = ["kind_biped", "kind_quadruped", "log_mass", "log_leg", "speed", "log_gain", "urgency"]
HANG = 2.1                      # how much lower the feet are when hanging from the hands at an edge, leg lengths
Z_CLIP = 12.0                   # deeper drops are treated as this deep


class Knob:
    def __init__(self, name, lo, hi, neutral, kinds, doc):
        self.name, self.lo, self.hi, self.neutral, self.kinds, self.doc = name, lo, hi, neutral, kinds, doc


B_, Q_ = ("biped",), ("biped", "quadruped")
KNOBS = [
    Knob("take", 0.0, 1.0, 0.0, Q_, "a step or two on (leg lengths) before taking off"),
    Knob("apex", -0.6, 2.0, 0.2, Q_, "height of the arc above the highest of take-off, landing and what is passed over"),
    Knob("hands", 0.0, 1.0, 0.0, B_, "hands on the obstacle or the edge: vault over it or pull up onto it"),
    Knob("tuck", 0.0, 1.0, 0.0, Q_, "legs drawn up in the air"),
    Knob("flex", 0.0, 1.0, 0.3, Q_, "how deep the legs give on landing"),
    Knob("roll", 0.0, 1.0, 0.0, B_, "roll on landing to spread the impact"),
    Knob("lower", 0.0, 1.0, 0.0, B_, "lower oneself from the edge on the hands before letting go"),
    Knob("arms_out", 0.0, 1.0, 0.0, B_, "arms out for balance on landing"),
    Knob("lean", -0.2, 0.6, 0.1, Q_, "trunk lean forward at landing, rad"),
]
KI = {k.name: i for i, k in enumerate(KNOBS)}
MARGIN = 0.15
HIDDEN = (96, 96)


CAND = ["x", "z", "water", "top_before", "low_before", "water_before", "first_edge", "room", "next_step", "top_at"]


def input_names():
    return list(BODY_IN) + [c.name for c in CHANNELS] + ["c_" + f for f in CAND]


def output_names():
    return ["score"] + [k.name for k in KNOBS]


# ---------------------------------------------------------------------------------------------- encoding
def encode_channels_t(ch: torch.Tensor) -> torch.Tensor:
    enc = []
    for j, c in enumerate(CHANNELS):
        v = ch[:, j]
        e = torch.log(v.clamp(min=1e-3) / c.neutral) / c.scale if c.log else (v - c.neutral) / c.scale
        enc.append(e.clamp(-3, 3))
    return torch.stack(enc, 1)


def encode_shared_t(kind, mass, leg, speed, gain, urg, ch):
    body = torch.stack([(kind == 0).float(), (kind == 1).float(), torch.log(mass / 30.0) / 2.0, torch.log(leg / 0.5),
                        speed.clamp(0, 3), torch.log(gain), urg], 1)
    return torch.cat([body, encode_channels_t(ch)], 1)


def cand_features(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Geometry of each landing place k, from the profile (leg lengths): the same few numbers for any
    scene. (B, K) -> (B, K, len(CAND)); also used for a single profile (K,) -> (K, len(CAND))."""
    single = z.ndim == 1
    z = np.atleast_2d(z).astype(np.float32); w = np.atleast_2d(w).astype(np.float32)
    n = z.shape[0]; xs = np.array(XS, np.float32)
    f = np.zeros((n, K, len(CAND)), np.float32)
    feat = (np.abs(z) > 0.3) | (w > 0.3)
    for k in range(K):
        zb = np.concatenate([np.zeros((n, 1), np.float32), z[:, :k]], 1)        # the ground passed over, from the feet on
        wb = np.concatenate([np.zeros((n, 1), np.float32), w[:, :k]], 1)
        top = zb.max(1); low = zb.min(1)
        arg = zb.argmax(1)
        x_top = np.where(top > 0.3, np.where(arg > 0, xs[np.maximum(arg - 1, 0)], 0.0), 0.0)
        fe = feat[:, :k + 1]
        first = np.where(fe.any(1), xs[np.argmax(fe, 1)], xs[k])
        same = np.abs(z[:, k + 1:] - z[:, k:k + 1]) < 0.3
        run = np.cumprod(same, 1).sum(1) if k + 1 < K else np.zeros(n)
        room = run * DX + np.where(run == K - 1 - k, 1.0, 0.0)
        nxt = (z[:, k + 1] if k + 1 < K else z[:, k]) - z[:, k]
        f[:, k] = np.stack([np.full(n, xs[k] / 5.0), np.clip(z[:, k] / 3, -3, 3), np.clip(w[:, k] / 2, 0, 3), np.clip(top / 3, -3, 3),
                            np.clip(low / 3, -3, 3), np.clip(wb.max(1) / 2, 0, 3), first / 5.0, np.clip(room / 5, 0, 1.5),
                            np.clip(nxt / 3, -3, 3), x_top / 5.0], 1)
    return f[0] if single else f


def encode_shared_np(kind: str, mass, leg, speed, gain, urgency, ch: dict) -> np.ndarray:
    from .net import encode_channels
    body = [1.0 if kind == "biped" else 0.0, 1.0 if kind == "quadruped" else 0.0, math.log(mass / 30.0) / 2.0,
            math.log(leg / 0.5), min(3.0, max(0.0, speed)), math.log(gain), urgency]
    return np.concatenate([np.array(body, np.float32), encode_channels(ch)]).astype(np.float32)


# ---------------------------------------------------------------------------------------------- the teacher
def relu(x): return torch.clamp(x, min=0.0)
def sig(x): return torch.sigmoid(x)


def _c(ch, name): return ch[:, CH[name]]


def cost(kind, mass, speed, gain, urg, ch, z, w, kn, parts=False):
    """Cost of landing at each profile sample with the knobs ``kn`` (B, nK); returns (B, K) and the
    cost of refusing (B,). Everything is per body in leg lengths, g = 1."""
    Bn = z.shape[0]
    Bp = (kind == 0).float()[:, None]; Qd = (kind == 1).float()[:, None]
    knv = kn if kn.dim() == 3 else kn[:, None, :].expand(Bn, K, kn.shape[-1])
    g = lambda n: knv[:, :, KI[n]]
    take, apex, hands, tuck, flex, roll, lower, arms, lean = (g(n) for n in ("take", "apex", "hands", "tuck", "flex", "roll", "lower", "arms_out", "lean"))
    xs = torch.tensor(XS)[None, :]                                      # (1, K)
    zc = z.clamp(min=-Z_CLIP)
    # --- the body's state
    mu = _c(ch, "friction")[:, None]
    legpain = torch.maximum(_c(ch, "pain_leg_L"), _c(ch, "pain_leg_R"))[:, None]
    armpain = torch.maximum(_c(ch, "pain_arm_L"), _c(ch, "pain_arm_R"))[:, None]
    load = _c(ch, "load")[:, None]; fat = _c(ch, "fatigue")[:, None]; bl = _c(ch, "blood_lost")[:, None]
    dark = _c(ch, "dark")[:, None]; dizzy = _c(ch, "dizzy")[:, None]; sink = _c(ch, "sink")[:, None]
    wet = _c(ch, "wet")[:, None]; temp = _c(ch, "air_temp")[:, None]; ins = _c(ch, "insulation")[:, None]
    strength = (1 - 0.65 * bl) * (1 - 0.45 * fat) * (1 - 0.35 * _c(ch, "core_cold")[:, None])
    legs = strength * (1 - 0.55 * legpain) * (1 - 0.35 * load) * (1 - 0.5 * sink)
    cap = gain[:, None] * legs                                          # vertical take-off speed the legs can give
    vtot = 1.5 * cap
    v0 = speed[:, None]
    noise = 0.06 + 0.12 * dark + 0.15 * dizzy + 0.05 * fat              # how well the jump is judged and executed
    # --- run-up: the ground up to the take-off must be walkable
    before = sig((take - xs) / 0.05)                                     # (B, K): samples run over
    approach = (before * (3.0 * relu(zc.abs() - 0.3) + 1.0 * relu(w - 0.6))).sum(1, keepdim=True)
    # --- landing at sample k
    xL = xs.expand(Bn, K); zL = zc; wL = w
    D = xL - take                                                        # (B, K)
    behind = 6.0 * relu(0.3 - D) / 0.3
    drop = relu(-zL)
    lowr = lower * Bp * sig((drop - 0.3) / 0.08)                         # lowering oneself only means something above a drop
    zS = -lowr * torch.minimum(drop, torch.full_like(drop, HANG))        # hanging from the edge lowers the start
    hang_far = 3.0 * lower * Bp * relu(D - 0.6)                          # one lets go straight down from the edge
    # what lies under the flight, and where the hands can go
    xi = xs[:, None, :]                                                  # (1, 1, K): every sample against every landing
    inflight = sig((xi - take[:, :, None] - 0.1) / 0.04) * sig((xL[:, :, None] - 0.1 - xi) / 0.04)    # (B, K, K)
    zi = zc[:, None, :]
    ztop = (zi * inflight + (-20.0) * (1 - inflight)).amax(2)           # the highest thing passed over
    zmax = torch.maximum(ztop, zL)                                       # top within reach of the hands
    zmin = (zi * inflight + 20.0 * (1 - inflight)).amin(2)
    za = torch.maximum(torch.maximum(zS, zL), ztop) + apex               # apex: clearance over the highest point
    Tu = torch.sqrt(2 * relu(za - zS) + 1e-4); Td = torch.sqrt(2 * relu(za - zL) + 1e-4); T = Tu + Td
    vh = relu(D) / T
    t_i = ((xi - take[:, :, None]) / vh[:, :, None].clamp(min=1e-3)).clamp(min=0)
    t_i = torch.minimum(t_i, T[:, :, None])
    y_i = zS[:, :, None] + Tu[:, :, None] * t_i - 0.5 * t_i ** 2         # height of the feet over each sample
    hand_ok = sig((zmax - 0.25) / 0.06) * sig((2.4 - zmax) / 0.08) * Bp
    arm = strength * (1 - 0.7 * armpain) * (1 - 0.4 * load)
    h_eff = hands * hand_ok * arm
    clear = y_i + 0.35 * tuck[:, :, None] + 0.55 * h_eff[:, :, None] - zi - 0.06
    hit = (inflight * torch.nn.functional.softplus(-clear * 20) / 20).sum(2) * 6.0 * (0.6 + vh)
    # can the legs (and arms) power it?
    pull = 1.3 * h_eff
    rise = relu(za - zS - pull) * (1 - lowr)
    vz = torch.sqrt(2 * rise + 1e-4)
    vh_cap = torch.sqrt(relu(vtot ** 2 - vz ** 2) + 1e-4) + 0.9 * v0 + 0.3 * h_eff
    m_v = cap - vz; m_h = vh_cap - vh
    margin = -0.05 * torch.logsumexp(torch.stack([-m_v, -m_h], 0) / 0.05, 0)
    p_short = sig(-margin / noise)
    dvh = relu(vh - 0.9 * v0)
    mu_t = dvh / (vz + 0.4) + 0.05
    p_slip_t = sig((mu_t - mu) / 0.03) * (1 - lowr)
    p_fail = 1 - (1 - p_short) * (1 - p_slip_t)
    wmin = (w[:, None, :] * inflight).amax(2)
    fail_inj = 0.6 + 0.5 * relu(zmax.clamp(max=3)) + 0.6 * relu(relu(-zmin) - 2.5 * wmin).clamp(max=Z_CLIP) ** 0.8
    # landing
    H = relu(za - zL)
    in_water = sig((wL - 0.15) / 0.04)
    Hs = relu(H - 2.6 * wL)                                              # the water brakes the fall before the bottom
    room_same = sig((0.3 - (zc[:, None, :] - zL[:, :, None]).abs()) / 0.05)   # (B, landing, j)
    after = (xi > xL[:, :, None]).float()
    run = torch.cumprod(room_same * after + (1 - after), 2) * after
    room = (run.sum(2) * DX + (xL >= XS[-1] - 1e-6).float() * 1.0)
    roll_ok = sig((room - 0.9) / 0.1) * (1 - in_water) * Bp
    stroke = 0.05 + 0.3 * flex + 0.5 * roll * roll_ok
    F = 1 + Hs / stroke
    tol = 10.0 * (1 - 0.5 * legpain) * (1 - 0.25 * fat) * (1 + 0.4 * Qd) * ((mass[:, None] / 70.0) ** (-1 / 3)).clamp(0.7, 3.0)
    inj = 4.0 * sig((F - tol) / (0.12 * tol)) + 0.02 * F * (1 + load) + 0.15 * relu(H - 10) * in_water
    mu_l = vh / (torch.sqrt(2 * H + 1e-4) + 1.0) * (1 - 0.4 * flex) * (1 - 0.3 * lean) * (1 - 0.6 * roll * roll_ok) + 0.05
    p_slip_l = sig((mu_l - mu) / 0.03) * (1 - in_water)
    over = vh * 0.5 * (1 - 0.5 * flex) + 0.9 * roll
    p_off = sig((over - room) / 0.08) * (1 - in_water)
    z_next = torch.cat([zc[:, 1:], zc[:, -1:]], 1)
    off_inj = 0.4 + 0.4 * relu(zL - z_next).clamp(max=6)
    bal = (0.5 * dizzy + 0.5 * dark + 0.3 * bl + 0.6 * relu(0.35 - mu)) * (1 - 0.5 * arms * Bp) * (0.3 + vh) / (room + 0.3)
    lean_bad = 0.3 * (lean - 0.25 * vh.clamp(max=1.5)) ** 2
    land = inj + p_slip_l * 0.8 * (1 + load) + p_off * off_inj + 0.5 * bal + lean_bad
    # water: cold, and deep water with a load, a useless arm or failing strength
    deep = sig((wL - 1.2) / 0.15)
    water = in_water * (relu(14 - temp) / 20 * (1 - 0.4 * ins) * (1 - wet) * 1.5) \
        + deep * (0.08 + 3.0 * load + 2.0 * armpain + 1.5 * bl + 1.0 * fat + 0.25 * ins)
    effort = (0.12 * (vz ** 2 + dvh ** 2) * (1 + load) + 0.15 * hands + 0.3 * lowr + 0.06 * tuck + 0.04 * flex + 0.3 * roll + 0.05 * arms
              + 0.08 * T + 0.35 * take) * (1 + fat)
    progress = urg[:, None] * 1.2 * xL.clamp(max=4.0) / 4.0
    c = approach + behind + hang_far + hit + p_fail * fail_inj * (1 + load) + (1 - p_fail) * land + water + effort - progress
    # knobs a body cannot use are pinned
    pin = torch.zeros(Bn, K)
    for j, k in enumerate(KNOBS):
        if "quadruped" not in k.kinds: pin = pin + 4.0 * Qd * ((knv[:, :, j] - k.neutral) / (k.hi - k.lo)) ** 2
    c = c + pin
    c_ref = 0.3 + 0.0 * urg                                              # staying: nothing gained, nothing risked
    if parts:
        return c, c_ref, dict(p_fail=p_fail, inj=inj, hit=hit, water=water, effort=effort, F=F, H=H, vh=vh, T=T, za=za, zS=zS, room=room)
    return c, c_ref


# ---------------------------------------------------------------------------------------------- scenes
def random_scenes(n, rng: np.random.Generator):
    """Ground profiles in leg lengths: flat, a step up, a drop (with or without water below), a wall with
    the ground beyond higher or lower, a gap, a bank down to water; some combined, some rough."""
    z = np.zeros((n, K), np.float32); w = np.zeros((n, K), np.float32); thin = np.zeros((n, K), bool)
    xs = np.array(XS)
    kind = rng.choice(6, n, p=[0.08, 0.2, 0.24, 0.26, 0.14, 0.08])
    edge = rng.uniform(0.2, 2.4, n)
    for i in range(n):
        e = edge[i]; k = kind[i]; beyond = xs > e
        if k == 1:                                                       # step up
            z[i, beyond] = rng.uniform(0.15, 3.6)
        elif k == 2:                                                     # drop, maybe into water
            d = math.exp(rng.uniform(math.log(0.2), math.log(16.0)))
            z[i, beyond] = -d
            if rng.uniform() < 0.5: w[i, beyond] = min(d + 3.0, math.exp(rng.uniform(math.log(0.15), math.log(6.0))))
        elif k == 3:                                                     # wall
            hgt = rng.uniform(0.15, 3.2); th = rng.uniform(0.15, 1.4)
            on = beyond & (xs <= e + th); z[i, on] = hgt; thin[i, on] = th < 0.8
            after = xs > e + th
            z[i, after] = rng.choice([0.0, rng.uniform(-3.0, 0.6)])
            if rng.uniform() < 0.15: w[i, after] = rng.uniform(0.1, 3.0); z[i, after] = min(z[i, after][0], 0.0) - w[i, after][0] * 0.5
        elif k == 4:                                                     # gap
            wd = rng.uniform(0.2, 3.5); dp = rng.uniform(0.3, 10.0)
            on = beyond & (xs <= e + wd); z[i, on] = -dp
            if rng.uniform() < 0.4: w[i, on] = rng.uniform(0.2, dp + 1)
            z[i, xs > e + wd] = rng.uniform(-0.6, 0.6)
        elif k == 5:                                                     # rough ground and stairs
            z[i] = np.cumsum(rng.normal(0, 0.12, K) * (rng.uniform(0, 1, K) < 0.5))
        if rng.uniform() < 0.1: w[i] = np.maximum(w[i], rng.uniform(0.02, 0.25))      # puddles
    return z, w, thin


def random_bodies(n, rng):
    kind = (rng.uniform(0, 1, n) < 0.35).astype(np.int64)
    mass = np.where(kind == 0, np.exp(rng.uniform(math.log(15), math.log(130), n)), np.exp(rng.uniform(math.log(2.5), math.log(650), n)))
    leg = np.where(kind == 0, np.exp(rng.uniform(math.log(0.45), math.log(1.05), n)), np.exp(rng.uniform(math.log(0.14), math.log(1.1), n)))
    gain = np.where(kind == 0, np.exp(rng.uniform(math.log(0.6), math.log(1.4), n)), np.exp(rng.uniform(math.log(0.9), math.log(3.6), n)))
    speed = np.where(rng.uniform(0, 1, n) < 0.35, 0.0, rng.uniform(0, 1.6, n) ** 1.3)
    urg = rng.uniform(0, 1, n) ** 0.7
    return kind, mass, leg, speed, gain, urg


_SCENES = {}


def scene_pool(n=200000, seed=7):
    """Profiles with their per-place features, computed once (the features are the slow part)."""
    if (n, seed) not in _SCENES:
        z, w, thin = random_scenes(n, np.random.default_rng(seed))
        _SCENES[(n, seed)] = (z, w, thin, cand_features(z, w))
    return _SCENES[(n, seed)]


def make_batch(n, rng, pool, scenes=None):
    from .train import jitter_channels
    kind, mass, leg, speed, gain, urg = random_bodies(n, rng)
    ch = jitter_channels(pool[rng.integers(0, len(pool), n)], rng)
    if scenes is None:
        z, w, thin = random_scenes(n, rng); f = cand_features(z, w)
    else:
        idx = rng.integers(0, len(scenes[0]), n); z, w, thin, f = (a[idx] for a in scenes)
    make_batch.feats = torch.tensor(f)
    mask = ~(thin & (rng.uniform(0, 1, (n, K)) < 0.5))                  # some thin wall tops cannot be stood on
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    return t(kind, torch.long), t(mass), t(leg), t(speed), t(gain), t(urg), t(ch), t(z), t(w), t(mask, torch.bool)


# ---------------------------------------------------------------------------------------------- network
class Net(torch.nn.Module):
    """One small MLP applied to every landing place: (body, channels, that place's geometry) -> a score
    against staying (score 0) and the jump program for landing there."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        sizes = [len(input_names()), *hidden, 1 + len(KNOBS)]
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.05)
            b = self.layers[-1].bias; b.zero_()
            for j, k in enumerate(KNOBS):
                s_ = neutral_sig(k); b[1 + j] = math.log(s_ / (1 - s_))

    def forward(self, shared, feats, noise=0.0):
        Bn = shared.shape[0]
        x = torch.cat([shared[:, None, :].expand(Bn, K, shared.shape[1]), feats], 2)
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1: x = torch.tanh(x)
        score, kz = x[:, :, 0], x[:, :, 1:]
        self.kz = kz
        if noise > 0: kz = kz + noise * torch.randn_like(kz)
        lo = torch.tensor([k.lo for k in KNOBS]); hi = torch.tensor([k.hi for k in KNOBS]); m = MARGIN * (hi - lo)
        raw = lo - m + (hi - lo + 2 * m) * torch.sigmoid(kz)
        self.overshoot = (raw - raw.clamp(lo, hi)) / (hi - lo)
        logits = torch.cat([torch.zeros(Bn, 1), score], 1)
        return logits, raw + (torch.maximum(torch.minimum(raw, hi), lo) - raw).detach()


def neutral_sig(k):
    return (k.neutral - k.lo + MARGIN * (k.hi - k.lo)) / ((1 + 2 * MARGIN) * (k.hi - k.lo))


def masked(logits, mask):
    full = torch.cat([torch.ones(mask.shape[0], 1, dtype=torch.bool), mask], 1)
    return logits.masked_fill(~full, -1e4)


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(HERE, "results", "leap_net.json")


def train(steps=6000, batch=2048, lr=3e-3, seed=0, out=DEFAULT_OUT, log=True):
    from .train import condition_pool
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    t0 = time.time(); pool = condition_pool(40000, seed); scenes = scene_pool()
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    for it in range(steps):
        kind, mass, leg, speed, gain, urg, ch, z, w, mask = make_batch(batch, rng, pool, scenes)
        frac = max(0.0, 1 - it / (0.7 * steps))
        logits, kn = net(encode_shared_t(kind, mass, leg, speed, gain, urg, ch), make_batch.feats, noise=0.4 * frac)
        c, c_ref = cost(kind, mass, speed, gain, urg, ch, z, w, kn)
        allc = torch.cat([c_ref[:, None], c], 1)
        allc = 6.0 * torch.tanh(allc / 6.0)                                # bounded: a hopeless option must not swamp the gradient
        lg = masked(logits, mask)
        p = torch.softmax(lg, 1)
        # the choice learns which option is cheapest with the knobs as they are now (soft arg-min target);
        # the knobs learn to make the chosen options cheap, with some weight on the options the net prefers
        q = torch.softmax(masked(-allc.detach() / 0.05, mask), 1)
        # every landing place has its own jump program: all of them learn, the plausible ones most
        wk = 0.2 / K + 0.4 * torch.softmax(masked(-allc.detach() / 0.3, mask)[:, 1:], 1) + 0.4 * p.detach()[:, 1:]
        exp_c = (wk * allc[:, 1:]).sum(1)
        ce = -(q * torch.log_softmax(lg, 1)).sum(1)
        loss = exp_c.mean() + 0.3 * ce.mean() + 2.0 * (net.overshoot ** 2).sum(2).mean() + 0.01 * (relu(net.kz.abs() - 4.0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (it % 250 == 0 or it == steps - 1):
            hard = allc.gather(1, lg.argmax(1, keepdim=True))[:, 0]
            print(f"step {it:5d} expected {float(exp_c.detach().mean()):.4f} chosen {float(hard.detach().mean()):.4f} refuse {float((lg.argmax(1) == 0).float().mean()):.2f}", flush=True)
    secs = time.time() - t0
    save(net, out, dict(trained=dict(steps=steps, batch=batch, seed=seed, seconds=round(secs, 1))))
    if log: print(f"saved {out} in {secs:.0f} s")
    return net


def save(net, path, extra=None):
    d = dict(format="skeleto-nn/leap-net/1", inputs=input_names(), outputs=output_names(), samples=dict(n=K, spacing=DX), candidate_features=CAND,
             knobs=[dict(name=k.name, lo=k.lo, hi=k.hi, neutral=k.neutral, kinds=list(k.kinds), doc=k.doc) for k in KNOBS],
             channels=[dict(name=c.name, lo=c.lo, hi=c.hi, neutral=c.neutral, scale=c.scale, log=c.log) for c in CHANNELS],
             margin=MARGIN, hang=HANG, z_clip=Z_CLIP,
             body=dict(kinds=list(LEAP_KINDS), mass_ref=30.0, mass_div=2.0, leg_ref=0.5, speed_max=3.0),
             layers=[dict(W=l.weight.detach().cpu().numpy().round(6).tolist(), b=l.bias.detach().cpu().numpy().round(6).tolist()) for l in net.layers])
    if extra: d.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path, "w"))


def load(path=DEFAULT_OUT) -> Net:
    d = json.load(open(path)); net = Net()
    with torch.no_grad():
        for l, src in zip(net.layers, d["layers"]):
            l.weight.copy_(torch.tensor(src["W"])); l.bias.copy_(torch.tensor(src["b"]))
    return net


# ---------------------------------------------------------------------------------------------- runtime (numpy)
class LeapNet:
    """numpy forward pass and the decision: what an engine port does."""
    def __init__(self, path=DEFAULT_OUT):
        d = json.load(open(path)) if isinstance(path, str) else path
        self.layers = [(np.array(l["W"], np.float32), np.array(l["b"], np.float32)) for l in d["layers"]]

    def raw(self, x):
        h = x
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1: h = np.tanh(h)
        return h

    def decide(self, kind, mass, leg, speed, gain, urgency, ch: dict, z, w, mask=None):
        """z, w in metres (heights relative to the feet, water depth), sampled every DX * leg ahead.
        speed in m/s. Returns the decision in metres and seconds."""
        zn = np.asarray(z, np.float32) / leg; wn = np.asarray(w, np.float32) / leg
        shared = encode_shared_np(kind, mass, leg, speed / math.sqrt(9.81 * leg), gain, urgency, ch)
        feats = cand_features(zn, wn)
        outs = [self.raw(np.concatenate([shared, feats[k]])) for k in range(K)]
        logits = np.array([0.0] + [o[0] for o in outs], np.float32)
        if mask is not None: logits[1:][~np.asarray(mask, bool)] = -1e4
        a = int(np.argmax(logits))
        res = dict(logits=logits.tolist())
        if a == 0:
            res.update(action="refuse", label="refuse", knobs={k.name: k.neutral for k in KNOBS}); return res
        cell = a - 1
        kz = outs[cell][1:]; kn = {}
        for j, k in enumerate(KNOBS):
            s_ = 1 / (1 + math.exp(-kz[j])); m = MARGIN * (k.hi - k.lo)
            v = min(k.hi, max(k.lo, k.lo - m + (k.hi - k.lo + 2 * m) * s_))
            kn[k.name] = v if (kind in k.kinds) else k.neutral
        res.update(action="jump", cell=cell, knobs=kn, **arc(kn, cell, zn, wn, leg, kind))
        res["label"] = label(res, zn, wn, kn)
        return res


def arc(kn, cell, zn, wn, leg, kind="biped"):
    """The flight that the knobs describe, in metres and seconds (same formulas as ``cost``)."""
    xL = XS[cell]; zL = max(float(zn[cell]), -Z_CLIP); take = kn["take"]
    low = kn["lower"] if kind == "biped" else 0.0
    drop = max(-zL, 0.0)
    low = low / (1 + math.exp(-(drop - 0.3) / 0.08))
    zS = -low * min(drop, HANG)
    top = max([float(zn[i]) for i in range(cell) if XS[i] > take + 0.1] + [-20.0])
    za = max(zS, zL, top) + kn["apex"]
    Tu = math.sqrt(2 * max(za - zS, 0) + 1e-4); Td = math.sqrt(2 * max(za - zL, 0) + 1e-4); T = Tu + Td
    s = math.sqrt(leg / 9.81)
    return dict(take_m=take * leg, land_m=xL * leg, land_z_m=zL * leg, start_z_m=zS * leg, apex_m=za * leg, flight_s=T * s,
                rise_s=Tu * s, water_m=float(wn[cell]) * leg)


def label(res, zn, wn, kn):
    c = res["cell"]; zL = zn[c]; inbetween = [zn[i] for i in range(c) if XS[i] > kn["take"] + 0.1]
    top = max(inbetween + [zL]) if inbetween else zL
    if kn["lower"] > 0.5 and zL < -0.5: return "hang-drop"
    if wn[c] > 0.8 and zL < -0.3: return "jump into water"
    if kn["hands"] > 0.5 and zL > 0.4: return "pull up onto"
    if kn["hands"] > 0.5 and top > 0.25: return "vault"
    if zL > 0.3: return "jump up onto"
    if zL < -0.5: return "jump down"
    if top > 0.25: return "hop over"
    if min(inbetween + [0.0]) < -0.3: return "leap the gap"
    return "hop"


# ---------------------------------------------------------------------------------------------- scenes in metres
def profile(leg, *features):
    """Build (z, w, thin) in metres from features: ("wall", at, height, thick, beyond), ("step", at, up),
    ("drop", at, depth, water), ("gap", at, width, depth, water)."""
    xs = np.array(XS) * leg; z = np.zeros(K); w = np.zeros(K); thin = np.zeros(K, bool)
    for f in features:
        k, at = f[0], f[1]; b = xs > at
        if k == "wall":
            _, _, h, th, beyond = f; on = b & (xs <= at + th); z[on] = h; thin[on] = th < 0.8 * leg; z[xs > at + th] = beyond
        elif k == "step": z[b] = f[2]
        elif k == "drop": z[b] = -f[2]; w[b] = f[3]
        elif k == "gap":
            _, _, wd, dp, wa = f; on = b & (xs <= at + wd); z[on] = -dp; w[on] = wa
    return z, w, thin


SCENES = [
    ("flat street", [], ()),
    ("kerb up 0.2 m", [("step", 0.6, 0.2)], ()),
    ("churchyard wall 0.85 m", [("wall", 0.6, 0.85, 0.4, 0.0)], ()),
    ("wall 1.3 m", [("wall", 0.6, 1.3, 0.4, 0.0)], ()),
    ("wall 2.2 m", [("wall", 0.6, 2.2, 0.4, 0.0)], ()),
    ("wall 0.85 m, 1 m drop beyond", [("wall", 0.6, 0.85, 0.4, -1.0)], ()),
    ("platform up 0.7 m", [("step", 0.6, 0.7)], ()),
    ("ledge up 1.4 m", [("step", 0.6, 1.4)], ()),
    ("ledge up 2.6 m", [("step", 0.6, 2.6)], ()),
    ("drop 1 m", [("drop", 0.6, 1.0, 0.0)], ()),
    ("drop 2.5 m", [("drop", 0.6, 2.5, 0.0)], ()),
    ("drop 6 m onto stone", [("drop", 0.6, 6.0, 0.0)], ()),
    ("quay 2 m into deep water", [("drop", 0.6, 2.0, 3.0)], ()),
    ("cliff 8 m into deep water", [("drop", 0.6, 8.0, 4.0)], ()),
    ("cliff 8 m into shallow water", [("drop", 0.6, 8.0, 0.4)], ()),
    ("ditch 1.5 m wide", [("gap", 0.6, 1.5, 1.2, 0.0)], ()),
    ("stream 2.5 m wide", [("gap", 0.6, 2.5, 1.0, 0.8)], ()),
]

SITUATIONS = [("normal", []), ("ice", ["ice"]), ("sprained left ankle", [("sprain", 1.0, {"side": "L"})]),
              ("broken right arm", [("broken_arm", 1.0, {"side": "R"})]), ("heavy load", ["heavy_load"]),
              ("night", ["night"]), ("exhausted", ["exhausted"]), ("frost", ["frost"])]


def probe(path=DEFAULT_OUT, kind="biped", mass=70.0, leg=0.9, speed=0.0, gain=1.0, urgency=0.6, situations=None):
    from .conditions import channels_from, CONDITIONS
    net = LeapNet(path); rows = []
    for sname, conds in situations or SITUATIONS:
        conds = [c for c in conds if (c if isinstance(c, str) else c[0]) in CONDITIONS]
        ch = channels_from(conds)
        for name, feats, _ in SCENES:
            z, w, thin = profile(leg, *feats)
            d = net.decide(kind, mass, leg, speed, gain, urgency, ch, z, w, ~thin)
            rows.append((sname, name, d))
    return rows


def check(path=DEFAULT_OUT, n=600, seed=321, iters=300, restarts=4):
    """Per-case optimum: for every landing place, the best knobs by gradient descent; then the best place."""
    from .train import condition_pool
    rng = np.random.default_rng(seed); pool = condition_pool(4000, seed + 1)
    kind, mass, leg, speed, gain, urg, ch, z, w, mask = make_batch(n, rng, pool)
    net = load(path)
    with torch.no_grad():
        logits, kn = net(encode_shared_t(kind, mass, leg, speed, gain, urg, ch), torch.tensor(cand_features(z.numpy(), w.numpy())))
        c, c_ref = cost(kind, mass, speed, gain, urg, ch, z, w, kn)
        allc = torch.cat([c_ref[:, None], c], 1)
        a = masked(logits, mask).argmax(1)
        c_net = allc.gather(1, a[:, None])[:, 0]
    # optimum: free knobs for every (case, landing place), several random starts, then the best place
    lo = torch.tensor([k.lo for k in KNOBS]); hi = torch.tensor([k.hi for k in KNOBS])
    best = None
    for r in range(restarts):
        zz = (torch.randn(n, K, len(KNOBS)) * 1.5).requires_grad_(True)
        opt = torch.optim.Adam([zz], lr=0.08)
        for _ in range(iters):
            cc, _ = cost(kind, mass, speed, gain, urg, ch, z, w, lo + (hi - lo) * torch.sigmoid(zz))
            opt.zero_grad(); cc.sum().backward(); opt.step()
        with torch.no_grad():
            cc, _ = cost(kind, mass, speed, gain, urg, ch, z, w, lo + (hi - lo) * torch.sigmoid(zz))
            best = cc if best is None else torch.minimum(best, cc)
    with torch.no_grad():
        best_k = best.masked_fill(~mask, 1e9)
        c_opt = torch.minimum(best_k.min(1).values, c_ref)
        opt_a = torch.where(c_ref <= best_k.min(1).values, torch.zeros(n, dtype=torch.long), best_k.argmin(1) + 1)
    gain_possible = float((c_ref - c_opt).mean()); gap = float((c_net - c_opt).mean())
    res = dict(cost_refuse=float(c_ref.mean()), cost_net=float(c_net.mean()), cost_optimum=float(c_opt.mean()),
               share_of_possible_gain=1 - gap / max(gain_possible, 1e-9),
               same_decision=float(((a == 0) == (opt_a == 0)).float().mean()), same_landing=float((a == opt_a).float().mean()))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--out", default=DEFAULT_OUT); ap.add_argument("--check", action="store_true"); ap.add_argument("--probe", action="store_true")
    ap.add_argument("--kind", default="biped"); ap.add_argument("--mass", type=float, default=70.0); ap.add_argument("--leg", type=float, default=0.9)
    ap.add_argument("--gain", type=float, default=1.0); ap.add_argument("--speed", type=float, default=0.0)
    a = ap.parse_args()
    if a.check: check(a.out)
    elif a.probe:
        for s, n, d in probe(a.out, a.kind, a.mass, a.leg, a.speed, a.gain):
            k = d["knobs"]
            extra = "" if d["action"] == "refuse" else f"land {d['land_m']:.2f} m (z {d['land_z_m']:+.2f}) apex {d['apex_m']:.2f} hands {k['hands']:.2f} tuck {k['tuck']:.2f} flex {k['flex']:.2f} roll {k['roll']:.2f} lower {k['lower']:.2f} arms {k['arms_out']:.2f} take {d['take_m']:.2f}"
            print(f"{s:20s} {n:32s} {d['label']:16s} {extra}")
    else: train(a.steps, a.batch, out=a.out)
