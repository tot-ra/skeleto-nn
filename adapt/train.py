"""Train the style network by gradient descent through the cost model.

    python -m adapt.train                          # from scratch, about a minute on a laptop CPU
    python -m adapt.train --init results/style_net.json --steps 800    # warm start after adding a channel or condition
    python -m adapt.train --check                  # evaluate a weight file against per-case optimal styles

Every batch draws bodies (sparrow to horse), commanded speeds, flying or not, and situations: random
combinations of the conditions in ``conditions.py`` at random intensities, sides, wound places and wind
directions, mixed with random channel values so the space between named conditions is covered too.
The loss is the mean cost of the network's style (``model.evaluate``); there are no target motions.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
import numpy as np
import torch
from .schema import CHANNELS, STYLE, BODY, KINDS, CH
from .conditions import CONDITIONS, REGIONS, channels_from
from .model import evaluate
from . import net as N

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(HERE, "results", "style_net.json")

# body ranges per kind: (mass kg, leg length m), log-uniform
BODY_RANGES = {"biped": ((12.0, 140.0), (0.45, 1.05)), "quadruped": ((2.5, 650.0), (0.14, 1.1)), "bird": ((0.015, 6.0), (0.03, 0.4))}
FR_MAX = {"biped": 2.2, "quadruped": 3.0, "bird": 1.0}


def sample_bodies(n, rng: np.random.Generator):
    kind = rng.choice(len(KINDS), n, p=[0.45, 0.35, 0.2])
    mass = np.zeros(n); leg = np.zeros(n); air = np.zeros(n); fr = np.zeros(n)
    for i, k in enumerate(KINDS):
        m = kind == i; cnt = int(m.sum())
        (m0, m1), (l0, l1) = BODY_RANGES[k]
        mass[m] = np.exp(rng.uniform(math.log(m0), math.log(m1), cnt))
        leg[m] = np.exp(rng.uniform(math.log(l0), math.log(l1), cnt))
        u = rng.uniform(0, 1, cnt)
        fr[m] = np.where(u < 0.15, 0.0, rng.uniform(0, 1, cnt) ** 1.5 * FR_MAX[k])
        if k == "bird": air[m] = (rng.uniform(0, 1, cnt) < 0.5).astype(float)
    return kind, mass, leg, air, fr


def random_condition(rng: random.Random):
    name = rng.choice(list(CONDITIONS))
    opt = dict(side=rng.choice("LR"), dir=rng.uniform(-math.pi, math.pi))
    if CONDITIONS[name].get("region"):
        opt["region"] = rng.choice([r.replace("_S", "") for r in REGIONS])
    return (name, rng.uniform(0.3, 1.0), opt)


def condition_pool(n, seed=0, exclude_channels=()):
    rng = random.Random(seed)
    out = np.zeros((n, len(CHANNELS)), np.float32)
    for i in range(n):
        u = rng.random()
        k = 0 if u < 0.12 else (1 if u < 0.45 else (2 if u < 0.8 else 3))
        ch = channels_from([random_condition(rng) for _ in range(k)])
        out[i] = [ch[c.name] for c in CHANNELS]
    return out


def jitter_channels(base: np.ndarray, rng: np.random.Generator, p_free=0.25):
    """Random channel values on top of the condition pool: some cases fully random, others nudged."""
    x = base.copy(); n = len(x)
    lo = np.array([c.lo for c in CHANNELS]); hi = np.array([c.hi for c in CHANNELS]); neu = np.array([c.neutral for c in CHANNELS])
    free = rng.uniform(0, 1, n) < p_free
    for j, c in enumerate(CHANNELS):
        pick = free & (rng.uniform(0, 1, n) < 0.35)
        if c.log:
            vals = np.exp(rng.uniform(math.log(c.lo), math.log(c.hi), n))
        elif c.name.startswith("wound"):
            vals = rng.uniform(c.lo, c.hi, n)
        elif c.lo < 0 and c.name.startswith("wind"):
            vals = rng.uniform(-1, 1, n) * rng.uniform(0, 1, n)
        elif c.name == "air_temp":
            vals = rng.uniform(-30, 40, n)
        else:
            vals = rng.uniform(0, 1, n) ** 1.3 * (c.hi - c.lo) + c.lo
        x[pick, j] = vals[pick]
    # states that come with a situation over time
    st = rng.uniform(0, 1, n) < 0.3
    for name in ("core_cold", "blood_lost", "fatigue"):
        j = CH[name]; x[st, j] = np.maximum(x[st, j], rng.uniform(0, 0.8, n)[st] * (rng.uniform(0, 1, n)[st] < 0.4))
    return np.clip(x, lo, hi).astype(np.float32)


def torch_encode(kind, mass, leg, air, fr, ch):
    kind_oh = torch.nn.functional.one_hot(kind, len(KINDS)).float()
    body_vec = torch.cat([kind_oh, (torch.log(mass / 30.0) / 2.0)[:, None], torch.log(leg / 0.5)[:, None], air[:, None], fr.clamp(max=3.0)[:, None]], 1)
    enc = []
    for j, c in enumerate(CHANNELS):
        v = ch[:, j]
        e = torch.log(v.clamp(min=1e-3) / c.neutral) / c.scale if c.log else (v - c.neutral) / c.scale
        enc.append(e.clamp(-3, 3))
    return torch.cat([body_vec, torch.stack(enc, 1)], 1)


def body_dict(kind, mass, leg, air, fr):
    d = {"kind_" + k: (kind == i).float() for i, k in enumerate(KINDS)}
    d.update(mass=mass, leg=leg, airborne=air, froude=fr)
    return d


class Net(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden=N.HIDDEN):
        super().__init__()
        sizes = [n_in, *hidden, n_out]
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        with torch.no_grad():                    # start at the neutral style: last layer outputs the neutral logits
            self.layers[-1].weight.mul_(0.05)
            neu = torch.tensor([N.neutral_sig(k) for k in STYLE])
            self.layers[-1].bias.copy_(torch.log(neu / (1 - neu)))

    def forward(self, x, noise=0.0):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1: x = torch.tanh(x)
        self.logits = x
        if noise > 0: x = x + noise * torch.randn_like(x)       # exploration: keeps saturated knobs learnable
        lo = torch.tensor([k.lo for k in STYLE]); hi = torch.tensor([k.hi for k in STYLE])
        m = N.MARGIN * (hi - lo)
        raw = lo - m + (hi - lo + 2 * m) * torch.sigmoid(x)    # the sigmoid spans a slightly wider range ...
        self.overshoot = (raw - raw.clamp(lo, hi)) / (hi - lo)
        return raw + (torch.maximum(torch.minimum(raw, hi), lo) - raw).detach()   # ... clamped, gradient passes straight through


def load_init(net: Net, path):
    d = json.load(open(path)); names = d["inputs"]; want = N.input_names()
    with torch.no_grad():
        for l, src in zip(net.layers, d["layers"]):
            W = torch.tensor(src["W"]); b = torch.tensor(src["b"])
            if l is net.layers[0] and names != want:
                W2 = torch.zeros(l.weight.shape)
                for j, n in enumerate(want):
                    if n in names: W2[:, j] = W[:, names.index(n)]
                W = W2
            outs = d["outputs"]
            if l is net.layers[-1] and outs != [k.name for k in STYLE]:
                W2 = l.weight.clone(); b2 = l.bias.clone()
                for j, k in enumerate(STYLE):
                    if k.name in outs: W2[j] = W[outs.index(k.name)]; b2[j] = b[outs.index(k.name)]
                W, b = W2, b2
            l.weight.copy_(W); l.bias.copy_(b)


def make_batch(n, rng, pool):
    kind, mass, leg, air, fr = sample_bodies(n, rng)
    ch = jitter_channels(pool[rng.integers(0, len(pool), n)], rng)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
    return t(kind, torch.long), t(mass), t(leg), t(air), t(fr), t(ch)


def train(steps=2500, batch=4096, lr=3e-3, seed=0, init=None, out=DEFAULT_OUT, pool_n=60000, log=True, noise=1.0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    t0 = time.time()
    pool = condition_pool(pool_n, seed)
    net = Net(len(BODY) + len(CHANNELS), len(STYLE))
    if init: load_init(net, init)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    hist = []
    for it in range(steps):
        kind, mass, leg, air, fr, ch = make_batch(batch, rng, pool)
        st = net(torch_encode(kind, mass, leg, air, fr, ch), noise=noise * max(0.0, 1 - it / (0.7 * steps)))
        cost, parts, _ = evaluate(body_dict(kind, mass, leg, air, fr), ch, st)
        loss = cost.mean() + 2.0 * (net.overshoot ** 2).sum(1).mean() + 0.01 * (torch.relu(net.logits.abs() - 4.0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 250 == 0 or it == steps - 1:
            hist.append((it, float(loss.detach())))
            if log: print(f"step {it:5d} cost {float(loss):.4f}  " + " ".join(f"{k}={float(v.mean()):.3f}" for k, v in parts.items()), flush=True)
    secs = time.time() - t0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    N.to_json(net.layers, out, extra=dict(trained=dict(steps=steps, batch=batch, seed=seed, seconds=round(secs, 1), final_cost=hist[-1][1], warm_start=bool(init))))
    if log: print(f"saved {out} in {secs:.0f} s")
    return net, hist, secs


def optimal_styles(kind, mass, leg, air, fr, ch, iters=400, lr=0.05):
    """Per-case optimum of the cost (free parameters, no network): the bar the network is measured against."""
    n = len(kind)
    neu = torch.tensor([(k.neutral - k.lo) / (k.hi - k.lo) for k in STYLE]).clamp(0.05, 0.95)
    z = torch.log(neu / (1 - neu)).repeat(n, 1).clone().requires_grad_(True)
    lo = torch.tensor([k.lo for k in STYLE]); hi = torch.tensor([k.hi for k in STYLE])
    opt = torch.optim.Adam([z], lr=lr)
    body = body_dict(kind, mass, leg, air, fr)
    for _ in range(iters):
        st = lo + (hi - lo) * torch.sigmoid(z)
        c, _, _ = evaluate(body, ch, st)
        opt.zero_grad(); c.sum().backward(); opt.step()
    st = (lo + (hi - lo) * torch.sigmoid(z)).detach()
    return st, evaluate(body, ch, st)[0].detach()


def check(path=DEFAULT_OUT, n=2000, seed=123):
    rng = np.random.default_rng(seed)
    pool = condition_pool(4000, seed + 1)
    kind, mass, leg, air, fr, ch = make_batch(n, rng, pool)
    net = Net(len(BODY) + len(CHANNELS), len(STYLE)); load_init(net, path)
    with torch.no_grad():
        st = net(torch_encode(kind, mass, leg, air, fr, ch))
        c_net = evaluate(body_dict(kind, mass, leg, air, fr), ch, st)[0]
        neu = torch.tensor([k.neutral for k in STYLE]).repeat(n, 1)
        c_neu = evaluate(body_dict(kind, mass, leg, air, fr), ch, neu)[0]
    _, c_opt = optimal_styles(kind, mass, leg, air, fr, ch)
    gap = float((c_net - c_opt).mean()); gain = float((c_neu - c_opt).mean())
    res = dict(cost_neutral=float(c_neu.mean()), cost_net=float(c_net.mean()), cost_optimum=float(c_opt.mean()),
               share_of_possible_gain=1 - gap / max(gain, 1e-9))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init"); ap.add_argument("--out", default=DEFAULT_OUT); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check: check(a.out)
    else: train(a.steps, a.batch, a.lr, a.seed, a.init, a.out)
