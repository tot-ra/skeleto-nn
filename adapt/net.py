"""The style network: a small MLP from (body, channels) to motor-style knobs, and its portable form.

Training uses torch (``train.py``). The weights are saved as JSON together with the input and output
schema, so the forward pass runs anywhere with a few lines of code: ``StyleNet`` here (numpy) and
``motion_style_net.gd`` in the game. Inputs are encoded by ``encode_inputs``; outputs pass a sigmoid
and are scaled into each knob's range, so any output is a valid style.

A weight file trained before a channel existed loads into a newer schema by giving the new channel a
zero input column: the old network behaves exactly as before, and a warm-started retrain only has to
learn what the new column means.
"""
from __future__ import annotations
import json, math
import numpy as np
from .schema import CHANNELS, STYLE, BODY, KINDS

HIDDEN = (64, 64)
MARGIN = 0.15        # the output sigmoid spans [lo - m, hi + m] with m = MARGIN (hi - lo), then is clamped to [lo, hi]


def encode_channels(ch: dict) -> np.ndarray:
    out = []
    for c in CHANNELS:
        v = float(ch.get(c.name, c.neutral))
        e = math.log(max(v, 1e-3) / c.neutral) / c.scale if c.log else (v - c.neutral) / c.scale
        out.append(min(3.0, max(-3.0, e)))
    return np.array(out, np.float32)


def encode_body(kind: str, mass: float, leg: float, airborne: bool, froude: float) -> np.ndarray:
    k = [1.0 if kind == n else 0.0 for n in KINDS]
    return np.array(k + [math.log(mass / 30.0) / 2.0, math.log(leg / 0.5) / 1.0, 1.0 if airborne else 0.0,
                         min(froude, 3.0)], np.float32)


def encode_inputs(body_vec: np.ndarray, ch_vec: np.ndarray) -> np.ndarray:
    return np.concatenate([body_vec, ch_vec]).astype(np.float32)


def input_names(channels=None) -> list:
    return list(BODY) + [c.name for c in (channels or CHANNELS)]


def scale_out(sig: np.ndarray) -> np.ndarray:
    lo = np.array([k.lo for k in STYLE], np.float32); hi = np.array([k.hi for k in STYLE], np.float32)
    m = MARGIN * (hi - lo)
    return np.clip(lo - m + (hi - lo + 2 * m) * sig, lo, hi)


class StyleNet:
    """numpy forward pass of an exported weight file."""
    def __init__(self, path_or_dict):
        d = json.load(open(path_or_dict)) if isinstance(path_or_dict, str) else path_or_dict
        self.meta = d
        names = d["inputs"]; want = input_names()
        layers = [(np.array(l["W"], np.float32), np.array(l["b"], np.float32)) for l in d["layers"]]
        W0, b0 = layers[0]
        if names != want:                       # widen or reorder: a channel unknown to the file gets a zero column
            W = np.zeros((W0.shape[0], len(want)), np.float32)
            for j, n in enumerate(want):
                if n in names: W[:, j] = W0[:, names.index(n)]
            layers[0] = (W, b0)
        outs = d["outputs"]; self.out_idx = [outs.index(k.name) if k.name in outs else -1 for k in STYLE]
        self.layers = layers

    def raw(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1: h = np.tanh(h)
        return h

    def __call__(self, body_vec, ch_vec) -> dict:
        z = self.raw(encode_inputs(body_vec, ch_vec))
        sig = np.array([1 / (1 + math.exp(-z[i])) if i >= 0 else neutral_sig(k) for i, k in zip(self.out_idx, STYLE)], np.float32)
        vals = scale_out(sig)
        return {k.name: float(v) for k, v in zip(STYLE, vals)}

    def style(self, kind, mass, leg, airborne, froude, ch: dict) -> dict:
        return self(encode_body(kind, mass, leg, airborne, froude), encode_channels(ch))


def neutral_sig(k):
    return (k.neutral - k.lo + MARGIN * (k.hi - k.lo)) / ((1 + 2 * MARGIN) * (k.hi - k.lo))


def to_json(torch_layers, path, extra=None):
    d = dict(format="skeleto-nn/style-net/1", inputs=input_names(), outputs=[k.name for k in STYLE],
             channels=[dict(name=c.name, lo=c.lo, hi=c.hi, neutral=c.neutral, scale=c.scale, log=c.log, doc=c.doc) for c in CHANNELS],
             knobs=[dict(name=k.name, lo=k.lo, hi=k.hi, neutral=k.neutral, kinds=list(k.kinds), air=k.air, doc=k.doc) for k in STYLE],
             margin=MARGIN, body=dict(kinds=list(KINDS), mass_ref=30.0, mass_div=2.0, leg_ref=0.5, leg_div=1.0, froude_max=3.0),
             layers=[dict(W=l.weight.detach().cpu().numpy().round(6).tolist(), b=l.bias.detach().cpu().numpy().round(6).tolist()) for l in torch_layers])
    if extra: d.update(extra)
    json.dump(d, open(path, "w"))
    return d
