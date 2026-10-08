"""Bone-injury model for falls and blows. The same numbers score the physical reflex (evolution target) and the kinematic falls.

A bone breaks when the force through it exceeds a tolerance, and it tolerates less when the load arrives fast (bone is brittle at a
high loading rate). A body that yields while it stops (muscles lengthening under tension, joints flexing, rolling) spreads the same
momentum over a longer stroke and a longer time: lower peak force, lower rate, so a much lower fracture risk than a rigid stop.
  stopping a mass m moving at v over a stroke s:  F = m v^2 / (2 s)      (rigid bone-on-floor s ~ 1 cm, a flexing limb or a roll 10-30 cm)
Forces are in body weights (mg). The head has the lowest tolerance and the highest weight in the score."""
from __future__ import annotations
import math
import numpy as np

REGIONS = ("head", "torso", "arms", "legs")
TOL = dict(head=5.0, torso=8.0, arms=4.0, legs=10.0)      # peak force (mg) for ~63% fracture risk under slow loading (skull 4-6 kN, ribs, wrist 2.5-3.5 kN, femur and tibia 5-10 kN)
WEIGHT = dict(head=3.0, torso=1.5, arms=1.0, legs=1.0)    # how bad it is to injure the region
RATE0 = 500.0                                              # loading rate (mg/s) that doubles the brittleness
RATE_CAP = 3.0

def risk(peak, rate, region):
    k = min(RATE_CAP, 1.0 + max(rate, 0.0) / RATE0)
    return 1.0 - math.exp(-(max(peak, 0.0) * k / TOL[region]) ** 3)

def loading_rate(series, dt, win=0.02):
    """Largest rise of the force over a window of ~20 ms, in mg/s."""
    s = np.asarray(series, float); n = max(1, int(round(win / dt)))
    if len(s) <= n: return float(s.max() / max(len(s) * dt, 1e-6)) if len(s) else 0.0
    return float(np.max((s[n:] - s[:-n]) / (n * dt)))

def score(series_by_region, dt):
    """series: {region: forces in mg per step}. Returns (cost, parts). Cost ~ weighted fracture risk + a small pain term."""
    parts = {}; cost = 0.0
    for r in REGIONS:
        s = np.asarray(series_by_region.get(r, []), float)
        if len(s) == 0: parts[r] = dict(peak=0.0, rate=0.0, risk=0.0); continue
        pk = float(s.max()); rt = loading_rate(s, dt); rk = risk(pk, rt, r)
        pain = float(np.sum(np.maximum(s - 1.0, 0.0)) * dt)            # impulse above body weight, mg*s
        parts[r] = dict(peak=pk, rate=rt, risk=rk, pain=pain)
        cost += WEIGHT[r] * rk + 0.05 * WEIGHT[r] * pain
    return cost, parts

def stroke_force(v, stroke, mass, g=9.81):
    """Peak force in body weights when `mass` moving at v is stopped over `stroke`."""
    return (v * v / (2 * max(stroke, 1e-3))) / g
