"""Line muscles of the human leg: where they start and end on the bones, what they wrap around, how strong they are.

Each muscle is a MuJoCo spatial tendon from its origin to its insertion through via points and over wrapping
cylinders (the kneecap in front of the knee, the femoral condyles behind it, the femoral head at the hip), pulled
by a Hill-type force computed in ``physics.PhysChar.step``:

  F = strength x F0 x (a x fl(l) x fv(v) + fpe(l))

  a     activation: follows the network's excitation (0..1) with a 10 ms rise and a 40 ms fall
  fl    active force-length: a muscle is strongest near its optimal fibre length, weak stretched or bunched up
  fv    force-velocity (Hill): weaker while shortening fast, up to 1.4 times stronger while being stretched
  fpe   passive elastic force of a stretched muscle: what stops a hip from resting on its extension stop, what a
        straight knee does to the hamstrings when the hip flexes

A muscle crosses one joint or two (hamstrings: hip and knee; rectus femoris: hip and knee; gastrocnemius: knee and
ankle). Moment arms come from the geometry and change with the joint angles, as they do in a body. Fibre lengths
are normalised per muscle from the joint ranges it spans (rigid tendon): the shortest length the joints allow is 0.6
optimal fibre lengths, the longest 1.45. Strength F0 is set so that, standing, each muscle gives its share of the
joint strength already in ``bodies.human`` (knee extension about 250 N m: vasti 0.7, rectus femoris 0.3, ...).

Coordinates: every bone frame at rest has x forward, y to the left, z up; the hip joint is at the thigh origin, the
knee at the shank origin, the ankle at the foot origin. The spec is written for the left leg; the right is its mirror.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

FL_LO, FL_HI = 0.6, 1.45      # normalised fibre length at the shortest and longest the joint ranges allow
FL_W = 0.45                   # width of the active force-length curve (Gaussian)
FPE_K, FPE_E0 = 5.0, 0.6      # passive force: (exp(K (l - 1) / E0) - 1) / (exp(K) - 1); F0 at 1.6 optimal lengths
V_MAX = 10.0                  # optimal fibre lengths per second: shortening speed at which the force falls to nothing
TAU_ACT, TAU_DEACT = 0.010, 0.040   # s: activation rises in 10 ms, falls in 40 (Thelen 2003, Millard 2013)


def fl_active(lt):
    return np.exp(-((lt - 1.0) / FL_W) ** 2)


def fl_passive(lt):
    x = np.clip(lt, 1.0, 1.75)
    return (np.exp(FPE_K * (x - 1.0) / FPE_E0) - 1.0) / (np.exp(FPE_K) - 1.0)


@dataclass
class Muscle:
    name: str                        # e.g. "VAS_L"
    title: str                       # anatomical name for figures
    path: list                       # ("site", bone, xyz) or ("wrap", geom, sidesite) items, origin first
    primary: tuple                   # (bone, axis, sign): the joint and direction its strength is calibrated on
    share: float                     # share of that joint direction's strength
    tendon_return: float = 0.0       # share of the braking work its tendon stores and gives back (Achilles, patellar)
    side: str = "L"


@dataclass
class MuscleSet:
    muscles: list = field(default_factory=list)
    wraps: list = field(default_factory=list)       # (bone, geom name, pos, radius, half length)
    sites: list = field(default_factory=list)       # (bone, site name, pos): side sites of the wraps
    covered: set = field(default_factory=set)       # (bone, axis) joint axes moved by muscles (no joint motor there)


def human_legs(c) -> MuscleSet:
    """The leg muscles of a ``bodies.human`` creature (10 per leg)."""
    sk = c.skel
    s = c.params.get("height", 1.75) / 1.75
    Lt = sk.length[sk.idx["thigh_L"]]; Ls = sk.length[sk.idx["shank_L"]]
    hl = float(sk.attach[sk.idx["thigh_L"]][1])          # the hip joint's lateral offset in the pelvis frame
    leg = next(l for l in c.legs if l.side == "L")
    hx, ah = float(leg.heel[0]), -float(leg.heel[2])     # heel behind the ankle, ankle height over the sole
    out = MuscleSet()
    for side, my in (("L", 1.0), ("R", -1.0)):
        P, T, S, F = "pelvis", "thigh_" + side, "shank_" + side, "foot_" + side
        v = lambda x, y, z: (x * s, my * y * s, z * s)
        vy = lambda x, y, z: (x, my * y, z)                # already in metres
        # wrapping surfaces: femoral head (hip), kneecap (in front of the knee), femoral condyles (behind it)
        out.wraps += [(P, f"w_hip_{side}", vy(0.0, hl, 0.0), 0.045 * s, 0.05 * s),
                      (T, f"w_patella_{side}", vy(0.0, 0.0, -Lt), 0.045 * s, 0.05 * s),
                      (T, f"w_condyle_{side}", vy(0.0, 0.0, -Lt), 0.025 * s, 0.05 * s)]
        out.sites += [(P, f"ss_hip_front_{side}", vy(0.10 * s, hl, 0.0)), (P, f"ss_hip_back_{side}", vy(-0.10 * s, hl, 0.0)),
                      (T, f"ss_knee_front_{side}", vy(0.10 * s, 0.0, -Lt)), (T, f"ss_knee_back_{side}", vy(-0.10 * s, 0.0, -Lt))]
        site = lambda bone, xyz: ("site", bone, xyz)
        heel = vy(hx - 0.005 * s, 0.0, -ah + 0.035 * s)    # calcaneal tuberosity, above the heel's ground point
        M = lambda *a, **k: Muscle(a[0] + "_" + side, *a[1:], side=side, **k)
        out.muscles += [
            M("ILIO", "iliopsoas", [site(P, vy(0.01 * s, hl - 0.035 * s, 0.11 * s)), ("wrap", f"w_hip_{side}", f"ss_hip_front_{side}"),
                                    site(T, v(0.012, -0.012, -0.07))], (T, "Y", -1), 0.7),
            M("GMAX", "gluteus maximus", [site(P, vy(-0.075 * s, hl - 0.045 * s, 0.07 * s)), ("wrap", f"w_hip_{side}", f"ss_hip_back_{side}"),
                                          site(T, v(-0.025, 0.012, -0.12))], (T, "Y", 1), 0.6),
            M("HAM", "hamstrings", [site(P, vy(-0.05 * s, hl - 0.015 * s, -0.055 * s)), ("wrap", f"w_condyle_{side}", f"ss_knee_back_{side}"),
                                    site(S, v(-0.022, 0.0, -0.05))], (T, "Y", 1), 0.4),
            M("RF", "rectus femoris", [site(P, vy(0.045 * s, hl, 0.045 * s)), ("wrap", f"w_hip_{side}", f"ss_hip_front_{side}"),
                                       site(T, vy(0.05 * s, 0.0, -0.5 * Lt)), ("wrap", f"w_patella_{side}", f"ss_knee_front_{side}"), site(S, v(0.035, 0.0, -0.07))],
              (S, "Y", -1), 0.3, tendon_return=0.3),
            M("VAS", "vasti", [site(T, v(0.03, 0.0, -0.18)), ("wrap", f"w_patella_{side}", f"ss_knee_front_{side}"),
                               site(S, v(0.035, 0.0, -0.07))], (S, "Y", -1), 0.7, tendon_return=0.3),
            M("GAS", "gastrocnemius", [site(T, vy(-0.02 * s, 0.0, -Lt + 0.045 * s)), ("wrap", f"w_condyle_{side}", f"ss_knee_back_{side}"),
                                       site(F, heel)], (F, "Y", 1), 0.35, tendon_return=0.5),
            M("SOL", "soleus", [site(S, vy(-0.022 * s, 0.0, -0.35 * Ls)), site(F, heel)], (F, "Y", 1), 0.65, tendon_return=0.5),
            M("TA", "tibialis anterior", [site(S, vy(0.025 * s, 0.0, -0.3 * Ls)), site(S, vy(0.035 * s, 0.0, -Ls + 0.035 * s)),
                                          site(F, vy(0.06 * s, -0.01 * s, -ah + 0.045 * s))], (F, "Y", -1), 1.0),
            M("GMED", "gluteus medius", [site(P, vy(-0.01 * s, hl + 0.06 * s, 0.12 * s)), site(T, v(0.0, 0.045, -0.04))],
              (T, "X", my), 0.4),
            M("ADD", "adductors", [site(P, vy(0.03 * s, hl - 0.07 * s, -0.03 * s)), site(T, v(0.01, -0.015, 0.0)[:2] + (-0.45 * Lt,))],
              (T, "X", -my), 0.4),
        ]
        out.covered |= {(T, "Y"), (T, "X"), (S, "Y"), (F, "Y")}
    return out


SETS = {"human_legs": human_legs}
