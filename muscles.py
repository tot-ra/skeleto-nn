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
# Fatigue of every muscle (three-compartment model, Xia and Frey-Law 2008): of its motor units a share is fatigued (MF);
# the rest can be recruited. Units working at activation a tire at F a per second, tired ones recover at R per second,
# and a muscle cannot be activated beyond 1 - MF. Rates per joint fitted to endurance times (Frey-Law, Looft and
# Heitsman 2012); held at full effort a shoulder gives out within a minute, an ankle in some three, and the level a
# muscle can keep up indefinitely is R / (F + R): about 9 % of its strength at the knee, 15 % in the trunk
FATIGUE = {"ankle": (0.00589, 0.00058), "knee": (0.01500, 0.00149), "trunk": (0.00755, 0.00128),
           "shoulder": (0.01820, 0.00168), "elbow": (0.00912, 0.00094), "hand": (0.00980, 0.00064)}


def fatigue_class(bone):
    """The joint whose fatigue rates a muscle gets, from the bone its primary action turns (hips use the knee's)."""
    b = bone[:-2] if bone.endswith(("_L", "_R")) else bone
    if b.startswith(("foot", "toes", "meta", "paw")):
        return "ankle"
    if b.startswith(("uarm", "wing_a", "scap", "clav")):
        return "shoulder"
    if b.startswith(("farm", "wing_b")):
        return "elbow"
    if b.startswith(("hand", "wing_c")):
        return "hand"
    if b.startswith(("lumbar", "thorax", "chest", "spine", "tail", "neck", "head", "pelvis")):
        return "trunk"
    return "knee"


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
    check: tuple = None              # degrees of the primary joint where it must pull its way (default: the inner range)
    group: str = ""                  # muscles whose summed strength is set together (``MuscleSet.targets``)


@dataclass
class MuscleSet:
    muscles: list = field(default_factory=list)
    wraps: list = field(default_factory=list)       # (bone, geom name, pos, radius, half length)
    sites: list = field(default_factory=list)       # (bone, site name, pos): side sites of the wraps
    covered: set = field(default_factory=set)       # (bone, axis) joint axes moved by muscles (no joint motor there)
    targets: dict = field(default_factory=dict)     # group -> summed strength of one side's muscles in body weights


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


class Builder:
    """Writes a muscle set for the left side and mirrors it: bone names carry ``@`` for the side, points are given as a
    share ``t`` along the bone plus an offset in metres for a reference body (scaled by ``scale``; y is mirrored)."""

    def __init__(self, c, scale):
        self.c = c; self.sk = c.skel; self.s = scale; self.ms = MuscleSet(); self.sd = "L"; self.my = 1.0

    def side(self, sd):
        self.sd = sd; self.my = 1.0 if sd == "L" else -1.0
        return self

    def b(self, bone):
        return bone.replace("@", self.sd)

    def pt(self, bone, t=0.0, off=(0.0, 0.0, 0.0)):
        i = self.sk.idx[self.b(bone)]
        p = self.sk.dir[i] * self.sk.length[i] * t + np.array([off[0], self.my * off[1], off[2]]) * self.s
        return tuple(float(x) for x in p)

    def site(self, bone, t=0.0, off=(0.0, 0.0, 0.0)):
        return ("site", self.b(bone), self.pt(bone, t, off))

    def sn(self, bone, t, n=0.0, y=0.0):
        """A site ``n`` metres off the bone's axis in the side-view plane, on the cranial or dorsal side for n > 0 (for a
        bone pointing down: in front; for a paw pointing forward: on top), and ``y`` to the side; ``t`` < 0 is
        above the joint along the bone (the point of the hock)."""
        i = self.sk.idx[self.b(bone)]; d = self.sk.dir[i]
        perp = np.array([-d[2], 0.0, d[0]]); perp = perp if (perp[0] > 1e-6 or (abs(perp[0]) <= 1e-6 and perp[2] > 0)) else -perp
        p = d * self.sk.length[i] * t + (perp * n + np.array([0.0, self.my * y, 0.0])) * self.s
        return ("site", self.b(bone), tuple(float(x) for x in p))

    def wrap(self, bone, key, t, off, r, side_off, kind="cylinder", axis=(0, 1, 0)):
        """A wrapping surface on ``bone`` and the side site that tells a path which way round it goes."""
        nm = f"w_{key}_{self.sd}"; ss = f"ss_{key}_{self.sd}"
        if not any(w[1] == nm for w in self.ms.wraps):
            self.ms.wraps.append((self.b(bone), nm, self.pt(bone, t, off), r * self.s, 0.04 * self.s, kind, axis))
        k = 0
        while any(x[1] == f"{ss}{k}" for x in self.ms.sites):
            if any(x[1] == f"{ss}{k}" and x[2] == self.pt(bone, t, side_off) for x in self.ms.sites):
                return ("wrap", nm, f"{ss}{k}")
            k += 1
        self.ms.sites.append((self.b(bone), f"{ss}{k}", self.pt(bone, t, side_off)))
        return ("wrap", nm, f"{ss}{k}")

    group = ""

    def add(self, key, title, path, primary, share, ret=0.0, check=None):
        bone, ax, sg = primary
        if self.sd == "R" and ax in "XZ":                     # rotations about X and Z flip in the mirror
            sg = -sg
            check = check and (-check[1], -check[0])
        self.ms.muscles.append(Muscle(f"{key}_{self.sd}", title, path, (self.b(bone), ax, sg), share, ret, self.sd, check, self.group))

    def near(self, child, off=(0.0, 0.0, 0.0)):
        """A site on the parent of ``child``, at the joint where ``child`` attaches plus an offset (the hip on the pelvis,
        the shoulder on the chest)."""
        i = self.sk.idx[self.b(child)]; par = self.sk.bones[self.sk.parent[i]].name
        p = np.array(self.sk.attach[i]) + np.array([off[0], self.my * off[1], off[2]]) * self.s
        return ("site", par, tuple(float(x) for x in p))

    def cover(self, bone, axes):
        for ax in axes:
            if ax in self.sk.bones[self.sk.idx[self.b(bone)]].lim:
                self.ms.covered.add((self.b(bone), ax))


def human_upper(c) -> MuscleSet:
    """Trunk and arm muscles of a ``bodies.human`` creature: abdominal wall, back extensors, obliques; deltoid, chest,
    back and arm flexors and extensors (13 per side). Hip rotation, the shoulder girdle, the twist of the spine, the
    wrist, neck and head keep joint motors."""
    s = c.params.get("height", 1.75) / 1.75
    B = Builder(c, s); sk = c.skel
    Lp = sk.length[sk.idx["pelvis"]]
    spine = ("lumbar", "lumbar1", "lumbar2", "thorax")
    for sd in "LR":
        B.side(sd)
        # trunk: rectus abdominis in front of every vertebral joint, erector spinae behind (lumbar and thoracic parts),
        # the external oblique on the side
        B.add("RA", "rectus abdominis", [B.site("pelvis", 0.0, (0.07, 0.03, 0.0))] + [B.site(b, 0.5, (0.11, 0.035, 0.0)) for b in spine]
              + [B.site("chest", 0.2, (0.12, 0.04, 0.0))], ("lumbar", "Y", 1), 0.45)
        B.add("ESL", "erector spinae, lumbar", [B.site("pelvis", 1.0, (-0.06, 0.03, -0.04))] + [B.site(b, 0.5, (-0.065, 0.03, 0.0)) for b in spine[:3]]
              + [B.site("thorax", 0.6, (-0.065, 0.03, 0.0))], ("lumbar", "Y", -1), 0.6)
        B.add("EST", "erector spinae, thoracic", [B.site("lumbar1", 0.5, (-0.065, 0.03, 0.0)), B.site("lumbar2", 0.5, (-0.07, 0.03, 0.0)),
                                                  B.site("thorax", 0.5, (-0.075, 0.03, 0.0)), B.site("chest", 0.8, (-0.08, 0.03, 0.0))],
              ("thorax", "Y", -1), 0.6)
        B.add("EO", "external oblique", [B.site("pelvis", 1.0, (0.0, 0.13, 0.0))] + [B.site(b, 0.5, (0.04, 0.13, 0.0)) for b in spine[1:]]
              + [B.site("chest", 0.3, (0.08, 0.12, 0.0))], ("lumbar", "X", -1), 0.5)
        for b in spine + ("chest",):
            B.cover(b, "YX")
        # shoulder: the deltoid runs over the humeral head (a via point on the head, so it turns with the arm) to the
        # deltoid tuberosity; chest and back muscles adduct. A sphere wrap at the joint let the paths jump from one side
        # of the head to the other, flipping the force
        head = lambda side_off: B.wrap("clav_@", "humhead", 1.0, (0, 0, 0), 0.04, side_off, kind="sphere")
        B.add("DELTA", "deltoid, front", [B.site("clav_@", 0.75, (0.05, 0.0, 0.035)), head((0.09, 0.0, 0.0)),
                                          B.site("uarm_@", 0.4, (0.018, 0.012, 0.0))], ("uarm_@", "Y", -1), 0.5,
              check=(-110, 70))      # higher, the shoulder blade turns to lift the arm (the clavicle motors here)
        B.add("DELTP", "deltoid, back", [B.site("clav_@", 0.85, (-0.06, 0.0, 0.035)), head((-0.09, 0.0, 0.0)),
                                         B.site("uarm_@", 0.4, (-0.012, 0.012, 0.0))], ("uarm_@", "Y", 1), 0.5,
              check=(-110, 70))
        B.add("DELTM", "deltoid, middle", [B.site("clav_@", 1.0, (0.0, 0.015, 0.045)), head((0.0, 0.09, 0.0)),
                                           B.site("uarm_@", 0.4, (0.0, 0.016, 0.0))], ("uarm_@", "X", 1), 0.7)
        B.add("PEC", "pectoralis major", [B.site("chest", 0.55, (0.10, 0.03, 0.0)), B.site("uarm_@", 0.0, (0.02, 0.0, -0.07))],
              ("uarm_@", "X", -1), 0.6, check=(-5, 95))     # above that it lifts the arm further (action inversion)
        B.add("LAT", "latissimus dorsi", [B.site("lumbar2", 0.5, (-0.07, 0.03, 0.0)), B.site("thorax", 0.7, (-0.07, 0.10, 0.0)),
                                          B.site("uarm_@", 0.0, (0.005, -0.012, -0.08))], ("uarm_@", "X", -1), 0.4)
        B.cover("uarm_@", "YX")
        # elbow: biceps (from the shoulder blade) and brachialis bowstring in front of the elbow, triceps wraps the olecranon
        elbow = lambda side_off: B.wrap("uarm_@", "elbow", 1.0, (0, 0, 0), 0.02, side_off)
        B.add("BIC", "biceps", [B.site("clav_@", 1.0, (0.03, -0.01, -0.02)), B.site("uarm_@", 0.5, (0.03, 0.0, 0.0)),
                                B.site("farm_@", 0.0, (0.025, 0.0, -0.055))], ("farm_@", "Y", -1), 0.5)
        B.add("BRA", "brachialis", [B.site("uarm_@", 0.55, (0.02, 0.0, 0.0)), B.site("farm_@", 0.0, (0.022, 0.0, -0.05))],
              ("farm_@", "Y", -1), 0.5)
        B.add("TRI", "triceps", [B.site("clav_@", 1.0, (-0.02, -0.01, -0.03)), B.site("uarm_@", 0.5, (-0.028, 0.0, 0.0)), elbow((-0.06, 0.0, 0.0)),
                                 B.site("farm_@", 0.0, (-0.025, 0.0, 0.015))], ("farm_@", "Y", 1), 1.0)
        B.cover("farm_@", "Y")
    return B.ms


def zigzag_leg(B, root, th, sh, me, ft, k=1.0, tag=""):
    """A hind leg folded in a zig-zag (quadruped hind leg, bird leg): hip, stifle (knee), hock (intertarsal joint),
    toes. ``root``: the bone the leg hangs from; offsets are for a 30 kg dog and scale with ``B.s``; ``k`` scales them
    for thinner limbs. Every joint has a wrapping surface (femoral head, kneecap and condyles, the hock's retinaculum
    and the point of the hock, the sesamoids of the toe joint) so a path cannot slip to the wrong side of a bent
    joint. 11 muscles."""
    o = lambda x, y, z: (x * k, y * k, z * k)
    W = lambda bone, key, r, side: B.wrap(bone, key + tag, 1.0, (0, 0, 0), r * k, o(*side))
    hip = lambda side: _joint_wrap(B, th, "hip" + tag, 0.028 * k, o(*side))
    B.add("GLUT" + tag, "gluteals", [B.near(th, o(0.09, 0.0, 0.05)), hip((-0.05, 0, 0.05)), B.sn(th, 0.12, -0.02 * k, 0.012 * k)], (th, "Y", 1), 0.5)
    B.add("HAMS" + tag, "hamstrings", [B.near(th, o(-0.06, -0.005, -0.01)), hip((-0.06, 0, -0.02)), B.sn(th, 0.5, -0.035 * k),
                                        W(th, "cond", 0.018, (-0.06, 0, 0)),
                                        B.sn(sh, 0.3, -0.01 * k)], (th, "Y", 1), 0.5)
    B.add("ILIO" + tag, "iliopsoas", [B.near(th, o(0.12, -0.03, 0.06)), hip((0.06, 0, -0.02)), B.sn(th, 0.15, 0.012 * k, -0.01 * k)], (th, "Y", -1), 0.7)
    B.add("RF" + tag, "rectus femoris", [B.near(th, o(0.035, 0.0, -0.01)), B.sn(th, 0.5, 0.03 * k), W(th, "stifle", 0.03, (0.08, 0, 0)),
                                          B.sn(sh, 0.12, 0.025 * k)], (sh, "Y", -1), 0.25)
    B.add("QUAD" + tag, "vasti (quadriceps)", [B.sn(th, 0.2, 0.025 * k), W(th, "stifle", 0.03, (0.08, 0, 0)), B.sn(sh, 0.12, 0.025 * k)],
          (sh, "Y", -1), 0.75)
    calc = B.sn(me, -0.12, -0.04 * k)                       # the point of the hock (calcaneal tuber)
    B.add("GAS" + tag, "gastrocnemius", [B.sn(th, 0.9, -0.025 * k), W(th, "cond", 0.018, (-0.06, 0, 0)), B.sn(sh, 0.6, -0.03 * k),
                                          W(sh, "hockpt", 0.03, (-0.07, 0, 0.02)), calc], (me, "Y", 1), 0.6, ret=0.5)
    B.add("SDF" + tag, "superficial digital flexor", [B.sn(th, 0.9, -0.02 * k), W(th, "cond", 0.018, (-0.06, 0, 0)), B.sn(sh, 0.6, -0.028 * k),
                                                      W(sh, "hockpt", 0.03, (-0.07, 0, 0.02)), B.sn(me, 0.5, -0.02 * k),
                                                      W(me, "sesam", 0.012, (-0.04, 0, -0.04)), B.sn(ft, 0.7, -0.012 * k)], (ft, "Y", 1), 0.7, ret=0.5)
    B.add("TC" + tag, "tibialis cranialis", [B.sn(sh, 0.15, 0.025 * k), W(sh, "retin", 0.015, (0.06, 0, 0)), B.sn(me, 0.2, 0.015 * k)],
          (me, "Y", -1), 1.0)
    B.add("DE" + tag, "long digital extensor", [B.sn(sh, 0.1, 0.02 * k), W(sh, "retin", 0.015, (0.06, 0, 0)), B.sn(me, 0.5, 0.015 * k),
                                                W(me, "fetlock", 0.012, (0.04, 0, 0.04)), B.sn(ft, 0.7, 0.012 * k)], (ft, "Y", -1), 1.0)
    B.add("ABD" + tag, "hip abductors", [B.near(th, o(0.03, 0.0, 0.08)), B.sn(th, -0.05, 0.0, 0.045 * k)], (th, "X", 1), 0.5)
    B.add("ADD" + tag, "adductors", [B.near(th, o(-0.02, -0.04, -0.04)), B.sn(th, 0.5, 0.0, -0.015 * k)], (th, "X", -1), 0.5)
    B.cover(th, "YX"); B.cover(sh, "Y"); B.cover(me, "Y"); B.cover(ft, "Y")


def _joint_wrap(B, child, key, r, side_off, axis=(0, 1, 0)):
    """A wrapping cylinder on the parent of ``child`` at the joint (femoral head, humeral head)."""
    i = B.sk.idx[B.b(child)]; par = B.sk.bones[B.sk.parent[i]].name
    nm = f"w_{key}_{B.sd}"; at = tuple(float(x) for x in B.sk.attach[i])
    if not any(w[1] == nm for w in B.ms.wraps):
        B.ms.wraps.append((par, nm, at, r * B.s, 0.04 * B.s, "cylinder", axis))
    p = tuple(float(a + b) for a, b in zip(at, (side_off[0] * B.s, B.my * side_off[1] * B.s, side_off[2] * B.s)))
    for x in B.ms.sites:
        if x[0] == par and x[2] == p:
            return ("wrap", nm, x[1])
    ss = f"ss_{key}_{B.sd}{sum(1 for x in B.ms.sites if x[1].startswith(f'ss_{key}_{B.sd}'))}"
    B.ms.sites.append((par, ss, p))
    return ("wrap", nm, ss)


def quadruped_set(c) -> MuscleSet:
    """Muscles of a ``bodies.quadruped``: 11 per hind leg, 11 per fore leg, back and belly (3 per side)."""
    s = c.params["spec"]["scale"]
    B = Builder(c, s); sk = c.skel
    spine = [b.name for b in sk.bones if b.name.startswith("spine")]
    cd = sk.bones[sk.idx["chest"]].radius / 1.15 / s         # half the chest depth for the reference size (0.11 m)
    sp = c.params["spec"]
    B.ms.targets = {"hind": sp["muscle_bw_hind"], "fore": sp["muscle_bw_fore"]}
    for sd in "LR":
        B.side(sd)
        B.group = "hind"
        zigzag_leg(B, "pelvis", "thighH_@", "shankH_@", "metaH_@", "pawH_@", tag="H")
        B.group = "fore"
        # fore leg. There is no shoulder blade bone: the shoulder joint sits on the chest, the muscles of the
        # shoulder blade start there
        th, sh, me, ft = "thighF_@", "shankF_@", "metaF_@", "pawF_@"
        W = lambda bone, key, r, side: B.wrap(bone, key, 1.0, (0, 0, 0), r, side)
        sho = lambda side: _joint_wrap(B, th, "shoulder", 0.028, side)
        B.add("BRC", "brachiocephalicus", [B.site("neck", 0.5, (0.0, 0.03, 0.0)), sho((0.06, 0, 0.02)), B.sn(th, 0.6, 0.025)], (th, "Y", -1), 0.6)
        B.add("SSP", "supraspinatus", [B.near(th, (-0.02, 0.0, 0.10)), sho((0.06, 0, 0.04)), B.sn(th, 0.05, 0.03)], (th, "Y", -1), 0.4)
        B.add("LATF", "latissimus dorsi", [B.site("chest", 0.0, (0.0, 0.04, 0.9 * cd)), sho((-0.06, 0, 0.0)), B.sn(th, 0.25, -0.012, -0.01)],
              (th, "Y", 1), 0.5)
        B.add("TER", "teres major and deltoid", [B.near(th, (-0.08, 0.0, 0.08)), sho((-0.06, 0, 0.02)), B.sn(th, 0.3, -0.015, 0.01)], (th, "Y", 1), 0.5)
        B.add("TRIF", "triceps", [B.near(th, (-0.07, 0.0, 0.06)), B.sn(th, 0.6, -0.03), B.sn(sh, -0.15, -0.035)], (sh, "Y", 1), 1.0)
        B.add("BICF", "biceps", [B.near(th, (0.03, 0.0, 0.01)), B.sn(th, 0.5, 0.025), B.sn(sh, 0.3, 0.025)], (sh, "Y", -1), 0.6)
        B.add("BRF", "brachialis", [B.sn(th, 0.4, 0.012, 0.012), B.sn(sh, 0.22, 0.02)], (sh, "Y", -1), 0.4)
        B.add("ECR", "extensor carpi radialis", [B.sn(th, 0.95, 0.02), W(sh, "carpf", 0.015, (0.05, 0, 0)), B.sn(me, 0.2, 0.015)], (me, "Y", -1), 1.0)
        B.add("FCU", "carpal flexors", [B.sn(th, 0.95, -0.015, -0.01), W(sh, "carpb", 0.015, (-0.05, 0, 0)), B.sn(me, 0.25, -0.02)], (me, "Y", 1), 1.0)
        B.add("DFF", "digital flexors", [B.sn(th, 0.95, -0.02), W(sh, "carpb", 0.015, (-0.05, 0, 0)), B.sn(me, 0.5, -0.02),
                                         W(me, "sesamF", 0.012, (-0.04, 0, -0.04)), B.sn(ft, 0.7, -0.012)], (ft, "Y", 1), 0.7, ret=0.5)
        B.add("CDE", "common digital extensor", [B.sn(th, 0.95, 0.015, 0.01), W(sh, "carpf", 0.015, (0.05, 0, 0)), B.sn(me, 0.5, 0.015),
                                                 W(me, "fetlockF", 0.012, (0.04, 0, 0.04)), B.sn(ft, 0.7, 0.012)], (ft, "Y", -1), 1.0)
        B.add("ABDF", "shoulder abductors", [B.near(th, (0.0, 0.03, 0.06)), B.sn(th, 0.2, 0.0, 0.02)], (th, "X", 1), 0.5)
        B.add("PECF", "pectorals", [B.near(th, (0.03, -0.07, -0.05)), B.sn(th, 0.3, 0.0, -0.015)], (th, "X", -1), 0.5)
        B.cover(th, "YX"); B.cover(sh, "Y"); B.cover(me, "Y"); B.cover(ft, "Y")
        # trunk: epaxial muscles over the back (extension), rectus abdominis under the belly (flexion), obliques
        # for bending sideways; the axial twist keeps joint motors
        B.group = ""
        B.add("EPAX", "epaxial (longissimus)", [B.site("pelvis", 0.4, (0.0, 0.03, 0.9 * cd))] + [B.site(b, 0.5, (0.0, 0.03, 0.95 * cd)) for b in spine]
              + [B.site("chest", 0.6, (0.0, 0.03, cd))], (spine[0], "Y", -1), 0.6)
        B.add("RECT", "rectus abdominis", [B.site("pelvis", 0.6, (0.0, 0.02, -0.8 * cd))] + [B.site(b, 0.5, (0.0, 0.02, -1.1 * cd)) for b in spine]
              + [B.site("chest", 0.6, (0.0, 0.02, -1.1 * cd))], (spine[0], "Y", 1), 0.5)
        B.add("OBLQ", "obliques", [B.site("pelvis", 0.7, (0.0, 0.9 * cd, 0.0))] + [B.site(b, 0.5, (0.0, 1.0 * cd, 0.0)) for b in spine]
              + [B.site("chest", 0.4, (0.0, 1.0 * cd, 0.0))], (spine[0], "Z", 1), 0.6)
        for b in spine + ["chest"]:
            B.cover(b, "YZ")
        # tail: the sacrocaudal muscles in two runs along the tail (from the pelvis over the first two joints, from the
        # second vertebra over the last two), above (levators), below (depressors) and at the side (lateral flexors),
        # so a cat can swing and curl it in an S; the twist keeps joint motors
        tail = [b.name for b in sk.bones if b.name.startswith("tail")]
        if len(tail) >= 4:
            rt = sk.bones[sk.idx["tail0"]].radius / s * 1.3
            runs = ((None, tail[0:2]), (tail[1], tail[2:4]))     # (bone the run starts on, the bones it pulls)
            for k, (start, chain) in enumerate(runs):
                for key, title, off, prim in (("TLEV", "tail levator", (0.0, 0.4, 1.0), (chain[0], "Y", 1)),
                                              ("TDEP", "tail depressor", (0.0, 0.4, -1.0), (chain[0], "Y", -1)),
                                              ("TLAT", "tail lateral flexor", (0.0, 1.0, 0.0), (chain[0], "Z", -1))):
                    o = tuple(v * rt for v in off)
                    # the base run starts on the sacrum and the pelvic floor ahead of the tail's root, so a tail raised or
                    # lowered far still has leverage
                    head = (B.near("tail0", (2.5 * rt, o[1], o[2] * 1.6)) if start is None
                            else B.site(start, 0.6, o))
                    B.add(f"{key}{k}", title + (" (base)" if k == 0 else " (tip)"),
                          [head] + [B.site(b, 0.5, o) for b in chain], prim, 0.5 if key != "TLAT" else 1.0)
            for b in tail:
                B.cover(b, "YZ")
    return B.ms


def bird_set(c) -> MuscleSet:
    """Muscles of a ``bodies.bird``: the flight muscles on the keel (pectoralis pulls the wing down; supracoracoideus
    lifts it through the triosseal canal, a pulley over the shoulder), wing protractor and retractor, elbow and wrist
    flexors and extensors, and the leg (11 per leg). The wing's twist and the trunk keep joint motors."""
    ch = c.params["spec"]["scale"]                          # 1 for a crow
    B = Builder(c, ch); sk = c.skel
    for sd in "LR":
        B.side(sd)
        wa, wb, wc = "wing_a_@", "wing_b_@", "wing_c_@"
        keel = lambda t, z: B.site("chest", t, (0.0, 0.004, -z))
        # the shoulder: a pulley across the stroke (axis forward) and one across the sweep (axis up), so paths stay on
        # their side of the joint through a full beat
        stroke = lambda side: _joint_wrap(B, wa, "wstroke", 0.008, side, axis=(1, 0, 0))
        sweep = lambda side: _joint_wrap(B, wa, "wsweep", 0.008, side, axis=(0, 0, 1))
        B.add("PECT", "pectoralis", [keel(0.4, 0.05), stroke((0.0, 0.0, -0.03)), B.site(wa, 0.18, (0.004, 0.0, -0.008))], (wa, "X", -1), 1.0)
        B.add("SUPRA", "supracoracoideus", [keel(0.55, 0.04), B.near(wa, (0.004, -0.006, 0.012)), stroke((0.0, 0.0, 0.03)),
                                            B.site(wa, 0.08, (0.0, 0.0, 0.009))], (wa, "X", 1), 1.0)
        B.add("PROT", "wing protractors", [B.near(wa, (0.03, -0.01, 0.01)), sweep((0.03, 0.0, 0.0)), B.site(wa, 0.3, (0.007, 0.0, 0.0))], (wa, "Z", -1), 1.0)
        B.add("RETR", "wing retractors", [B.near(wa, (-0.04, -0.01, 0.015)), sweep((-0.03, 0.0, 0.0)), B.site(wa, 0.2, (-0.007, 0.0, 0.0))], (wa, "Z", 1), 1.0)
        B.cover(wa, "XZ")
        elbow = lambda side: _joint_wrap(B, wb, "welbow", 0.004, side, axis=(0, 0, 1))
        B.add("BICW", "biceps", [B.site(wa, 0.08, (0.006, 0.0, 0.0)), elbow((0.02, 0.0, 0.0)), B.site(wb, 0.15, (0.005, 0.0, 0.0))], (wb, "Z", -1), 1.0)
        B.add("TRIW", "triceps", [B.site(wa, 0.15, (-0.006, 0.0, 0.0)), elbow((-0.02, 0.0, 0.0)), B.site(wb, 0.08, (-0.006, 0.0, 0.0))],
              (wb, "Z", 1), 1.0)
        B.cover(wb, "Z")
        wrist = lambda side: _joint_wrap(B, wc, "wwrist", 0.004, side, axis=(0, 0, 1))
        B.add("EMR", "extensor metacarpi radialis", [B.site(wa, 0.95, (0.005, 0.0, 0.0)), elbow((0.02, 0.0, 0.0)), B.site(wb, 0.6, (0.005, 0.0, 0.0)),
                                                     wrist((0.02, 0.0, 0.0)), B.site(wc, 0.1, (0.004, 0.0, 0.0))], (wc, "Z", -1), 1.0)
        B.add("FCU", "flexor carpi ulnaris", [B.site(wb, 0.3, (-0.004, 0.0, 0.0)), wrist((-0.02, 0.0, 0.0)), B.site(wc, 0.1, (-0.004, 0.0, 0.0))],
              (wc, "Z", 1), 1.0)
        B.cover(wc, "Z")
        zigzag_leg(B, "pelvis", "thigh_@", "shank_@", "meta_@", "toes_@", k=0.25)
    return B.ms


def merge(*sets):
    out = MuscleSet()
    for m in sets:
        out.muscles += m.muscles; out.wraps += m.wraps; out.sites += m.sites; out.covered |= m.covered
    return out


SETS = {"human_legs": human_legs, "human": lambda c: merge(human_legs(c), human_upper(c)), "quadruped": quadruped_set,
        "bird": bird_set}
