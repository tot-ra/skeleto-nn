"""Parametric bodies: human, quadruped (dog, cat, horse, ...), bird, snake. A species or a person is a set
of numbers; the same planner and the same physics executor run all of them."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from skeleton import Skeleton, Bone, AX, rot

X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)

def nrm(v):
    v = np.array(v, float); return tuple(v / np.linalg.norm(v))

@dataclass
class Leg:
    name: str
    side: str
    chain: list            # bone indices: [thigh, shank, (meta), foot]
    anchor: int            # trunk bone the hip hangs from
    heel: np.ndarray       # contact points in the last bone's local frame
    ball: np.ndarray
    reach: float = 0.0     # fully stretched length from hip to ankle
    fore: bool = False
    toes: int | None = None    # toe bone (biped): bends so the toes stay on the ground while the heel rises

@dataclass
class Arm:
    name: str
    side: str
    chain: list            # [upper, lower, hand]
    reach: float = 0.0
    clav: int | None = None    # clavicle bone the upper arm hangs from

@dataclass
class Creature:
    skel: Skeleton
    kind: str                         # biped | quadruped | bird | snake
    legs: list = field(default_factory=list)
    arms: list = field(default_factory=list)
    spine: list = field(default_factory=list)    # pelvis -> chest
    neck: list = field(default_factory=list)
    head: int = -1
    tail: list = field(default_factory=list)
    wings: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    z0: float = 0.0                   # root height when standing at rest

    def leg_len(self):
        return float(np.mean([l.reach for l in self.legs])) if self.legs else 1.0

# ---------------------------------------------------------------------------------------------
@dataclass
class HumanBody:
    """Everything that makes one person differ from another. 1.0 = the 75 kg, 1.75 m adult."""
    height: float = 1.75
    mass: float = 75.0
    leg_ratio: float = 1.0       # legs relative to torso
    belly: float = 0.0           # kg carried in front of the abdomen
    armour: float = 0.0          # kg on chest and thighs
    pack: float = 0.0            # kg on the back
    dress: float = 0.0           # 0..1 skirt: restricts stride, adds drag
    strength: float = 1.0
    stoop: float = 0.0           # rest kyphosis, radians
    width: float = 1.0           # shoulder and hip width
    vigor: float = 1.0           # muscle power for the body size: 0.45 starved, 0.55 old, 1.0 adult, 1.5 energetic child
    peg: str = ""              # "L", "R" or "LR": the shank and foot of that leg are replaced by a rigid stick
    outfit: str = "none"        # key of OUTFITS: clothing that restricts joints, adds mass or changes the foot
    sex: str = "m"                # "m" | "f": changes shoulder/hip proportions and how a blow to the groin is felt

# What clothing does: range-of-motion factors per bone prefix (or explicit {axis: (lo, hi)}), extra mass in kg per limb segment,
# and gait parameters the planner reads. Nothing is animated: the limits, masses and the foot shape change and the planner copes.
OUTFITS = {
    "none": dict(),
    "skirt_tight": dict(rom={"thigh": {"Y": (-62, 20), "X": (-6, 10), "Z": (-8, 8)}, "shank": 1.0}, mass={"thigh": 0.5}, params=dict(stride_scale=0.62, v_max=1.9, balance=0.9, arm_swing=0.9)),
    "skirt_wide": dict(rom={"thigh": {"Y": (-105, 25), "X": (-20, 40), "Z": (-30, 30)}}, mass={"thigh": 1.1}, params=dict(stride_scale=0.92, v_max=3.4, arm_swing=0.9)),
    "trousers_tight": dict(rom={"thigh": {"Y": (-85, 22), "X": (-12, 30), "Z": (-22, 22)}, "shank": {"Y": (0, 125)}}, params=dict(stride_scale=0.9, v_max=3.6)),
    "boots_heavy": dict(rom={"foot": {"Y": (-18, 30), "X": (-8, 8), "Z": (-8, 8)}, "toes": {"Y": (-22, 12)}}, mass={"foot": 1.4, "shank": 0.5}, params=dict(stride_scale=0.94, toe_amp=0.45)),
    "heels_high": dict(rom={"foot": {"Y": (-10, 50), "X": (-8, 8), "Z": (-8, 8)}}, params=dict(stride_scale=0.74, v_max=1.8, heel_pitch=0.50, balance=0.55, toe_amp=0.25)),
    "chainmail": dict(rom={"lumbar": 0.85, "thorax": 0.85, "chest": 0.85, "clav": 0.85, "uarm": 0.9}, mass={"chest": 6.0, "pelvis": 1.5, "uarm": 0.8, "thigh": 0.6}, params=dict(stride_scale=0.97, arm_swing=0.9)),
    "plate": dict(rom={"lumbar": 0.35, "thorax": 0.3, "chest": 0.3, "neck": 0.5, "clav": 0.35, "uarm": 0.55, "farm": {"Y": (-125, 0)}, "thigh": {"Y": (-80, 15), "X": (-12, 28), "Z": (-18, 18)},
                  "shank": {"Y": (0, 120)}, "foot": {"Y": (-18, 35), "X": (-8, 8), "Z": (-8, 8)}, "toes": {"Y": (-30, 15)}},
                  mass={"chest": 11.0, "pelvis": 3.0, "uarm": 1.8, "farm": 1.2, "thigh": 2.4, "shank": 2.0, "foot": 1.2}, params=dict(stride_scale=0.84, v_max=3.8, balance=1.2, arm_swing=0.45, toe_amp=0.55)),
}

def _apply_outfit(bones, name):
    o = OUTFITS.get(name, {})
    for bn in bones:
        for prefix, spec in o.get("rom", {}).items():
            if bn.name.startswith(prefix):
                if isinstance(spec, dict): bn.lim = {ax: spec.get(ax, bn.lim[ax]) for ax in bn.lim}
                else: bn.lim = {ax: (lo * spec, hi * spec) for ax, (lo, hi) in bn.lim.items()}
    extra = []
    for bone, kg in o.get("mass", {}).items():
        if bone in ("chest", "pelvis"): extra.append((bone, 0.5, (0, 0, 0), kg, "outfit"))
        else: extra += [(bone + "_L", 0.5, (0, 0, 0), kg, "outfit"), (bone + "_R", 0.5, (0, 0, 0), kg, "outfit")]
    return extra, o.get("params", {})

def human(b: HumanBody | None = None, name="human", rig: dict | None = None) -> Creature:
    b = b or HumanBody()
    if b.vigor < 1.0 and b.stoop == 0.0:
        import dataclasses; b = dataclasses.replace(b, stoop=0.5 * (1 - b.vigor))     # weak bodies stand stooped
    rig = rig or {}
    s = b.height / 1.75
    sl = s * b.leg_ratio ** 0.5; st = s / b.leg_ratio ** 0.5     # leg scale, torso scale
    w = b.width * s ** 0.5
    hipw = 1.08 if b.sex == "f" else 1.0; shw = 0.93 if b.sex == "f" else 1.0
    mscale = (b.mass - b.belly - b.armour - b.pack) / 75.0
    tau = b.strength * mscale * s
    R = lambda r: r * s ** 0.5 * mscale ** 0.3
    L = {
        "pelvis": 0.11 * st, "lumbar": 0.17 * st, "chest": 0.27 * st, "neck": 0.08 * st, "head": 0.21 * st,
        "thigh": 0.43 * sl, "shank": 0.43 * sl, "foot": 0.21 * s, "uarm": 0.30 * s, "farm": 0.27 * s, "hand": 0.09 * s,
    }
    for k_ in ("pelvis", "lumbar", "chest", "neck", "head", "thigh", "shank", "foot", "uarm", "farm", "hand"):
        if k_ in rig: L[k_] = rig[k_]
    ah = rig.get("ah", 0.075 * s)     # ankle height above the sole
    hip_lat = rig.get("hip_lat", 0.095 * w * hipw); sh_lat = rig.get("shoulder_lat", 0.19 * w * shw); sh_t = rig.get("shoulder_t", 0.93); heel_x = rig.get("heel", -0.055 * s)
    m = lambda x: x * mscale
    # trunk: three lumbar vertebral groups, a lower thoracic group and the chest (rib cage + upper thoracic)
    lumb_lim = {"Y": (-9, 14), "X": (-6, 6), "Z": (-9, 9)}; thor_lim = {"Y": (-6, 12), "X": (-9, 9), "Z": (-15, 15)}
    spine_h = []; prev_h = "pelvis"
    for nm_, ln_, ms_, rr_, lm_, wd_ in (("lumbar", L["lumbar"] / 3, 8 / 3, 0.10, lumb_lim, 0.14), ("lumbar1", L["lumbar"] / 3, 8 / 3, 0.10, lumb_lim, 0.14), ("lumbar2", L["lumbar"] / 3, 8 / 3, 0.10, lumb_lim, 0.14),
                                         ("thorax", L["chest"] * 0.45, 9.0, 0.12, thor_lim, 0.16), ("chest", L["chest"] * 0.55, 11.0, 0.13, thor_lim, 0.17)):
        spine_h.append(Bone(nm_, prev_h, nrm((np.sin(b.stoop * .4), 0, 1)), ln_, m(ms_), R(rr_), order="YXZ", lim=dict(lm_), group="trunk", tau=250 * tau * 1.4 / 2.5 * (1.0 if "lumbar" in nm_ else 1.6),
                         shape="ellipsoid", size=(0.10 * w, wd_ * w, ln_ / 2 + 0.02))); prev_h = nm_
    bones = [
        Bone("pelvis", None, Z, L["pelvis"], m(11), R(0.10), group="trunk", shape="ellipsoid", size=(0.10 * w, 0.15 * w, L["pelvis"] / 2 + 0.03)),
        *spine_h,
        Bone("neck", "chest", nrm((np.sin(b.stoop * .3), 0, 1)), L["neck"], m(1.0), R(0.045), order="YXZ",
             lim={"Y": (-40, 45), "X": (-25, 25), "Z": (-50, 50)}, group="neck", tau=40 * tau),
        Bone("head", "neck", Z, L["head"], m(4.5), R(0.085), order="YXZ",
             lim={"Y": (-40, 40), "X": (-30, 30), "Z": (-60, 60)}, group="head", tau=40 * tau, shape="ellipsoid", size=(0.09 * s, 0.075 * s, L["head"] / 2 + 0.01)),
        Bone("thigh_L", "pelvis", (0, 0, -1), L["thigh"], m(8.0), R(0.07), t=0.0, offset=(0, hip_lat, 0), order="ZXY",
             lim={"Y": (-120, 25), "X": (-20, 45), "Z": (-35, 35)}, side="L", group="leg", tau=330 * tau),
        Bone("shank_L", "thigh_L", (0, 0, -1), L["shank"], m(3.5), R(0.05), order="YXZ", lim={"Y": (0, 150)}, side="L", group="leg", tau=250 * tau),
        Bone("foot_L", "shank_L", X, L["foot"] * 0.76, m(1.0), R(0.04), order="YXZ",
             lim={"Y": (-30, 50), "X": (-20, 20), "Z": (-15, 15)}, side="L", group="foot", tau=110 * tau, shape="box", size=(L["foot"] * 0.38 + 0.03 * s, 0.045 * s, 0.03 * s)),
        Bone("toes_L", "foot_L", X, L["foot"] * 0.24, m(0.2), R(0.03), order="YXZ",
             lim={"Y": (-70, 35)}, side="L", group="foot", tau=25 * tau, shape="box", size=(L["foot"] * 0.12 + 0.01 * s, 0.045 * s, 0.02 * s)),
        Bone("clav_L", "chest", (0, 1, 0), 0.16 * w * s ** 0.5, m(0.6), R(0.02), t=sh_t, offset=(0, 0.03 * w, 0), order="XZY",
             lim={"X": (-12, 38), "Z": (-22, 28)}, side="L", group="arm", tau=60 * tau),
        Bone("uarm_L", "clav_L", (0, 0, -1), L["uarm"], m(2.1), R(0.045), t=1.0, order="ZXY",
             lim={"Y": (-170, 70), "X": (-5, 140), "Z": (-70, 70)}, side="L", group="arm", tau=70 * tau),
        Bone("farm_L", "uarm_L", (0, 0, -1), L["farm"], m(1.5), R(0.038), order="YXZ", lim={"Y": (-150, 0)}, side="L", group="arm", tau=45 * tau),
        Bone("hand_L", "farm_L", (0, 0, -1), L["hand"], m(0.5), R(0.04), order="YXZ",
             lim={"Y": (-60, 60), "X": (-30, 30)}, side="L", group="arm", tau=8 * tau),
    ]
    sk_extra, outfit_params = _apply_outfit(bones, b.outfit)
    if b.belly: sk_extra.append(("lumbar1", 0.4, (0.13 * s, 0, 0), b.belly, "belly"))
    if b.armour: sk_extra += [("chest", 0.5, (0.02, 0, 0), b.armour * 0.6, "armour"),
                              ("thigh_L", 0.5, (0, 0, 0), b.armour * 0.1, "armour"), ("thigh_R", 0.5, (0, 0, 0), b.armour * 0.1, "armour"),
                              ("pelvis", 0.5, (0, 0, 0), b.armour * 0.2, "armour")]
    if b.pack: sk_extra.append(("chest", 0.5, (-0.17 * s, 0, 0), b.pack, "pack"))
    if b.dress: sk_extra += [("thigh_L", 0.7, (0, 0, 0), 1.2 * b.dress, "dress"), ("thigh_R", 0.7, (0, 0, 0), 1.2 * b.dress, "dress")]
    sk = Skeleton(name, bones, sk_extra)
    _normalize_mass(sk, b.mass + sum(e[3] for e in sk_extra if e[4] == "outfit"))
    c = Creature(sk, "biped")
    i = sk.idx
    c.spine = [i[n_] for n_ in ("pelvis", "lumbar", "lumbar1", "lumbar2", "thorax", "chest")]; c.neck = [i["neck"]]; c.head = i["head"]
    for side in "LR":
        ch = [i["thigh_" + side], i["shank_" + side], i["foot_" + side]]
        c.legs.append(Leg("leg_" + side, side, ch, i["pelvis"], np.array([heel_x, 0, -ah]), np.array([L["foot"] * 0.76, 0, -ah]),
                          reach=L["thigh"] + L["shank"], toes=i["toes_" + side]))
        c.arms.append(Arm("arm_" + side, side, [i["uarm_" + side], i["farm_" + side], i["hand_" + side]], reach=L["uarm"] + L["farm"] + L["hand"], clav=i["clav_" + side]))
    c.params = dict(height=b.height, mass=b.mass, dress=b.dress, ankle_h=ah, body=b.__dict__.copy(),
                    stand_ratio=0.99, step_ratio=0.62, arm_swing=1.0, skirt=b.dress, sex=b.sex)
    c.params.update(outfit_params)
    v_ = b.vigor
    c.params.update(vigor=v_, jump_gain=1.0 * v_ ** 0.6, v_sprint=6.2 * v_ ** 0.7, acc_scale=v_ ** 0.5)
    if v_ < 1.0:                                           # weak: slower, shorter steps, more time on both feet, stooped, small arm swing
        c.params["v_max"] = min(c.params.get("v_max") or 99.0, 6.2 * v_ ** 0.7 * 0.55)           # even a fast walk is a run for them
        c.params["stride_scale"] = c.params.get("stride_scale", 1.0) * v_ ** 0.35
        c.params["duty_add"] = 0.14 * (1 - v_); c.params["arm_swing"] = c.params.get("arm_swing", 1.0) * (0.45 + 0.55 * v_)
        c.params["balance"] = c.params.get("balance", 1.0) * (0.55 + 0.45 * v_); c.params["toe_amp"] = c.params.get("toe_amp", 1.0) * (0.5 + 0.5 * v_)
    elif v_ > 1.2:                                         # energetic: springy steps
        c.params["bounce"] = 0.05 * (v_ - 1.0) / 0.5
    for side in b.peg:                                  # the shank and foot of this leg are a rigid stick with a point contact
        for nm in ("foot_", "toes_"):
            bn = sk.bones[sk.idx[nm + side]]; bn.lim = {ax: (0.0, 0.0) for ax in "XYZ"}
        ft = sk.bones[sk.idx["foot_" + side]]; ft.length = 0.03; ft.radius = 0.012; ft.mass = 0.2; ft.shape = "capsule"; ft.size = ()
        tp = sk.bones[sk.idx["toes_" + side]]; tp.length = 0.004; tp.radius = 0.008; tp.mass = 0.05; tp.shape = "capsule"; tp.size = ()
        sh = sk.bones[sk.idx["shank_" + side]]; sh.length = sh.length + ah - 0.012; sh.radius = 0.018; sh.mass = 1.0
        sk.refresh()
        leg = next(l for l in c.legs if l.side == side); leg.heel = np.array([-0.004, 0, -0.012]); leg.ball = np.array([0.026, 0, -0.012])
        c.params.setdefault("leg_gait", {})[side] = dict(duty=0.78, hike=0.07, lean=0.075)
    if b.peg: c.params["toe_amp"] = 0.4
    c.z0 = _rest_height(c)
    return c

def _normalize_mass(sk: Skeleton, total: float):
    """Scale bone masses so bones plus extras add up to `total` kg."""
    extra = sum(e[3] for e in sk.extra)
    sk.mass *= max(total - extra, 0.1) / sk.mass.sum()

def _rest_height(c: Creature):
    sk = c.skel
    S, E, Rw = sk.fk(np.zeros(3), np.eye(3), sk.zeros())
    low = 1e9
    for l in c.legs:
        f = l.chain[-1]
        for p in (l.heel, l.ball):
            low = min(low, (S[f] + Rw[f] @ p)[2])
    return -low

def human_game_rig(name="game_human") -> Creature:
    """A human whose proportions equal a rig that was measured at rest (KayKit-style proportions)."""
    rig = dict(pelvis=0.044, lumbar=0.308, chest=0.162, neck=0.058, head=0.163, thigh=0.39, shank=0.253, foot=0.30, uarm=0.28, farm=0.27, hand=0.10,
               ah=0.147, hip_lat=0.101, shoulder_lat=0.14, shoulder_t=0.51, heel=-0.07)
    c = human(HumanBody(height=1.62, mass=70.0), name=name, rig=rig)
    return c

# ---------------------------------------------------------------------------------------------
@dataclass
class QuadSpec:
    name: str = "dog"
    mass: float = 30.0
    scale: float = 1.0         # linear scale w.r.t. the dog template
    leg_scale: float = 1.0     # extra leg length factor
    cannon: float = 1.0        # distal leg segment length factor (horse: long)
    spine_flex: float = 1.0    # 1 = dog, 1.6 = cat, 0.35 = horse
    neck_len: float = 1.0
    tail_len: float = 1.0
    tail_n: int = 4
    head_len: float = 1.0
    body_len: float = 1.0
    chest_depth: float = 1.0
    colour: str = "dog"
    # --- trunk anatomy: how many mobile joints between pelvis and shoulders, and how far the whole column bends (degrees, total)
    n_spine: int = 5                    # mobile trunk joints (lumbar + thoracolumbar, the last one is the chest)
    ext_rom: float = 30.0               # dorsiflexion (arching the back down) of the whole column
    flex_rom: float = 45.0              # ventroflexion (hunching the back up)
    lat_rom: float = 55.0               # lateral bending
    twist_rom: float = 20.0             # axial twist
    core_power: float = 1.0             # trunk torque capacity relative to mass^(2/3)
    n_neck: int = 3                     # cervical joints that move independently
    neck_rom: float = 95.0              # total sagittal range of the neck (deg)
    neck_nod: float = 0.06              # head/neck nod amplitude at a walk (rad)
    neck_speed_drop: float = 0.1        # how far the neck stretches forward and down at speed (rad)
    tail_type: str = "wag"              # wag | balance | hair | curl | none
    foot_type: str = "paw"              # paw | hoof
    jump_pitch: float = 0.65            # max trunk rotation about the hind feet at take-off (rad)
    jump_gain: float = 2.2              # take-off speed a hind-limb push can give, in units of sqrt(g * leg length)

# Numbers follow comparative anatomy: cats have ~20 mobile thoraco-lumbar vertebrae and bend ~110 degrees in the sagittal plane,
# dogs ~100 degrees less than that, horses have a nearly rigid thoracolumbar column (~25 degrees) with a mobile lumbosacral joint.
# jump_gain is the muscle side of the story: relative hind-limb power (cat ~3.4 sqrt(g L), dog ~2.2, horse ~1.35, pig ~1.0).
QUADS = {
    "dog": QuadSpec("dog", 30.0, 1.0),
    "cat": QuadSpec("cat", 4.5, 0.58, tail_n=6, leg_scale=0.95, spine_flex=1.7, tail_len=1.3, neck_len=0.8, head_len=0.7, body_len=1.05,
                    n_neck=3, neck_rom=110.0, neck_nod=0.04, neck_speed_drop=0.05, tail_type="balance", n_spine=8, ext_rom=45.0, flex_rom=75.0, lat_rom=95.0, twist_rom=45.0, core_power=1.5, jump_gain=3.4, jump_pitch=1.05),
    "horse": QuadSpec("horse", 500.0, 2.45, tail_n=5, leg_scale=1.22, cannon=1.45, spine_flex=0.35, neck_len=1.3, tail_len=0.8, head_len=1.2, body_len=0.95, chest_depth=0.78,
                      n_neck=5, neck_rom=140.0, neck_nod=0.14, neck_speed_drop=0.30, tail_type="hair", foot_type="hoof", n_spine=3, ext_rom=9.0, flex_rom=14.0, lat_rom=16.0, twist_rom=6.0, core_power=0.9, jump_gain=1.35, jump_pitch=0.3),
    "wolf": QuadSpec("wolf", 40.0, 1.15, leg_scale=1.1, cannon=1.1, spine_flex=1.1, tail_len=1.0,
                     n_neck=4, neck_rom=100.0, neck_nod=0.07, neck_speed_drop=0.10, tail_type="wag_low", n_spine=6, ext_rom=32.0, flex_rom=52.0, lat_rom=60.0, twist_rom=22.0, core_power=1.15, jump_gain=2.5, jump_pitch=0.7),
    "pig": QuadSpec("pig", 100.0, 1.0, tail_n=4, leg_scale=0.55, cannon=0.6, spine_flex=0.5, neck_len=0.5, tail_len=0.3, head_len=1.1, body_len=1.15, chest_depth=1.45,
                    n_neck=2, neck_rom=50.0, neck_nod=0.10, neck_speed_drop=0.05, tail_type="curl", foot_type="hoof", n_spine=3, ext_rom=10.0, flex_rom=18.0, lat_rom=20.0, twist_rom=8.0, core_power=0.8, jump_gain=1.0, jump_pitch=0.25),
}

def quadruped(spec: QuadSpec | str = "dog", load: float = 0.0, name=None, missing: str = "", peg: str = "") -> Creature:
    sp = QUADS[spec] if isinstance(spec, str) else spec
    s = sp.scale; mscale = sp.mass / 30.0
    tau = mscale ** (2 / 3.) * 1.0
    d = lambda v: nrm(v)
    ls = s * sp.leg_scale
    cd = 0.11 * s * sp.chest_depth
    Lp, Lsp, Lc = 0.20 * s * sp.body_len, 0.22 * s * sp.body_len, 0.22 * s * sp.body_len
    fl = sp.spine_flex
    m = lambda x: x * mscale
    nj = max(2, sp.n_spine); nsb = nj - 1                      # mobile joints; nsb "spine" bones plus the chest
    per = lambda deg: deg / nj
    slim = {"Y": (-per(sp.ext_rom), per(sp.flex_rom)), "X": (-per(sp.twist_rom), per(sp.twist_rom)), "Z": (-per(sp.lat_rom), per(sp.lat_rom))}
    stau = 400.0 * tau * sp.core_power / nj
    spine_bones = []
    prev_s = "pelvis"
    for k in range(nsb):
        nm = "spine" if k == 0 else f"spine{k}"
        spine_bones.append(Bone(nm, prev_s, X, Lsp / nsb, m(5.5 / nsb), cd * 1.05, order="YXZ", lim=dict(slim), group="trunk", tau=stau)); prev_s = nm
    spine_bones.append(Bone("chest", prev_s, X, Lc, m(7.0), cd * 1.15, order="YXZ", lim=dict(slim), group="trunk", tau=stau))
    nn = max(1, sp.n_neck); nlim = {"Y": (-0.45 * sp.neck_rom / nn, 0.55 * sp.neck_rom / nn), "X": (-30 / nn * 1.5, 30 / nn * 1.5), "Z": (-50 / nn * 1.4, 50 / nn * 1.4)}
    neck_bones = []; prev_n = "chest"
    for k in range(nn):
        nm = "neck" if k == 0 else f"neck{k}"
        neck_bones.append(Bone(nm, prev_n, d((1, 0, 0.7 if k == 0 else 0.35)), 0.20 * s * sp.neck_len / nn, m(1.8 / nn), cd * 0.55, t=0.95 if k == 0 else 1.0, order="YXZ", lim=dict(nlim), group="neck", tau=60 * tau / nn)); prev_n = nm
    bones = [
        Bone("pelvis", None, X, Lp, m(5.0), cd, group="trunk"),
        *spine_bones,
        *neck_bones,
        Bone("head", prev_n, d((1, 0, -0.35)), 0.20 * s * sp.head_len, m(2.0), 0.05 * s, order="YXZ", lim={"Y": (-45, 45), "X": (-30, 30), "Z": (-50, 50)}, group="head", tau=30 * tau),
    ]
    prev = "pelvis"; tl = 0.09 * s * sp.tail_len * 4 / sp.tail_n
    for k in range(sp.tail_n):
        bones.append(Bone(f"tail{k}", prev, d((-1, 0, 0.35 if k == 0 else 0.0)), tl, m(0.35 / sp.tail_n * 4), 0.02 * s, t=0.0 if k == 0 else 1.0,
                          offset=(0, 0, 0.02 * s) if k == 0 else (0, 0, 0), order="YXZ", lim={"Y": (-60, 60), "Z": (-60, 60), "X": (-20, 20)}, group="tail", tau=8 * tau))
        prev = f"tail{k}"
    cn = sp.cannon
    # hind leg: femur, tibia, metatarsus, paw (digitigrade zig-zag)
    hind = [("thigh", d((0.40, 0, -0.92)), 0.22 * ls, 4.0, {"Y": (-80, 60), "X": (-25, 25), "Z": (-25, 25)}, 260),
            ("shank", d((-0.60, 0, -0.80)), 0.21 * ls, 1.4, {"Y": (-70, 90)}, 150),
            ("meta", d((0.05, 0, -1.0)), 0.12 * ls * cn, 0.5, {"Y": (-100, 55)}, 70),
            ("paw", X, 0.075 * s, 0.3, {"Y": (-45, 70), "X": (-10, 10)}, 40)]
    fore = [("thigh", d((-0.20, 0, -0.98)), 0.17 * ls, 3.0, {"Y": (-80, 80), "X": (-30, 30), "Z": (-25, 25)}, 260),
            ("shank", d((0.20, 0, -0.98)), 0.20 * ls, 1.2, {"Y": (-30, 140)}, 150),
            ("meta", d((0, 0, -1.0)), 0.10 * ls * cn, 0.5, {"Y": (-30, 120)}, 70),
            ("paw", X, 0.06 * s, 0.3, {"Y": (-45, 70), "X": (-10, 10)}, 40)]
    if sp.foot_type == "hoof":          # one toe in a hoof: a short stiff last segment instead of a flexible pad
        hind[3] = ("paw", X, 0.075 * s * 0.8, 0.5, {"Y": (-15, 35), "X": (-4, 4)}, 60)
        fore[3] = ("paw", X, 0.06 * s * 0.8, 0.5, {"Y": (-15, 35), "X": (-4, 4)}, 60)
    for tag, defn, anchor, t, off in (("H", hind, "pelvis", 0.15, (0, 0.085 * s * sp.chest_depth, -0.02 * s)),
                                       ("F", fore, "chest", 0.80, (0, 0.085 * s * sp.chest_depth, -0.04 * s))):
        prev = None
        for part, dr, ln, ms, lim, tq in defn:
            nm = f"{part}{tag}_L"
            bones.append(Bone(nm, prev or anchor, dr, ln, m(ms), 0.035 * s * (1.0 if part != "paw" else 0.8),
                              t=t if prev is None else 1.0, offset=off if prev is None else (0, 0, 0), order="ZXY" if part == "thigh" else "YXZ",
                              lim=lim, side="L", group="leg" if part != "paw" else "foot", tau=tq * tau))
            prev = nm
    extra = []
    if load: extra.append(("spine", 0.5, (0, 0, cd + 0.02), load, "load"))
    sk = Skeleton(name or sp.name, bones, extra)
    _normalize_mass(sk, sp.mass + load)
    c = Creature(sk, "quadruped"); i = sk.idx
    c.spine = [i["pelvis"]] + [i["spine" if k == 0 else f"spine{k}"] for k in range(nsb)] + [i["chest"]]; c.neck = [i["neck" if k == 0 else f"neck{k}"] for k in range(nn)]; c.head = i["head"]
    c.tail = [i[f"tail{k}"] for k in range(sp.tail_n)]
    paw_r = 0.025 * s
    for tag, anchor, fore_ in (("H", "pelvis", False), ("F", "chest", True)):
        for side in "LR":
            ch = [i[f"thigh{tag}_{side}"], i[f"shank{tag}_{side}"], i[f"meta{tag}_{side}"], i[f"paw{tag}_{side}"]]
            pl = sk.length[ch[-1]]
            c.legs.append(Leg(f"{tag}{side}", side, ch, i[anchor], np.array([-0.01 * s, 0, -paw_r]), np.array([pl, 0, -paw_r]),
                              reach=float(sum(sk.length[k] for k in ch[:-1])), fore=fore_))
    c.params = dict(mass=sp.mass, spec=sp.__dict__.copy(), stand_ratio=1.0, spine_flex=fl, tail=sp.tail_n, jump_gain=sp.jump_gain, jump_pitch=sp.jump_pitch, tail_type=sp.tail_type, neck_nod=sp.neck_nod, neck_speed_drop=sp.neck_speed_drop, foot_type=sp.foot_type,
                    ext_rom=math.radians(sp.ext_rom), flex_rom=math.radians(sp.flex_rom))
    if missing: c.params["nwb"] = missing                 # one leg is gone (or held up): a tripod gait
    for nm_ in ([peg] if peg else []):                      # the last segments of this leg are a rigid stick
        for part in ("meta", "paw"):
            bn = sk.bones[sk.idx[f"{part}{nm_[0]}_{nm_[1]}"]]; bn.lim = {ax: (0.0, 0.0) for ax in bn.lim}; bn.radius = 0.012 * s
        c.params.setdefault("leg_gait", {})[nm_] = dict(duty=0.8, hike=0.04 * s)
    _ground_legs(c)
    c.z0 = _rest_height(c)
    return c

def _ground_legs(c: Creature):
    """Shift each leg's hip attach so every foot rests on the same ground level in the rest pose."""
    sk = c.skel
    S, E, Rw = sk.fk(np.zeros(3), np.eye(3), sk.zeros())
    z = {}
    for l in c.legs:
        f = l.chain[-1]
        z[l.name] = min((S[f] + Rw[f] @ l.heel)[2], (S[f] + Rw[f] @ l.ball)[2])
    for l in c.legs:
        sk.attach[l.chain[0]][2] -= (z[l.name] - min(z.values()))     # lift legs that are too low by moving the hip down
        # (a leg whose foot rests higher than the others gets its hip lowered so the foot meets the ground)

# ---------------------------------------------------------------------------------------------
@dataclass
class BirdSpec:
    name: str = "crow"
    mass: float = 0.45
    scale: float = 1.0
    leg_scale: float = 1.0
    neck_len: float = 1.0
    wing_len: float = 1.0
    tail_len: float = 1.0

BIRDS = {
    "crow": BirdSpec("crow", 0.45, 1.0),
    "gull": BirdSpec("gull", 0.9, 1.25, 0.9, 1.0, 1.35, 0.9),
    "heron": BirdSpec("heron", 1.9, 2.1, 1.9, 2.2, 1.4, 0.7),
    "sparrow": BirdSpec("sparrow", 0.03, 0.42, 0.8, 0.5, 0.8, 1.0),
    "duck": BirdSpec("duck", 1.1, 1.2, 0.55, 1.0, 1.0, 0.7),
}

def bird(spec: BirdSpec | str = "crow", name=None) -> Creature:
    sp = BIRDS[spec] if isinstance(spec, str) else spec
    s = 0.40 * sp.scale; mscale = sp.mass
    d = lambda v: nrm(v)
    m = lambda x: x * sp.mass
    rb = 0.075 * s / 0.4
    bones = [
        Bone("pelvis", None, d((1, 0, 0.12)), 0.12 * s / 0.4, m(0.30), rb * 0.9, group="trunk", shape="ellipsoid", size=(0.10 * s / 0.4, 0.07 * s / 0.4, 0.06 * s / 0.4)),
        Bone("chest", "pelvis", d((1, 0, 0.12)), 0.12 * s / 0.4, m(0.30), rb, order="YXZ", lim={"Y": (-35, 35), "X": (-25, 25), "Z": (-30, 30)}, group="trunk", tau=2.0 * m(1) * s),
    ]
    prev = "chest"
    nl = 0.065 * s / 0.4 * sp.neck_len
    for k in range(4):
        bones.append(Bone(f"neck{k}", prev, d((0.35, 0, 1) if k == 0 else (0.4, 0, 1) if k == 1 else (0.2, 0, 1) if k == 2 else (-0.1, 0, 1)),
                          nl, m(0.025), 0.022 * s / 0.4, t=1.0, order="YXZ", lim={"Y": (-60, 60), "X": (-20, 20), "Z": (-40, 40)}, group="neck", tau=0.6 * m(1) * s))
        prev = f"neck{k}"
    bones.append(Bone("head", prev, d((1, 0, -0.1)), 0.10 * s / 0.4, m(0.05), 0.03 * s / 0.4, order="YXZ", lim={"Y": (-50, 50), "X": (-40, 40), "Z": (-60, 60)}, group="head", tau=0.3 * m(1) * s))
    prev = "pelvis"
    for k in range(2):
        bones.append(Bone(f"tail{k}", prev, d((-1, 0, -0.1)), 0.10 * s / 0.4 * sp.tail_len, m(0.015), 0.02 * s / 0.4, t=0.0 if k == 0 else 1.0, order="YXZ",
                          lim={"Y": (-50, 50), "X": (-30, 30), "Z": (-40, 40)}, group="tail", tau=0.3 * m(1) * s))
        prev = f"tail{k}"
    ls = s / 0.4 * sp.leg_scale
    leg = [("thigh", d((0.5, 0, -0.85)), 0.06 * ls, 0.02, {"Y": (-60, 70), "X": (-25, 40), "Z": (-25, 25)}),
           ("shank", d((-0.55, 0, -0.84)), 0.10 * ls, 0.02, {"Y": (-70, 90)}),
           ("meta", d((0.15, 0, -0.99)), 0.09 * ls, 0.01, {"Y": (-100, 55)}),
           ("toes", X, 0.05 * ls, 0.005, {"Y": (-40, 70), "X": (-10, 10)})]
    prev = None
    for part, dr, ln, ms, lim in leg:
        nm = f"{part}_L"
        bones.append(Bone(nm, prev or "pelvis", dr, ln, m(ms), 0.012 * ls * (2.0 if part == "thigh" else 1.0), t=0.5 if prev is None else 1.0,
                          offset=(0, 0.045 * s / 0.4, -0.03 * s / 0.4) if prev is None else (0, 0, 0), order="ZXY" if part == "thigh" else "YXZ", lim=lim, side="L",
                          group="leg" if part != "toes" else "foot", tau=1.5 * m(1) * s))
        prev = nm
    wl = 0.16 * s / 0.4 * sp.wing_len; ch = s / 0.4
    bones += [
        Bone("wing_a_L", "chest", d((0, 1, 0)), wl, m(0.03), 0.02 * s / 0.4, t=0.7, offset=(0, 0.03 * s / 0.4, 0.02 * s / 0.4), order="XZY",
             lim={"X": (-100, 100), "Z": (-90, 100), "Y": (-80, 80)}, side="L", group="wing", tau=1.0 * m(1) * s, shape="box", size=(0.055 * ch, wl / 2, 0.004 * ch)),
        Bone("wing_b_L", "wing_a_L", d((0, 1, 0)), wl * 0.9, m(0.025), 0.016 * s / 0.4, order="XZY", lim={"X": (-30, 30), "Z": (-165, 20), "Y": (-30, 30)}, side="L", group="wing", tau=0.5 * m(1) * s, shape="box", size=(0.05 * ch, wl * 0.45, 0.003 * ch)),
        Bone("wing_c_L", "wing_b_L", d((0, 1, 0)), wl * 1.0, m(0.02), 0.012 * s / 0.4, order="XZY", lim={"X": (-40, 40), "Z": (-20, 155), "Y": (-20, 20)}, side="L", group="wing", tau=0.3 * m(1) * s, shape="box", size=(0.04 * ch, wl * 0.5, 0.003 * ch)),
    ]
    sk = Skeleton(name or sp.name, bones)
    _normalize_mass(sk, sp.mass)
    c = Creature(sk, "bird"); i = sk.idx
    c.spine = [i["pelvis"], i["chest"]]; c.neck = [i[f"neck{k}"] for k in range(4)]; c.head = i["head"]
    c.tail = [i["tail0"], i["tail1"]]
    c.wings = [[i["wing_a_L"], i["wing_b_L"], i["wing_c_L"]], [i["wing_a_R"], i["wing_b_R"], i["wing_c_R"]]]
    for side in "LR":
        ch = [i["thigh_" + side], i["shank_" + side], i["meta_" + side], i["toes_" + side]]
        pl = sk.length[ch[-1]]
        c.legs.append(Leg("leg_" + side, side, ch, i["pelvis"], np.array([-0.005, 0, -0.008 * ls]), np.array([pl, 0, -0.008 * ls]),
                          reach=float(sum(sk.length[k] for k in ch[:-1]))))
    c.params = dict(mass=sp.mass, spec=sp.__dict__.copy(), stand_ratio=1.0)
    c.z0 = _rest_height(c)
    return c

# ---------------------------------------------------------------------------------------------
def snake(n=22, length=1.6, mass=0.8, name="snake") -> Creature:
    seg = length / n; r = 0.02 * (length / 1.6) ** 0.5
    bones = [Bone("seg0", None, X, seg, mass / n, r, group="trunk")]
    for k in range(1, n):
        taper = 0.55 + 0.45 * np.sin(np.pi * min(1.0, (k + 2) / n) * 0.85) if k < n - 1 else 0.4
        bones.append(Bone(f"seg{k}", f"seg{k-1}", X, seg, mass / n, r * min(1.0, taper + 0.2), order="ZYX",
                          lim={"Z": (-40, 40), "Y": (-30, 30), "X": (-10, 10)}, group="trunk", tau=2.0 * mass))
    sk = Skeleton(name, bones)
    c = Creature(sk, "snake"); c.spine = list(range(n)); c.head = 0
    c.params = dict(mass=mass, length=length)
    c.z0 = r
    return c
