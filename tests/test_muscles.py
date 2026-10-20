"""Line muscles of every family: each pulls its joint the anatomical way over its range, and moves it when excited."""
import numpy as np
import pytest

from bodies import human
from muscles import human_legs
from physics import PhysChar
from muscle_check import arms_at


def build(fam="human", species=None):
    from muscles import SETS
    if fam == "human":
        c = human()
    elif fam == "quadruped":
        from bodies import quadruped; c = quadruped(species)
    else:
        from bodies import bird; c = bird(species)
    pc = PhysChar(c, torque=True, muscles=SETS[fam](c))
    idx = {(pc.sk.bones[i].name, "XYZ"[a]): k for k, (i, a) in enumerate(pc.dof_list)}
    return c, pc, idx


@pytest.mark.parametrize("fam,species", [("human", None), ("quadruped", "dog"), ("quadruped", "horse"), ("quadruped", "cat"),
                                         ("bird", "crow"), ("bird", "heron")])
def test_primary_moment_arms_keep_their_sign_over_the_range(fam, species):
    """Every muscle pulls its joint the anatomical way over the inner range; at the very ends of a range some lose
    their leverage or turn (a deltoid with the arm overhead, the pectoralis above 90 degrees, a biceps fully bent:
    action inversion, as in a body)."""
    _c, pc, idx = build(fam, species)
    assert not pc.m_bad, pc.m_bad
    for j, mu in enumerate(pc.ms.muscles):
        bn, ax, sg = mu.primary; k = idx[(bn, ax)]
        row = []
        lo, hi = (np.radians(mu.check) if mu.check else (pc.lo[k], pc.hi[k]))
        for x in np.linspace(lo, hi, 7):
            q = np.zeros(pc.nj); q[k] = x
            a, _l = arms_at(pc, q)
            row.append(sg * a[j, k])
        assert min(row[1:-1]) > 0, (mu.name, row)


def test_two_joint_muscles_act_on_both_joints():
    _c, pc, idx = build()
    arm = dict(zip(pc.mnames, pc.arm0))
    assert arm["HAM_L"][idx[("thigh_L", "Y")]] > 0.02 and arm["HAM_L"][idx[("shank_L", "Y")]] > 0.015    # hip ext, knee flex
    assert arm["RF_L"][idx[("thigh_L", "Y")]] < -0.02 and arm["RF_L"][idx[("shank_L", "Y")]] < -0.02     # hip flex, knee ext
    assert arm["GAS_L"][idx[("shank_L", "Y")]] > 0.015 and arm["GAS_L"][idx[("foot_L", "Y")]] > 0.03     # knee flex, plantarflex


def test_excited_muscle_moves_a_hanging_leg():
    c, pc, idx = build()
    pc.model.opt.gravity[:] = 0.0
    nf = len(pc.free)
    for name, key, sign in (("VAS_L", ("shank_L", "Y"), -1), ("HAM_L", ("shank_L", "Y"), 1), ("TA_L", ("foot_L", "Y"), -1),
                            ("SOL_L", ("foot_L", "Y"), 1), ("ILIO_L", ("thigh_L", "Y"), -1)):
        q0 = np.zeros(pc.nj); q0[idx[("shank_L", "Y")]] = 0.5
        pc.set_state(np.array([0, 0, 2.0]), np.eye(3), q0)
        a = np.zeros(pc.n_act); a[nf + pc.mnames.index(name)] = 0.5
        for _ in range(4):
            pc.step(a)
        assert sign * (pc.qvec()[idx[key]] - q0[idx[key]]) > 0.02, name
        assert pc.m_act[pc.mnames.index(name)] > 0.4


def test_mirror_maps_muscles_by_side():
    from motor.biped import BipedEnv
    BipedEnv.MUSCLES = "human_legs"
    try:
        env = BipedEnv(seed=0)
        a = np.zeros(env.act_dim); nf = len(env.pc.free)
        a[nf + env.pc.mnames.index("VAS_L")] = 1.0
        assert env.mirror_act(a)[nf + env.pc.mnames.index("VAS_R")] == 1.0
    finally:
        BipedEnv.MUSCLES = None
