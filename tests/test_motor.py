"""The evaluation system measures the consequence each term names, and the biped family mirrors exactly."""
import numpy as np

from bodies import HumanBody
from motor.biped import BipedEnv, SORES
from motor import evaluate as ev

CALM = dict(body=HumanBody(), friction=1.0, wind=np.zeros(3), goal=np.zeros(2))


def standing(env, steps=30, **kw):
    sores = kw.pop("sores", {k: 0.0 for k in SORES})
    env.reset(sores=sores, profile=dict(ev.DEFAULT_WEIGHTS), **{**CALM, **kw})
    env.assist = 1.0
    total = {t: 0.0 for t in ev.TERMS}
    for _ in range(steps):
        _o, _r, _d, info = env.step(np.zeros(env.act_dim))
        for t in ev.TERMS:
            total[t] += info["costs"][t] / steps
    return total


def test_mirror_is_an_involution_and_matches_the_reflected_world():
    env = BipedEnv(seed=3)
    o = env.reset(**CALM)
    assert np.abs(env.mirror_obs(env.mirror_obs(o)) - o).max() == 0.0
    a = np.random.default_rng(0).normal(size=env.act_dim)
    assert np.abs(env.mirror_act(env.mirror_act(a)) - a).max() == 0.0


def test_a_sore_leg_hurts_when_it_carries_weight():
    env = BipedEnv(seed=1)
    calm = standing(env)
    sore = standing(env, sores={**{k: 0.0 for k in SORES}, "leg_L": 1.0})
    assert sore["pain"] > calm["pain"] + 0.2


def test_moving_costs_energy_and_strain_and_shakes_the_head():
    env = BipedEnv(seed=1)
    env.reset(sores={k: 0.0 for k in SORES}, profile=dict(ev.DEFAULT_WEIGHTS), **CALM)
    env.assist = 1.0
    still = {t: 0.0 for t in ev.TERMS}; flail = {t: 0.0 for t in ev.TERMS}
    for k in range(20):
        _o, _r, _d, info = env.step(np.zeros(env.act_dim))
        for t in ev.TERMS: still[t] += info["costs"][t]
    for k in range(20):
        _o, _r, _d, info = env.step(np.full(env.act_dim, 0.4 * (-1) ** k))
        for t in ev.TERMS: flail[t] += info["costs"][t]
    assert flail["energy"] > 3 * still["energy"]
    assert flail["strain"] > still["strain"]
    assert flail["smooth"] > still["smooth"]


def test_a_drop_onto_the_ground_is_scored_as_injury():
    env = BipedEnv(seed=1)
    env.reset(sores={k: 0.0 for k in SORES}, profile=dict(ev.DEFAULT_WEIGHTS), **CALM)
    p, R = env.pc.root_pose()
    env.pc.set_state(p + np.array([0, 0, 1.5]), R, env.pc.qvec())        # 1.5 m above the ground, legs straight
    worst = 0.0
    for _ in range(30):
        _o, _r, _d, info = env.step(np.zeros(env.act_dim))
        worst = max(worst, info["costs"]["injury"])
    assert worst > 1.0


def test_profile_vector_is_zero_for_the_default_weights():
    assert np.abs(ev.profile_vector(ev.DEFAULT_WEIGHTS)).max() == 0.0
