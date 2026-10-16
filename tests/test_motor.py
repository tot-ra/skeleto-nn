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


def test_exploration_noise_does_not_cost_more_energy_than_a_brisk_walk():
    # muscle activation dynamics (50 ms): commands that flip every step must not make the body vibrate
    from motor.quadruped import QuadrupedEnv
    for E in (BipedEnv, QuadrupedEnv):
        env = E(seed=1, spread=0.0)
        env.reset(goal=np.zeros(2), friction=1.0, wind=np.zeros(3))
        rng = np.random.default_rng(0); e = []
        for _ in range(30):
            _o, _r, _d, info = env.step(rng.normal(0, 0.08, env.act_dim))
            e.append(info["costs"]["energy"])
        assert np.mean(e) < ev.SUSTAIN, (E.__name__, np.mean(e))


def test_quadruped_mirror_matches_the_reflected_world():
    from motor.quadruped import QuadrupedEnv
    env = QuadrupedEnv(seed=3, spread=0.0)
    env.reset(goal=np.array([0.7, 0.3]), wind=np.array([2.0, 5.0, 0.0]), friction=1.0)
    rng = np.random.default_rng(0)
    q = np.clip(rng.normal(0, 0.3, env.J), env.pc.lo, env.pc.hi); qd = rng.normal(0, 1, env.J)
    env.pc.set_state(np.array([0, 0, 1.0]), np.eye(3), q, np.array([0.5, 0.4, 0.1]), np.array([0.3, 0.2, 0.5]), qd)
    env.prev_a = rng.normal(0, 0.5, env.J); env.sores_now["leg_FL"] = 0.7
    o1 = env.obs()
    env.pc.set_state(np.array([0, 0, 1.0]), np.eye(3), q[env.m_perm] * env.m_sign, np.array([0.5, -0.4, 0.1]),
                     np.array([-0.3, 0.2, -0.5]), qd[env.m_perm] * env.m_sign)
    env.prev_a = env.mirror_act(env.prev_a); env.sores_now["leg_FL"] = 0.0; env.sores_now["leg_FR"] = 0.7
    env.goal_world = env.goal_world * np.array([1, -1]); env.wind.w = env.wind.w * np.array([1, -1, 1])
    assert np.abs(env.mirror_obs(o1) - env.obs()).max() < 1e-5


def test_bird_flies_under_the_harness_and_falls_without_it():
    from motor.bird import BirdEnv
    env = BirdEnv(seed=1, spread=0.0)
    o = env.reset()
    assert np.abs(env.mirror_obs(env.mirror_obs(o)) - o).max() == 0.0
    for assist, expect_fall in ((1.0, False), (0.0, True)):
        env.assist = assist; env.reset(); fell = False
        for _ in range(150):
            _o, _r, d, info = env.step(np.zeros(env.act_dim))
            if d:
                fell = info["fell"]          # held up but not flying ends the episode too, as a stall
                break
        assert fell == expect_fall


def test_standing_still_when_asked_to_walk_ends_the_episode_but_standing_when_asked_to_stand_does_not():
    env = BipedEnv(seed=2)
    for goal, should_end in ((np.array([1.2, 0.0]), True), (np.zeros(2), False)):
        env.moving_start = 0.0
        env.reset(sores={k: 0.0 for k in SORES}, profile=dict(ev.DEFAULT_WEIGHTS), **{**CALM, "goal": goal})
        env.assist = 1.0
        ended = False
        for _ in range(90):
            _o, _r, done, info = env.step(np.zeros(env.act_dim))
            if done:
                ended = info["stalled"] and not info["fell"]
                break
        assert ended == should_end
