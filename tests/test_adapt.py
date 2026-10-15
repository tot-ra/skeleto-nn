"""Condition-adaptive motion: the network, its behaviours, the physiology, the planner hooks, the schema growth."""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
from adapt.net import StyleNet, encode_body, encode_channels
from adapt.conditions import channels_from, CONDITIONS
from adapt.schema import CHANNELS, STYLE, BODY, neutral_channels
from adapt.physiology import BodyState, Exposure, heat_balance
from adapt import train as T

NET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "style_net.json")
net = StyleNet(NET)
NEUTRAL = net.style("biped", 70, 0.9, False, 0.25, neutral_channels())


def person(conds, fr=0.25):
    """Style relative to the neutral situation, as the driver applies it."""
    st = net.style("biped", 70, 0.9, False, fr, channels_from(conds))
    base = net.style("biped", 70, 0.9, False, fr, neutral_channels())
    return {k.name: k.neutral + st[k.name] - base[k.name] for k in STYLE}


def test_numpy_forward_equals_torch():
    m = T.Net(len(BODY) + len(CHANNELS), len(STYLE)); T.load_init(m, NET)
    ch = channels_from(["ice", ("bleeding", 1.0, dict(region="thigh", side="L"))])
    x = torch.tensor(np.concatenate([encode_body("biped", 70, 0.9, False, 0.25), encode_channels(ch)]))[None]
    a = m(x).detach().numpy()[0]
    b = np.array([net.style("biped", 70, 0.9, False, 0.25, ch)[k.name] for k in STYLE])
    assert np.abs(a - b).max() < 1e-4


def test_ice_short_flat_careful_steps_arms_out():
    s = person(["ice"])
    assert s["stride"] < 0.92 and s["foot_flat"] > 0.7 and s["care"] > 0.6 and s["arm_out"] > 0.4 and s["crouch"] > 0.1, s


def test_hand_goes_to_the_wound():
    s = person([("bleeding", 1.0, dict(region="thigh", side="L"))])
    assert s["press_L"] > 0.6 and s["press_L"] > s["press_R"] + 0.3 and s["speed_frac"] < 0.9, s


def test_guard_a_broken_arm_and_limp_on_a_sprain():
    s = person([("broken_arm", 1.0, dict(side="R"))]); assert s["hug_R"] > 0.7, s
    s = person([("sprain", 1.0, dict(side="L"))]); assert s["limp"] > 0.5, s
    s = person([("sprain", 1.0, dict(side="R"))]); assert s["limp"] < -0.5, s


def test_cold_and_wet_hug_hunch_shiver():
    s = person(["drenched", ("frost", 0.5)])
    assert s["tremor"] > 0.5 and s["hug_L"] > 0.5 and s["hug_R"] > 0.5, s
    assert person(["heat"])["tremor"] < 0.1


def test_lean_into_the_wind():
    assert person([("gale", 1.0, dict(dir=math.pi / 2))])["trunk_roll"] > 0.1
    assert person([("gale", 1.0, dict(dir=-math.pi / 2))])["trunk_roll"] < -0.1
    assert person([("gale", 0.9, dict(dir=0.0))])["trunk_pitch"] > 0.1


def test_animals_and_birds_use_the_same_network():
    dog = net.style("quadruped", 25, 0.5, False, 0.4, channels_from(["ice"]))
    dog0 = net.style("quadruped", 25, 0.5, False, 0.4, neutral_channels())
    assert dog["foot_flat"] - dog0["foot_flat"] > 0.5 and dog["care"] - dog0["care"] > 0.5
    wet = net.style("bird", 0.45, 0.1, True, 0.5, channels_from(["drenched"]))
    dry = net.style("bird", 0.45, 0.1, True, 0.5, neutral_channels())
    assert wet["flap_rate"] * wet["flap_amp"] > dry["flap_rate"] * dry["flap_amp"] + 0.05
    walk_ice = net.style("bird", 0.45, 0.1, False, 0.4, channels_from(["ice"]))
    assert walk_ice["arm_out"] > 0.4                          # wings out for balance


def test_condition_added_after_training_works_without_retraining():
    CONDITIONS["test_grease"] = dict(doc="test", effects={"friction": ("min", 0.11)})
    try:
        s = person(["test_grease"])
        assert s["foot_flat"] > 0.7 and s["stride"] < 0.92
    finally:
        del CONDITIONS["test_grease"]


def test_old_weight_file_loads_into_a_wider_schema():
    d = json.load(open(NET))
    j = d["inputs"].index("smoke")
    d2 = dict(d); d2["inputs"] = [n for n in d["inputs"] if n != "smoke"]
    W = np.array(d["layers"][0]["W"]); d2["layers"] = [dict(W=np.delete(W, j, 1).tolist(), b=d["layers"][0]["b"])] + d["layers"][1:]
    old = StyleNet(d2)
    ch = channels_from(["ice"])
    a = old.style("biped", 70, 0.9, False, 0.25, ch); b = net.style("biped", 70, 0.9, False, 0.25, ch)
    assert max(abs(a[k] - b[k]) for k in a) < 1e-5            # smoke is neutral here, so the zero column changes nothing


def test_physiology_rain_cools_and_hugging_slows_it():
    env = channels_from([("frost", 0.5)])
    def run(st):
        b = BodyState()
        for _ in range(600): b.step(1.0, env, Exposure(rain=1.0), st, "biped", False, 0.3)
        return b
    plain = run({}); hug = run(dict(hug_L=1.0, hug_R=1.0, trunk_pitch=0.3, tremor=0.8))
    assert plain.wet > 0.8 and plain.core_cold > 0.3, (plain.wet, plain.core_cold)
    assert hug.core_cold < plain.core_cold - 0.1


def test_pressure_slows_blood_loss():
    def run(press):
        b = BodyState(); b.wound("thigh", "L", rate=0.8)
        for _ in range(300): b.step(1.0, neutral_channels(), Exposure(), {}, "biped", False, 0.3, press)
        return b.blood_lost
    assert run(0.9) < 0.6 * run(0.0)


def test_driver_on_ice_takes_shorter_steps_and_neutral_is_unchanged():
    from bodies import human
    from planner import Walker, Cmd
    from terrain import Terrain
    from adapt.driver import StyleDriver
    def walk(conds, drive=True):
        w = Walker(human(), Terrain())
        d = StyleDriver(w, net, conditions=conds) if drive else None
        steps = 0
        for _ in range(int(6 / w.dt)):
            w.step(Cmd(v=np.array([1.3, 0.0])))
            steps += sum(1 for ls in w.legs if ls.td_event)
        return float(w.pos[0]) / max(steps, 1), d
    dry, d0 = walk([])
    plain, _ = walk([], drive=False)
    ice, d1 = walk(["ice"])
    assert abs(dry - plain) < 0.02, (dry, plain)
    assert ice < 0.9 * dry, (ice, dry)
    assert all(abs(d0.style[k.name] - k.neutral) < 0.03 for k in STYLE)
