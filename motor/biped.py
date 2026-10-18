"""The biped family: any human body, in any environment, with a goal and a motivation profile, in MuJoCo.

Everything shared with the other families (observation, mirror maps, muscle activation, harness, wind, reward) is in
``motor/family.py``. What is human:

  body        height, log mass, leg length, muscle, flexibility, pack, belly and armour (share of the mass), sex
              (``bodies.human``: 24 bones, 53 joint axes; the hand is one rigid segment for now)
  sores       pain per region: each leg, each arm, the torso, the head
  start       mid-stride: one thigh forward, the other back with the knee bent
  goal speed  in m/s
"""
from __future__ import annotations
import math
import numpy as np

from bodies import human, HumanBody
from motor.family import FamilyEnv, ALIVE, FALL_COST, heading, yaw_of  # noqa: F401  (re-exported for older callers)

SORES = ("leg_L", "leg_R", "arm_L", "arm_R", "torso", "head")


def sample_body(rng, spread=1.0):
    """A random person: child to tall adult, starved to athlete, stiff to supple, with or without a load."""
    h = float(np.clip(1.72 + rng.normal(0, 0.12) * spread - (0.45 * spread if rng.random() < 0.12 * spread else 0.0), 1.1, 2.0))
    bmi = float(np.clip(rng.normal(23.5, 3.0 * spread), 16.0, 34.0))
    mass = bmi * h * h
    muscle = float(np.clip(1.0 + rng.normal(0, 0.18) * spread, 0.5, 1.4))
    flex = float(np.clip(1.0 + rng.normal(0, 0.1) * spread, 0.7, 1.25))
    load = rng.random()
    pack = float(rng.uniform(4, 25)) if load < 0.15 * spread else 0.0
    armour = float(rng.uniform(5, 20)) if 0.15 * spread <= load < 0.25 * spread else 0.0
    belly = float(rng.uniform(2, 10)) if rng.random() < 0.1 * spread else 0.0
    sex = "f" if rng.random() < 0.5 else "m"
    return HumanBody(height=h, mass=mass + pack + armour + belly, pack=pack, armour=armour, belly=belly,
                     muscle=muscle, strength=muscle, flex=flex, sex=sex)


def body_vector(b: HumanBody, leg):
    return np.array([b.height - 1.7, math.log(b.mass / 75.0), leg - 0.86, b.muscle - 1.0, b.flex - 1.0,
                     b.pack / b.mass, b.belly / b.mass, b.armour / b.mass, 1.0 if b.sex == "f" else 0.0], np.float32)




class BipedEnv(FamilyEnv):
    family = "biped"
    BODY_DIM = 9
    SOFT_RANGE = {"arm": 0.2}   # the last fifth of an arm joint's range resists: arms cannot hang on their stops
    # exploration floor 0.10: with the 0.06 a walker had settled to, the arms kept the pose they had learned on their stops
    # for rounds after the stops became soft and costly; nothing ever tried lowering them
    MIN_LOGSTD = -2.3
    ENDURE = True        # a held-out arm tires its shoulder within minutes; a walker's arms hang and swing
    ACTION_BOUND = 0.95  # arms back inside the clip range, where exploration changes the command

    def default_body(self):
        return HumanBody()

    def sample_body(self, rng, spread):
        return sample_body(rng, spread)

    def make_creature(self, body):
        return human(body)

    def body_vector(self, b, c):
        return body_vector(b, c.leg_len())

    def sores(self):
        return SORES

    def stride_pose(self, rng, speed):
        front, back = ("L", "R") if rng.random() < 0.5 else ("R", "L")
        stride = min(1.0, 0.4 + 0.25 * speed) * rng.uniform(0.6, 1.0)
        # thigh about Y: negative swings it forward; shank about Y: positive bends the knee
        return {(f"thigh_{front}", 1): -0.45 * stride, (f"thigh_{back}", 1): 0.30 * stride,
                (f"shank_{front}", 1): 0.10, (f"shank_{back}", 1): 0.35 * stride}
