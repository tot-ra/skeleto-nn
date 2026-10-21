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
import math, os
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
    # goal speeds up to 5 m/s: up to 3 (the family default) a shuffle with almost no flight still met the goal (cadence
    # 257 a minute, 7 % flight at 3 m/s); people run from about 2 m/s, and at 4 to 5 a run without flight is not possible
    MAX_SPEED = 5.0
    SOFT_RANGE = {"arm": 0.2,   # the last fifth of an arm joint's range resists: arms cannot hang on their stops
                  # and the last tenth of the ankle's: a muscle walker stood leaning on the dorsiflexion stop (17 N m on
                  # average, up to 97), calves silent and the shin muscle pulling the foot up, the ligaments carrying it
                  # for free. The toes keep a hard stop: they bend back under the body's weight in every push-off
                  "foot": 0.1, "toes": 0.0,
                  # and the spine's, neck's and head's: the walker stooped with every lumbar and thoracic joint on its
                  # flexion and side-bend stops (25 to 45 N m each on the ligaments, back muscles barely working) and the
                  # head hanging down and turned. A stooped back held on its ligaments aches within minutes
                  "trunk": 0.2, "neck": 0.15, "head": 0.15}
    # exploration floor 0.10: with the 0.06 a walker had settled to, the arms kept the pose they had learned on their stops
    # for rounds after the stops became soft and costly; nothing ever tried lowering them
    MIN_LOGSTD = -2.3
    # the Achilles tendon returns about half of the ankle's braking work in running, the patellar tendon about a third of
    # the knee's: without them a bouncing run cost more than a shuffle (cadence 257 a minute, 8 % flight at 2.9 m/s)
    TENDONS = {"foot": 0.5, "shank": 0.3}
    HEAD_UP = True
    ENDURE = True        # a held-out arm tires its shoulder within minutes; a walker's arms hang and swing
    ACTION_BOUND = 0.95  # arms back inside the clip range, where exploration changes the command
    # a moving start puts both feet down in a stride, joints still, the whole body at up to the goal speed: at 3 to 4 m/s
    # about half of those starts fell within a second (from rest the same networks never fell, but never sped up past
    # a walk either), and the falls taught that going fast means falling. Run speeds are reached by speeding up
    MOVING_START_MAX = 1.5
    LEG_CONTACTS = True
    # a third of moving starts from a recorded stride (CMU walks and runs fitted to the body, ``motor.rsi``): from the
    # made-up mid-stride pose a walker never got past about 0.9 m/s without falling (2 of 6 at 0.9, 3 of 6 at 1.1)
    RSI = os.path.join(os.path.dirname(__file__), "..", "data", "mocap", "cmu_rsi.npz")
    RSI_SHARE = 0.35
    AHEAD_SPEED = 0.5     # m/s above the curriculum's top speed
    # with starts capped, both networks sped up to about 1.8 m/s and no further (a quick walk, 2 to 6 % flight): every
    # try at a run had fallen within a second. Goals past a walk get a harness that only keeps the trunk upright, so a
    # run can be practised before it can be balanced; it fades with time (ppo --run-assist0, --run-fade)
    RUN_FROM = 2.2
    # experiment (BIPED_TORQUE=1): muscles as torques. With target angles at a fixed stiffness the trained walkers held the
    # knee on its stop all the time (peak knee speed 0.3 rad/s; a person's swing knee bends 60 degrees at 6 to 7): a
    # relaxed knee could not swing as a pendulum, any bend had to be paid for, a straight leg cost nothing. Walked with
    # stiff legs (the waddle, feet barely off the ground); such legs cannot run
    TORQUE = bool(os.environ.get("BIPED_TORQUE"))
    # experiment (BIPED_ACHILLES=1): the Achilles tendon as a real spring, taut once the ankle dorsiflexes past neutral
    # (negative foot Y); 2 strengths per radian is about 220 N m/rad, 75 N m at 20 degrees, a third of a run's push-off
    SPRINGS = {"foot_": ("Y", 2.0, 0.0)} if os.environ.get("BIPED_ACHILLES") else {}
    # BIPED_MUSCLES=1: line muscles on the legs (``muscles.human_legs``: 10 per leg with origins, insertions and wraps,
    # Hill force-length-velocity, passive stretch, activation dynamics); joint torques at every joint or axis they do
    # not span. Even as torques, the hip sat on its extension stop and swung little: one-joint torques have no
    # hamstrings tying hip to knee, no stretched hip flexor pulling the thigh forward, no calf spanning knee and ankle
    # (BIPED_MUSCLES=legs: the legs only; any other value: legs, trunk and arms, ``muscles.human_upper``)
    MUSCLES = ({"legs": "human_legs"}.get(os.environ.get("BIPED_MUSCLES"), "human")
               if os.environ.get("BIPED_MUSCLES") else None)

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

    def amp_features(self):
        """The pose and motion features the motion-prior discriminator compares with recordings (``motor.amp``)."""
        from motor.amp import features
        pc = self.pc; sk = pc.sk; d = pc.data
        S, E, _R = pc.bone_ends(); i = sk.idx
        pts = dict(chest=E[i["chest"]])
        for l in self.c.legs:                              # the ball of the foot on the sole, as a recording marks it
            s = l.side; f = i["foot_" + s]
            pts.update({f"hip_{s}": S[i["thigh_" + s]], f"knee_{s}": E[i["thigh_" + s]], f"ankle_{s}": E[i["shank_" + s]],
                        f"ball_{s}": S[f] + _R[f] @ l.ball, f"shoulder_{s}": S[i["uarm_" + s]], f"elbow_{s}": E[i["uarm_" + s]],
                        f"wrist_{s}": E[i["farm_" + s]]})
        _p, R = pc.root_pose()
        leg = sk.length[i["thigh_L"]] + sk.length[i["shank_L"]]
        return features(pts, d.qvel[0:3].copy(), float((R @ d.qvel[3:6])[2]), leg)

