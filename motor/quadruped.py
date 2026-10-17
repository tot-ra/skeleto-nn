"""The quadruped family: dog, cat, wolf, goat, horse, pig, bear and everything between, one policy.

One network needs one joint layout, so every quadruped is built on a canonical skeleton (``CANON``: four spine
joints, three neck joints, four tail joints, four legs of thigh, shank, metapodial and paw or hoof). Species differ in
everything else: mass and size, leg length and the long cannon bone of a horse, how far the back bends (a species'
total spinal range is spread over the four joints, so a cat's column still bends five times as far as a horse's),
paws or hooves, power for the size. ``sample_body`` draws a species and varies it.

Goal speeds are in units of sqrt(g * leg length) (Froude speed): a trot is about 0.7, a gallop 1.5 and more, for a
cat and a horse alike.
"""
from __future__ import annotations
import dataclasses
import math
import numpy as np

from bodies import quadruped, QUADS
from motor.family import FamilyEnv, G

CANON = dict(n_spine=4, n_neck=3, tail_n=4)
SPECIES = ("dog", "cat", "wolf", "goat", "horse", "pig", "bear")
# drawn by weight: the species that fail most come up more often (in run 6 the horse finished 0 of 13 training
# episodes, the cat 3 of 16, the goat 8 of 20, the bear and wolf about half)
SPECIES_WEIGHT = np.array([1.0, 1.6, 1.0, 1.4, 2.2, 1.0, 1.0]); SPECIES_WEIGHT /= SPECIES_WEIGHT.sum()


def canonical(spec):
    return dataclasses.replace(spec, **CANON)


class QuadrupedEnv(FamilyEnv):
    family = "quadruped"
    BODY_DIM = 16
    MAX_SPEED = 2.0      # Froude: a gallop

    def default_body(self):
        return canonical(QUADS["dog"])

    def sample_body(self, rng, spread):
        sp = QUADS[SPECIES[int(rng.choice(len(SPECIES), p=SPECIES_WEIGHT))]] if rng.random() < spread else QUADS["dog"]
        jitter = lambda: math.exp(rng.normal(0, 0.08 * spread))
        sp = dataclasses.replace(sp, mass=sp.mass * jitter() ** 3, scale=sp.scale * jitter(), leg_scale=sp.leg_scale * jitter(),
                                 vigor=float(np.clip(sp.vigor * jitter() ** 2, 0.5, 1.4)))
        return canonical(sp)

    def make_creature(self, body):
        return quadruped(body)

    # Each species gets its own share of the curriculum's goal speeds, kept per worker: with one shared curriculum the
    # species that fell most (horse, cat) pulled the goal speeds down for all, and the dog stopped training at speed
    def speed_scale(self):
        if not hasattr(self, "sp_scale"):
            self.sp_scale = {}
        return self.sp_scale.get(self.c.skel.name, 1.0)

    def episode_done(self, failed, goal):
        k = self.c.skel.name; s = self.speed_scale()
        if failed:
            s = max(0.3, s - 0.05)
        elif goal > 0.5:
            s = min(2.0, s + 0.03)
        self.sp_scale[k] = s

    def body_vector(self, sp, c):
        return [math.log(sp.mass / 30.0), math.log(sp.scale), sp.leg_scale - 1.0, sp.cannon - 1.0, sp.spine_flex - 1.0,
                sp.neck_len - 1.0, sp.tail_len - 1.0, sp.head_len - 1.0, sp.body_len - 1.0, sp.chest_depth - 1.0,
                1.0 if sp.foot_type == "hoof" else 0.0, sp.ext_rom / 45.0, sp.flex_rom / 75.0, sp.lat_rom / 95.0,
                sp.core_power - 1.0, sp.vigor - 1.0]

    def sores(self):
        return ("leg_HL", "leg_HR", "leg_FL", "leg_FR", "torso", "head")

    def speed_unit(self):
        return math.sqrt(G * self.leg)

    def stride_pose(self, rng, speed):
        # a trot: diagonal pairs together, one pair forward, the other back (thigh about Y: negative swings forward)
        swing = min(0.5, 0.25 + 0.1 * speed / self.speed_unit()) * rng.uniform(0.6, 1.0)
        a, b = ("L", "R") if rng.random() < 0.5 else ("R", "L")
        return {(f"thighH_{a}", 1): -swing, (f"thighF_{b}", 1): -swing,
                (f"thighH_{b}", 1): swing, (f"thighF_{a}", 1): swing}
