# Motor networks: one trained policy per body family

Plan and decision: Reval Rebel ADR 0056 ("Neural motor control, one network per body family"). This page
describes what exists in `motor/` and what does not yet.

The aim is to replace every non-network method of this repository (emergent rules, goal scripts, kinematic
models, evolved controllers) with networks trained in physics. Each body family (biped, quadruped, mixed bodies
such as monkeys, bears, kangaroos and penguins, birds, snakes) gets its own policy. Within a family the body, the
environment, the goal and a motivation profile are inputs, so one network covers every individual and situation of
that family. A second, smaller network will later copy each policy without physics, so that a game can run it.

## The evaluation system (`motor/evaluate.py`)

Behaviour is never judged by how it looks. Every control step (30 Hz) the MuJoCo simulation is measured and each
term turns one consequence into a cost per second, with the body's own scale divided out:

| term | measured as |
|---|---|
| goal | how well the goal is met (stage 1: the velocity to hold, facing where you go; progress along the wanted direction already counts) |
| upright | a constant reward per second for staying up and whole; a fall ends the episode and costs about five seconds of a met goal. Without it, falling early looked cheaper than paying the costs of staying up |
| injury | fracture risk per region (`injury.py`: tolerance per region, a fast load is more brittle), from the peak contact force of every physics sub-step; an event, so a certain leg fracture costs about three seconds of goal |
| pain | weight carried by a sore leg, work of a sore arm, load on a sore trunk or head; joints pressed into the last 5 % of their range by their own muscle |
| strain | muscle torque against its capacity, squared, weighted by the mass each joint carries |
| energy | metabolic power: positive work at 25 % efficiency, negative work, isometric holding |
| fatigue | an energy reserve drains above the sustainable power (about a brisk walk) and refills below it; a drained body is weaker |
| balance | fall risk: the capture point (where the body would have to step to stop) further than half a leg from the nearest foot; pelvis tilt |
| gaze | head spin |
| smooth | change of the muscle commands between steps |
| heat, breath | reserved for stage 3 (cold, water, smoke); zero now |

Wind is a force in the simulation (drag on every segment), not a term: what it costs shows up as energy, strain and
fall risk. The weights of the terms are a motivation profile, drawn at random for every episode and given to the
policy as input, so the same network can be careful or desperate.

## The biped family (`motor/biped.py`)

* **Body:** the parametric human of `bodies.py` (24 bones, 53 joint axes): height 1.1 to 2.0 m, body mass index,
  muscle, flexibility, sex, a pack, armour, a belly. **No fingers yet**: the hand is one rigid segment (see Limits).
* **Environment:** ground friction (ice to dry stone), steady wind from any side.
* **Sores:** pain in one of the legs, arms, the trunk or the head.
* **Goal:** a velocity to hold (standing still, walking, running, any direction).
* **Observation (202 numbers):** gravity in the pelvis frame, pelvis velocity and spin, pelvis height, joint angles
  and speeds, the previous command, foot contacts, the energy reserve, the goal in the heading frame, the body, the
  environment, the sores, the motivation profile.
* **Action (53 numbers):** a target angle for every joint muscle (PD, torque limited by strength and the reserve),
  as a share of the joint's range from the standing pose.

## Training (`motor/ppo.py`)

PPO in worker processes (numpy copies of the policy) with the update in torch, after Yu, Turk and Liu 2018
("Learning Symmetric and Low-Energy Locomotion"): no reference motion; a mirror loss makes left and right act alike
(`mirror_obs`, `mirror_act`, checked against a reflected world in `tests/test_motor.py`); a harness holds the
pelvis up and upright and fades out only while the goal is met and falls are rare; the variety of bodies and
situations grows as the harness fades. The harness never pushes toward the goal: a first run with a pushing harness
learned to lean stiffly and be pushed, then fell like a plank once the harness weakened.

    python -m motor.ppo --family biped --name biped_s1 --workers 14 --hours 3
    python -m motor.show runs/motor/biped_s1/ckpt.pt          # GIFs and summary.json for ten review situations
    python -m pytest tests/test_motor.py

About 8,000 to 9,000 simulated steps per second with 14 workers on an M5 Pro.

## Status

Stage 1 (evaluation system, biped environment, trainer, renderer) is built and tested. No policy walks yet, and
nothing here is used by a game yet. The first two runs (GIFs in the README, `showcase/motor/`):

| run | setup | what it learned |
|---|---|---|
| 1 | reward = goal minus costs; harness held the pelvis up and pushed it toward the goal velocity | to stiffen, lean and be pushed; it fell like a plank as the harness faded to 40 % |
| 2 | plus a reward for staying upright and a fall cost; the goal pays for partial progress; the harness only holds, never pushes | to stand still under the full harness (0.04 m/s when asked for 1.3 m/s); the harness never faded because the goal was never met, and without it the policy falls within 2 s |

Both are the usual local optima of reinforcement learning without reference motion. Next: a speed curriculum that
starts from standing and slow walking, a harness that fades with time as well as with success, and episodes that
start mid-stride.

## Limits

* The hand is one rigid segment in the physics. The game rig has 15 finger joints per hand; the physics body gets
  fingers when grasping, bracing and climbing goals arrive (stage 3), warm-started from the policy without them.
* Flat ground only; no terrain height field in the observation yet.
* Heat and breath are not measured yet.
