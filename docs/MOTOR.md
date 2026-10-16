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

## Body families (`motor/family.py`, `motor/biped.py`, `motor/quadruped.py`)

Every family shares one environment (`family.py`): the physics body, the observation, the mirror maps, muscle
activation, the harness, wind and the reward. A family says only what differs: its individuals, their body vector,
named sores, a mid-stride start pose and the unit of goal speed.

* **Observation:** gravity in the root frame, root velocity and spin, root height over leg length, joint angles and
  speeds, the muscle targets (activation state), foot contacts, the energy reserve, the goal velocity in the heading
  frame, the body vector, ground friction and wind in the heading frame, sores, the motivation profile.
* **Action:** a target angle for every joint muscle (PD, torque limited by strength and the reserve), as a share of
  the joint's range from the standing pose. **Muscle activation:** the target follows the command with a 50 ms time
  constant, as real muscle activation does. Without it, the exploration noise of training made the stiff PD muscles
  vibrate, and that vibration cost 13 to 20 times the energy of a brisk walk: falling early paid better than living.
  With it, the same noise costs 0.06 to 0.09 (a test checks it stays below a brisk walk).

| family | bodies | sizes | goal speed unit |
|---|---|---|---|
| biped | `bodies.human`: 24 bones, 53 joint axes; height 1.1 to 2.0 m, body mass index, muscle, flexibility, sex, a pack, armour, a belly; **no fingers yet** (see Limits) | obs 202, act 53 | m/s |
| quadruped | `bodies.quadruped` on a canonical skeleton (4 spine, 3 neck, 4 tail joints, four legs of thigh, shank, metapodial, paw or hoof): dog, cat, wolf, goat, horse, pig, bear and variations; a species' total spinal range is spread over the four spine joints, so a cat's back still bends five times as far as a horse's | obs 244, act 64 | sqrt(g x leg length) |

Mixed bodies (monkey, bear on two legs, kangaroo, penguin), birds (with the air model of `air.py`) and snakes are not
built yet.

## Training (`motor/ppo.py`, `motor/pbt.py`)

PPO in worker processes (numpy copies of the policy) with the update in torch, after Yu, Turk and Liu 2018
("Learning Symmetric and Low-Energy Locomotion"): no reference motion; a mirror loss makes left and right act alike
(`mirror_obs`, `mirror_act`, checked against a reflected world for both families in `tests/test_motor.py`).

Curriculum:

* **Narrow start:** at first one default body (an adult, a dog) on dry, still ground, walking forward. Variety
  (bodies and species, ice, wind, sores, directions, motivation profiles) grows as the goal is met.
* **Speed:** goal speeds start below 0.8 units and grow while the goal is met.
* **Moving starts:** 70 % of episodes with a goal speed start already moving at that speed, mid-stride (a human with
  one thigh forward, a quadruped in a trot pose), so the body has to step or fall from the first moment.
* **Harness:** it holds the root up and upright, never pushes toward the goal, and fades with time as well as with
  success, so standing still under it never becomes a home.

**Population-based training** (`motor/pbt.py`): several members train side by side; every round (15 minutes) each is
scored on fixed situations *without the harness*, and the worst is replaced by a copy of the best with mutated
settings: learning rate, harness fade, mirror-loss weight, and the reward itself (what staying up is worth, what a
fall costs, how much the comfort costs weigh). Bad runs die off; good ones spread. `history.json` records every round.

    python -m motor.pbt --family biped --name biped_pbt --members 3 --workers 3 --round-min 15 --rounds 16
    python -m motor.ppo --family quadruped --name dog --workers 15 --hours 3       # a single run
    python -m motor.show runs/motor/biped_pbt/m0/ckpt.pt [out_dir] [--assist 0.4]  # GIFs and summary.json
    python -m pytest tests/test_motor.py

About 1,300 to 2,500 simulated steps per second per worker on an M5 Pro; the physics step is more than half of it.

## Status

Stage 1 (evaluation system, biped and quadruped environments, trainer, population training, renderer) is built and tested. No policy walks yet, and
nothing here is used by a game yet. The first two runs (GIFs in the README, `showcase/motor/`):

| run | setup | what it learned |
|---|---|---|
| 1 | reward = goal minus costs; harness held the pelvis up and pushed it toward the goal velocity | to stiffen, lean and be pushed; it fell like a plank as the harness faded to 40 % |
| 2 | plus a reward for staying upright and a fall cost; the goal pays for partial progress; the harness only holds, never pushes | to stand still under the full harness (0.04 m/s when asked for 1.3 m/s); the harness never faded because the goal was never met, and without it the policy falls within 2 s |

Both are the usual local optima of reinforcement learning without reference motion, and both were made worse by
the vibration energy that muscle activation now removes. The curriculum, moving starts, activation and population
training above are the answer; runs with them are in progress for bipeds and quadrupeds.

## Limits

* The hand is one rigid segment in the physics. The game rig has 15 finger joints per hand; the physics body gets
  fingers when grasping, bracing and climbing goals arrive (stage 3), warm-started from the policy without them.
* Flat ground only; no terrain height field in the observation yet.
* Heat and breath are not measured yet.
