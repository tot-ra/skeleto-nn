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

| bird | `bodies.bird`, one skeleton for every species (4 neck, 2 tail joints, three wing and four leg segments a side): crow, gull, heron, sparrow, duck and variations; wings and tail are flat plates in the air of `air.py` (lift with stall, drag, wind); nothing about flapping is authored | obs 211, act 56 | sqrt(g x wingspan) |

Birds start in the air at about the goal velocity (horizontal velocity and a climb rate), 3 to 6 wingspans up; touching
the ground ends the flight and its force is scored as injury; balance is scored by tilt only (a capture point means
nothing in the air); the harness carries part of the weight and holds the body level. Take-off, landing, perching,
walking and swimming birds come later.

The bird network also reads a **wingbeat clock** (sine and cosine of a phase) and sets, as one extra output, how fast
it advances (half to twice the species' natural frequency, 4.5 Hz for a crow); what the wings do at each phase is the
network's. Cruise goals start at 2.6 units, just above a crow's minimum flying speed.

**Air model fixes (run 5).** The first bird runs could not have learned to fly: the air model made flight impossible.
Measured with a held crow at 9 to 12 m/s:

| problem | measured | fix (`air.py`, `physics.py`, `motor/evaluate.py`) |
|---|---|---|
| every bone a bare cylinder broadside to the air | body drag 2.6 N, glide ratio 1.6 (a crow: about 8) | feathered trunk, thighs, neck and head are a streamlined body; bare legs scaled to a real tarsus: glide ratio 6 to 7 at 8 m/s |
| the attached-flow force along the plate normal (no leading-edge suction) | a beating wing made no thrust in any pattern | lift perpendicular to the relative wind, induced drag from the aspect ratio: a downstroke leans the force forward (Knoller-Betz) |
| steady stall at 17 degrees | the tip of a beating wing stalls on every downstroke | a beating wing's leading-edge vortex: lift saturating at C_L 1.6, stall at about 35 degrees (Dickinson 1999, Muijres 2012) |
| a folded wing kept its full area | the upstroke pulled the bird down as hard as the downstroke lifted it | a strip's area follows its spanwise extent: a wing folded or swept back lifts less |
| wing joints with a limb's 80 ms response and heavy armature | the shoulder reached 40 % of a 4.5 Hz beat | wing joints: 20 ms response, light armature |
| energy counted every swing of the wing instant by instant, in walking units | flapping cost about 8 times a walk's sustainable power | flight power units (wingspan), a flier's sustainable power, wing joints elastic (work averaged over a beat) |

After the fixes a searched open-loop wingbeat (CMA-ES, used only to check feasibility; nothing of it is in the
network) gives net thrust (+0.46 N) with lift above the weight (5.5 N against 4.4 N): sustained flapping flight is
possible in the simulation, and run 5 trains the bird network from scratch on it.

Mixed bodies (monkey, bear on two legs, kangaroo, penguin) and snakes are not
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
* **Stalls end episodes:** a moving goal met this little (`STALL_GOAL`) for about a second ends the episode with the
  fall cost. Standing still when asked to move is a dead end, not a safe choice.
* **Turning is progress:** a body turning toward a goal behind or beside it earns a little goal score and is not
  ended as a stall halfway round.
* **Slow, reversible curriculum:** goal speeds grow by 0.005 units per successful iteration and shrink when most
  episodes fail (falls and stalls both count).
* **Variety shrinks too** when most episodes fail, back to where it started.
* **Birds glide first:** the first flight goals sink at about a crow's glide ratio (0.45 units), so gravity pays for
  the speed; they level out as the curriculum's speed knob grows, so flapping has to pay. The harness follows the
  asked-for path without carrying any weight (carried weight is what drives a glide; carrying 70 % of it stalled
  every glide), holds the body level with a stiffness of m g x wingspan per radian (one scaled by the trunk's inertia
  let it roll over), and when most flights fail it comes back fully (people and quadrupeds: up to 30 %).
* **Speed cap per family** (`MAX_SPEED`): 3 m/s for people, 2 Froude (a gallop) for quadrupeds, 4 units for birds.
  `--reset-curriculum` keeps a checkpoint's weights and starts the harness and the goals over.

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

Stage 1 (evaluation system, biped, quadruped and bird environments, trainer, population training, renderer) is built and
tested. The biped and the quadruped walk on their own; birds do not fly yet. Nothing here is used by a game yet. Runs
so far (GIFs in the README, `showcase/motor/`):

| run | setup | what it learned |
|---|---|---|
| 1 | reward = goal minus costs; harness held the pelvis up and pushed it toward the goal velocity | to stiffen, lean and be pushed; it fell like a plank as the harness faded to 40 % |
| 2 | plus a reward for staying upright and a fall cost; the goal pays for partial progress; the harness only holds, never pushes | to stand still under the full harness (0.04 m/s when asked for 1.3 m/s); the harness never faded because the goal was never met, and without it the policy falls within 2 s |
| 3 | population training (3 bipeds, 2 quadrupeds, 16 rounds, about 4 h); curriculum, moving starts, activation | to stand perfectly on its own, in every situation and at any asked speed (0.00 m/s when asked for 1.3 m/s); from a moving start it falls within 1 s. Scores flat at 0.36 to 0.38 |
| 4 | run 3's weights; stalls end episodes; speed caps; curriculum restarted (4 h) | **biped: the first walk**: upright, alternating steps for the full 10 s at 0.65 m/s when asked for 1.3; turns, ice, a pack, a child's body still fall. Quadruped: still stands; its curriculum ran away again because a stall did not count as a failure (fixed: the curriculum now counts falls and stalls). Birds: fall within 1.5 s without the harness. After 4 h: the biped survives the full test at every forward speed (about 1 m/s when asked for 1.3, 1.3 to 1.5 when asked for 2.5), on ice and as a 1.4 m person; the quadruped walks at 0.5 to 0.8 Froude and on ice; both fail sideways goals (the stall rule ended them while turning; fixed) and the quadruped fails the cat and horse |
| 5 | turning counts as progress, slower curriculum; birds from scratch on the corrected air model with a wingbeat clock (3 h) | **biped:** walks at 1.36 m/s when asked for 1.3 and runs at 2.87 when asked for 3.2, for the full 10 s; also on ice, with a sprained ankle, in a gale, as a child and as an elder; a turn from a walk and a 20 kg pack still fall. **Quadruped:** dog (1.0 m/s), lame dog and cat walk for the full 10 s, head carried low; horse, pig and a dog on ice fall; its curriculum stalled at the slowest goals because the full variety of species made most episodes fail. **Birds:** the network beat its wings as slowly as allowed, lost speed under the harness and flipped without it (score 0.24 falling to 0.09); the harness never came back above 30 % |
| 6 | birds glide first (goals sink at a glide ratio of 7 and level out as the curriculum grows), the harness follows the asked-for path and comes back fully when flights fail; quadruped variety shrinks when most episodes fail; pack test added for people | in progress |

Runs 1 to 3 are the usual local optima of reinforcement learning without reference motion. Run 3 had two causes:
the curriculum had run away under an earlier goal score that paid standing still as a slow walk (quadrupeds were
asked for up to 4 Froude, twice a gallop), and staying up earned the alive reward every step while trying to walk
risked the fall cost. Run 4 ends stalled episodes, caps goal speeds per family and
starts the curriculum over from run 3's weights.

## Limits

* The hand is one rigid segment in the physics. The game rig has 15 finger joints per hand; the physics body gets
  fingers when grasping, bracing and climbing goals arrive (stage 3), warm-started from the policy without them.
* Flat ground only; no terrain height field in the observation yet.
* Heat and breath are not measured yet.
