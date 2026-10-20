# ☠️ skeleto-nn

Neural network powered character motion for **any skeleton**: people, 🐕 🐈‍⬛ 🐎 🦅 🐍

A body is a tree of bones with joint limits, masses and a few anatomical and physiological numbers; what it does is
decided by small networks that are trained against the physical and physiological consequences of what they do
(injury, pain, strain, energy, fatigue, falls, heat loss, blood loss, wind), moving the body through its muscles.
Nothing is replayed: in training a critic compares the motion with references (recorded human walks and runs, the
old procedural gaits and flight for animals and birds), but at run time a network only excites muscles in the physics.

| network | what it decides | inputs | weights | state |
|---|---|---|---|---|
| **Style network** (`adapt/`, [docs/ADAPT.md](docs/ADAPT.md)) | how a body moves in its situation: stride, crouch, lean, arms out or hugged, a hand on a wound, limp, shiver, wingbeat | body, 24 situation channels (friction, sinking feet, wetness, cold, wind, darkness, smoke, blood loss, pain per region, fatigue, load ...) | [`results/style_net.json`](results/style_net.json), MLP 31-64-64-24, about 7,800 weights | works: 99.9 % of a per-case optimisation; trained in about 25 s on a CPU |
| **Jump network** (`adapt/leap.py`, [docs/ADAPT.md](docs/ADAPT.md#jumping-by-context-adaptleappy)) | whether and how to jump: hop, vault with the hands, pull up onto a ledge, drop, jump into water, or refuse | the ground ahead as a profile (height and water depth every quarter leg length), body, the same 24 channels | [`results/leap_net.json`](results/leap_net.json), MLP scored per landing place, about 14,000 weights | works: 91 % of a per-case optimisation, 95 % the same jump-or-stay decision |
| **Motor networks**, one per body family (`motor/`, [docs/MOTOR.md](docs/MOTOR.md)) | every joint, 30 times a second | body, environment, goal, the body's state, a motivation profile | none yet | in training: people and quadrupeds move by line muscles and still walk under a part-weight harness; birds fly on joint torques (see each section) |

The style and jump networks are trained on differentiable consequence models (fast and exact, but reduced-order);
the motor networks are trained by reinforcement learning in MuJoCo. Their forward passes are a few lines of code in
any language; a GDScript port runs both finished networks in the Reval Rebel game.

## How the bodies move

The motor networks do not turn joints: they excite muscles. Every limb and the trunk are moved by line muscles
(`muscles.py`) that run from an origin to an insertion on the bones, over wrapping surfaces at the joints (femoral
head, kneecap, condyles, the point of the hock, the bird's triosseal canal), and pull with Hill's force: activation
(10 ms up, 40 ms down), force-length, force-velocity, and the passive stretch of a long muscle. Moment arms follow
from the geometry and change with the angle; two-joint muscles (hamstrings, rectus femoris, gastrocnemius, biceps,
triceps, the digital flexors) tie their joints together as in a body. Each muscle's strength is its share of the
joint's, and a quadruped's legs are scaled to an estimate from muscle mass and fibre length: a dog's hind leg about 40
body weights of force, its fore leg 30, a cat's 95 and 60, a horse's 20 and 12. Line width in the atlases goes with
strength (diameter ~ square root of force). `python muscle_check.py` and `tests/test_muscles.py` check that every
muscle pulls its joint the anatomical way over the range; the atlases come from the physics model itself
(`python muscle_atlas.py`).

## People

### Muscles

<img src="showcase/motor/muscle_atlas_human.png" width="100%"><br><b>22 muscles per side</b>: legs, trunk, shoulder, arm

<img src="showcase/motor/muscle_twitch_human.gif" width="60%"><br>Each muscle on its own: the body pinned by the pelvis, the muscle excited and let go; red and thick while it contracts, pale and thin when relaxed. No network involved.

### Motor network

<!-- motor-status:biped:start -->
**Current network** (`python -m motor.showcase`, updated 2026-10-21 01:16): 19,000 training iterations, score 0.82 of 1.3 (survival share x (0.3 + goal)); 44 line muscles, shown with the 30% harness it still trains with (it carries part of the weight and keeps the pelvis upright).

<table><tr>
<td width="25%" valign="top"><img src="showcase/motor/latest/biped_walk_slow.gif" width="100%"><br><b>walk 0.8 m/s</b>: held 8.0 s, 0.72 m/s, goal 0.77</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/biped_walk.gif" width="100%"><br><b>walk 1.3 m/s</b>: held 8.0 s, 0.99 m/s, goal 0.57</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/biped_turn.gif" width="100%"><br><b>walk to the left</b>: fell at 1.2 s, 1.05 m/s, goal 0.55</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/biped_stand.gif" width="100%"><br><b>standing still</b>: held 8.0 s, 0.03 m/s, goal 0.91</td>
</tr></table>

<img src="showcase/motor/latest/biped_emg.png" width="100%">

Activation of every left-side muscle through the walk above (like an EMG), with the foot contacts below.
<!-- motor-status:biped:end -->

### Style network: ground, weather, injury and state

The style network sets how a body moves in its situation. In these GIFs its output drives the procedural walker of this repository; that walker no longer runs anything itself and serves the motor networks only as a training reference.

<table>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_ground.gif" width="100%"><br><b>adapt_ground</b><br>dry, wet cobbles, mud, deep snow, ice, wet ice</td><td width="50%" valign="top"><img src="showcase/adapt/adapt_injury.gif" width="100%"><br><b>adapt_injury</b><br>bleeding thigh and chest, sprain, broken arm, bruised ribs, drunk, exhausted</td></tr>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_weather.gif" width="100%"><br><b>adapt_weather</b><br>frost, gale from the side, headwind, night, smoke</td><td width="50%" valign="top"><img src="showcase/adapt/adapt_rain_cold.gif" width="100%"><br><b>adapt_rain_cold</b><br>rain soaks the body, the core cools, the network answers (body time 40x)</td></tr>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_bleed_time.gif" width="100%"><br><b>adapt_bleed_time</b><br>a cut at 1 s: hand to the wound, blood lost grows (body time 20x)</td><td width="50%" valign="top"><img src="showcase/adapt/adapt_unseen.gif" width="100%"><br><b>adapt_unseen</b><br>conditions written after training, and combinations</td></tr>
</table>

### Jump network: what a 58 kg teenager does in front of ...

`python -m adapt.leap --probe` (each cell is the network's choice; the full table is in
[docs/ADAPT.md](docs/ADAPT.md#jumping-by-context-adaptleappy)):

| ahead | normal | ice | sprained ankle | heavy load |
|---|---|---|---|---|
| wall 0.85 m | vault, hands on the top | refuse | refuse | refuse |
| wall 1.3 m | vault | refuse | refuse | refuse |
| ledge 1.4 m | pull up | refuse | pull up | refuse |
| drop 1 m | drop, knees give | refuse | drop | drop |
| drop 2.5 m or 6 m onto stone | refuse | refuse | refuse | refuse |
| quay over deep water, summer | jump in | refuse | jump in | refuse |
| cliff over shallow water | refuse | refuse | refuse | refuse |

## Four-legged animals

### Muscles

<img src="showcase/motor/muscle_atlas_quadruped.png" width="100%"><br><b>quadrupeds</b>: 11 per leg, back, belly and tail; the hind legs push, the fore legs carry; joint ranges from goniometry (a dog's stifle bends 42 to 162 degrees)

<img src="showcase/motor/muscle_twitch_quadruped.gif" width="60%"><br>Each muscle on its own: the body pinned by the pelvis, the muscle excited and let go; red and thick while it contracts, pale and thin when relaxed. No network involved.

### Motor network

One network for dog, cat, wolf, goat, horse, pig and bear on one canonical skeleton. A cat is also dropped upside
down from 2 to 5 body lengths and has to land on its feet, turning with its spine and tail.

<!-- motor-status:quadruped:start -->
**Current network** (`python -m motor.showcase`, updated 2026-10-21 01:17): 22,030 training iterations, score 0.18 of 1.3 (survival share x (0.3 + goal)); 66 line muscles, shown with the 30% harness it still trains with (it carries part of the weight and keeps the pelvis upright).

<table><tr>
<td width="25%" valign="top"><img src="showcase/motor/latest/quadruped_dog_walk.gif" width="100%"><br><b>dog, walk</b>: held 8.0 s, 0.46 m/s, goal 0.26</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/quadruped_cat_walk.gif" width="100%"><br><b>cat, walk</b>: fell at 0.7 s, 0.54 m/s, goal 0.27</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/quadruped_horse_walk.gif" width="100%"><br><b>horse, walk</b>: fell at 0.4 s, 1.20 m/s, goal 0.52</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/quadruped_cat_drop.gif" width="100%"><br><b>cat dropped upside down</b>: fell at 0.5 s, 0.14 m/s, goal 0.90</td>
</tr></table>

<img src="showcase/motor/latest/quadruped_emg.png" width="100%">

Activation of every left-side muscle through the walk above (like an EMG), with the foot contacts below.
<!-- motor-status:quadruped:end -->

### Style network

<table>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_animals.gif" width="100%"><br><b>adapt_animals</b><br>dog on ice and in snow, soaked cat, horse in mud, lame horse</td><td></td></tr>
</table>

## Birds

### Muscles

<img src="showcase/motor/muscle_atlas_bird.png" width="100%"><br><b>birds</b>: the pectoralis pulls the wing down; the supracoracoideus, under it on the keel, lifts the wing through a pulley over the shoulder

<img src="showcase/motor/muscle_twitch_bird.gif" width="60%"><br>Each muscle on its own: the body pinned by the pelvis, the muscle excited and let go; red and thick while it contracts, pale and thin when relaxed. No network involved.

### Motor network

Crow, gull, heron, sparrow and duck. Birds still fly on joint torques (the bird muscles are drawn but not yet in the
flight environment); the old procedural flap-and-glide flight is the style reference.

<!-- motor-status:bird:start -->
**Current network** (`python -m motor.showcase`, updated 2026-10-21 01:16): 58,730 training iterations, score 0.50 of 1.3 (survival share x (0.3 + goal)); joint torques, shown with the 13% harness it still trains with (it carries part of the weight and keeps the pelvis upright).

<table><tr>
<td width="25%" valign="top"><img src="showcase/motor/latest/bird_crow_cruise.gif" width="100%"><br><b>crow, level flight</b>: fell at 5.0 s, height 5.1 → 0.2 m, 7.4 m/s, goal 0.32</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/bird_crow_turn.gif" width="100%"><br><b>crow, turn</b>: fell at 5.0 s, height 4.9 → 0.2 m, 7.4 m/s, goal 0.32</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/bird_crow_climb.gif" width="100%"><br><b>crow, climb</b>: held 8.0 s, height 5.2 → 1.3 m, 7.2 m/s, goal 0.28</td>
<td width="25%" valign="top"><img src="showcase/motor/latest/bird_gull.gif" width="100%"><br><b>gull</b>: held 8.0 s, height 8.9 → 2.4 m, 7.8 m/s, goal 0.34</td>
</tr></table>
<!-- motor-status:bird:end -->

### Style network

<table>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_birds_ground.gif" width="100%"><br><b>adapt_birds_ground</b><br>crow on ice (wings out), soaked, hurt foot</td><td width="50%" valign="top"><img src="showcase/adapt/adapt_birds_flight.gif" width="100%"><br><b>adapt_birds_flight</b><br>calm, wet plumage, headwind, tailwind, carrying prey, hurt wing</td></tr>
</table>

## Status

| part | state |
|---|---|
| Style network for people, quadrupeds and birds | works; 99.9 % of per-case optimum on held-out cases; tests |
| Jump network for people and quadrupeds | works; 91 % of per-case optimum, 95 % same decision; numpy equals torch; tests |
| Evaluation system (`motor/evaluate.py`): injury, pain, strain, energy, fatigue, balance, gaze | works; each term responds to its own consequence (tests) |
| Line muscles (`muscles.py`): Hill model, wraps, two-joint muscles, strength from muscle mass | works for people, quadrupeds and birds; attachments checked over the joint ranges (tests) |
| Motion prior (`motor/amp.py`): recorded human motion, the old procedural gaits and flight as references | works in training |
| People motor network (muscles) | walks under a 30 % harness; learning to stand straight on its own muscles rather than its ligaments |
| Quadruped motor network (muscles) | dog, cat and horse walk under a 30 % harness; cat drop landings in training |
| Bird motor network | crow flies on its own; flight style from the reference; muscles not yet in flight |
| Mixed-body (monkey, bear on two legs, kangaroo, penguin) and snake motor networks | not started |
| A distilled pose network that a game runs without physics | not started |

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MUJOCO_GL=glfw                           # macOS; on Linux use egl or osmesa

python -m adapt.train                           # style network (about 25 s, CPU); --init FILE --steps 800 to warm start
python -m adapt.train --check                   # against per-case optimal styles on held-out cases
python -m adapt.probe --body dog                # what the network does in named situations
python -m adapt.leap                            # jump network (about 12 min, CPU); --check, --probe
python -m adapt.export_engine OUT_DIR           # weights, condition registry and golden vectors for ports

python -m motor.pbt --family biped --name biped_pbt --members 3 --workers 3   # population training (also --family quadruped)
python -m motor.ppo --family biped --name biped --workers 15 --hours 4   # a single run
python -m motor.show runs/motor/biped/ckpt.pt [out_dir] [--assist 0.4]   # GIFs and summary.json, ten review situations

python -m pytest tests/test_adapt.py tests/test_leap.py tests/test_motor.py
```

Needs ffmpeg for GIFs. Python 3.10+.

## Using the networks in an engine

`python -m adapt.export_engine OUT_DIR` writes `style_net.json` and `leap_net.json` (weights with their input and
output schema), the condition registry, and golden vectors (inputs with the expected outputs) so that a port can be
checked number for number. A port is a matrix multiply, a tanh and a sigmoid per layer; see `adapt/net.py` for the
numpy version. The motor networks will be exported the same way once a policy works, as a distilled pose network
that needs no physics engine.

## Layout

| path | role |
|---|---|
| `adapt/` | style network and jump network: schema, conditions, consequence models, training, physiology, export ([docs/ADAPT.md](docs/ADAPT.md)) |
| `motor/` | motor networks: evaluation system, body-family environments, PPO, review renderer ([docs/MOTOR.md](docs/MOTOR.md)) |
| `results/` | trained weights (`style_net.json`, `leap_net.json`) |
| `showcase/` | GIFs of the style network (`adapt/`), the muscles and the current motor networks (`motor/`) |
| `tests/` | tests |

## License

MIT, see `LICENSE`.
