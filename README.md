# ☠️ skeleto-nn

Neural network powered character motion for **any skeleton**: people, 🐕 🐈‍⬛ 🐎 🦅 🐍

A body is a tree of bones with joint limits, masses and a few anatomical and physiological numbers; what it does is
decided by small networks that are trained, without motion capture, against the physical and physiological
consequences of what they do (injury, pain, strain, energy, fatigue, falls, heat loss, blood loss, wind).

| network | what it decides | inputs | weights | state |
|---|---|---|---|---|
| **Style network** (`adapt/`, [docs/ADAPT.md](docs/ADAPT.md)) | how a body moves in its situation: stride, crouch, lean, arms out or hugged, a hand on a wound, limp, shiver, wingbeat | body, 24 situation channels (friction, sinking feet, wetness, cold, wind, darkness, smoke, blood loss, pain per region, fatigue, load ...) | [`results/style_net.json`](results/style_net.json), MLP 31-64-64-24, about 7,800 weights | works: 99.9 % of a per-case optimisation; trained in about 25 s on a CPU |
| **Jump network** (`adapt/leap.py`, [docs/ADAPT.md](docs/ADAPT.md#jumping-by-context-adaptleappy)) | whether and how to jump: hop, vault with the hands, pull up onto a ledge, drop, jump into water, or refuse | the ground ahead as a profile (height and water depth every quarter leg length), body, the same 24 channels | [`results/leap_net.json`](results/leap_net.json), MLP scored per landing place, about 14,000 weights | works: 91 % of a per-case optimisation, 95 % the same jump-or-stay decision |
| **Motor networks**, one per body family (`motor/`, [docs/MOTOR.md](docs/MOTOR.md)) | every joint, 30 times a second | body, environment, goal, the body's state, a motivation profile | none yet | in progress: the evaluation system, the biped environment and the trainer work; **no policy walks yet** (see below) |

The style and jump networks are trained on differentiable consequence models (fast and exact, but reduced-order);
the motor networks are trained by reinforcement learning in MuJoCo. Their forward passes are a few lines of code in
any language; a GDScript port runs both finished networks in the Reval Rebel game.

## Showcase

### Style network: one network for ground, weather, injury and state

The network sets the style; the bodies are drawn by the kinematic baseline walker of this repository (see
[Baselines](#baselines-code-that-is-not-a-network)).

<table>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_ground.gif" width="100%"><br><b>adapt_ground</b><br>dry, wet cobbles, mud, deep snow, ice, wet ice</td><td width="50%" valign="top"><img src="showcase/adapt/adapt_injury.gif" width="100%"><br><b>adapt_injury</b><br>bleeding thigh and chest, sprain, broken arm, bruised ribs, drunk, exhausted</td></tr>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_weather.gif" width="100%"><br><b>adapt_weather</b><br>frost, gale from the side, headwind, night, smoke</td><td width="50%" valign="top"><img src="showcase/adapt/adapt_rain_cold.gif" width="100%"><br><b>adapt_rain_cold</b><br>rain soaks the body, the core cools, the network answers (body time 40x)</td></tr>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_bleed_time.gif" width="100%"><br><b>adapt_bleed_time</b><br>a cut at 1 s: hand to the wound, blood lost grows (body time 20x)</td><td width="50%" valign="top"><img src="showcase/adapt/adapt_animals.gif" width="100%"><br><b>adapt_animals</b><br>dog on ice and in snow, soaked cat, horse in mud, lame horse</td></tr>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_birds_ground.gif" width="100%"><br><b>adapt_birds_ground</b><br>crow on ice (wings out), soaked, hurt foot</td><td width="50%" valign="top"><img src="showcase/adapt/adapt_birds_flight.gif" width="100%"><br><b>adapt_birds_flight</b><br>calm, wet plumage, headwind, tailwind, carrying prey, hurt wing</td></tr>
<tr><td width="50%" valign="top"><img src="showcase/adapt/adapt_unseen.gif" width="100%"><br><b>adapt_unseen</b><br>conditions written after training, and combinations</td><td></td></tr>
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

### Motor networks: the first biped training runs (not working yet)

Reinforcement learning without reference motion finds cheap but wrong answers first. These are the policies of the
first two runs, kept to show where the work stands ([docs/MOTOR.md](docs/MOTOR.md) has the details and the fixes):

<table>
<tr><td width="33%" valign="top"><img src="showcase/motor/biped_run1_walk.gif" width="100%"><br><b>run 1</b>, asked to walk at 1.3 m/s, harness at 40 %: the harness pushed toward the goal, so the policy learned to lean stiffly and be pushed; it falls like a plank</td>
<td width="33%" valign="top"><img src="showcase/motor/biped_run2_walk.gif" width="100%"><br><b>run 2</b>, same request, full harness: with a reward for staying up and a harness that only holds, it learned to stand still (0.04 m/s)</td>
<td width="33%" valign="top"><img src="showcase/motor/biped_run2_walk_no_harness.gif" width="100%"><br><b>run 2</b> without the harness: it falls within 2 s</td></tr>
</table>

## Status

| part | state |
|---|---|
| Style network for people, quadrupeds and birds | works; 99.9 % of per-case optimum on held-out cases; tests |
| Jump network for people and quadrupeds | works; 91 % of per-case optimum, 95 % same decision; numpy equals torch; tests |
| Motor networks: evaluation system (`motor/evaluate.py`) | works; each term responds to its own consequence (tests) |
| Motor networks: biped environment, mirror symmetry, PPO with symmetry loss and harness (`motor/biped.py`, `motor/ppo.py`) | works; about 9,000 simulated steps per second on 15 CPU workers |
| Motor networks: a biped policy that walks | **not yet**: run 1 leaned and was pushed, run 2 stands still (above). Next: a speed curriculum from standing, a harness that fades with time, starts from mid-stride |
| Quadruped, mixed-body (monkey, bear, kangaroo, cat, penguin), bird and snake motor networks | not started |
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

python -m motor.ppo --family biped --name biped --workers 15 --hours 4   # motor network
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

## Baselines: code that is not a network

The repository started as a workbench of other methods, and their code is kept as baselines to compare the motor
networks against, and as the presentation layer of the style network's GIFs. They are no longer documented or
showcased, and they will be deleted family by family as the motor networks replace them:

| module | method |
|---|---|
| `planner.py`, `terrain.py`, `nav.py`, `controls.py`, `jumper.py`, `climb.py`, `aids.py` | kinematic walker: gait from the Froude number, footholds, a jump self-model |
| `goals.py`, `posing.py`, `skills.py`, `catalog_hang.py` | authored goal sequences solved by IK (chair, ladder, door) |
| `catalog_home.py` (swim strokes), `catalog_birds.py` (flapping, landing), `catalog_extra.py` (snake) | hand-built phase models |
| `reflex.py`, `slide.py`, `swimmer.py`, `flier.py`, `swim_evolve.py`, `flier_evolve.py`, `results/reflex_*.json`, `results/slide_controller.json`, `results/swim_*.json`, `results/bird_*.json` | controller parameters evolved by CMA-ES |
| `env.py`, `train.py`, `eval_policy.py`, `ref.py` | PPO tracking the walker's reference (never converged) |
| `hit.py`, `combat.py`, `catalog*.py`, `shots.py`, `bake_clip.py`, `export_clip.py`, `make_showcase.py`, `phys_gifs.py` | reactions, scenes, clip baking, the old showcase |

Shared by everything and kept: `skeleton.py` and `bodies.py` (bones, parametric bodies), `physics.py`, `water.py`,
`air.py` (MuJoCo bodies with water and air), `injury.py` (the bone-injury model the evaluation system uses),
`render.py` (GIF rendering).

## Layout

| path | role |
|---|---|
| `adapt/` | style network and jump network: schema, conditions, consequence models, training, physiology, export ([docs/ADAPT.md](docs/ADAPT.md)) |
| `motor/` | motor networks: evaluation system, body-family environments, PPO, review renderer ([docs/MOTOR.md](docs/MOTOR.md)) |
| `results/` | trained weights (`style_net.json`, `leap_net.json`) and baseline parameter files |
| `showcase/` | GIFs of the style network (`adapt/`) and of the motor training runs (`motor/`) |
| `tests/` | tests |

## License

MIT, see `LICENSE`.
