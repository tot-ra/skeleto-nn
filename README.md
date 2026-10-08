# skeleto-nn

Procedural character motion for **any skeleton**: people, dogs, cats, horses, birds, snakes. No animation clips, no per-situation scripts for locomotion: a body is a tree of bones with joint limits, masses and a few anatomical numbers, and one planner makes it walk, run, jump, climb, duck, react to blows and fall.

![walk and run](showcase/sim/walk_run.gif)

**Showcase (about 60 GIFs): open [`showcase/index.html`](showcase/index.html).**

| | |
|---|---|
| ![stairs](showcase/sim/stairs_up_down.gif) | ![forest](showcase/sim/run_forest.gif) |
| ![cat jump](showcase/sim/quad_jump_species.gif) | ![climb](showcase/sim/climb_rock.gif) |
| ![dodge](showcase/sim/dodge_reactions.gif) | ![weapons](showcase/sim/weapon_reactions.gif) |

## What is in it

* **A planner for legged bodies** (`planner.py`). Gait (walk, trot, canter, gallop) is picked from the Froude number, so a dwarf, a horse and a cat choose their own cadence. Feet are planted in the world and swing between footholds searched on the terrain (stairs, kerbs, logs are just terrain). The pelvis height is the highest the stance legs can reach (so it vaults and dips by itself), the trunk leans to keep the centre of mass over the feet (a belly, a pack or armour changes the posture without extra code), a ceiling lowers the body, a jump is a power-limited ballistic arc.
* **Anatomy as parameters** (`bodies.py`). Human: 3 lumbar + lower-thoracic + chest joints, clavicles, a toe bone in each foot. Quadrupeds: dog, cat, wolf, horse, pig differ in number of spine joints and range of the column, neck joints and nod, tail type (balance, wag, low, hair, curl), hoof or paw, hind-limb power for jumping. Birds, a snake. People differ by height, mass, leg ratio, belly, pack, armour, sex, and **outfit** (tight skirt, wide skirt, tight trousers, heavy boots, high heels, chainmail, plate: joint limits, extra mass, foot shape, stride and speed).
* **Goal-space skills** (`goals.py`, `posing.py`, `skills.py`): door, sit, lie in a bed, get up from the floor, ladder, ride, swim, dive. Goals for pelvis, hands and feet plus IK.
* **Combat as reactions** (`combat.py`, `hit.py`). A blow (stick, axe, arrow) is aimed at a body part. The defender notices it after its own delay and picks, by predicting the geometry and the pain, between duck, side step, step back, hop, block with a forearm, a shield or a lifted knee, or taking it. At impact the hit is resolved from the real positions. Pain per region (head, torso, groin, arms, legs, knees) makes an arm unusable (the hand clutches, a weapon is dropped), makes a leg limp, doubles the body over, or knocks it down; a blow to the groin is felt about three times less by a woman.
* **Limping and walking aids** (`aids.py`): a sore leg, a rigid stick instead of a lower leg, crutches with one leg held up, a dog on three legs.
* **Climbing where only some points can be held** (`climb.py`): three points of contact stay, the fourth limb reaches for a free hold in range.
* **Moves in clutter**: running into a wall with and without the arms up, running through a forest and through a crowd, squeezing between people.
* **A bone-injury model** (`injury.py`). A bone breaks when the force through it exceeds a tolerance, and it tolerates less when the load arrives fast. A body that yields while it stops spreads the same momentum over a longer stroke: lower peak force, lower rate, lower risk. The head has the lowest tolerance and the highest weight. This one score drives everything that is learned or evolved here: the fall and landing reflex, the slope-slide controller, the PPO reward, and the choice of a reaction.
* **A physical body** (`physics.py`, MuJoCo): the same skeleton as a torque-limited PD ragdoll. `reflex.py` evolves (CMA-ES) a small fall and landing controller against the injury score. Knock-downs in the demos are played from this physical fall.

## Honest status

| Part | How it is made | State |
|---|---|---|
| Walk, run, stairs, slopes, ducking, navigation, jumping, crowds, forest | planner, emergent from body and terrain | works, stick-figure level polish |
| Door, chair, bed, floor, ladder, ride | goals + IK | works, scripted goal sequences |
| Swimming, diving, bird flight and landings, snake | kinematic models | works, hand-built |
| Combat reactions, pain, limping, knock-downs | geometry + pain model + physical fall | works |
| Fall and landing reflex | physics, evolved on the injury score | see numbers below |
| Walking policy that tracks the planner under pushes (PPO) | physics, learned | **does not converge** without an assist harness (about two thirds of episodes fall when the assist is 0); code kept, no checkpoint shipped |

### Fall and landing reflex, 240 held-out shoves and drops (drops of 0.4 m to 2.6 m)

Mean fracture risk per region from the injury model (0 = none, 1 = certain); cost is the weighted score the controller was evolved on.

| controller | injury cost | head | torso | arms | legs |
|---|---|---|---|---|---|
| statue (stiff, no reflex) | 3.72 | 0.46 | 0.67 | 0.44 | 0.71 |
| hand-made starting pose | 3.81 | 0.55 | 0.48 | 0.73 | 0.53 |
| evolved on bone-injury score | 1.61 | 0.02 | 0.01 | 0.61 | 0.78 |

The evolved reflex nearly removes head and torso injury (risk 0.46 and 0.67 for a stiff body, 0.02 and 0.01 evolved). Arms and legs still take large peak forces: the legs carry 12 to 15 body weights in drops of 1 to 2.5 m against 34 to 66 for the stiff body, which is better but still above the leg tolerance of 10, so the limb columns stay high. That is the weak part and the next thing to improve.

### Standing on a slippery slope, 40 held-out slopes (8 to 25 degrees, friction 0.12 to 0.4)

| controller | stays up 5 s | mean time up | mean slide |
|---|---|---|---|
| stiff body, no controller | 0% | 1.1 s | 0.44 m |
| evolved controller | 20% | 1.8 s | 0.25 m |

A partial result: the evolved controller helps, most slopes still end in a fall.


## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MUJOCO_GL=glfw          # macOS; on Linux use egl or osmesa
python catalog.py list                         # all shots
python catalog.py gif walk_run runs/walk_run.gif
python catalog.py sheet duel runs/duel.png 8   # contact sheet
python make_showcase.py                        # rebuild runs/showcase/index.html from runs/gifs3
python tests/test_skeleton.py && python tests/test_physics.py   # planner FK equals MuJoCo FK
python reflex.py --gens 300 --pop 28 --procs 8 --out runs/reflex.json
python reflex_eval.py runs/reflex.json runs/reflex_eval
```

Needs ffmpeg for GIFs. Python 3.10+.

## Layout

| File | Role |
|---|---|
| `skeleton.py`, `bodies.py`, `ik.py` | bones, forward kinematics, parametric bodies and outfits, analytic IK |
| `planner.py`, `terrain.py`, `nav.py` | the walker, height-field terrain with ceilings and pits, A* |
| `goals.py`, `posing.py`, `skills.py`, `climb.py` | goal-space skills |
| `hit.py`, `combat.py`, `injury.py` | pain, knock-down, reactions, injury score |
| `physics.py`, `reflex.py`, `slide.py`, `env.py`, `train.py` | MuJoCo body, evolved controllers, PPO tracker |
| `shots.py`, `catalog*.py`, `render.py` | scenes and GIF rendering |
| `export_clip.py`, `retarget_blender.py`, `game_gif.py` | optional: retarget a motion onto your own rigged glTF in Blender |

See [`docs/DESIGN.md`](docs/DESIGN.md) for how it works and where it is weak.

## Use it on your own rig

`bodies.py` describes a creature; `human_game_rig()` shows how the proportions of an existing rig are passed in. The planner outputs, per frame, world rotations of every bone; `retarget_blender.py` applies them to a rig by bone-name map (edit `MAP`). Nothing in the planner assumes a particular rig.

## License

MIT, see `LICENSE`.
