# Native pretrained motion runtime

[Current game integration and GIF (2026-10-25)](../README.md#game-integration-2026-10-25).

This is the game deployment path. It uses pretrained NVIDIA MotionBricks weights through
the Apache-2.0 `motion-bricks.cpp` port and MIT GGML. It does not depend on Python, PyTorch,
MuJoCo, a running server, or network access while the game runs. Python and CMake are build tools only.
The separate `motor/` muscle-learning experiment and its checkpoints are preserved.

The validated platform is macOS arm64, CPU inference with two threads. Other platforms
need their own native build and verification; no Apple GPU or cross-platform speed claim is made.

## Build and verify

From this repository, with Git, CMake and the platform C++ compiler installed:

```sh
python -m runtime.build
python -m runtime.test_native --worker runs/native_runtime/skeleto_motion_worker \
  --bundle runs/native_runtime
```

The first command downloads the explicit source revisions and SHA-256-checked weights
into `runs/`, then builds a self-contained CPU helper. Existing source directories with
different revisions are left intact and cause an error. Python ML packages are not needed.

Reval Rebel keeps its sources, weights, build tree and installed bundle outside the game
tree in `~/.local/share/rebel-reval/neural/` (`native/`, `worker-build/`, `runtime/`;
override with `REBEL_NEURAL_STORE` or `XDG_DATA_HOME`). The game's
`tools/setup_neural_runtime.py --skeleto /path/to/skeleto-nn` runs this build with those
paths; the equivalent direct commands are:

```sh
STORE=~/.local/share/rebel-reval/neural
python -m runtime.build --source $STORE/native --output $STORE/runtime --build-dir $STORE/worker-build
python -m runtime.test_native --worker $STORE/runtime/skeleto_motion_worker --bundle $STORE/runtime
```

`build.py` checks the source revisions and the model file sizes/SHA-256 against the
upstream distribution manifest. It produces a standalone executable, four model files,
three style assets, a versioned `runtime.json` and the required license/attribution notices.
Do not commit these approximately 0.74 GB build outputs to source control.

To install the verified bundle into an already exported game app:

```sh
python -m runtime.package --source ~/.local/share/rebel-reval/neural/runtime \
  --destination /path/to/RevalRebel.app
```

This puts the bundle in `Contents/Resources/neural_runtime`. Sign the completed app after
packaging. Reval Rebel's `tools/export_neural_game.py` performs export, packaging and ad-hoc
signing together. For a non-macOS executable folder, the same package command installs
`neural_runtime/` alongside the executable.

## Runtime contract

Start `skeleto_motion_worker BUNDLE_ROOT THREADS`. The worker emits one JSON metadata line
with `protocol: "skeleto-nn/motion/1"`, `protocol_version: 2`, the loaded `styles`, 30 FPS
and the 34-joint G1 hierarchy/neutral positions. Then send one ASCII request per line
(v1 form; see [Protocol v2](#protocol-v2-2026-11-06) for `hello` and `plan2`):

```text
plan ID ADVANCE SPEED MOVE_X MOVE_Z FACE_X FACE_Z MODE PHASE
plan 23 0 0.7 0 1 0 1 locomotion 0
```

`ADVANCE` is an index into the preceding generated plan within the same mode, not wall-clock time; it selects
the next four context frames. Speeds are metres per second in the source body's scale.
The consumer must account for character scale and its chosen coordinate frame. Reval Rebel sends body-local movement with facing +Z; its actor owns yaw, and the retargeter removes predicted root yaw. `ID` also supplies the
random seed; a command stream is reproducible on the same build/backend.

Modes:

| Mode | Network input and use |
| --- | --- |
| `locomotion` | Previous generated pose, desired velocity/facing, supplied idle/walk style. |
| `jump_start` | Crouched takeoff goal keyframes. |
| `jump_air` | Tucked-leg goal keyframes. |
| `jump_land` | Landing give, reducing with `PHASE` in 0..1. |
| `door` | Right-arm reaching goal keyframes. |
| `ladder` | Alternating arm and knee goals from rung phase in 0..1. Exact contacts stay in the game. |

Locomotion calls the pretrained agent. Action modes call the pretrained stateless
in-betweening API with four source and four goal frames and a 24-frame horizon. Goal
keyframes are absolute neutral-relative constraints, not offsets compounded onto the last action, learned muscle policies or playback clips. The target root stays upright, and switching modes resets the agent context to idle. The network
generates the transition; the game still owns its action clock, collision decisions,
jump trajectory, and exact hand/foot contacts. Do not describe this as learned physics.

Each reply contains `id`, `mode`, `style`, `masked` (constrained joint count, 0 for plain
locomotion and for the full-body v1 modes), `frames`, `inference_ms`, `roots[frames*3]` and
`rotations[frames*34*4]` (local XYZW quaternions). Coordinates are Y-up, Z-forward, metres.
The consumer must validate lengths, finite numbers and quaternion norms, buffer playback,
and keep the game thread independent of inference. Invalid input returns an error object;
startup failure exits nonzero. `quit` or stdin EOF exits. The game must own and close the child.

## Protocol v2 (2026-11-06)

Every v1 line above is accepted unchanged. A v2 worker additionally answers two commands:

```text
hello
plan2 ID ADVANCE SPEED MOVE_X MOVE_Z FACE_X FACE_Z MODE PHASE STYLE MASK GOALS
plan2 23 10 0.5 0 1 0 1 locomotion 0 slow_walk arm_r right_shoulder_pitch_skel=-0.434966,0,0,0.900447;right_elbow_skel=0.223106,0,0,0.974794
```

`hello` (or `version`) replies with one line `{"hello":true,"protocol_version":2,"v1":true,
"styles":[...],"style_speeds":{...},"groups":{...},"modes":[...]}`: the style names loaded
from `BUNDLE_ROOT/styles/*.mbstyle` (the shipped bundle has `idle`, `slow_walk`, `walk`),
their configured paces in m/s (0, 0.3, 1.0), the joint groups a mask may name and the
modes. The reply carries no `protocol` key, so a v1 client's metadata check never sees it.

- `STYLE` is a loaded style name or `-` (v1 choice: `idle` under 0.05 m/s, else `walk`).
  Standing still always uses `idle`. `SPEED -1` with a named style walks at that style's
  configured pace (v1 lines still reject negative speeds). With an explicit speed the
  bundled `slow_walk` produces the same joint poses as `walk`: the two exemplars share
  their pose frames and differ only in root pace and configured speed, so the style
  changes stride timing only at the style pace (knee period 47 frames at 0.3 m/s versus
  31 at 1.0 m/s in `test_native`).
- `MASK` is `none`, `all`, or a comma list of joint groups: `legs` (14 joints), `arms`,
  `arm_l`, `arm_r` (8 each), `spine` (3), `head` (accepted; the G1 body has no head joints,
  so it constrains nothing). The constrained set is the mask plus every joint named in
  `GOALS` and in the mode's table (below).
- `GOALS` is `-` or `joint=qx,qy,qz,qw;...`: absolute neutral-relative local XYZW rotations
  (unit within 0.01) the constrained joints must reach at the horizon end (frame 24). A
  constrained joint without a goal goes to neutral.

With a partial mask the worker runs its locomotion plan as usual, then the stateless
in-betweener between four source frames (the plan start; a joint constrained in the
previous delivery continues from the pose the game consumed, so a held goal does not
restart from the arm swing) and four target frames (the plan's own frames 20..23 with
the constrained joints replaced by their goals). The delivered 24 frames take every
unmasked joint and the root from the locomotion plan and only the constrained joints
from the generated motion; the agent's next context is the locomotion plan, so the legs
of a plan with an `arm_r` goal are identical to the plan without it. When the goals stop,
one 24-frame plan blends the released joints back into the walk before plain 40-frame
plans resume. `mb_inference_request_set_mask` upstream is per boundary slot (eight flags a
field), not per joint, which is why joint selection happens in the boundary composition;
the worker still uses it for the fixed 24-frame duration.

The six v1 modes are now goal tables in the worker (`mode_table`): `door` is the right
shoulder pitch -0.9 rad and elbow 0.45 rad, `ladder` alternating shoulder/elbow/hip/knee
goals from the rung phase, the jumps hip/knee/ankle bends with a root height. A v1 line
or `MASK all` constrains the whole body to neutral plus the table and the table's root
height, exactly as before (the 256-plan regression is unchanged). `plan2 ... door 0 walk
arm_r -` keeps the door table but constrains only the right arm over the walk.
Mode changes still reset the agent context; a goal overlay in `locomotion` does not.

Malformed v2 lines (unknown style, group or joint, a non-unit or three-component goal,
missing fields, trailing tokens) answer `{"error":...}` and the worker keeps serving.

## Sustained gait regression

```sh
python -m runtime.test_native --worker runs/native_runtime/skeleto_motion_worker \
  --bundle runs/native_runtime --long
```

This checks 256 consecutive real-weight plans across walking, repeated ladder/door/jump goals,
and walking again. It rejects accumulated knee angles and falling root orientations. Without
`--long` the script runs the v1 contract (six modes, invalid input, clean exit) and the v2
checks: handshake and style list, `slow_walk` at its pace travels slower with a longer knee
period than `walk`, an `arm_r` goal leaves hips, knees and the root of ten consecutive plans
within 2 degrees (measured 0) of the same stream without the goal while the shoulder reaches
the goal, the release plan, the door through v2 versus v1, and eight malformed v2 lines
followed by a served request (30 checks, 2026-11-06). The game
also checks startup turns before metadata arrives, 90/180-degree turns, slope-frame consistency,
foot-contact release, and a long route at its actual 1.5 m/s walk speed. The source body's walking
range is limited to 2 m/s by the game adapter; its 8 m/s sprint uses the authored run clip.
No pretrained weights were retrained for these fixes.

## Reval Rebel integration

`scripts/motion/neural_motion_driver.gd` drives the hero only and smoothly yields to authored
combat, sprinting, sitting/lying, swimming, riding and other actions. Playback uses the protocol
frame rate (30 FPS), including action plans. `motion_contact_modifier.gd` places feet on the
city's actual `walk_height` surfaces and makes the right hand follow a door leaf's handle.
The fixed smithy ladder supplies rung targets and phase; the game owns ascent, descent, stamina and its loft layer.
Slopes and existing stair flights use the same ground query, not a new movement mechanic.
Jump poses blend with the game's existing jump clips and wall-hand constraints. Fingers,
head and exact scene contacts are not generated by this G1 model. Missing/incompatible
runtime assets preserve normal animation playback.

Commercial permission comes from the specific NVIDIA model license, not this repository's
MIT code license. Keep `UPSTREAM_LICENSE`, `NOTICE`, `MOTIONBRICKS_CPP_LICENSE`, `GGML_LICENSE`
and `SKELETO_LICENSE` with redistributed bundles. No raw BONES-SEED or research mocap is
copied by this pipeline. See the upstream model license for its full terms.
