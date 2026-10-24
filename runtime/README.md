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
To reuse the source/weights already tested in Reval Rebel instead:

```sh
python -m runtime.build --source ../rebel-reval/build/neural_motion/native \
  --output ../rebel-reval/build/neural_runtime
python -m runtime.test_native --worker ../rebel-reval/build/neural_runtime/skeleto_motion_worker \
  --bundle ../rebel-reval/build/neural_runtime
```

`build.py` checks the source revisions and the model file sizes/SHA-256 against the
upstream distribution manifest. It produces a standalone executable, four model files,
three style assets, a versioned `runtime.json` and the required license/attribution notices.
Do not commit these approximately 0.74 GB build outputs to source control.

To install the verified bundle into an already exported game app:

```sh
python -m runtime.package --source ../rebel-reval/build/neural_runtime \
  --destination /path/to/RevalRebel.app
```

This puts the bundle in `Contents/Resources/neural_runtime`. Sign the completed app after
packaging. Reval Rebel's `tools/export_neural_game.py` performs export, packaging and ad-hoc
signing together. For a non-macOS executable folder, the same package command installs
`neural_runtime/` alongside the executable.

## Runtime contract

Start `skeleto_motion_worker BUNDLE_ROOT THREADS`. The worker emits one JSON metadata line
with `protocol: "skeleto-nn/motion/1"`, 30 FPS and the 34-joint G1 hierarchy/neutral positions.
Then send one ASCII request per line:

```text
plan ID ADVANCE SPEED MOVE_X MOVE_Z FACE_X FACE_Z MODE PHASE
plan 23 0 0.7 0 1 0 1 locomotion 0
```

`ADVANCE` is an index into the preceding generated plan, not wall-clock time; it selects
the next four context frames. Speeds are metres per second in the source body's scale.
The consumer must account for character scale and initial heading. `ID` also supplies the
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
keyframes are constraints, not learned muscle policies or playback clips. The network
generates the transition; the game still owns its action clock, collision decisions,
jump trajectory, and exact hand/foot contacts. Do not describe this as learned physics.

Each reply contains `id`, `mode`, `frames`, `inference_ms`, `roots[frames*3]` and
`rotations[frames*34*4]` (local XYZW quaternions). Coordinates are Y-up, Z-forward, metres.
The consumer must validate lengths, finite numbers and quaternion norms, buffer playback,
and keep the game thread independent of inference. Invalid input returns an error object;
startup failure exits nonzero. `quit` or stdin EOF exits. The game must own and close the child.

## Reval Rebel integration

`scripts/motion/neural_motion_driver.gd` drives the hero only and smoothly yields to authored
combat, swimming, riding and other actions. `motion_contact_modifier.gd` places feet on the
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
