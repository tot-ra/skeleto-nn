# Design

The aim is a motion system that does not need a clip for each situation. The body is described by numbers; what it does comes from a small set of general mechanisms. Where a situation is a script (a door, a bed), the script is a list of goals for the pelvis, hands and feet, not joint angles, and the same IK and balance code carries it out.

## 1. Bodies (`bodies.py`, `skeleton.py`)

A creature is a tree of rigid bones with hinge degrees of freedom, limits, masses and torque limits. People, quadrupeds, birds and a snake are parameter sets.

* **Human**: pelvis, three lumbar joints, a lower thoracic joint, the chest, neck, head; a clavicle on each side; thigh, shank, foot and a toe bone; upper arm, forearm, hand. Height, mass, leg ratio, belly, pack, armour, width, sex and outfit are parameters. 53 degrees of freedom.
* **Quadrupeds** (`QuadSpec`): number of mobile spine joints and the sagittal, lateral and twist range of the whole column (cat 8 joints and 120 degrees, dog 5 and 75, horse 3 and 23, pig 3 and 28), neck joints and range, nod and stretch at speed, tail type and tail joints, hoof or paw, trunk torque, and hind-limb power for jumping. The numbers follow comparative anatomy; they are parameters, not measurements of an individual animal.
* **Tail types** are damped chains with a species table: `balance` (cat: strong counter-swing to the turn, slow idle sway, curl at the tip), `wag` (dog: signal wag scaled by mood), `wag_low` (wolf), `hair` (horse: heavy, lifts with speed, flicks at rest), `curl` (pig: stiff spiral).
* **Outfits** change joint limits (a tight skirt limits hip flexion and abduction, plate armour stiffens the spine, shoulder and ankle), add mass on the limbs they cover, and set gait parameters (stride scale, maximum speed, heel pitch of a high heel, balance). The planner copes: stride, speed, toe roll and trunk response follow.

## 2. The planner (`planner.py`)

Nothing in it is specific to stairs, slopes, an obstacle or a species.

* Gait parameters follow from the Froude number v²/(g h): duty factor, stride length, and for quadrupeds the blend walk, trot, canter, gallop.
* Each foot is planted in the world during stance and swings between footholds that are searched on the terrain (flat, away from edges, reachable). Swing height comes from the terrain between lift-off and landing.
* The trunk height is the highest value the stance legs can reach, so the pelvis vaults over a planted leg and dips in double support.
* The trunk pitch moves the whole-body centre of mass over the feet. A belly, a pack or armour therefore change posture without extra code.
* A ceiling lowers and tilts the trunk (ducking). The toes bend so they stay flat on the ground while the heel rises.
* **Jumps are power limited.** The legs can give a vertical take-off speed up to `jump_gain * sqrt(g L)` and a total speed 1.5 times that; a run-up adds horizontal speed. The planner searches for the lowest apex that fits both limits and refuses a jump that does not (a standing jump of 3.5 m, a wall that is too high). Cats, dogs and horses differ in `jump_gain`; the trunk rears and the spine coils and extends with the species range.
* A push is a velocity change plus a spring-damper whip of the trunk; recovery steps follow from foot placement on the predicted hip position.

## 3. Goal-space skills (`goals.py`, `posing.py`, `skills.py`)

A motion is a list of keyed goals (pelvis pose, feet, hands) blended with minimum jerk. `Poser` solves feet and hands by analytic IK; the shoulder girdle follows the reach. These are authored sequences and are labelled as such in the showcase.

## 4. Pain, blows and reactions (`hit.py`, `combat.py`)

* `apply_hit(region, direction, strength)` is the single entry point. Regions: head, torso, groin, arm L/R, leg L/R, knee L/R. Strength is a velocity change in m/s.
* Pain per region decays over seconds. It changes what the body can do: an injured arm cannot hold a goal (the hand goes to the hurt place and a weapon drops), an injured leg limits speed and the body leans off it, torso pain doubles the body over with the hands at the belly, head pain makes the walk weave and puts a hand on the head, groin pain bends the body and slows it (felt about three times less by a woman).
* A hard enough blow, or too much pain, knocks the body down. The fall is simulated on the physical body with the evolved reflex (section 6) and played back; the get-up is a goal sequence that starts from the pose the body ended in and does not load an injured limb.
* **Reactions.** A strike (stick, axe, arrow) is a hand path with a fixed aim after launch. The defender notices it after its own delay and chooses by **expected pain**: for each candidate (duck, side step, step back, hop back, hop up, block with a forearm, a shield, or a lifted knee, none) the geometry at impact is predicted and the cost is the effort of the move plus the weighted pain of the blow if the move fails. Strong blows to the head justify big moves; a weak jab to an arm is taken. At impact the hit is resolved from the real positions of the stick or arrow and the body: miss, blocked (with its own damage factor) or hit.
* The attacker keeps re-aiming until the blow is launched, so an early dodge is followed; the best dodge is late.

## 5. The injury model (`injury.py`)

Forces are in body weights. A region's fracture risk is `1 - exp(-(F k / F_tol)^3)` where `k = min(3, 1 + rate / 500)` and the rate is the largest rise of the force over 20 ms in body weights per second: fast loading is more dangerous. Tolerances (slow loading, about 63% risk): head 5, torso 8, arms 4, legs 10 body weights (order of magnitude from skull 4 to 6 kN, ribs, wrist 2.5 to 3.5 kN, femur and tibia 5 to 10 kN). The score weights the head 3, torso 1.5, arms and legs 1, plus a small term for force above body weight over time (pain).

Stopping a mass m at speed v over a stroke s takes F = m v² / (2 s): a rigid stop (1 cm) and a yielding one (30 cm, flexing limb or roll) differ by a factor of thirty in force. This is the formal version of "a direct blow to a bone breaks it, a blow absorbed gradually by the muscles does not", and the head is protected because it is the cheapest place to be wrong.

Everything that is learned or evolved uses this score: `reflex.py` (fall and landing), `slide.py` (standing on a slippery slope), `env.py` (PPO reward has a pain term), and the reaction choice above.

## 6. The physical body and the evolved reflex (`physics.py`, `reflex.py`)

The same skeleton is built as a MuJoCo ragdoll with torque-limited PD joints. The reflex is about 50 numbers: a pose that depends linearly on the direction of the fall, the air pose before touching down, a trigger angle, stiffness of the whole body, of the arms and of the legs, and a time constant after which the limbs go soft so that the stroke is long. CMA-ES minimises the injury score over random shoves and drops (fresh scenarios each generation). Contact forces are sampled at every physics sub-step so that peaks and loading rates are real. The ground and body contacts are softened to a time constant of 20 ms (soft tissue, soles): a perfectly rigid contact would add a force spike that no real body has.

## 7. Climbing (`climb.py`)

Holds are points with a kind (hand, foot, both). Three points of contact stay; the fourth limb is chosen and sent to the best free hold that is in reach of its shoulder or hip, usable by that limb, keeps the feet below the hands and does not overreach. The pelvis target is near the wall, over the feet, within reach of every hold used. When nothing is reachable the climber waits, then accepts a poor hold (a foot hold as a handhold) before reporting that it is stuck. A greedy search can end in a dead end; the random walls in the demo are climbable, a hard route would need a search over several moves.

## 7b. Limping and aids (`aids.py`, `hit.py`, `bodies.py`)

A leg can be sore (chronic pain makes the stance on it shorter and the trunk lean over it), replaced below the knee by a rigid stick (`HumanBody(peg="L")`: no ankle, a point contact, shorter stance, a swing that goes out and round), or carry no weight (`nwb`, held up). With a leg out of use, `Crutches` plants the tips while the good leg swings and moves them ahead while it is down; the hands are goals on the grips. A dog can have a missing or a stick leg (`quadruped("dog", missing="HL")`, `peg="HL"`).

## 8. Verification

* `tests/test_skeleton.py`, `tests/test_physics.py`: forward kinematics of the planner equal MuJoCo's for the human, dog and bird.
* `tests/test_env.py`, `tests/test_reflex.py`: the tracking environment and the reflex run.
* Every shot in the showcase is rendered by `catalog.py`; `reflex_eval.py` evaluates the reflex on 240 held-out scenarios and compares it with a stiff body and a hand-made pose.

## 9. Limits

* Kinematic scenes use a stick-figure body; the retarget to a rigged mesh is a basic world-rotation delta method.
* The PPO tracker does not hold without an assist force; the evolved reflex is a small controller, not a general physical locomotion controller.
* Doors, beds, ladders and rides are authored goal sequences; swimming, diving, bird flight and the snake are hand-built kinematic models.
* Hit resolution is geometric (capsules and sampled stick points), not a contact simulation; pain and injury numbers are plausible, not clinical.
* Jump power, tolerances and pain gains are order-of-magnitude values from the literature, tuned for plausibility.
* Finger, jaw, eye, spine-per-vertebra and muscle-by-muscle models are not included; "muscles" appear as torque limits, joint ranges and stiffness.
