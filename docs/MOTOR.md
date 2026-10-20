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
| strain | muscle torque against its capacity, squared, weighted by the mass each joint carries; for people also local muscle endurance (Frey-Law and Avin 2010: a share f of a muscle's strength can be held for about 20 f^-1.7 s; the sustained load, low-passed over a second, counts, so a held-out arm tires and a stride's legs rest between pushes) |
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

That margin turned out too thin to learn: gliders trained on it never found a wingbeat. A new search at 1x wing
strength found 0.58 N of net thrust (13 % of the weight), with the shoulder and outer wing at their torque limit most of
the beat; at 2x it found 2.49 N (56 %), about a real bird's margin for a climb. The wing joints are now twice as strong
(`bodies.WING_STRENGTH`), standing in for what flat plates lack: a feathered wing's twist and change of shape.

Two more physics faults turned up in bird training: a heron's wing twist (a long bone turning about itself) was so
stiff for its tiny inertia that the explicit muscle spring blew the simulation up within a few steps (wing joints now
get the armature that keeps kp dt^2 / inertia below 1); and the sparrow, beating at about 13 Hz, gets only two network
commands per beat at 30 control steps a second, so it is left out of training until birds get a faster control rate.

**Flight energy, run 7.** The trained crow beat its wings at the fastest rate allowed (8.7 Hz, twice a crow's),
because the wing work was counted as fully elastic and averaged over a beat at the control rate: a wasteful beat cost
nothing. An energy audit of a 10 s flight: the muscles did 38 W of work, the air took 4.8 W, the bird lost 1.1 W of
height and speed; the rest went into swinging the wings to and fro. Now the wing work is summed over every physics
sub-step, tendons return 60 % of the braking work (`ELASTIC_RETURN`), and the flight power unit is what the most
efficient searched wingbeat costs in this physics (`FLIGHT_UNIT`; the flat-plate wing has no downwash, so absolute
watts are about 15 times a real crow's). Sustainable is 1.4 units (1.2 at first: level flight costs about 1.5, and a
rested crow sank and unlearned flying), the flight reserve drains in about 20 s at twice that, and every flight starts
with a reserve of 70 to 100 % (40 % at first: too weak to hold height at all): the bird has to balance speed, height and energy. The
goal is a height to hold (at the asked climb rate), not only a climb rate, so drifting up is as wrong as sinking.
Every fine-tuning on these rules unlearned level flight within a round, whatever else was changed (learning rate,
value warm-up, a longer horizon, the action-bound pull): the reward depended on the height to hold, but the bird only
saw its absolute height and the asked-for climb rate, so it could not predict its own reward. It now observes the
climb rate wanted at this moment (the correction toward the height included); ten minutes of training from the last
level flier then kept it flying level and brought it back to its height instead of overshooting it. Birds also look
about 11 s ahead (`GAMMA` 0.997; a slow sink crashes long after the energy it saves) and get no action-bound pull.
Even so, longer bird training kept losing level flight within a few rounds (each fix held for a while), so bird
training is paused: the published crow is the round 44 flier that holds its height. Birds need structural work first:
a faster control rate (3 to 7 commands per beat now) and a wing model with downwash (flapping costs about 15 times a
real crow's power), so that level flight is not a knife edge that every training change pushes off.
**Control rate per family.** Birds now command their joints 60 times a second (people and quadrupeds 30; the
physics runs at 180 Hz either way): at 30, a 4.5 Hz wingbeat got 7 commands and a 9 Hz one 3. Episode lengths, test
durations and the discount are in seconds, not steps.
**Downwash.** Induced drag was counted per wing strip from its instantaneous lift coefficient, so every stroke paid as
if the whole wing were at maximum lift. The wing now sits in one downwash for the whole bird (momentum theory,
following the lift over about a beat), and the induced drag comes from the lift tilting back in it. The energy unit was
also recalibrated: it had been a wingbeat searched for thrust alone (about 200 W); a search for the cheapest level
flight (thrust and lift just enough) finds one at 81 W metabolic for a crow, which is the unit now (`FLIGHT_UNIT` 6).
A real crow needs about 10 W: the open-loop sine beats of the search and the flat plates leave a factor of eight.
On the new air model, with one simulation worker per bird network (the machine's budget was halved), the birds fell
over five rounds (0.67 to 0.47); bird training is paused with the last pre-downwash flier restored, until there is
compute to retrain them on it.
Wing twist is barely used yet (8 degrees); at 30 control steps a second and 4 to 9 Hz the network has 3 to 7
commands per beat. A faster bird control rate and a wing model with downwash are the next steps.

**Quadruped anatomy.** The horse has a deep chest, a long neck rising steeply from the withers and a slanted head;
the cat has longer hind legs and a heavier, stronger tail to balance with. A passive nuchal ligament (a spring of 1.5
joint strengths per radian in every quadruped's neck, free of energy) carries the head: the horse's raised neck held up
by muscle alone kept its base at 71 % of its strength, and it could not walk; with the ligament, 46 %.

**The horse and the shared network.** The quadruped network walks the dog, cat and pig but kept the horse almost
standing (0.2 m/s at any asked speed); drawing the horse three times as often only made the others worse. Trained on
horses alone (`QUAD_ONLY=horse`), the same network walked the horse within 20 minutes (0.8 to 1.05 m/s for the full
test). The body can walk; the shared network does not make room for it. Distillation (`motor/distill.py`, DAgger:
the shared network is rolled out over all species, 40 % of episodes on the horse, and every state is labelled by the
horse specialist or by a frozen copy of the shared network itself) put it in: 240,000 labels later the shared network
walks the horse for the full 10 s at 1.14 m/s (before: a fall within 5 s at 0.26 m/s), trots it at 1.38 m/s for 4.5 s,
and walks and trots the dog, cat and pig as before (only the dog on ice got worse: a fall at 5.7 s).
Two rounds of ordinary training then washed it half out again (the horse back to 0.64 m/s), so quadruped training
now kickstarts (`--kick horse=<specialist>`): every horse state's specialist action pulls the policy toward it in
the loss, all the time.

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
  the speed; they level out as the curriculum's speed knob grows (level after one unit of growth), so flapping has to pay.
  The bird curriculum moves on while fewer than half the flights fail (people and quadrupeds: 30 %) and the mean goal
  score is above 0.35 (others: 0.5; the bird's climb rate is scored tightly).
  Variety (species, wind, turns) is held at how level the goals are (first 0.2 ahead of it; halfway to level flight
  that let gulls, herons and wind fail half the flights and hold the curriculum there): harder species made most
  flights fail, which held the goals at glides, so flapping was never needed.
  Bird exploration noise is kept at 0.2 or more (log std -1.6): with the usual 0.08 the gliders never found a wingbeat
  and the goals stopped about a tenth of the way from a glide to level flight.
  More noise made the wings move but not beat: noise drawn independently every step does not add up to a rhythm.
  So 60 % of a bird's exploration noise is locked to its wingbeat clock (A sin(phase) + B cos(phase), A and B drawn
  per episode): every step's noise is still N(0, std^2), but over an episode it is a coherent wingbeat to try.
  The climb rate is scored on its own, with a tolerance of 0.15 units: inside the speed tolerance a glide sinking
  0.05 units faster than asked lost under 1 % of the goal score, so holding height never paid until the ground was hit. The harness follows the
  asked-for path without carrying any weight (carried weight is what drives a glide; carrying 70 % of it stalled
  every glide), holds the body level with a stiffness of m g x wingspan per radian (one scaled by the trunk's inertia
  let it roll over), and when most flights fail it comes back fully (people and quadrupeds: up to 30 %).
* **Quadruped species drawn by weight:** the species that fall most (horse, cat, goat) come up more often. A goal
  speed per species (down after every fall, up after a met episode) was tried for eight rounds and dropped: it held
  every species at the speed where it fell about a third of the time, and the dog stopped training at test speeds.
* **Arms re-learned from rest:** soft arm ranges, the action-bound pull and full-weight costs moved the arms only
  slowly: the shoulder command sat at the end of its range, its muscle pressing at 78 % of its strength into the soft
  stop (the arms were a balance pole: hung down by force, the walker fell in 3 of 4 runs at 3 m/s). With local muscle
  endurance in the strain cost and the arm rows of the output layer set to zero (the arms start hanging, the rest of
  the network kept), one round later the arms hang beside the body with the elbows bent 20 degrees and swing 16
  degrees with the stride; strain fell from 1.4 to 0.4 per second and the walk, run, ice, pack and turn all hold.
* **Leg tendons:** the walker stepped 206 times a minute at 1.3 m/s (a person about 110) and ran with 8 % flight at
  2.9 m/s (a runner 30 to 40 %), at about twice a person's cost of transport: all the legs' braking work was lost, so a
  shuffle was cheaper than a bounce. The ankle's tendons now return half of its braking work and the knee's a third
  (Achilles and patellar tendon), summed over every physics sub-step (`TENDONS`, `Evaluator.tendons`).
* **Direction-dependent joint strength:** every joint was equally strong both ways, the hip at 330 N m and the ankle
  at 110: the legs swung forward twice as easily as a person's and the calf pushed off at half a person's strength, so
  short, quick steps paid and a run with flight did not (asked for 4.5 m/s, 3.4 at 274 steps a minute with 10 % flight).
  Now the hip flexes at about 150 N m and extends at 250, the knee extends at 250 and flexes at 130, the ankle pushes
  off at about 200 and lifts the toes at 50 (`Bone.tau_dir`); strain counts against the strength in the direction the
  torque acts. People's goal speeds also go up to 5 m/s now (a 4 m/s run in the review).
* **Mean actions stay inside [-1, 1]:** the environment clips actions there; a policy mean beyond it gets no
  gradient from the exploration noise. A walker's arm actions sat at 1.3 to 1.9 and the arms stayed on their stops for
  rounds after the stops were made costly; a small loss term now pulls the mean back inside.
* **People stay at 30 Hz control:** a network trained from scratch on the realistic body walked like a person (137
  steps a minute, 0.55 to 0.6 m steps at 1.3 m/s) but fell within a second of each try at a run with flight. The same
  weights fine-tuned at 60 Hz for about two hours (discount 0.995, the same horizon in seconds) went back to a shuffle:
  180 steps a minute and 0.37 m steps at 1.3 m/s, up to 766 a minute when asked for 3.5. A faster command rate made
  quick small steps easier, not a run.
* **People's moving starts are capped at a walk** (`MOVING_START_MAX`, 1.5 m/s): a moving start puts both feet down
  in a stride, joints still, the whole body at up to the goal speed. At 3 to 4 m/s about half of those starts fell
  within a second, while from rest the same networks never fell but never sped up past a walk (2 to 4 % flight): the
  falls taught that going fast means falling. Run speeds are now reached by speeding up.
* **Achilles spring (experiment, `BIPED_ACHILLES=1`):** a one-sided spring on the ankle (`SPRINGS`,
  `physics.PhysChar`), taut once the shank tips over the planted foot, about 220 N m/rad for an adult. Two hours of
  fine-tuning with it changed nothing: both networks still topped out at a quick walk of about 1.7 to 1.8 m/s.
* **Run harness** (`RUN_FROM`, `ppo --run-assist0/--run-fade`): with starts capped, the networks sped up to about
  1.8 m/s and no further, since every try at a run had fallen within a second. Goals past 2.2 m/s get a harness that
  only keeps the trunk upright (no height, no push), so a run can be practised before it can be balanced; it fades
  with time. In about 1,500 iterations the network came to lean on it (it fell at 4 m/s within 3 s once the harness was
  off) and still did not run.
* **Fatigue is not what stops a run:** the walker spends 6.6 to 7.7 W/kg against a sustainable 10; a run at two to two
  and a half times that would leave about 0.65 of the reserve after a 15 s episode.
* **Stiff knees are:** with joints driven toward target angles at a fixed stiffness, the trained walker held its knee
  target on the straight stop all the time (peak knee speed 0.3 rad/s; a person's swing knee bends about 60 degrees at
  6 to 7 rad/s). A relaxed knee could not swing back as a pendulum, every bend had to be held against the stiffness,
  and a leg locked straight cost nothing: the waddle with the feet barely off the ground, and legs that cannot run.
* **Muscle torque (experiment, `BIPED_TORQUE=1`):** the actions are activations; the torque is the activation times
  the strength in that direction, weakened while the muscle shortens and up to 1.4 times stronger while it is
  stretched (Hill's relation at the joint, `physics.hill`: zero force at 20 rad/s, about 65 % at 4 rad/s and 40 % at
  8). A relaxed joint is free. Trained from scratch (`runs/motor/biped_torque`). The first run bent its knees
  freely (peak 6 rad/s) but, with the harness gone within about 700 iterations, settled into a deep crouch (knees at
  80 to 127 degrees, energy cost four times the straight-legged walker's) and never stood up out of it. Restarted with
  the harness fading about twenty times slower (`--fade 0.0001 --fade-ok 0.0002`). Under that slow harness the
  network folded its legs (knees at 150 degrees, feet off the ground) and hung from the harness's spring to the
  standing height: free for muscle torques, and without falls the curriculum raised the goal speeds anyway. For muscle
  torques the harness now carries a fixed share of the weight (up to 60 %) with no height spring. That run walked
  with the harness at half and human knees (0 to 28 degrees at up to 8 rad/s), but the hip sat at 18 to 26 degrees,
  on its extension stop (25), and hardly swung; without the harness it fell within a second.
* **Line muscles on the legs (`BIPED_MUSCLES=1`, `muscles.py`):** one torque per joint axis has no anatomy: no
  hamstrings tying hip extension to knee flexion, no rectus femoris or gastrocnemius across two joints, no stretched
  hip flexor pulling an extended thigh forward. Each leg now has 10 muscles (iliopsoas, gluteus maximus and medius,
  adductors, hamstrings, rectus femoris, vasti, gastrocnemius, soleus, tibialis anterior) as MuJoCo spatial tendons
  from origin to insertion, over wrapping cylinders at the femoral head, the kneecap and the femoral condyles. The
  force is Hill's: activation (10 ms up, 40 ms down) x force-length x force-velocity, plus the passive force of a
  stretched muscle; moment arms come from the geometry and change with the angle. Joint motors stay on the axes the
  muscles do not span (hip rotation, ankle inversion, toes, trunk, arms). Energy is counted per muscle: the work of
  the contractile part, holding cost from activation (a muscle shortening fast spends its activation for less force),
  and the Achilles and patellar tendons store a share of the braking work for the next push.
  `python muscle_check.py` prints every moment arm over the joint ranges (a bad attachment shows as a wrong sign;
  `tests/test_muscles.py` checks them); `python muscle_atlas.py` draws the attachments and a twitch of each muscle.
  Standing, the relaxed muscles give 5 to 9 N m (iliopsoas pulling the hip forward, calf the heel up): a hanging
  relaxed leg rests with the knee a little bent and the toes down. Trained from scratch (`runs/motor/biped_muscle`).
* **Muscles everywhere (`muscles.py`, `BIPED_MUSCLES=1`, `QUAD_MUSCLES=1`):** people also have trunk and arm
  muscles (rectus abdominis, lumbar and thoracic erector spinae, external oblique; front, middle and back deltoid,
  pectoralis major, latissimus dorsi, biceps, brachialis, triceps: 22 per side, 73 actions with the joint motors left
  for hip rotation, shoulder girdle, spinal twist, wrist, neck and head). Quadrupeds get 11 per leg (gluteals,
  hamstrings, iliopsoas, rectus femoris, vasti, gastrocnemius, superficial digital flexor, tibialis cranialis, long
  digital extensor, abductors, adductors; fore: brachiocephalicus, supraspinatus, latissimus, teres, triceps,
  biceps, brachialis, carpal extensor and flexors, digital flexor and extensor, abductors, pectorals) plus epaxial,
  rectus abdominis and obliques. Birds: pectoralis and supracoracoideus on the keel (the supracoracoideus runs up
  through the triosseal canal, a pulley over the shoulder, and lifts the wing), protractors, retractors, elbow and
  wrist flexors and extensors, and the leg (birds do not train on them yet). Every joint a path crosses has a
  wrapping surface; without them the checker found paths slipping to the wrong side of a bent joint (a dog's carpal
  flexors extended the bent wrist, the shoulder's sphere wrap let the deltoid jump sides). At the very ends of a
  range a few muscles turn, as in a body (a deltoid with the arm overhead, the pectoralis above 90 degrees of
  abduction): the test checks the inner range.
* **Quadruped anatomy, found while attaching the muscles:** the joint ranges let a dog's stifle straighten fully
  (180 degrees between the bones; a dog's stops at about 162) and a horse's hock too, and the elbow bent only 30
  degrees the anatomical way but 140 the other; the trained joint-motor dog walked with straight knees. Ranges now
  come from the angle between the bones (Jaegger et al. 2002 for Labradors: elbow 36 to 165, carpus 32 to 196,
  stifle 42 to 162, hock 39 to 164; cat and horse their own). The joint strengths summed to about 2.5 times what the
  muscle mass of a real leg gives (a dog's hind leg about 40 body weights of muscle force, from 3 kg of muscle with
  6 cm fibres at 25 N/cm2; a cat 95, a horse 20); the muscle model scales each leg to those (``QuadSpec.muscle_bw_*``)
  and hind legs are stronger than fore legs (``hind_power``, ``fore_power``; a horse's lower legs are mostly tendon,
  ``distal_power``).
* **Motion prior from recordings (`--amp data/mocap/cmu_amp.npz`, `motor/amp.py`):** reward alone kept finding
  gaits no person uses. A discriminator (AMP, Peng et al. 2021) learns to tell the policy's transitions from 99
  recorded walks, runs, turns and stops of six people (CMU Motion Capture Database, `data/mocap/get_cmu.sh`); how
  human a transition looks is a reward of up to `--amp-w` (2) per second next to the goal and the costs. Only training
  uses the recordings: the network still drives the muscles in the physics, nothing is replayed. The features are
  segment directions (thighs, shanks, feet, upper arms, forearms, trunk) in the heading frame plus the hip velocity in
  sqrt(g leg) and the turn rate: with positions over leg length, the model standing still differed from the recordings
  by up to three standard deviations through its proportions alone (a longer trunk), which a discriminator would have
  used instead of the motion. Started from the muscle run at iteration 801.
* **Speed cap per family** (`MAX_SPEED`): 3 m/s for people, 2 Froude (a gallop) for quadrupeds, 4 units for birds.
  `--reset-curriculum` keeps a checkpoint's weights and starts the harness and the goals over.

**Population-based training** (`motor/pbt.py`): several members train side by side; every round (15 minutes) each is
scored on fixed situations *without the harness*, and the worst is replaced by a copy of the best with mutated
settings: learning rate, harness fade, mirror-loss weight, and the reward itself (what staying up is worth, what a
fall costs, how much the comfort costs weigh). Bad runs die off; good ones spread. `history.json` records every round. With only two members the review score is too
noisy to choose between them: the bird population's settings random-walked to the bounds (alive and cost scale at
their minimum) and the birds unlearned holding height, while the same weights from twenty rounds earlier still held
it in the same environment. Two-member runs now train apart (`--replace 0`) with default settings.

    python -m motor.pbt --family biped --name biped_pbt --members 3 --workers 3 --round-min 15 --rounds 16
    python -m motor.ppo --family quadruped --name dog --workers 15 --hours 3       # a single run
    python -m motor.show runs/motor/biped_pbt/m0/ckpt.pt [out_dir] [--assist 0.4]  # GIFs and summary.json
    python -m motor.showcase [--force]       # best network of each family -> showcase/motor/latest + README status
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
| 6 | birds glide first and level out; harness follows the path without carrying weight; exploration noise floor and noise locked to the wingbeat clock; wings twice as strong; heron blow-ups fixed; sparrow left out; climb rate scored on its own; quadruped torque scales with lever length, hard species drawn more often; pack test for people | **birds: the crow flies by flapping** (round 20, about 8 h of bird training): asked to hold height at 2.8 units it loses about 1.5 m, then climbs at about 0.3 m/s, for 15 s from every start; the climb and turn scenes last 10 s; side wind, gull and heron fall. **People:** best score 1.10 of 1.3; the 20 kg pack now walks for the full test; turning from a walk still falls. **Quadrupeds:** flat at 0.70 to 0.80; the horse now stands in its pose but the network has not learned it |

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
