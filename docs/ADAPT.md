# Condition-adaptive motion (`adapt/`)

One small neural network adapts the motion of any body (person, dog, cat, horse, crow, gull) to the
situation it is in: ice, mud, deep snow, rain and wet clothes, frost, wind, smoke, darkness, a bleeding
wound, a sprain, a broken arm, bruised ribs, drink, exhaustion, a load. No situation has its own motion
code or clip. A new kind of ground, weather or injury is a line of data, and usually needs no retraining.

![ground](../showcase/adapt/adapt_ground.gif)

## The idea in four parts

1. **Situations are data that change physical channels** (`adapt/conditions.py`). "Ice" lowers the
   friction to 0.09 and the air to -5 C; "deep snow" makes the feet sink 0.45 leg lengths; "bleeding"
   opens a wound at a place on the body with a rate and some pain; "gale" adds wind from a direction.
   There are 24 channels (`adapt/schema.py`): friction, sink, wetness, insulation, air temperature,
   core cooling, wind ahead and sideways, bleeding rate, blood lost, wound position (x, y, z on the
   body), pain in each leg and arm, the trunk and the head, fatigue, dizziness, load, darkness, smoke.
2. **The network sees channels, never names.** Input: the channels plus the body (biped, quadruped
   or bird; mass; leg length; flying or not; commanded speed as a Froude number). Output: 24 generic
   motor-style knobs that every body plan understands: speed taken, stride, ground-contact share,
   stance width, crouch, trunk pitch and roll, arms held out, each arm hugged to the body, each hand
   pressed on the wound point, head down, shoulders raised, shiver, flat-footed placement, swing
   clearance, limp side, caution, tail tuck, fur or feather fluff, wingbeat rate, amplitude and glide
   share. A 31-64-64-24 MLP, about 7,800 weights; one evaluation is about 7,800 multiply-adds, so a
   game can update hundreds of creatures a few times a second.
3. **It is trained on consequences, not on example motions** (`adapt/model.py`). A differentiable
   reduced-order model scores a style in a situation by what would physically follow: the required
   friction (half the step over the hip height) against the available friction and the injury of a
   fall from the height of the centre of mass; tripping where the feet sink or the ground is unseen;
   sideways balance (dizziness, blood loss, gusts, shivering) against the base of support; the wind
   moment on the trunk; heat lost through the exposed surface (temperature difference, wetness, wind,
   insulation) against heat made (basal, motion, shivering); blood lost per second (flow, a hand's
   pressure where it can reach, heart rate); pain from loading a sore leg, swinging a sore arm, jolting
   the ribs or the head; smoke breathed at head height; lift against weight in flight; metabolic
   power; not reaching the commanded speed. Hugging, hunching, pressing a wound, limping and holding
   the arms out on ice are never written anywhere: they are what lowers that sum.
4. **The style drives the existing planner** (`adapt/driver.py`). Gait knobs go to `Walker.style`
   (stride, duty, stance width, crouch, trunk lean, foot roll, clearance, head, shoulders, tail,
   shiver), arm knobs become arm goals solved by the planner's IK (a hand on the wound, an arm across
   the chest, arms out to the side), speed and limp go into the command and the leg duty. What is
   applied is `neutral + net(situation) - net(no situation)` for that body and speed, so in a normal
   situation the planner's motion is exactly unchanged.

## Interactive environment: the body keeps state

`adapt/physiology.py` advances the channels that are the body's own state: wetness (rain unless
sheltered, water, drying with warmth, wind and a fire), core cooling (from the same heat balance and
the style actually used, so hugging and shivering really slow it), blood lost (less with a hand on
the wound, more with effort; wounds clot faster under pressure), fatigue and pain. The loop is
environment -> channels -> network -> style -> heat, blood, effort -> state -> channels.

![rain and cold over time](../showcase/adapt/adapt_rain_cold.gif)

## What it learned (person walking at Froude 0.25, relative to normal)

`python -m adapt.probe` prints this table for any body; a selection:

| situation | what the network does |
|---|---|
| wet cobbles | flatter foot placement, a little more care |
| ice, wet ice | stride 0.83, feet flat, careful, crouched 0.3, arms out, more time on both feet, wider stance |
| deep snow | high swing clearance (0.75), slower, flat feet |
| mud | more clearance, shorter steps |
| frost, light clothes | shivers, arms a little in |
| soaked at about 5 C | both arms hugged, hunched, head down, shoulders up, full shiver |
| bleeding left thigh | left hand presses the wound, trunk bends to reach it, limps off the left leg, slower |
| bleeding right chest | left hand presses (the right arm cannot reach its own chest well), right arm guards |
| sprained left ankle | limp off the left, flat foot, shorter steps |
| broken right arm | right arm held to the body, left arm supports it |
| bruised ribs | both arms guard the trunk, slight hunch, slower |
| drunk | wider stance |
| gale from the left | leans into the wind, crouches, wider stance |
| headwind | leans forward |
| darkness | higher swing clearance |
| smoke | crouches and slows down (head out of the smoke, less breathing) |

A dog on ice places its feet flat and carefully with more time on the ground; a soaked cat or dog in
the cold hunches, shivers, tucks its tail and fluffs its fur; a horse lame on the left limps; a crow
on ice spreads its wings for balance; a crow with wet plumage or carrying prey beats its wings faster
and wider; a crow with a hurt wing flaps less; with a tailwind it glides more.

![injuries](../showcase/adapt/adapt_injury.gif)
![animals](../showcase/adapt/adapt_animals.gif)
![birds on the ground](../showcase/adapt/adapt_birds_ground.gif)
![birds in flight](../showcase/adapt/adapt_birds_flight.gif)
![weather](../showcase/adapt/adapt_weather.gif)
![bleeding over time](../showcase/adapt/adapt_bleed_time.gif)
![conditions added after training](../showcase/adapt/adapt_unseen.gif)

## Adding a situation

* **A new condition over existing channels** (a greasy kitchen floor, slush, sticky pitch, fish slime,
  a hailstorm as cold + wind + wet): add an entry to `CONDITIONS` in `conditions.py`. No retraining.
  `adapt_unseen` shows conditions defined after the network was trained.
* **A new channel** (a new physical effect, say "heat radiating from one side" or "deep water to the
  waist"): append a `Channel` to `schema.py`, add a cost term with `@term` to `model.py` that says
  what the channel does physically, add the conditions that set it, then warm-start:
  `python -m adapt.train --init results/style_net.json --steps 800` (about 10 s). Old weight files load
  into the wider schema with a zero column for the new channel, so nothing changes until it is retrained.
* **A new style knob** (say "drag a foot", "hold the hat"): append a `Knob` to `schema.py`, use it in a
  cost term and in the driver (or the engine), then retrain from scratch (about 25 s). The training is
  `python -m adapt.train` (2,500 Adam steps of 4,096 random bodies and situations, CPU).
* `python -m adapt.train --check` compares the network with a per-case optimisation of the same cost
  on 2,000 held-out cases: it reaches 99.9 % of the possible cost reduction (cost from 10.3 with the
  neutral style to 1.69 with the network, against 1.63 for the per-case optimum).

## Jumping by context (`adapt/leap.py`)

A second network decides jumps from the same channels. What lies ahead is a profile: every quarter leg
length for five leg lengths, the height of whatever is solid there relative to the feet and the depth of
water on it. One MLP (41 inputs: the body, the 24 channels and ten numbers about one landing place, such
as its height and water, the highest and lowest thing passed over, where the first edge is and how much
room there is after landing; two hidden layers of 96) is evaluated for every sample as a landing place
and gives a score against staying put and a jump program: a step before take-off, the arc height over
the highest point passed, hands on the obstacle, tucked legs, how deep the landing gives, a roll,
lowering oneself from the edge first, arms out, trunk lean. The best place wins, or staying.

It is trained the same way, through a differentiable model of consequences (`cost`): the take-off speed
the legs can give (weakened by pain, blood loss, fatigue, a load, sinking feet) plus what the arms can
pull on a top within reach; the take-off slipping where friction is low; the feet clearing what lies
between (the hands and tucked legs help); the landing force (drop height over the stroke of the knees, a
roll or the water) against a bone tolerance that is lower with a sore leg and higher for small and
four-legged bodies; slipping or running off a narrow top; cold water, deep water with a load, a useless
arm or blood loss; effort, time and the ground gained. A first version with one flat network over the
whole profile reached only 30 to 60 % of the possible gain and mostly refused; scoring each landing
place with one shared network reaches 91 % (`python -m adapt.leap --check`: cost 0.198 against 0.187 for
a per-case optimisation and 0.30 for staying; same jump-or-stay decision in 95 % of 600 held-out cases).
Training: `python -m adapt.leap` (12,000 steps of 3,072 random bodies, situations and scenes, about 12
minutes on a CPU).

`python -m adapt.leap --probe` (a 58 kg, 0.86 m leg teenager who wants across):

| scene | normal | ice | sprained ankle | heavy load |
|---|---|---|---|---|
| churchyard wall 0.85 m | vault (hands 0.78), lands 2.3 m on | refuse | refuse | refuse |
| wall 1.3 m | vault (hands 0.98) | refuse | refuse | refuse |
| wall 2.2 m | refuse | refuse | refuse | refuse |
| ledge up 1.4 m | pull up onto (hands 1.0) | refuse | pull up onto | refuse |
| drop 1 m | jump down, full knee give | refuse | jump down | jump down |
| drop 2.5 m, drop 6 m onto stone | refuse | refuse | refuse | refuse |
| quay 2 m over deep water | jump into water | refuse | jump into water | refuse |
| cliff 8 m over shallow water | refuse | refuse | refuse | refuse |

Quadrupeds are in the same network (a dog with a jump gain of 2.5 at a 0.6 m fence is an export golden
case). Limits: the hang-drop exists in the program but never wins over a refusal at the drops tested;
there is no fall: a chosen jump is assumed to succeed; birds fly instead.

## Running it in an engine

`python -m adapt.export_engine OUT_DIR` writes `style_net.json` (weights with the input and output schema),
`motion_conditions.json` (the condition registry, body regions, region -> pain channel) and
`style_net_golden.json` (bodies, conditions, channels, expected raw and calibrated styles, and a ten-minute
physiology run), so a port can prove it computes the same numbers. It also writes `leap_net.json` and `leap_net_golden.json`
(scenes in metres with the expected decision, knobs and arc). The Reval Rebel game runs the jump network for
its hero (vault a churchyard wall, jump off a deck into the sea, refuse a bank). The Reval Rebel game runs a GDScript
port (forward pass 0.14 ms; 40 creatures 0.34 ms per frame with caching) on its humanoid rigs, skinned
animals and flying birds; a condition list may also carry measured values (`{"air_temp": -6.0}`).

## Honest limits

* The teacher is a reduced-order model with order-of-magnitude constants, not a muscle simulation.
  It decides *which way* and *how much* a style changes; the planner and the engine decide what that
  looks like. Constants are tuned so that the neutral situation reproduces the planner's normal gait.
* Physiology times (cooling in minutes, a serious wound lethal in about a quarter of an hour) are
  tuned for readability in a game, not for medicine. The demos run body time 20 to 40 times faster.
* Feather and fur fluff is a knob but the stick-figure renderer cannot show it; the game scales the
  plumage or fur mesh with it.
* The style modulates a kinematic planner; it does not make a physical controller (the PPO tracker
  in `train.py` still does not converge). The physical fall reflex (`reflex.py`) is unchanged.
* Licences: training uses PyTorch (BSD), numpy, MuJoCo (Apache-2.0); no motion-capture data and no
  pretrained weights from elsewhere, so the weights are this repository's own (MIT).
