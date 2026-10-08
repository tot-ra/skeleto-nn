"""Build runs/showcase/index.html: every GIF with a one-line caption and an honest label of how the motion is made.
python make_showcase.py   (reads runs/gifs3, runs/game, runs/reflex_eval; copies them next to the page)"""
import os, shutil, glob, html, json
import argparse, subprocess
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--slim", action="store_true", help="re-encode GIFs smaller"); ap.add_argument("--no-game", action="store_true", help="leave out renders on a proprietary rig")
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); runs = os.path.join(here, "runs"); out = args.out or os.path.join(runs, "showcase")
os.makedirs(out, exist_ok=True)

def put(src, dst):
    if args.slim:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-vf", "fps=12,scale=400:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=32:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle", dst], check=True)
    else: shutil.copy(src, dst)

EMERGENT, GOAL, KIN, PHYS = "emergent", "goal script", "kinematic model", "physics, evolved"
SECTIONS = [
 ("People: walking, running, terrain, obstacles (one planner, gait from the Froude number)", "sim", [
  ("walk_run", EMERGENT, "stand, walk, fast walk, run, back to walk, stop"), ("stairs_up_down", EMERGENT, "stairs are just terrain (the steps span the whole width: the only way is over)"),
  ("slope_up_down", EMERGENT, "slopes: trunk pitch and foot rotation follow the ground"), ("step_over_log", EMERGENT, "step over a log"), ("duck_under_beam", EMERGENT, "auto-duck below a ceiling"),
  ("obstacle_course", EMERGENT, "uneven ground, a log, pits (jumped with the run-up the body needs), a low wall, a waist-high wall (vaulted), a wall that is too high (gone round), a pit that is too wide (stops, afraid)"),
  ("around_obstacles", EMERGENT, "A* around walls"), ("run_forest", EMERGENT, "run through a forest, duck under branches"), ("run_crowd", EMERGENT, "run through a crowd"), ("crowd", EMERGENT, "14 people cross a plaza"),
  ("crowd_squeeze", EMERGENT, "squeeze through people: shoulders turn, hands go up"), ("run_into_wall", EMERGENT, "run into a wall: seen early, late, not at all"),
  ("start_from_squat", EMERGENT, "from a squat to a run: stand first, drive out, or a sprinter's start"), ("joystick_control", EMERGENT, "stick input with inertia: brake, pivot, corner, accelerate"), ("push_recovery", EMERGENT, "shoved: recovery steps")]),
 ("People: muscle, energy, flexibility, age, pregnancy, clothing, injury", "sim", [
  ("bodies_walk", EMERGENT, "tall, dwarf, belly, armour, pack, skirt"), ("personas_course", EMERGENT, "child, adult, elder, starved, athlete on the same course: what each jumps, vaults, refuses"),
  ("personas_race", EMERGENT, "a 36 s all-out run: the reserve of energy sets the pace that can be held"), ("outfits_walk", EMERGENT, "tight skirt, wide skirt, trousers, boots, heels, chainmail, plate"),
  ("pregnant_stairs", EMERGENT, "stairs in a skirt: not pregnant vs heavily pregnant"), ("limp_people", EMERGENT, "sore leg, rigid stick for a lower leg, crutches"), ("bump_head", EMERGENT, "hit the head on a beam")]),
 ("Jumping: how far the body knows it can jump (self-model learned by practice, fear)", "sim", [
  ("jump_gap", EMERGENT, "gaps of 0.8, 1.8, 2.8 and 3.4 m: the run-up follows the gap"), ("jump_fear", EMERGENT, "knowing the limit: jumps 3.4 m, is afraid of 4.4 m, an over-confident one balks at the edge, an over-cautious one will not try"), ("jump_wall", EMERGENT, "standing jump over a wall")]),
 ("Climbing and hanging", "sim", [
  ("climb_rock", EMERGENT, "rock face: only the coloured holds can be used"), ("climb_tree", EMERGENT, "tree: branch stubs are the only holds"), ("ladder_climb", GOAL, "ladder"),
  ("pullup_bar", GOAL, "pull-ups on a bar, legs help"), ("pullup_pipe", GOAL, "pull-ups on a pipe"), ("pullup_rings", GOAL, "pull-ups on rings"), ("mantle_ledge", GOAL, "from a hang to standing on a ledge"), ("rope_climb", GOAL, "climbing a rope with hands and feet")]),
 ("Everyday situations", "sim", [
  ("door_open", GOAL, "open a door"), ("sit_stand", GOAL, "sit on a chair and stand"), ("sit_table", GOAL, "pull the chair out, sit, scoot in, eat, leave (table and chair are real obstacles)"),
  ("bed_lie_rise", GOAL, "lie in a bed, get up"), ("get_up_floor", GOAL, "get up from the floor"), ("rider", GOAL, "ride a horse (see the horse section)")]),
 ("Blows, pain, reactions", "sim", [
  ("hit_reactions_a", EMERGENT, "hit on head, torso, arm: stagger, doubled over, drop the weapon, clutch"), ("hit_reactions_b", PHYS, "leg: limp; hard head blow: physical fall, lie, get up"),
  ("hit_reactions_c", PHYS, "groin (man vs woman) and knee"), ("dodge_reactions", EMERGENT, "notice in time: side step, duck, hop, block; too late: hit"),
  ("weapon_reactions", EMERGENT, "axe vs shield, axe on a bare head, arrows dodged, caught by a shield, taken; shin block"), ("duel", EMERGENT, "two fighters choose targets and reactions on their own")]),
 ("Blows with different weapons and limbs; damage from animals", "sim", [
  ("weapons_human", EMERGENT, "sword, axe, spear and fist: swing or straight thrust, hit or dodged"), ("kick_human", EMERGENT, "a kick to the thigh: the victim limps"),
  ("wolf_bite", EMERGENT, "a wolf lunges and bites the forearm: the arm is hurt"), ("bear_swipe", EMERGENT, "a bear's paw swipe to the head: a knock-down"), ("horse_kick", EMERGENT, "a hind hoof kick: a hard blow to the torso")]),
 ("Cat", "sim", [("cat_jump_up", EMERGENT, "jump up and down: eight spine joints coil and stretch"), ("slalom_cat", EMERGENT, "slalom: tight turns, balance tail")]),
 ("Dog", "sim", [("slalom_dog", EMERGENT, "slalom at a run"), ("limp_dogs", EMERGENT, "sore leg, three legs, a stick for a lower leg"), ("animal_ages", EMERGENT, "old dog, puppy, kid goat")]),
 ("Horse (and rider)", "sim", [
  ("horse_gaits", EMERGENT, "walk, trot, canter, gallop"), ("slalom_horse", EMERGENT, "slalom: the big body swings wide"), ("horse_jumps", EMERGENT, "fences of 1.0, 1.6 and 2.7 m: jumps, jumps, refuses"),
  ("horse_load", EMERGENT, "with 0, 90, 180 kg on the back"), ("horse_gait_tail", EMERGENT, "5-joint neck nod, hair tail"), ("rider", GOAL, "rider on the saddle: walk to gallop"), ("rider_course", GOAL, "horse and rider: slalom and a fence")]),
 ("Other animals, side by side", "sim", [
  ("animals_trot", EMERGENT, "dog, cat, horse, wolf, pig trot"), ("quadruped_speeds", EMERGENT, "walk to gallop by Froude number"), ("quad_species_tails", EMERGENT, "balance tail, wag, low tail, curl"),
  ("quad_jump_species", EMERGENT, "same relative obstacle: cat, dog, wolf, pig"), ("animals_stairs_log", EMERGENT, "stairs and a log, four legs")]),
 ("Birds and snake", "sim", [
  ("birds_walk", EMERGENT, "bird gaits"), ("bird_flight", KIN, "flapping flight model"), ("bird_land_ground", KIN, "landing on the ground"), ("bird_land_branch", KIN, "landing on a branch"), ("bird_land_water", KIN, "landing on water"),
  ("bird_fly", PHYS, "physical flight: wings and tail are plates in the air; gliding about 3 s (the stroke is not good yet)"), ("bird_land", PHYS, "physical landing: not solved (hits hard)"), ("bird_takeoff", PHYS, "physical take-off: not learned"),
  ("snake_around", KIN, "serpentine motion between walls")]),
 ("Water", "sim", [
  ("swim_freestyle", KIN, "swimming with the head above water (hand-built)"), ("dive_swim", KIN, "diving (hand-built)"),
  ("swim_surface", PHYS, "physical swimmer at the surface with an oxygen budget: a poor stroke, but it does breathe"), ("swim_under_bottom", PHYS, "physical swimmer to a goal on the bottom"), ("swim_under_far", PHYS, "physical swimmer to a far goal at depth")]),
 ("Physical falls (MuJoCo, evolved on the bone-injury score)", "sim", [
  ("shove_forward", PHYS, "shove: stiff body vs evolved reflex"), ("drop_2m", PHYS, "drop of 2 m"), ("drop_2p5m_fwd", PHYS, "drop of 2.5 m with forward speed"),
  ("preg_shove_forward", PHYS, "pregnant: the general reflex vs the reflex evolved for this body (the belly is scored most)"), ("preg_drop_1m", PHYS, "pregnant: a drop"), ("slide_slope", PHYS, "standing on a slippery slope")]),
]
GALLERY = []
def items_html(rows, sub):
    s = ""
    for name, label, cap in rows:
        src = None
        for d_ in ("gifs3", "phys", "reflex_eval", "preg_eval", "gifs2"):
            cand = os.path.join(runs, d_, name + ".gif")
            if os.path.exists(cand): src = cand; break
        if src is None: continue
        os.makedirs(os.path.join(out, sub), exist_ok=True); put(src, os.path.join(out, sub, name + ".gif"))
        GALLERY.append((sub, name, label, cap))
        s += f'<figure><img loading="lazy" src="{sub}/{name}.gif"><figcaption><b>{html.escape(name)}</b> <span class="tag {label.split()[0]}">{label}</span><br>{html.escape(cap)}</figcaption></figure>\n'
    return s
body = ""
MD = []
for title, sub, rows in SECTIONS:
    n0 = len(GALLERY)
    body += f"<h2>{html.escape(title)}</h2><div class=grid>\n{items_html(rows, sub)}</div>\n"
    items = GALLERY[n0:]
    MD.append(f"### {title}\n")
    cells = [f'<td width="50%" valign="top"><img src="showcase/{sb}/{nm}.gif" width="100%"><br><b>{nm}</b> <i>({lb})</i><br>{html.escape(cp)}</td>' for (sb, nm, lb, cp) in items]
    rows_md = ["<tr>" + "".join(cells[i:i + 2]) + ("<td></td>" if len(cells[i:i + 2]) == 1 else "") + "</tr>" for i in range(0, len(cells), 2)]
    MD.append("<table>\n" + "\n".join(rows_md) + "\n</table>\n")
# character model and physics
game = [] if args.no_game else sorted(glob.glob(os.path.join(runs, "game", "*.gif")))
if game:
    os.makedirs(os.path.join(out, "game"), exist_ok=True); g = ""
    for f in game:
        n = os.path.basename(f); put(f, os.path.join(out, "game", n)); g += f'<figure><img loading="lazy" src="game/{n}"><figcaption><b>{n[:-4]}</b> <span class="tag retarget">on the shipped rig</span></figcaption></figure>\n'
    body += f"<h2>The same motion on a rigged character mesh (Blender)</h2><div class=grid>\n{g}</div>\n"
phys = sorted(glob.glob(os.path.join(runs, "reflex_eval", "*.gif")))
summary = ""
sj = os.path.join(runs, "reflex_eval", "summary.json")
if os.path.exists(sj):
    d = json.load(open(sj)); rows = ""
    for k, v in d.items():
        a = v["all"]; rows += f"<tr><td>{html.escape(k)}</td><td>{a['cost']:.2f}</td><td>{a.get('head_risk', float('nan')):.2f}</td><td>{a.get('torso_risk', float('nan')):.2f}</td><td>{a.get('arms_risk', float('nan')):.2f}</td><td>{a.get('legs_risk', float('nan')):.2f}</td></tr>"
    summary = f"<table><tr><th>controller (240 held-out shoves and drops)</th><th>injury cost</th><th>head risk</th><th>torso</th><th>arms</th><th>legs</th></tr>{rows}</table>"
slide_tab = ""
sj2 = os.path.join(runs, "slide_eval", "summary.json")
if os.path.exists(sj2):
    d2 = json.load(open(sj2)); rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['stays_up_5s'] * 100:.0f}%</td><td>{v['mean_time_fraction_up'] * 5:.1f} s</td><td>{v['mean_slide_m']:.2f} m</td></tr>" for k, v in d2.items())
    slide_tab = f"<h3>Standing on a slippery slope (40 held-out slopes 8-25 degrees, friction 0.12-0.4)</h3><table><tr><th>controller</th><th>stays up 5 s</th><th>mean time up</th><th>mean slide</th></tr>{rows}</table><p>A partial result: the evolved controller helps but most slopes still end in a fall.</p>"
preg_tab = ""
sj3 = os.path.join(runs, "preg_eval", "summary.json")
if os.path.exists(sj3):
    d3 = json.load(open(sj3)); rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v['cost']:.2f}</td><td>{v['abdomen_risk']:.2f}</td><td>{v['head_risk']:.2f}</td><td>{v['torso_risk']:.2f}</td><td>{v['arms_risk']:.2f}</td><td>{v['legs_risk']:.2f}</td></tr>" for k, v in d3.items())
    preg_tab = f"<h3>The same falls for a heavily pregnant body (120 held-out shoves and drops, the belly weighs most in the score)</h3><table><tr><th>controller</th><th>injury cost</th><th>belly</th><th>head</th><th>torso</th><th>arms</th><th>legs</th></tr>{rows}</table>"
body += f"<h2>Numbers for the physical controllers</h2>{summary}{preg_tab}{slide_tab}"
page = f"""<!doctype html><meta charset=utf-8><title>Procedural motion showcase</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font:15px system-ui,sans-serif;margin:0 auto;max-width:1500px;padding:16px;background:#f6f6f2;color:#222}}h1{{margin:.2em 0}}h2{{margin-top:1.6em;border-bottom:1px solid #bbb}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}}figure{{margin:0;background:#fff;border:1px solid #ddd;padding:6px;border-radius:6px}}img{{width:100%;display:block}}
figcaption{{font-size:13px;padding-top:4px}}.tag{{font-size:11px;padding:1px 6px;border-radius:9px;background:#ddd;margin-left:6px}}.emergent{{background:#cfe8cf}}.goal{{background:#f2e3b8}}.kinematic{{background:#e3d0f0}}.physics{{background:#bcd7f2}}.retarget{{background:#eee}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}</style>
<h1>Procedural motion showcase</h1>
<p>Green = comes out of general rules (gait from body size, footholds from terrain, reactions chosen by predicting geometry). Yellow = goals for pelvis, hands and feet solved by IK. Purple = hand-built kinematic model. Blue = trained or evolved on a physical body.</p>
{body}"""
open(os.path.join(out, "index.html"), "w").write(page)
open(os.path.join(out, "gallery.md"), "w").write("\n".join(MD))
print("wrote", os.path.join(out, "index.html"))
