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
 ("Walking, running and terrain (one planner, gait from the Froude number)", "sim", [
  ("walk_run", EMERGENT, "stand, walk, fast walk, run, back to walk, stop"), ("stairs_up_down", EMERGENT, "stairs are just terrain: footholds searched, swing height from the terrain"),
  ("slope_up_down", EMERGENT, "slopes: trunk pitch and foot rotation follow the ground normal"), ("step_over_log", EMERGENT, "step over a log"), ("duck_under_beam", EMERGENT, "auto-duck below a ceiling"),
  ("around_obstacles", EMERGENT, "A* navigation around walls"), ("bodies_walk", EMERGENT, "different bodies (tall, dwarf, belly, armour, pack, skirt) pick their own posture"), ("crowd", EMERGENT, "14 people cross a plaza"), ("run_forest", EMERGENT, "run through a forest: A* around trunks, duck under low branches"), ("run_crowd", EMERGENT, "run through a crowd: speed follows the gap, shoulders turn"), ("outfits_walk", EMERGENT, "tight skirt, wide skirt, trousers, heavy boots, high heels, chainmail, plate: limits, mass, foot shape"), ("limp_people", EMERGENT, "sore leg, rigid stick for a lower leg, crutches with one leg held up"), ("limp_dogs", EMERGENT, "dog with a sore leg, on three legs, with a stick for the lower hind leg"), ("climb_rock", EMERGENT, "rock face: only the coloured holds can be used"), ("climb_tree", EMERGENT, "tree: branch stubs are the only holds")]),
 ("Jumping (power-limited: the planner refuses a jump the legs cannot make)", "sim", [
  ("jump_gap", EMERGENT, "1.2 m gap from a run-up"), ("jump_wall", EMERGENT, "standing jump over a 60 cm wall"), ("cat_jump_up", EMERGENT, "cat: 8 spine joints coil, stretch, tuck, land"),
  ("horse_jump_fence", EMERGENT, "horse: stiff column, small trunk rotation"), ("quad_jump_species", EMERGENT, "same relative obstacle: cat, dog, wolf, pig")]),
 ("Animals: spine, neck, tail and feet differ by species", "sim", [
  ("animals_trot", EMERGENT, "dog, cat, horse, wolf, pig trot"), ("quadruped_speeds", EMERGENT, "walk to gallop by Froude number"), ("horse_load", EMERGENT, "horse with 0, 90, 180 kg"),
  ("quad_species_tails", EMERGENT, "balance tail, wag, low tail, curl"), ("horse_gait_tail", EMERGENT, "5-joint neck nod, hair tail flags and flicks"), ("animals_stairs_log", EMERGENT, "stairs and a log, four legs")]),
 ("Birds and snake", "sim", [
  ("birds_walk", EMERGENT, "bird gaits"), ("bird_flight", KIN, "flapping flight model (beat rate from mass)"), ("bird_land_ground", KIN, "landing on the ground: flare, feet first, run out"),
  ("bird_land_branch", KIN, "landing on a branch: near-stall, toes grip"), ("bird_land_water", KIN, "landing on water: skid and float"), ("snake_around", KIN, "serpentine motion between walls")]),
 ("Everyday situations (goals for pelvis, hands and feet + IK)", "sim", [
  ("door_open", GOAL, "open a door"), ("sit_stand", GOAL, "sit on a chair and stand"), ("sit_table", GOAL, "sit at a table"), ("bed_lie_rise", GOAL, "lie down in a bed, get up"),
  ("get_up_floor", GOAL, "get up from the floor"), ("ladder_climb", GOAL, "ladder"), ("rider", GOAL, "ride a horse"), ("swim_freestyle", KIN, "swim, head above water"), ("dive_swim", KIN, "dive"),
  ("crowd_squeeze", EMERGENT, "squeeze through people: shoulders turn, hands go up"), ("bump_head", EMERGENT, "hit the head on a beam")]),
 ("Combat as a set of reactions (no scripted duel)", "sim", [
  ("run_into_wall", EMERGENT, "run into a wall: seen early, late, not at all (arms and knees lengthen the stop)"), ("weapon_reactions", EMERGENT, "axe vs shield, axe on a bare head, arrows dodged, caught by a shield, taken; shin block"), ("hit_reactions_c", PHYS, "groin (man vs woman) and knee"), ("hit_reactions_a", EMERGENT, "hit on head, torso, arm: stagger, doubled over, drop the weapon, clutch"), ("hit_reactions_b", PHYS, "leg: limp; hard head blow: physical fall, lie, get up; blow in the back"),
  ("dodge_reactions", EMERGENT, "notice in time: side step, duck, hop, block; too late: hit"), ("duel", EMERGENT, "two fighters choose targets and reactions on their own"), ("push_recovery", EMERGENT, "shoved: recovery steps")]),
]
def items_html(rows, sub):
    s = ""
    for name, label, cap in rows:
        src = os.path.join(runs, "gifs3", name + ".gif")
        if not os.path.exists(src): src = os.path.join(runs, "gifs2", name + ".gif")
        if not os.path.exists(src): continue
        os.makedirs(os.path.join(out, sub), exist_ok=True); put(src, os.path.join(out, sub, name + ".gif"))
        s += f'<figure><img loading="lazy" src="{sub}/{name}.gif"><figcaption><b>{html.escape(name)}</b> <span class="tag {label.split()[0]}">{label}</span><br>{html.escape(cap)}</figcaption></figure>\n'
    return s
body = ""
for title, sub, rows in SECTIONS:
    body += f"<h2>{html.escape(title)}</h2><div class=grid>\n{items_html(rows, sub)}</div>\n"
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
if phys:
    g = ""; os.makedirs(os.path.join(out, "physics"), exist_ok=True)
    for f in phys:
        n = os.path.basename(f); put(f, os.path.join(out, "physics", n)); g += f'<figure><img loading="lazy" src="physics/{n}"><figcaption><b>{n[:-4]}</b> <span class="tag physics">physics, evolved</span></figcaption></figure>\n'
    body += f"<h2>Physical fall and landing reflex (MuJoCo, CMA-ES on a bone-injury score)</h2>{summary}{slide_tab}<div class=grid>\n{g}</div>\n"
page = f"""<!doctype html><meta charset=utf-8><title>Procedural motion showcase</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font:15px system-ui,sans-serif;margin:0 auto;max-width:1500px;padding:16px;background:#f6f6f2;color:#222}}h1{{margin:.2em 0}}h2{{margin-top:1.6em;border-bottom:1px solid #bbb}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}}figure{{margin:0;background:#fff;border:1px solid #ddd;padding:6px;border-radius:6px}}img{{width:100%;display:block}}
figcaption{{font-size:13px;padding-top:4px}}.tag{{font-size:11px;padding:1px 6px;border-radius:9px;background:#ddd;margin-left:6px}}.emergent{{background:#cfe8cf}}.goal{{background:#f2e3b8}}.kinematic{{background:#e3d0f0}}.physics{{background:#bcd7f2}}.retarget{{background:#eee}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}</style>
<h1>Procedural motion showcase</h1>
<p>Green = comes out of general rules (gait from body size, footholds from terrain, reactions chosen by predicting geometry). Yellow = goals for pelvis, hands and feet solved by IK. Purple = hand-built kinematic model. Blue = trained or evolved on a physical body.</p>
{body}"""
open(os.path.join(out, "index.html"), "w").write(page)
print("wrote", os.path.join(out, "index.html"))
