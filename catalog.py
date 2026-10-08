"""Catalog of demonstration shots. python catalog.py list | sheet NAME out.png | gif NAME out.gif | all OUTDIR"""
from __future__ import annotations
import sys, math, os
import numpy as np
from bodies import *
from planner import Cmd, wrap
from shots import *
from nav import Navigator
import skills as SK
from render import z_to_quat

SHOTS = {}
def shot(fn):
    SHOTS[fn.__name__] = fn; return fn

def goto(nav, speed, final_heading=None):
    return lambda t, w, ctx: nav.cmd(w.pos, speed)

def straight_cmd(v, until=1e9, start=0.0):
    return lambda t, w, ctx: Cmd(v=np.array([v if start <= t < until else 0.0, 0.0]))

def nav_actor(c, tr, start, goal, speed, label="", **kw):
    nv = Navigator(tr, c, **{k: v for k, v in kw.items() if k in ("jump_ok", "radius")})
    nv.plan(np.array(start[:2]), np.array(goal[:2]))
    return ActorSpec(c, pos=start, cmd=goto(nv, speed), label=label)

# ---------------------------------------------------------------------------- basic gaits
@shot
def walk_run():
    c = human(); tr = Terrain()
    sched = [(0, 0.0), (0.5, 1.4), (4.0, 2.2), (6.0, 4.2), (9.0, 1.4), (11.5, 0.0)]
    def cmd(t, w, ctx):
        v = 0.0
        for ts, vs in sched:
            if t >= ts: v = vs
        return Cmd(v=np.array([v, 0.0]))
    return Shot("walk_run", "stand, walk, fast walk, run, back to walk, stop (gait from Froude number)", tr, [ActorSpec(c, cmd=cmd)], 13.0, dict(dist=5.0, elevation=-6))

@shot
def stairs_up_down():
    c = human(); tr = Terrain()
    xe = tr.add_stairs(3.0, 0.0, 9, 0.17, 0.30, 11.8)               # the whole width of the terrain: the only way on is over the steps
    tr.add_box(xe, xe + 3.0, -5.9, 5.9, 9 * 0.17)
    tr.add_stairs(xe + 3.0, 0.0, 9, 0.17, 0.30, 11.8, down=True)
    a = nav_actor(c, tr, (0.0, 0.0, 0.0), (xe + 3.0 + 9 * 0.30 + 1.5, 0.0), 1.1)
    return Shot("stairs_up_down", "stairs up and down: foot placement on treads, swing height from the terrain", tr, [a], 22.0, dict(dist=5.5, elevation=-5))

@shot
def slope_up_down():
    c = human(); tr = Terrain()
    tr.add_ramp(2.0, 8.0, -1.5, 1.5, 0.0, 1.5); tr.add_box(8.0, 11.0, -1.5, 1.5, 1.5); tr.add_ramp(11.0, 17.0, -1.5, 1.5, 1.5, 0.0)
    a = nav_actor(c, tr, (0.0, 0.0, 0.0), (19.0, 0.0), 1.4)
    return Shot("slope_up_down", "uphill and downhill (14 degrees): lean, shorter stride, foot follows the surface", tr, [a], 18.0, dict(dist=6.0, elevation=-4))

@shot
def step_over_log():
    c = human(); tr = Terrain(); tr.add_box(3.0, 3.35, -2, 2, 0.30)
    a = ActorSpec(c, cmd=straight_cmd(1.3, 6.5))
    return Shot("step_over_log", "step over a 30 cm log: swing apex follows the obstacle between foot and foothold", tr, [a], 6.5, dict(dist=4.2, elevation=-4))

@shot
def duck_under_beam():
    c = human(); tr = Terrain(); tr.add_beam(3.2, 4.6, -1.5, 1.5, 1.35)
    a = ActorSpec(c, cmd=straight_cmd(1.2, 8.0))
    return Shot("duck_under_beam", "low doorway (1.35 m): the trunk lowers and tilts, no stooping animation", tr, [a], 8.0, dict(dist=4.4, elevation=-4))

@shot
def around_obstacles():
    c = human(); tr = Terrain()
    tr.add_box(3.0, 3.6, -4.2, 1.2, 1.3); tr.add_box(6.5, 7.1, -1.2, 4.2, 1.3); tr.add_box(10.0, 10.6, -4.2, 1.2, 1.3)
    a = nav_actor(c, tr, (0.0, 0.0, 0.0), (13.5, 0.0), 1.5)
    return Shot("around_obstacles", "path planning (A*) on the same terrain: walk around walls, turn without sliding", tr, [a], 17.0, dict(dist=11.0, azimuth=120, elevation=-45, look_z=0.5))

def _run_jump(c, tr, x0, v, trigger_x, target, T, title, name, cam):
    """Run up (or stand), take off at trigger_x, jump to target: the planner refuses a jump the legs cannot power."""
    state = {}
    def cmd(t, w, ctx):
        if v > 0 and w.pos[0] < trigger_x and not state.get("go"): return Cmd(v=np.array([v, 0.0]), heading=0.0)
        if not state.get("go") and w.pos[0] >= trigger_x - 1e-6 and (v > 0 or t > 1.0):
            state["go"] = True; state["ok"] = w.start_jump(target)
        return Cmd(v=np.array([v if (state.get("go") and w.jumpplan is None and state.get("ok") is False) else 0.0, 0.0]), heading=0.0) if not (state.get("go") and w.jumpplan is not None) else Cmd()
    a = ActorSpec(c, pos=(x0, 0), cmd=cmd)
    return Shot(name, title, tr, [a], T, cam)

@shot
def jump_gap():
    tr = Terrain(); tr.add_pit(2.9, 4.1, -3, 3, 1.2)
    return _run_jump(human(), tr, -2.0, 4.0, 2.0, (5.0, 0.0), 5.0, "jump over a 1.2 m gap from a 4 m/s run-up: the take-off speed is limited by the legs, a standing jump this long is refused", "jump_gap", dict(dist=6.5, elevation=-6, follow=0))

@shot
def jump_wall():
    tr = Terrain(); tr.add_box(2.7, 3.1, -3, 3, 0.6)
    return _run_jump(human(), tr, 2.0, 0.0, 2.0, (3.8, 0.0), 3.5, "standing jump over a 60 cm wall (the apex and the distance are limited by the take-off speed the legs can give)", "jump_wall", dict(dist=5.5, elevation=-6))

@shot
def push_recovery():
    c = human(); tr = Terrain()
    ev = [(1.0, lambda w, ctx: w.apply_push((1, 0), 1.5)), (3.5, lambda w, ctx: w.apply_push((-1, 0.2), 1.3)), (6.0, lambda w, ctx: w.apply_push((0.1, -1), 1.1))]
    a = ActorSpec(c, cmd=lambda t, w, ctx: Cmd(), events=ev)
    return Shot("push_recovery", "shoved from behind, in the chest and sideways: spring-like trunk, recovery steps", tr, [a], 8.0, dict(dist=5.0, elevation=-6))

@shot
def bodies_walk():
    names = [("average", HumanBody()), ("tall", HumanBody(height=1.95, mass=88, leg_ratio=1.05)), ("dwarf", HumanBody(height=1.30, mass=60, leg_ratio=0.78, width=1.25)),
             ("belly", HumanBody(mass=115, belly=32.0)), ("armour+pack", HumanBody(mass=105, armour=24, pack=14)), ("skirt", HumanBody(mass=60, height=1.65, dress=1.0))]
    tr = Terrain(); acts = []
    for i, (n, b) in enumerate(names):
        c = human(b)
        acts.append(ActorSpec(c, pos=(0.0, -2.5 + i * 1.0), cmd=straight_cmd(1.4, 9.0), label=n))
    return Shot("bodies_walk", "same planner, six bodies: " + ", ".join(n for n, _ in names), tr, acts, 9.0, dict(dist=6.2, azimuth=55, elevation=-8, follow="all", look_z=0.85), size=(520, 300))

@shot
def animals_trot():
    tr = Terrain(); acts = []
    for i, n in enumerate(["cat", "dog", "pig", "horse"]):
        c = quadruped(n)
        acts.append(ActorSpec(c, pos=(0.0, -3.6 + i * 2.4), cmd=straight_cmd(1.6 if n != "horse" else 2.0, 9.0), label=n))
    return Shot("animals_trot", "cat, dog, pig, horse: gait and cadence follow from size alone", tr, acts, 9.0, dict(dist=7.5, azimuth=60, elevation=-8, follow="all", look_z=0.7), size=(520, 300))

@shot
def quadruped_speeds():
    c = quadruped("dog"); tr = Terrain()
    sched = [(0, 0.0), (0.5, 0.7), (3.5, 1.6), (6.5, 3.2), (9.0, 5.5), (12.0, 8.5), (15.0, 0.0)]
    def cmd(t, w, ctx):
        v = 0.0
        for ts, vs in sched:
            if t >= ts: v = vs
        return Cmd(v=np.array([v, 0.0]))
    return Shot("quadruped_speeds", "dog: walk, trot, canter, gallop selected by Froude number", tr, [ActorSpec(c, cmd=cmd)], 17.0, dict(dist=5.0, elevation=-6))

@shot
def horse_load():
    tr = Terrain(); acts = []
    for i, (n, ld) in enumerate([("empty", 0.0), ("pack 90 kg", 90.0), ("pack 180 kg", 180.0)]):
        c = quadruped("horse", load=ld)
        acts.append(ActorSpec(c, pos=(0, -3 + i * 3), cmd=straight_cmd(2.2, 9.0), label=n))
    return Shot("horse_load", "horse: empty, 90 kg and 180 kg on the back (the centre of mass moves, the gait adapts)", tr, acts, 9.0, dict(dist=9.0, azimuth=60, elevation=-8, follow="all", look_z=1.0), size=(520, 300))

@shot
def animals_stairs_log():
    tr = Terrain(); tr.add_stairs(2.0, 0, 6, 0.14, 0.34, 11.8); tr.add_box(2.0 + 6 * 0.34, 7.0, -5.9, 5.9, 6 * 0.14)
    c = quadruped("dog")
    a = nav_actor(c, tr, (0, 0, 0), (7.0, 0.0), 1.2)
    return Shot("animals_stairs_log", "dog on stairs", tr, [a], 9.0, dict(dist=5.0, elevation=-5))

@shot
def birds_walk():
    tr = Terrain(); acts = []
    for i, n in enumerate(["sparrow", "crow", "duck", "heron"]):
        c = bird(n)
        v = math.sqrt(0.22 * 9.81 * c.z0)
        acts.append(ActorSpec(c, pos=(0.0, -1.8 + i * 1.2), cmd=straight_cmd(v, 9.0), label=n))
    return Shot("birds_walk", "sparrow, crow, duck, heron at the same Froude number: cadence and stride scale with leg length", tr, acts, 9.0, dict(dist=4.6, azimuth=70, elevation=-8, follow="all", look_z=0.35), size=(520, 300))

@shot
def cat_jump_up():
    tr = Terrain(); tr.add_box(2.6, 3.6, -1.0, 1.0, 0.62)
    c = quadruped("cat")
    a = ActorSpec(c, pos=(1.5, 0), cmd=straight_cmd(0.0), events=[(1.0, lambda w, ctx: w.start_jump((3.0, 0.0))), (4.5, lambda w, ctx: w.start_jump((4.8, 0.0)))])
    return Shot("cat_jump_up", "cat: jump onto a 62 cm table, then jump down; the same flight planner as for a person", tr, [a], 8.0, dict(dist=3.4, azimuth=90, elevation=-5))

@shot
def horse_jump_fence():
    tr = Terrain(); tr.add_box(8.0, 8.15, -3, 3, 1.4, rgba=(0.6, 0.45, 0.3, 1))
    c = quadruped("horse")
    sched = lambda t: 4.2 if t < 6.0 else 3.0
    a = ActorSpec(c, pos=(0, 0), cmd=lambda t, w, ctx: Cmd(v=np.array([sched(t) if not w.jumpplan else 0.0, 0.0])), events=[(2.2, lambda w, ctx: None)])
    def hook(t, w, ctx):
        if "j" not in ctx.state and w.pos[0] > 6.2 and w.speed > 2.0 and w.jumpplan is None:
            ctx.state["j"] = True; w.start_jump((9.6, 0.0))
    a.events = []
    base_cmd = a.cmd
    def cmd(t, w, ctx):
        hook(t, w, ctx)
        if w.jumpplan is None and ctx.state.get("j") and t > 0: return Cmd(v=np.array([2.5, 0.0]))
        return base_cmd(t, w, ctx)
    a.cmd = cmd
    return Shot("horse_jump_fence", "horse: canter up to a 140 cm fence, take off, fly, land and canter on", tr, [a], 9.0, dict(dist=9.0, azimuth=90, elevation=-4))

@shot
def quad_jump_species():
    """Same relative obstacle (1.2 x hip height) for four species: the column and the hind-limb power decide the posture; a pig cannot clear it."""
    tr = Terrain(); acts = []
    for i, n in enumerate(["cat", "dog", "wolf", "pig"]):
        c = quadruped(n); h = 1.2 * c.z0; y = -3.0 + i * 2.0
        tr.add_box(2.2, 2.35, y - 0.8, y + 0.8, h, rgba=(0.6, 0.45, 0.3, 1))
        d = 1.6 * c.z0 + 0.6
        def go(w, ctx, y=y, d=d, n=n): w.start_jump((2.28 + d, y))
        def cmd(t, w, ctx, y=y): return Cmd(v=np.array([1.8 if (not w.jumpplan and w.pos[0] < 1.4) else 0.0, 0.0]))
        a = ActorSpec(c, pos=(0.0, y), cmd=cmd, label=f"{n} ({c.params['spec']['n_spine']} spine joints, {c.params['spec']['flex_rom'] + c.params['spec']['ext_rom']:.0f} deg)")
        a.events = [(2.0, go)]
        acts.append(a)
    return Shot("quad_jump_species", "same obstacle relative to the body: cat (8 joints, 120 deg), dog, wolf, pig (3 stiff joints, weak push)", tr, acts, 6.0, dict(dist=7.5, azimuth=55, elevation=-16, follow="all", look_z=0.3), size=(640, 340))

# ---------------------------------------------------------------------------- combat
def _duel_state(ctx):
    return ctx.state.setdefault("duel", dict(arrive=None, hist={}, hit_done=set()))

@shot
def duel_scripted():
    tr = Terrain(); A = human(HumanBody(armour=8)); D = human(HumanBody(height=1.8, armour=10))
    props = [dict(geom='<geom type="capsule" fromto="0 0 -0.12 0 0 0.85" size="0.016" rgba="0.85 0.86 0.9 1"/><geom type="capsule" fromto="-0.07 0 0 0.07 0 0" size="0.014" rgba="0.3 0.25 0.2 1"/>'),
             dict(geom='<geom type="cylinder" size="0.30 0.02" rgba="0.55 0.18 0.15 1"/><geom type="sphere" size="0.045" pos="0 0 0.03" rgba="0.8 0.8 0.8 1"/>')]
    ATT = [2.2, 5.2]            # time after arrival of the two attacks
    def att_cmd(t, w, ctx):
        st = _duel_state(ctx); d = ctx.walkers[1]
        rel = d.pos - w.pos; dist = np.linalg.norm(rel[:2])
        head = math.atan2(rel[1], rel[0])
        v = np.zeros(2)
        if dist > 2.15 and st["arrive"] is None:
            v = rel[:2] / dist * 1.5
        elif st["arrive"] is None:
            st["arrive"] = t
        arms = None; ty = 0.0; pelvis = None
        if st["arrive"] is not None:
            for k, off in enumerate(ATT):
                t0 = st["arrive"] + off
                ph, p = SK.strike_phase(t, t0)
                if ph != "guard":
                    tgt = d.com + np.array([0, 0, 0.35])
                    hand, ph, p = SK.strike_hand(w, t, t0, tgt, "R")
                    arms = {"R": hand}
                    ty = -0.45 * (SK.minjerk(p) if ph == "windup" else 1.0 if ph in ("strike", "hold") else 1 - SK.minjerk(p)) * (1 if ph != "strike" else -1.4)
                    if ph == "strike": v = rel[:2] / dist * 1.1
                    bd = SK.blade_dir(w, hand, ph, p, tgt)
                    st["hist"][round(t * 60)] = (hand, bd)
                    if ph == "strike" and p > 0.9 and k not in st["hit_done"]:
                        st["hit_done"].add(k)
                        dirv = np.array([rel[0], rel[1], 0.0]) / dist
                        ctx.walkers[1].apply_push(dirv, 0.45 if k == 0 else 1.6)
                    break
            if arms is None:
                arms = {"R": SK.strike_hand(w, t, 1e9, d.com, "R")[0]}
                st["hist"][round(t * 60)] = (arms["R"], SK.blade_dir(w, arms["R"], "guard", 0, d.com))
        return Cmd(v=v, heading=head, arms=arms, torso_yaw=ty)
    def def_cmd(t, w, ctx):
        st = _duel_state(ctx); a = ctx.walkers[0]
        rel = a.pos - w.pos; head = math.atan2(rel[1], rel[0])
        arms = None
        if st["arrive"] is not None:
            # shield follows the incoming blade tip with 0.1 s delay; the second attack catches the defender off guard
            tt = round((t - 0.10) * 60)
            hd = st["hist"].get(tt)
            blocked = (t - st["arrive"]) < (ATT[1] - 0.8)
            if hd is not None and blocked:
                tip = hd[0] + hd[1] * 0.8
                pos, dirn = SK.shield_pose(w, tip, "L")
                arms = {"L": pos}
            elif blocked:
                arms = {"L": SK.shield_pose(w, a.com + np.array([0, 0, 0.3]), "L")[0]}
        return Cmd(heading=head, arms=arms)
    def prop_fn(t, ctx, sc, row):
        st = _duel_state(ctx)
        hd = st["hist"].get(round(t * 60))
        if hd is None:
            hd = (row[0]["E"][ctx.walkers[0].sk.idx["hand_R"]], np.array([0, 0, 1.0]))
        sc.set_prop(0, hd[0], z_to_quat(hd[1]))
        d = ctx.walkers[1]; sk = d.sk
        hl = row[1]["S"][sk.idx["hand_L"]]
        nrm = row[0]["root"] - row[1]["root"]; nrm[2] = 0
        sc.set_prop(1, hl + nrm / max(np.linalg.norm(nrm), 1e-6) * 0.05, z_to_quat(nrm + np.array([0, 0, 0.0])))
    return Shot("duel_scripted", "sword strike against a shield: hand goals in space, IK + footwork + spring reaction; the second blow lands", tr,
                [ActorSpec(A, pos=(0, 0), cmd=att_cmd, base=(0.6, 0.55, 0.5)), ActorSpec(D, pos=(4.5, 0), heading=math.pi, cmd=def_cmd, base=(0.45, 0.5, 0.6))],
                14.0, dict(dist=6.0, azimuth=100, elevation=-8, follow="all"), props=props, prop_fn=prop_fn, size=(520, 300))

# ---------------------------------------------------------------------------- sit and stand
@shot
def sit_stand():
    c = human(); tr = Terrain()
    chair = (3.0, 0.0); seat_h = 0.46
    props = [dict(geom=f'<geom type="box" size="0.24 0.24 0.02" pos="0 0 {seat_h-0.02}" rgba="0.5 0.35 0.2 1"/>' + ''.join(
        f'<geom type="box" size="0.02 0.02 {seat_h/2}" pos="{sx*0.21} {sy*0.21} {seat_h/2-0.04}" rgba="0.4 0.28 0.16 1"/>' for sx in (-1, 1) for sy in (-1, 1)) +
        f'<geom type="box" size="0.02 0.24 0.25" pos="-0.22 0 {seat_h+0.24}" rgba="0.5 0.35 0.2 1"/>')]
    goal = (chair[0] + 0.40, 0.0)
    nv = Navigator(tr, c); nv.plan(np.array([7.0, 0.4]), np.array(goal))
    def cmd(t, w, ctx):
        s = ctx.state; ph = s.get("ph", "walk")
        if ph == "walk":
            cm = nv.cmd(w.pos, 1.2)
            if np.linalg.norm(w.pos - np.array(goal)) < 0.2 and w.speed < 0.15:
                s["ph"] = "turn"; s["t"] = t
            return cm
        if ph == "turn":
            if abs(wrap(0.0 - w.heading)) < 0.08 and t - s["t"] > 0.8 and all(l.stance for l in w.legs):
                s["ph"] = "sit"; s["t"] = t
            return Cmd(heading=0.0)
        if ph == "sit":
            pel, tau = SK.sit_cmd(w, t, s["t"], chair, seat_h, 0.0)
            kh = {"L": SK.knee_hands(w, "L"), "R": SK.knee_hands(w, "R")}
            if tau >= 1.0: s["ph"] = "seated"; s["t"] = t
            return Cmd(heading=0.0, pelvis=pel, arms=kh)
        if ph == "seated":
            pel, tau = SK.sit_cmd(w, 9e9, 0.0, chair, seat_h, 0.0)
            kh = {"L": SK.knee_hands(w, "L"), "R": SK.knee_hands(w, "R")}
            if t - s["t"] > 2.2: s["ph"] = "rise"; s["t"] = t
            return Cmd(heading=0.0, pelvis=pel, arms=kh)
        if ph == "rise":
            pel, tau = SK.sit_cmd(w, t, s["t"], chair, seat_h, 0.0, stand=True, dur=1.4)
            if (t - s["t"]) > 1.4: s["ph"] = "leave"; s["nv"] = Navigator(tr, c); s["nv"].plan(w.pos, np.array([9.0, 0.0]))
            return Cmd(heading=0.0, pelvis=pel)
        return s["nv"].cmd(w.pos, 1.2)
    return Shot("sit_stand", "walk up to a chair, turn, sit down, stand up, leave: pelvis goal path with planted feet", tr,
                [ActorSpec(c, pos=(7.0, 0.4), heading=math.pi, cmd=cmd)], 20.0, dict(dist=5.0, azimuth=80, elevation=-6), props=props, prop_fn=lambda t, ctx, sc, row: sc.set_prop(0, [chair[0], chair[1], 0.0]))

# ---------------------------------------------------------------------------------------------
def build(name):
    return SHOTS[name]()

if __name__ == "__main__":
    sys.modules["catalog"] = sys.modules["__main__"]; import catalog_extra, catalog_home, catalog_birds, catalog_combat, catalog_moves
    cmd = sys.argv[1]
    if cmd == "list":
        print("\n".join(SHOTS))
    elif cmd == "sheet":
        run_shot(build(sys.argv[2]), sheet=sys.argv[3], n_sheet=int(sys.argv[4]) if len(sys.argv) > 4 else 8,
                 t_range=(float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 else None)
    elif cmd == "gif":
        run_shot(build(sys.argv[2]), gif=sys.argv[3])
    elif cmd == "all":
        os.makedirs(sys.argv[2], exist_ok=True)
        names = sys.argv[3:] or list(SHOTS)
        for n in names:
            print(n, flush=True)
            try: run_shot(build(n), gif=os.path.join(sys.argv[2], n + ".gif"))
            except Exception as e:
                import traceback; traceback.print_exc()
