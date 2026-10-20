"""Publish the current state of every family's motor network: GIFs of its current network, an EMG chart where it has
muscles, and a status block in the README, between the ``motor-status:<family>`` markers of the family's section.

    python -m motor.showcase [--force] [--only biped,bird]

For each family it loads the run's checkpoint (``runs/motor/<run>/ckpt.pt``, or the best member of the latest
population round when the run has a ``history.json``), plays a few fixed situations with the harness the run still
trains with (none once it has faded) and writes ``showcase/motor/latest/<family>_<situation>.gif`` and
``showcase/motor/latest/status.json``. The score is the mean over the shown situations of survival share x (0.3 +
goal), at most 1.3. A family is rendered again only when that score moved by more than ``MIN_CHANGE`` since the last
publication (or with ``--force``), so periodic runs do not fill the repository's history with the same GIFs.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("BIPED_MUSCLES", "1")              # the published people and quadrupeds move by line muscles
os.environ.setdefault("QUAD_MUSCLES", "1")
import json, shutil, sys, tempfile, time
import numpy as np

from motor.ppo import make_env
from motor.show import load, run, situations, render_clip

OUT = os.path.join("showcase", "motor", "latest")
MIN_CHANGE = 0.02
SIZE = (360, 225)
SECONDS = 8
FAMILIES = [  # (family, run, situations shown)
    ("biped", "biped_human", ["walk_slow", "walk", "turn", "stand"]),
    ("quadruped", "quad_muscle", ["dog_walk", "cat_walk", "horse_walk", "cat_drop"]),
    ("bird", "bird_amp", ["crow_cruise", "crow_turn", "crow_climb", "gull"]),
]
EMG = {"biped": 0.8, "quadruped": 0.5}   # walk speed of the EMG chart (people m/s, quadrupeds Froude units)


def checkpoint(run_name):
    """(path, rounds trained) of a run's current network."""
    base = os.path.join("runs", "motor", run_name)
    hist = os.path.join(base, "history.json")
    if os.path.exists(hist):
        h = json.load(open(hist))
        return os.path.join(base, f"m{int(h[-1]['best'])}", "ckpt.pt"), len(h)
    return os.path.join(base, "ckpt.pt"), None


def describe(family, res, frames, ctrl_hz=30):
    held = res["steps"] / ctrl_hz
    s = f"{'fell at' if res['fell'] else 'held'} {held:.1f} s"
    if family == "bird":
        z = [f[0][0][2] for f in frames]
        s += f", height {z[0]:.1f} → {z[-1]:.1f} m, {res['speed']:.1f} m/s"
    else:
        s += f", {res['speed']:.2f} m/s"
    return s + f", goal {res['goal']:.2f}"


def render_family(family, run_name, names, status, dry=False):
    path, rounds = checkpoint(run_name)
    tmp = os.path.join(tempfile.mkdtemp(), "ckpt.pt"); shutil.copy(path, tmp)   # training keeps writing the original
    ck, pi = load(tmp)
    assist = round(float(ck.get("assist", 0.0)), 2)
    env = make_env(family, seed=0)
    table = {name: (kw, title) for name, kw, title in situations(family, env)}
    clips, scores = [], []
    T = SECONDS * env.CTRL_HZ
    for name in names:
        kw, title = table[name]
        frames, res = run(env, pi, ck, kw, T=T, assist=assist)
        scores.append(res["steps"] / T * (0.3 + res["goal"]))
        gif = os.path.join(OUT, f"{family}_{name}.gif")
        if not dry:
            render_clip(env, frames, title + (f", harness {assist:.0%}" if assist else ""), gif, size=SIZE)
        clips.append(dict(name=name, title=title, gif=gif, result=describe(family, res, frames, env.CTRL_HZ)))
        print(family, name, clips[-1]["result"], flush=True)
    emg = None
    if family in EMG and env.pc.nm and not dry:
        from motor import emg as E
        sp = EMG[family] * (env.speed_unit() if family == "quadruped" else 1.0)
        _f, acts, contacts, v = E.record(env, pi, ck, sp, assist, 6 * env.CTRL_HZ)
        emg = os.path.join(OUT, f"{family}_emg.png")
        what = "dog" if family == "quadruped" else "walk"
        E.chart(env, acts, contacts, emg, f"Left side muscle activation, {what} at {v:.2f} m/s"
                + (f", harness {assist:.0%}" if assist else ""))
    st = dict(score=round(float(np.mean(scores)), 3), it=int(ck.get("it", 0)), rounds=rounds, assist=assist,
              muscles=int(env.pc.nm), clips=clips, emg=emg, updated=time.strftime("%Y-%m-%d %H:%M"))
    status[family] = st
    return st["score"]


def readme_block(family, st):
    tag = f"motor-status:{family}"
    head = (f"**Current network** (`python -m motor.showcase`, updated {st['updated']}): "
            f"{st['it']:,} training iterations, score {st['score']:.2f} of 1.3 (survival share x (0.3 + goal)); ")
    head += (f"{st['muscles']} line muscles" if st["muscles"] else "joint torques") + ", "
    head += (f"shown with the {st['assist']:.0%} harness it still trains with (it carries part of the weight and keeps "
             f"the pelvis upright)." if st["assist"] else "no harness.")
    out = [f"<!-- {tag}:start -->", head, "", "<table><tr>"]
    for c in st["clips"]:
        out.append(f'<td width="{100 // len(st["clips"])}%" valign="top"><img src="{c["gif"]}" width="100%">'
                   f'<br><b>{c["title"]}</b>: {c["result"]}</td>')
    out.append("</tr></table>")
    if st.get("emg"):
        out += ["", f'<img src="{st["emg"]}" width="100%">', "",
                "Activation of every left-side muscle through the walk above (like an EMG), with the foot contacts below."]
    out.append(f"<!-- {tag}:end -->")
    return "\n".join(out)


def main():
    force = "--force" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(OUT, "status.json")
    status = json.load(open(sp)) if os.path.exists(sp) else {}
    readme = open("README.md").read()
    changed = False
    for family, run_name, names in FAMILIES:
        if not os.path.exists(checkpoint(run_name)[0]) or (only and family not in only):
            continue
        old = status.get(family, {}).get("score")
        if not force and old is not None:
            probe = {}
            score = render_family(family, run_name, names, probe, dry=True)
            if abs(score - old) <= MIN_CHANGE:
                print(family, f"score {score:.3f} unchanged since last publication", flush=True)
                continue
        render_family(family, run_name, names, status); changed = True
    json.dump(status, open(sp, "w"), indent=1, ensure_ascii=False)
    for family, _r, _n in FAMILIES:
        tag = f"motor-status:{family}"
        if family in status and f"<!-- {tag}:start -->" in readme:
            a = readme.index(f"<!-- {tag}:start -->"); b = readme.index(f"<!-- {tag}:end -->") + len(f"<!-- {tag}:end -->")
            readme = readme[:a] + readme_block(family, status[family]) + readme[b:]
    open("README.md", "w").write(readme)
    print("changed" if changed else "unchanged")


if __name__ == "__main__":
    main()
