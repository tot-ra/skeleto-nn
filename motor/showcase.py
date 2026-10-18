"""Publish the current state of every family's motor network: GIFs of the best network so far and a status table in
the README, between the ``motor-status`` markers.

    python -m motor.showcase [--force] [--only biped,bird]

For each family it takes the best member of the latest population round (``runs/motor/<run>/history.json``), plays a
few fixed situations without the harness and writes ``showcase/motor/latest/<family>_<situation>.gif`` and
``showcase/motor/latest/status.json``. A family is rendered again only when its best score moved by more than
``MIN_CHANGE`` since the last publication (or with ``--force``), so periodic runs do not fill the repository's history
with the same GIFs.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "glfw")
import json, shutil, sys, tempfile, time
import numpy as np

from motor.ppo import make_env
from motor.show import load, run, situations, render_clip

OUT = os.path.join("showcase", "motor", "latest")
MIN_CHANGE = 0.02
SIZE = (360, 225)
SECONDS = 8
FAMILIES = [  # (family, population run, situations shown)
    ("biped", "biped_pbt", ["walk", "run", "ice", "pack", "turn"]),
    ("quadruped", "quad_pbt", ["dog_walk", "dog_trot", "cat_walk", "horse_walk"]),
    ("bird", "bird3_pbt", ["crow_cruise", "crow_turn", "crow_climb", "gull"]),
]
TITLES = {"biped": "People", "quadruped": "Four-legged animals", "bird": "Birds"}


def best_member(run_name):
    hist = json.load(open(os.path.join("runs", "motor", run_name, "history.json")))
    last = hist[-1]
    return last["round"], int(last["best"]), float(max(last["scores"])), len(hist)


def describe(family, res, frames, ctrl_hz=30):
    held = res["steps"] / ctrl_hz
    s = f"{'fell at' if res['fell'] else 'held'} {held:.1f} s"
    if family == "bird":
        z = [f[0][0][2] for f in frames]
        s += f", height {z[0]:.1f} → {z[-1]:.1f} m, {res['speed']:.1f} m/s"
    else:
        s += f", {res['speed']:.2f} m/s"
    return s + f", goal {res['goal']:.2f}"


def render_family(family, run_name, names, status):
    rnd, best, score, n = best_member(run_name)
    ckpt = os.path.join("runs", "motor", run_name, f"m{best}", "ckpt.pt")
    tmp = os.path.join(tempfile.mkdtemp(), "ckpt.pt"); shutil.copy(ckpt, tmp)   # training keeps writing the original
    ck, pi = load(tmp)
    env = make_env(family, seed=0)
    table = {name: (kw, title) for name, kw, title in situations(family, env)}
    clips = []
    for name in names:
        kw, title = table[name]
        frames, res = run(env, pi, ck, kw, T=SECONDS * env.CTRL_HZ)
        path = os.path.join(OUT, f"{family}_{name}.gif")
        render_clip(env, frames, title, path, size=SIZE)
        clips.append(dict(name=name, title=title, gif=path, result=describe(family, res, frames, env.CTRL_HZ)))
        print(family, name, clips[-1]["result"], flush=True)
    status[family] = dict(round=rnd, score=round(score, 3), rounds_total=n, clips=clips,
                          updated=time.strftime("%Y-%m-%d %H:%M"))


def readme_block(status):
    out = ["<!-- motor-status:start -->",
           "### Current state (updated automatically by `python -m motor.showcase`)",
           "",
           "The best network of each family so far, no harness, no clips: every joint from the network. The score is the "
           "population test mean (survival share x (0.3 + goal), at most 1.3).",
           ""]
    for family, _run, _names in FAMILIES:
        st = status.get(family)
        if not st:
            continue
        out.append(f"**{TITLES[family]}**: score {st['score']:.2f} after {st['rounds_total']} rounds "
                   f"(updated {st['updated']})")
        out.append("")
        out.append("<table><tr>")
        for c in st["clips"]:
            out.append(f'<td width="{100 // len(st["clips"])}%" valign="top"><img src="{c["gif"]}" width="100%">'
                       f'<br><b>{c["title"]}</b>: {c["result"]}</td>')
        out.append("</tr></table>")
        out.append("")
    out.append("<!-- motor-status:end -->")
    return "\n".join(out)


def main():
    force = "--force" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(OUT, "status.json")
    status = json.load(open(sp)) if os.path.exists(sp) else {}
    changed = False
    for family, run_name, names in FAMILIES:
        if not os.path.exists(os.path.join("runs", "motor", run_name, "history.json")) or (only and family not in only):
            continue
        _rnd, _best, score, _n = best_member(run_name)
        old = status.get(family, {}).get("score")
        if force or old is None or abs(score - old) > MIN_CHANGE:
            render_family(family, run_name, names, status); changed = True
        else:
            print(family, f"score {score:.3f} unchanged since last publication", flush=True)
    json.dump(status, open(sp, "w"), indent=1, ensure_ascii=False)
    readme = open("README.md").read()
    block = readme_block(status)
    if "<!-- motor-status:start -->" in readme:
        a = readme.index("<!-- motor-status:start -->"); b = readme.index("<!-- motor-status:end -->") + len("<!-- motor-status:end -->")
        readme = readme[:a] + block + readme[b:]
    else:
        anchor = "## Status"
        readme = readme.replace(anchor, block + "\n\n" + anchor, 1)
    open("README.md", "w").write(readme)
    print("changed" if changed else "unchanged")


if __name__ == "__main__":
    main()
