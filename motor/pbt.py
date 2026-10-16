"""Population-based training: several PPO members train side by side; bad ones die off and are replaced by
mutated copies of the best.

    python -m motor.pbt --family biped --name biped_pbt --members 3 --workers 5 --round-min 20 --rounds 9

Every round each member trains for ``--round-min`` minutes (``motor.ppo`` continuing from its checkpoint). Then every
member is tested on fixed review situations *without the harness* (``score``): what counts is what the policy can
do on its own. The worst members copy the checkpoint of the best (exploit) and change their training settings by a
random factor (explore): learning rate, how fast the harness fades, the weight of the mirror loss, and the reward
itself (what staying up is worth, what a fall costs, how much the comfort costs weigh). Gradient training
improves each member within a round; selection between rounds removes the runs that settled into a dead end
(leaning on the harness, standing still).

``runs/motor/<name>/history.json`` records every round: settings, scores, who was replaced by whom.
"""
from __future__ import annotations
import argparse, json, math, os, shutil, subprocess, sys, time
import numpy as np

from bodies import HumanBody

def tests(family):
    """(goal velocity in the family's speed units, extra reset arguments); each runs from a standing and a moving start."""
    if family == "bird":
        from bodies import BIRDS
        return [((2.0, 0.0), {}), ((3.0, 0.0), {}), ((2.5, 0.0), {"climb": 0.3}), ((2.5, 0.0), {"climb": -0.3}),
                ((2.5, 0.6), {}), ((2.5, 0.0), {"wind": np.array([0.0, -6.0, 0.0])}), ((2.5, 0.0), {"body": BIRDS["gull"]}),
                ((2.5, 0.0), {"body": BIRDS["sparrow"]})]
    if family == "quadruped":
        from bodies import QUADS
        from motor.quadruped import canonical
        return [((0.0, 0.0), {}), ((0.5, 0.0), {}), ((0.8, 0.0), {}), ((1.4, 0.0), {}), ((0.0, 0.5), {}),
                ((0.7, 0.0), {"friction": 0.15}), ((0.7, 0.0), {"body": canonical(QUADS["cat"])}),
                ((0.7, 0.0), {"body": canonical(QUADS["horse"])})]
    return [((0.0, 0.0), {}), ((0.8, 0.0), {}), ((1.3, 0.0), {}), ((2.5, 0.0), {}), ((0.0, 1.0), {}),
            ((1.2, 0.0), {"friction": 0.15}), ((1.0, 0.0), {"body": HumanBody(height=1.4, mass=38.0)})]


MUTABLE = {"lr": (5e-5, 1e-3), "fade": (0.0003, 0.006), "sym": (0.5, 10.0),
           # the reward itself evolves too: how much staying up is worth, what a fall costs, how much comfort weighs
           "alive": (0.3, 3.0), "fall_cost": (1.0, 15.0), "cost_scale": (0.3, 2.0)}
DEFAULTS = {"lr": 2e-4, "fade": 0.0015, "sym": 4.0, "alive": 1.0, "fall_cost": 5.0, "cost_scale": 1.0}


def score(ckpt, family="biped", T=240, seed=0):
    """Mean over the test cases of (share of the episode survived) x (0.3 + mean goal score), no harness."""
    from motor.ppo import make_env
    from motor.show import load
    from motor import evaluate as ev
    from motor.ppo import mlp_np
    ck, pi = load(ckpt)
    env = make_env(family, seed=seed)
    mean, std = ck["mean"], np.sqrt(ck["var"] + 1e-6)
    out = []
    for (g, extra) in tests(family):
        for moving in (0.0, 1.0):
            env.moving_start = moving
            body = extra.get("body", env.default_body())
            env._build(body)
            unit = 1.0 if family == "bird" else env.speed_unit()          # the bird env takes goals in speed units
            args = dict(body=body, friction=1.0, wind=np.zeros(3), sores={k: 0.0 for k in env.SORES},
                        profile=dict(ev.DEFAULT_WEIGHTS), goal=np.array(g, float) * unit)
            args.update({k: v for k, v in extra.items() if k != "body"})
            o = env.reset(**args); env.assist = 0.0
            goals, n = [], 0
            for _ in range(T):
                o, _r, d, info = env.step(mlp_np(np.clip((o - mean) / std, -10, 10).astype(np.float32), pi))
                goals.append(info["goal_now"]); n += 1
                if d:
                    break
            out.append(n / T * (0.3 + float(np.mean(goals))))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="biped"); ap.add_argument("--name", required=True)
    ap.add_argument("--members", type=int, default=3); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--round-min", type=float, default=20.0); ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--replace", type=int, default=1, help="members replaced per round")
    ap.add_argument("--reset-curriculum", action="store_true", help="first round: keep the weights, restart harness and goals")
    ap.add_argument("--assist0", type=float, default=1.0); ap.add_argument("--speed0", type=float, default=0.8)
    args = ap.parse_args()
    root = os.path.join("runs", "motor", args.name); os.makedirs(root, exist_ok=True)
    hist_path = os.path.join(root, "history.json")
    history = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    rng = np.random.default_rng(len(history))
    if history:
        settings = history[-1]["next"]
    else:
        settings = [{k: float(np.clip(v * math.exp(rng.uniform(-0.6, 0.6)) if m else v, *MUTABLE[k]))
                     for k, v in DEFAULTS.items()} for m in range(args.members)]
    first = len(history)
    for rnd in range(first, first + args.rounds):
        t0 = time.time(); procs = []
        for m, st in enumerate(settings):
            name = f"{args.name}/m{m}"
            ckpt = os.path.join("runs", "motor", name, "ckpt.pt")
            cmd = [sys.executable, "-m", "motor.ppo", "--family", args.family, "--name", name, "--workers", str(args.workers),
                   "--hours", str(args.round_min / 60.0), "--lr", f"{st['lr']:.6g}", "--fade", f"{st['fade']:.6g}",
                   "--sym", f"{st['sym']:.4g}", "--alive", f"{st['alive']:.4g}",
                   "--fall-cost", f"{st['fall_cost']:.4g}", "--cost-scale", f"{st['cost_scale']:.4g}",
                   "--assist0", str(args.assist0), "--speed0", str(args.speed0)]
            if os.path.exists(ckpt):
                cmd += ["--init", ckpt]
                if args.reset_curriculum and rnd == first:
                    cmd += ["--reset-curriculum"]
            log = open(os.path.join(root, f"m{m}.log"), "a")
            procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
        ckpts = [os.path.join("runs", "motor", f"{args.name}/m{m}", "ckpt.pt") for m in range(len(settings))]
        scores = [score(c, args.family) for c in ckpts]
        order = list(np.argsort(scores)[::-1])
        best = int(order[0]); replaced = []
        nxt = [dict(s) for s in settings]
        for loser in order[-args.replace:]:
            loser = int(loser)
            if loser == best:
                continue
            shutil.copy(ckpts[best], ckpts[loser])                      # exploit: the copy continues the best policy
            nxt[loser] = {k: float(np.clip(settings[best][k] * math.exp(rng.choice([-1, 1]) * rng.uniform(0.2, 0.5)),
                                           *MUTABLE[k])) for k in MUTABLE}  # explore: different settings
            replaced.append(dict(member=loser, copied=best))
        history.append(dict(round=rnd, minutes=round((time.time() - t0) / 60, 1), settings=settings, scores=scores,
                            best=best, replaced=replaced, next=nxt))
        json.dump(history, open(hist_path, "w"), indent=1)
        print(f"round {rnd}: scores {[round(s, 3) for s in scores]} best m{best} replaced {replaced}", flush=True)
        settings = nxt


if __name__ == "__main__":
    main()
