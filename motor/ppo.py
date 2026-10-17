"""PPO for a body family, scored by the evaluation system (``evaluate.py``), with a symmetry loss and a fading harness.

    python -m motor.ppo --family biped --name biped_s1 --workers 16 --hours 3 [--init runs/motor/x/ckpt.pt]

Recipe (Yu, Turk, Liu 2018, "Learning Symmetric and Low-Energy Locomotion"): no reference motion; the reward is
the goal minus the costs of the evaluation system; a mirror loss makes the left and right halves of the policy act
alike; a harness holds the body up and pushes it toward the goal at first and fades out as the policy learns to
stay up (the harness level is lowered whenever episodes run long enough and falls are rare).

Rollouts run in worker processes with a numpy copy of the policy; the update runs in torch.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, math, time
import multiprocessing as mp
import numpy as np

FAMILIES = {"biped": ("motor.biped", "BipedEnv"), "quadruped": ("motor.quadruped", "QuadrupedEnv"),
            "bird": ("motor.bird", "BirdEnv")}


def make_env(family, **kw):
    import importlib
    mod, cls = FAMILIES[family]
    return getattr(importlib.import_module(mod), cls)(**kw)


def mlp_np(x, layers, act=np.tanh):
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = act(x)
    return x


def worker(conn, family, seed, env_kw):
    env = make_env(family, seed=seed, **env_kw)
    rng = np.random.default_rng(seed + 7)
    obs = env.reset()
    conn.send(("ready", env.obs_dim, env.act_dim))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        if msg[0] == "assist":
            env.assist, env.spread, env.max_speed = msg[1], msg[2], msg[3]; conn.send(("ok",)); continue
        _, W, norm, n = msg
        mean, std = norm
        O, A, LP, R, D, V, infos, costs = [], [], [], [], [], [], [], []
        for _ in range(n):
            o = np.clip((obs - mean) / std, -10, 10).astype(np.float32)
            mu = mlp_np(o, W["pi"]); ls = W["logstd"]
            a = mu + np.exp(ls) * rng.standard_normal(mu.shape)
            lp = float(-0.5 * np.sum(((a - mu) / np.exp(ls)) ** 2 + 2 * ls + math.log(2 * math.pi)))
            v = float(mlp_np(o, W["v"])[0])
            obs2, r, done, info = env.step(a)
            O.append(o); A.append(a); LP.append(lp); V.append(v)
            if info["trunc"] and not done:
                o2 = np.clip((obs2 - mean) / std, -10, 10).astype(np.float32)
                r = r + 0.99 * float(mlp_np(o2, W["v"])[0])
            R.append(r); D.append(1.0 if (done or info["trunc"]) else 0.0)
            costs.append([info["costs"][t] for t in sorted(info["costs"])])
            if "ep_ret" in info:
                # a stall is a failure for the curriculum too, or standing still raises the goal speeds
                infos.append((info["ep_ret"], info["ep_len"], info["fell"] or info.get("stalled", False), info["goal"]))
            obs = env.reset() if (done or info["trunc"]) else obs2
        o = np.clip((obs - mean) / std, -10, 10).astype(np.float32)
        conn.send(("data", np.array(O, np.float32), np.array(A, np.float32), np.array(LP, np.float32),
                   np.array(R, np.float32), np.array(D, np.float32), np.array(V, np.float32),
                   float(mlp_np(o, W["v"])[0]), infos, np.mean(costs, 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="biped"); ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--hours", type=float, default=3.0); ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--init", default=None)
    ap.add_argument("--sym", type=float, default=4.0, help="weight of the mirror loss")
    ap.add_argument("--assist0", type=float, default=1.0); ap.add_argument("--spread", type=float, default=0.0,
                                                                          help="variety of bodies and situations at the start (grows to 1 with success)")
    ap.add_argument("--alive", type=float, default=None); ap.add_argument("--fall-cost", type=float, default=None)
    ap.add_argument("--cost-scale", type=float, default=1.0, help="multiplies every evaluation cost")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--reset-curriculum", action="store_true", help="with --init: restart harness, goal speeds and spread")
    ap.add_argument("--speed0", type=float, default=0.8, help="goal speeds at the start of the curriculum (m/s)")
    ap.add_argument("--fade", type=float, default=0.0015, help="harness lost per iteration regardless of success")
    args = ap.parse_args()
    import torch, torch.nn as nn
    torch.set_num_threads(2)                 # leave the cores to the simulation workers
    out = os.path.join("runs", "motor", args.name); os.makedirs(out, exist_ok=True)
    env_kw = dict(spread=args.spread, alive=args.alive, fall_cost=args.fall_cost, cost_scale=args.cost_scale)
    ref = make_env(args.family, seed=0, **env_kw)
    ctx = mp.get_context("spawn"); conns = []
    for i in range(args.workers):
        a, b = ctx.Pipe()
        ctx.Process(target=worker, args=(b, args.family, 1000 + i, env_kw), daemon=True).start(); conns.append(a)
    dims = [c.recv() for c in conns]; od, ad = dims[0][1], dims[0][2]
    print(f"{args.family}: obs {od}, act {ad}, {args.workers} workers", flush=True)

    def net(i, o, h=args.hidden):
        return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h // 2), nn.Tanh(), nn.Linear(h // 2, o))
    pi, vf = net(od, ad), net(od, 1)
    with torch.no_grad():
        pi[-1].weight.mul_(0.01); pi[-1].bias.zero_()
    logstd = nn.Parameter(torch.full((ad,), -2.5))
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()) + [logstd], lr=args.lr)
    mean = np.zeros(od, np.float32); var = np.ones(od, np.float32); cnt = 1e-4
    assist = args.assist0; it0 = 0; max_speed = args.speed0; spread = args.spread
    if args.init:
        ck = torch.load(args.init, weights_only=False)
        pi.load_state_dict(ck["pi"]); vf.load_state_dict(ck["vf"]); logstd.data = ck["logstd"]
        mean, var, cnt = ck["mean"], ck["var"], ck["cnt"]; assist = ck.get("assist", 0.0); it0 = ck.get("it", 0)
        max_speed = ck.get("max_speed", ref.MAX_SPEED); spread = ck.get("spread", 1.0)
        if args.reset_curriculum:            # keep the weights, start the harness and the goals over
            assist, max_speed, spread = args.assist0, args.speed0, args.spread
    m_perm = torch.tensor(ref.a_perm); m_sign = torch.tensor(ref.a_sign, dtype=torch.float32)

    def export():
        lay = lambda m: [(l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()) for l in m if isinstance(l, nn.Linear)]
        return dict(pi=lay(pi), v=lay(vf), logstd=logstd.detach().numpy().copy())

    def save(it):
        torch.save(dict(pi=pi.state_dict(), vf=vf.state_dict(), logstd=logstd.data, mean=mean, var=var, cnt=cnt, it=it,
                        assist=assist, max_speed=max_speed, spread=spread, family=args.family, obs_dim=od, act_dim=ad, hidden=args.hidden, args=vars(args)),
                   os.path.join(out, "ckpt.pt"))

    log = open(os.path.join(out, "log.csv"), "a"); t_start = time.time()
    gamma, lam = 0.99, 0.95
    cost_names = sorted(ref.ev.totals)
    for it in range(it0, it0 + args.iters):
        t0 = time.time()
        for c in conns: c.send(("assist", assist, spread, max_speed))
        for c in conns: c.recv()
        W = export(); norm = (mean, np.sqrt(var + 1e-6).astype(np.float32))
        for c in conns: c.send(("roll", W, norm, args.steps))
        res = [c.recv() for c in conns]
        O = np.concatenate([r[1] for r in res]); A = np.concatenate([r[2] for r in res]); LP = np.concatenate([r[3] for r in res])
        advs, rets = [], []
        for r in res:
            rew, dn, v, lastv = r[4], r[5], r[6], r[7]
            n = len(rew); adv = np.zeros(n, np.float32); g = 0.0
            for t in reversed(range(n)):
                nv = lastv if t == n - 1 else v[t + 1]; nd = 1.0 - dn[t]
                g = rew[t] + gamma * nv * nd - v[t] + gamma * lam * nd * g
                adv[t] = g
            advs.append(adv); rets.append(adv + v)
        ADV = np.concatenate(advs); RET = np.concatenate(rets)
        infos = [x for r in res for x in r[8]]
        cm = np.mean([r[9] for r in res], 0)
        raw = O * norm[1] + norm[0]
        bm, bv, bc = raw.mean(0), raw.var(0), len(raw); d = bm - mean; tot = cnt + bc
        mean = (mean + d * bc / tot).astype(np.float32); var = ((var * cnt + bv * bc + d ** 2 * cnt * bc / tot) / tot).astype(np.float32); cnt = tot
        To = torch.tensor(O); Ta = torch.tensor(A); Tlp = torch.tensor(LP)
        Tadv = torch.tensor((ADV - ADV.mean()) / (ADV.std() + 1e-6)); Tret = torch.tensor(RET)
        nm = torch.tensor(norm[0]); ns = torch.tensor(norm[1])
        N = len(O); mb = 4096; sym_l = 0.0
        for _ in range(5):
            perm = torch.randperm(N)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                mu = pi(To[idx])
                lp = (-0.5 * (((Ta[idx] - mu) / logstd.exp()) ** 2 + 2 * logstd + math.log(2 * math.pi))).sum(1)
                ratio = (lp - Tlp[idx]).exp()
                lpi = -torch.min(ratio * Tadv[idx], ratio.clamp(0.8, 1.2) * Tadv[idx]).mean()
                lv = 0.5 * ((vf(To[idx]).squeeze(1) - Tret[idx]) ** 2).mean()
                # mirror loss: the policy of the reflected state is the reflected action
                raw_o = To[idx] * ns + nm
                mo = torch.clamp((ref.mirror_obs(raw_o) - nm) / ns, -10, 10)
                lsym = ((pi(mo)[..., m_perm] * m_sign - mu) ** 2).mean()
                loss = lpi + 0.5 * lv + args.sym * lsym
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 1.0); opt.step()
                sym_l = float(lsym)
            with torch.no_grad():
                logstd.clamp_(-3.0, -1.5)
        n_ep = len(infos)
        el = np.mean([i[1] for i in infos]) if n_ep else float("nan")
        fell = np.mean([i[2] for i in infos]) if n_ep else float("nan")
        goal = np.mean([i[3] for i in infos]) if n_ep else float("nan")
        # curriculum: lower the harness when episodes run long and few fall
        # curriculum: the harness fades with time (so standing still under it never becomes a home) and faster while
        # the goal is met; goal speeds grow while it is met
        assist = max(0.0, assist - args.fade)
        # goal speeds and variety grow slowly while the goal is met (fast growth ran past what the body could learn
        # twice) and shrink back when most episodes fail
        if n_ep and goal > 0.5 and fell < 0.3:
            assist = max(0.0, assist - 0.01)
            max_speed = min(ref.MAX_SPEED, max_speed + 0.005)
            spread = min(1.0, spread + 0.003)
        elif n_ep and fell > 0.6:
            max_speed = max(args.speed0, max_speed - 0.01)
            spread = max(args.spread, spread - 0.003)
            if assist < ref.ASSIST_RECOVER:                  # how far the harness comes back is the family's
                assist = min(args.assist0, assist + 0.01)
        costs = " ".join(f"{k}={v:.3f}" for k, v in zip(cost_names, cm) if k not in ("heat", "breath"))
        line = (f"{it} sps={N / (time.time() - t0):.0f} ep_len={el:.0f} failed={fell:.2f} goal={goal:.3f} assist={assist:.2f} vmax={max_speed:.2f} spread={spread:.2f} "
                f"std={logstd.exp().mean().item():.3f} sym={sym_l:.4f} {costs} t={(time.time() - t_start) / 60:.1f}m")
        print(line, flush=True); log.write(line + "\n"); log.flush()
        if it % 10 == 9:
            save(it + 1)
        if (time.time() - t_start) / 3600 > args.hours:
            break
    save(it + 1)
    for c in conns: c.send(("stop",))


if __name__ == "__main__":
    main()
