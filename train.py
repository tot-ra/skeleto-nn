"""PPO for the imitation environment. Rollouts run in worker processes (numpy policy copy), updates in torch.

python train.py --name human_walk --workers 16 --iters 600 [--init runs/x/ckpt.pt]
"""
from __future__ import annotations
import os, sys, time, json, argparse, math, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import multiprocessing as mp
import numpy as np

# ------------------------------------------------------------------ body pools and clips
def build_pool(spec, seed, n_bodies, clips_per_body, clip_T):
    from bodies import human, HumanBody, quadruped, bird
    from ref import make_clip
    rng = np.random.default_rng(seed)
    pool = []
    for b in range(n_bodies):
        if spec["kind"] == "human":
            if b == 0 and spec.get("plain_first", True): hb = HumanBody()
            else:
                R = spec.get("range", "narrow")
                s = 0.06 if R == "narrow" else 0.16
                h = 1.75 * math.exp(rng.uniform(-s, s) * (3 if R != "narrow" else 1.5)) if R != "narrow" else 1.75 * (1 + rng.uniform(-s, s))
                mass = 75 * (h / 1.75) ** 2.3 * (1 + rng.uniform(-s, s) * 1.5)
                extra = rng.random()
                hb = HumanBody(height=h, mass=mass + (0 if R == "narrow" else 0), leg_ratio=1 + rng.uniform(-s, s),
                               belly=float(rng.uniform(0, 0.25) * mass * 0.5) if (R != "narrow" and extra < 0.3) else 0.0,
                               armour=float(rng.uniform(5, 22)) if (R != "narrow" and 0.3 <= extra < 0.55) else 0.0,
                               pack=float(rng.uniform(4, 16)) if (R != "narrow" and 0.55 <= extra < 0.7) else 0.0,
                               width=1 + rng.uniform(-s, s))
                hb.mass += hb.belly + hb.armour + hb.pack
            c = human(hb)
        elif spec["kind"] == "quadruped":
            c = quadruped(spec["species"])
        else:
            c = bird(spec["species"])
        clips = [make_clip(c, int(rng.integers(1 << 30)), T=clip_T, speeds=tuple(spec.get("speeds", (0.0, 0.5, 0.9, 1.3, 1.8, 2.6, 3.6)))) for _ in range(clips_per_body)]
        pool.append((c, clips))
    return pool

def worker(conn, spec, seed, n_bodies, clips_per_body, clip_T, env_kw):
    from env import Env
    pool = build_pool(spec, seed, n_bodies, clips_per_body, clip_T)
    envs = [Env(c, clips, seed=seed * 100 + i, **env_kw) for i, (c, clips) in enumerate(pool)]
    rng = np.random.default_rng(seed + 7)
    cur = envs[0]; obs = cur.reset()
    conn.send(("ready", cur.obs_dim, cur.act_dim))
    while True:
        msg = conn.recv()
        if msg[0] == "stop": break
        if msg[0] == "assist":
            for e in envs: e.assist = msg[1]
            conn.send(("ok",)); continue
        if msg[0] == "set_env_kw":
            for e in envs: e.push = msg[1]["push"]
            conn.send(("ok",)); continue
        _, W, norm, n = msg
        mean, std = norm
        def mlp(x, layers):
            for i, (Wm, bm) in enumerate(layers):
                x = x @ Wm + bm
                if i < len(layers) - 1: x = np.tanh(x)
            return x
        O, A, LP, R, D, V = [], [], [], [], [], []
        infos = []
        for _ in range(n):
            o = np.clip((obs - mean) / std, -10, 10).astype(np.float32)
            mu = mlp(o, W["pi"]); logstd = W["logstd"]
            a = mu + np.exp(logstd) * rng.standard_normal(mu.shape)
            lp = float(-0.5 * np.sum(((a - mu) / np.exp(logstd)) ** 2 + 2 * logstd + math.log(2 * math.pi)))
            v = float(mlp(o, W["v"])[0])
            obs2, r, done, info = cur.step(a)
            O.append(o); A.append(a); LP.append(lp); V.append(v)
            term = done or info["trunc"]
            if info["trunc"] and not done:
                o2 = np.clip((obs2 - mean) / std, -10, 10).astype(np.float32)
                r = r + 0.99 * float(mlp(o2, W["v"])[0])
            R.append(r); D.append(1.0 if term else 0.0)
            if "ep_ret" in info: infos.append((info["ep_ret"], info["ep_len"], info["fell"]))
            if term:
                cur = envs[int(rng.integers(len(envs)))]; obs = cur.reset()
            else:
                obs = obs2
        o = np.clip((obs - mean) / std, -10, 10).astype(np.float32)
        last_v = float(mlp(o, W["v"])[0])
        conn.send(("data", np.array(O, np.float32), np.array(A, np.float32), np.array(LP, np.float32), np.array(R, np.float32), np.array(D, np.float32), np.array(V, np.float32), last_v, infos))

# ------------------------------------------------------------------ learner
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--spec", default='{"kind":"human"}')
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--steps", type=int, default=512); ap.add_argument("--bodies", type=int, default=4); ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--clip-T", type=float, default=10.0)
    ap.add_argument("--init", default=None); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--push", default="0,0", help="probability,max velocity change (m/s)")
    ap.add_argument("--act-scale", type=float, default=0.30); ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=100.0); ap.add_argument("--assist-iters", type=int, default=0); ap.add_argument("--assist0", type=float, default=1.0)
    ap.add_argument("--speeds", default=None)
    args = ap.parse_args()
    import torch, torch.nn as nn
    torch.set_num_threads(4)
    spec = json.loads(args.spec)
    if args.speeds: spec["speeds"] = [float(x) for x in args.speeds.split(",")]
    pp = [float(x) for x in args.push.split(",")]
    env_kw = dict(push=(pp[0], pp[1]), act_scale=args.act_scale, kp_scale=args.kp)
    out = os.path.join("runs", args.name); os.makedirs(out, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs, conns = [], []
    for i in range(args.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, spec, 1000 + i, args.bodies, args.clips, args.clip_T, env_kw), daemon=True)
        p.start(); procs.append(p); conns.append(a)
    dims = [c.recv() for c in conns]
    od, ad = dims[0][1], dims[0][2]
    print("workers ready; obs", od, "act", ad, flush=True)

    def net(i, o, h=256):
        return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    pi, vf = net(od, ad), net(od, 1)
    with torch.no_grad():
        pi[-1].weight.mul_(0.01); pi[-1].bias.zero_()
    logstd = nn.Parameter(torch.full((ad,), -1.0))
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()) + [logstd], lr=args.lr)
    mean = np.zeros(od, np.float32); var = np.ones(od, np.float32); cnt = 1e-4
    it0 = 0
    if args.init:
        ck = torch.load(args.init, weights_only=False)
        pi.load_state_dict(ck["pi"]); vf.load_state_dict(ck["vf"]); logstd.data = ck["logstd"]; mean, var, cnt = ck["mean"], ck["var"], ck["cnt"]; it0 = ck.get("it", 0)

    def export():
        def lay(m): return [(l.weight.detach().numpy().T.astype(np.float32), l.bias.detach().numpy().astype(np.float32)) for l in m if isinstance(l, nn.Linear)]
        return dict(pi=lay(pi), v=lay(vf), logstd=logstd.detach().numpy().astype(np.float32))

    t_start = time.time(); log = open(os.path.join(out, "log.csv"), "a")
    gamma, lam = 0.99, 0.95
    for it in range(it0, it0 + args.iters):
        t0 = time.time()
        W = export(); norm = (mean, np.sqrt(var + 1e-6).astype(np.float32))
        if args.assist_iters:
            av = max(0.0, args.assist0 * (1.0 - (it - it0) / args.assist_iters))
            for c in conns: c.send(("assist", av))
            for c in conns: c.recv()
        for c in conns: c.send(("roll", W, norm, args.steps))
        res = [c.recv() for c in conns]
        O = np.concatenate([r[1] for r in res]); A = np.concatenate([r[2] for r in res]); LP = np.concatenate([r[3] for r in res])
        adv_l, ret_l = [], []
        for r in res:
            _, o, a, lp, rew, dn, v, lastv, info = r
            n = len(rew); adv = np.zeros(n, np.float32); g = 0.0
            for t in reversed(range(n)):
                nv = lastv if t == n - 1 else v[t + 1]
                nd = 1.0 - dn[t]
                delta = rew[t] + gamma * nv * nd - v[t]
                g = delta + gamma * lam * nd * g
                adv[t] = g
            adv_l.append(adv); ret_l.append(adv + v)
        ADV = np.concatenate(adv_l); RET = np.concatenate(ret_l)
        infos = [x for r in res for x in r[8]]
        # observation statistics (raw obs = norm obs * std + mean; recover the raw ones for the update of the running stats)
        raw = O * norm[1] + norm[0]
        bm, bv, bc = raw.mean(0), raw.var(0), len(raw)
        d = bm - mean; tot = cnt + bc
        mean = mean + d * bc / tot; var = (var * cnt + bv * bc + d ** 2 * cnt * bc / tot) / tot; cnt = tot
        # PPO update
        To = torch.tensor(O); Ta = torch.tensor(A); Tlp = torch.tensor(LP); Tadv = torch.tensor((ADV - ADV.mean()) / (ADV.std() + 1e-6)); Tret = torch.tensor(RET)
        N = len(O); mb = 4096
        for ep in range(6):
            perm = torch.randperm(N)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                mu = pi(To[idx]); lp = (-0.5 * (((Ta[idx] - mu) / logstd.exp()) ** 2 + 2 * logstd + math.log(2 * math.pi))).sum(1)
                ratio = (lp - Tlp[idx]).exp()
                l1 = ratio * Tadv[idx]; l2 = ratio.clamp(0.8, 1.2) * Tadv[idx]
                lpi = -torch.min(l1, l2).mean()
                lv = 0.5 * ((vf(To[idx]).squeeze(1) - Tret[idx]) ** 2).mean()
                loss = lpi + 0.5 * lv
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 1.0); opt.step()
            with torch.no_grad(): logstd.clamp_(-2.2, -0.3)
        n_ep = len(infos)
        er = np.mean([i[0] / max(i[1], 1) for i in infos]) if n_ep else float("nan")
        el = np.mean([i[1] for i in infos]) if n_ep else float("nan")
        fell = np.mean([i[2] for i in infos]) if n_ep else float("nan")
        line = f"{it} sps={N / (time.time() - t0):.0f} step_rew={RET.mean():.3f} ep_rew/step={er:.3f} ep_len={el:.0f} fell={fell:.2f} std={logstd.exp().mean().item():.3f} assist={(max(0.0, args.assist0 * (1.0 - (it - it0) / args.assist_iters)) if args.assist_iters else 0):.2f} eps={n_ep} elapsed={(time.time() - t_start) / 60:.1f}m"
        print(line, flush=True); log.write(line + "\n"); log.flush()
        if it % 10 == 9 or it == it0 + args.iters - 1 or (time.time() - t_start) / 3600 > args.hours:
            torch.save(dict(pi=pi.state_dict(), vf=vf.state_dict(), logstd=logstd.data, mean=mean, var=var, cnt=cnt, it=it + 1, spec=spec, args=vars(args)), os.path.join(out, "ckpt.pt"))
        if (time.time() - t_start) / 3600 > args.hours: break
    for c in conns: c.send(("stop",))

if __name__ == "__main__":
    main()
