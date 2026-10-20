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


def worker(conn, family, seed, env_kw, kick=None, amp=False):
    env = make_env(family, seed=seed, **env_kw)
    rng = np.random.default_rng(seed + 7)
    obs = env.reset()
    conn.send(("ready", env.obs_dim, env.act_dim))
    # Exploration locked to a rhythm (birds: the wingbeat clock): part of the noise is A sin(phase) + B cos(phase) with
    # A, B drawn per episode, so each step's noise is still N(0, std^2) but over an episode it is a coherent beat.
    # Noise independent at every step never added up to a wingbeat.
    share = getattr(env, "PHASE_NOISE", 0.0)
    # kickstarting: specialists for some bodies ("horse=ckpt"); their action at every state of that body goes back to the
    # learner, which pulls the policy toward it (a distilled skill washed out of a shared network in two rounds without)
    teachers = {}
    for spec in (kick or "").split(","):
        if "=" in spec:
            from motor.distill import Teacher
            name, path = spec.split("=", 1); teachers[name] = Teacher(path)
    AB = rng.standard_normal((2, env.act_dim))
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        if msg[0] == "assist":
            env.assist, env.spread, env.max_speed = msg[1], msg[2], msg[3]
            if len(msg) > 4: env.run_assist = msg[4]
            conn.send(("ok",)); continue
        _, W, norm, n = msg
        mean, std = norm
        O, A, LP, R, D, V, infos, costs = [], [], [], [], [], [], [], []
        K, KM, FP = [], [], []
        f = env.amp_features() if amp else None
        for _ in range(n):
            o = np.clip((obs - mean) / std, -10, 10).astype(np.float32)
            mu = mlp_np(o, W["pi"]); ls = W["logstd"]
            eps = rng.standard_normal(mu.shape)
            if share > 0:
                eps = math.sqrt(1 - share) * eps + math.sqrt(share) * (AB[0] * math.sin(env.phase) + AB[1] * math.cos(env.phase))
            a = mu + np.exp(ls) * eps
            lp = float(-0.5 * np.sum(((a - mu) / np.exp(ls)) ** 2 + 2 * ls + math.log(2 * math.pi)))
            v = float(mlp_np(o, W["v"])[0])
            t = teachers.get(env.c.skel.name) if teachers else None
            K.append(t.act(obs) if t is not None else np.zeros_like(mu)); KM.append(t is not None)
            obs2, r, done, info = env.step(a)
            if amp:                                          # the motion-prior transition: features before and after
                f2 = env.amp_features(); FP.append(np.concatenate([f, f2])); f = f2
            O.append(o); A.append(a); LP.append(lp); V.append(v)
            if info["trunc"] and not done:
                o2 = np.clip((obs2 - mean) / std, -10, 10).astype(np.float32)
                r = r + env.GAMMA * float(mlp_np(o2, W["v"])[0])
            R.append(r); D.append(1.0 if (done or info["trunc"]) else 0.0)
            costs.append([info["costs"][t] for t in sorted(info["costs"])])
            if "ep_ret" in info:
                # a stall is a failure for the curriculum too, or standing still raises the goal speeds
                infos.append((info["ep_ret"], info["ep_len"], info["fell"] or info.get("stalled", False), info["goal"]))
            if done or info["trunc"]:
                obs = env.reset(); AB = rng.standard_normal((2, env.act_dim))
                if amp: f = env.amp_features()
            else:
                obs = obs2
        o = np.clip((obs - mean) / std, -10, 10).astype(np.float32)
        conn.send(("data", np.array(O, np.float32), np.array(A, np.float32), np.array(LP, np.float32),
                   np.array(R, np.float32), np.array(D, np.float32), np.array(V, np.float32),
                   float(mlp_np(o, W["v"])[0]), infos, np.mean(costs, 0), np.array(K, np.float32), np.array(KM, bool),
                   np.array(FP, np.float32) if amp else None))


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
    ap.add_argument("--kick", default=None, help="kickstarting teachers, body=ckpt[,body=ckpt]")
    ap.add_argument("--kick-weight", type=float, default=1.0, help="weight of the pull toward the teachers' actions")
    ap.add_argument("--vwarm", type=int, default=0,
                    help="first iterations train only the value function (after the reward changed: a critic fitted to the "
                         "old reward gave misleading advantages and the policy unlearned a good skill before it caught up)")
    ap.add_argument("--reset-curriculum", action="store_true", help="with --init: restart harness, goal speeds and spread")
    ap.add_argument("--speed0", type=float, default=0.8, help="goal speeds at the start of the curriculum (m/s)")
    ap.add_argument("--fade", type=float, default=0.0015, help="harness lost per iteration regardless of success")
    ap.add_argument("--fade-ok", type=float, default=0.01, help="extra harness lost per iteration while the goal is met")
    ap.add_argument("--run-assist0", type=float, default=0.0,
                    help="run harness (goals past the family's RUN_FROM; upright only) when a run starts without one saved")
    ap.add_argument("--run-fade", type=float, default=0.0004, help="run harness lost per iteration")
    ap.add_argument("--amp", default=None, help="motion-prior transitions (motor/amp.py npz): a reward for looking human")
    ap.add_argument("--amp-w", type=float, default=2.0, help="motion-prior reward per second at a perfect score")
    args = ap.parse_args()
    import torch, torch.nn as nn
    torch.set_num_threads(2)                 # leave the cores to the simulation workers
    out = os.path.join("runs", "motor", args.name); os.makedirs(out, exist_ok=True)
    env_kw = dict(spread=args.spread, alive=args.alive, fall_cost=args.fall_cost, cost_scale=args.cost_scale)
    ref = make_env(args.family, seed=0, **env_kw)
    ctx = mp.get_context("spawn"); conns = []
    for i in range(args.workers):
        a, b = ctx.Pipe()
        ctx.Process(target=worker, args=(b, args.family, 1000 + i, env_kw, args.kick, bool(args.amp)), daemon=True).start(); conns.append(a)
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
    assist = args.assist0; it0 = 0; max_speed = args.speed0; spread = args.spread; run_assist = args.run_assist0
    if args.init:
        ck = torch.load(args.init, weights_only=False)
        pi.load_state_dict(ck["pi"]); vf.load_state_dict(ck["vf"]); logstd.data = ck["logstd"]
        mean, var, cnt = ck["mean"], ck["var"], ck["cnt"]; assist = ck.get("assist", 0.0); it0 = ck.get("it", 0)
        run_assist = ck.get("run_assist", args.run_assist0)
        max_speed = ck.get("max_speed", ref.MAX_SPEED); spread = ck.get("spread", 1.0)
        if args.reset_curriculum:            # keep the weights, start the harness and the goals over
            assist, max_speed, spread = args.assist0, args.speed0, args.spread
    m_perm = torch.tensor(ref.a_perm); m_sign = torch.tensor(ref.a_sign, dtype=torch.float32)
    disc = None
    if args.amp:
        from motor.amp import Discriminator
        disc = Discriminator(args.amp)
        if args.init and ck.get("disc") is not None:
            disc.load(ck["disc"])

    def export():
        lay = lambda m: [(l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()) for l in m if isinstance(l, nn.Linear)]
        return dict(pi=lay(pi), v=lay(vf), logstd=logstd.detach().numpy().copy())

    def save(it):
        torch.save(dict(pi=pi.state_dict(), vf=vf.state_dict(), logstd=logstd.data, mean=mean, var=var, cnt=cnt, it=it,
                        assist=assist, run_assist=run_assist, max_speed=max_speed, spread=spread, family=args.family, obs_dim=od, act_dim=ad, hidden=args.hidden, args=vars(args),
                        disc=disc.state() if disc else None),
                   os.path.join(out, "ckpt.pt"))

    log = open(os.path.join(out, "log.csv"), "a"); t_start = time.time()
    gamma, lam = ref.GAMMA, 0.95
    cost_names = sorted(ref.ev.totals)
    for it in range(it0, it0 + args.iters):
        t0 = time.time()
        for c in conns: c.send(("assist", assist, spread, max_speed, run_assist))
        for c in conns: c.recv()
        W = export(); norm = (mean, np.sqrt(var + 1e-6).astype(np.float32))
        for c in conns: c.send(("roll", W, norm, args.steps))
        res = [c.recv() for c in conns]
        O = np.concatenate([r[1] for r in res]); A = np.concatenate([r[2] for r in res]); LP = np.concatenate([r[3] for r in res])
        advs, rets = [], []
        amp_s = amp_d = (float("nan"),) * 2
        if disc is not None:                                 # the motion prior: how human each transition looks
            FPs = [r[12] for r in res]
            styles = [disc.reward(fp)[0] for fp in FPs]
            amp_s = float(np.mean(np.concatenate(styles))); amp_d = disc.update(np.concatenate(FPs))
        for k_r, r in enumerate(res):
            rew, dn, v, lastv = r[4], r[5], r[6], r[7]
            if disc is not None:
                rew = rew + args.amp_w * styles[k_r] / ref.CTRL_HZ
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
        TK = torch.tensor(np.concatenate([r[10] for r in res])); TKM = torch.tensor(np.concatenate([r[11] for r in res]))
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
                # keep the mean action inside [-1, 1]: the environment clips actions there, and a mean beyond it gets no
                # gradient from the exploration noise (a walker's arms sat at 1.3 to 1.9, pinned to the end of their
                # range, and never learned to come down after the range ends were made costly)
                # gently and from a bound per family (``ACTION_BOUND``): a bird's full-range strokes sat at the
                # end by design, and a pull of weight 1 from 1.0 took 26 of its 57 outputs off the ends within a round and
                # broke its wingbeat; a walker's arms need to come back inside 1, where the noise changes the command
                # summed over outputs (a mean over all of them diluted the pull on the few out of range about fifty times)
                lbound = (torch.relu(mu.abs() - ref.ACTION_BOUND) ** 2).sum(-1).mean()
                loss = lpi + 0.5 * lv + args.sym * lsym + 0.02 * lbound
                km = TKM[idx]
                if km.any():
                    loss = loss + args.kick_weight * ((mu[km] - TK[idx][km]) ** 2).mean() * km.float().mean()
                if it - it0 < args.vwarm:
                    loss = 0.5 * lv
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 1.0); opt.step()
                if ref.MIN_LOGSTD is not None:                   # keep exploring where the skill is far away
                    with torch.no_grad(): logstd.clamp_(min=ref.MIN_LOGSTD)
                if ref.pc.nm:                                    # muscle excitations explore on their own floor
                    with torch.no_grad(): logstd[len(ref.pc.free):].clamp_(min=ref.MUSCLE_LOGSTD)
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
        run_assist = max(0.0, run_assist - args.run_fade)
        # goal speeds and variety grow slowly while the goal is met (fast growth ran past what the body could learn
        # twice) and shrink back when most episodes fail
        if n_ep and goal > ref.GOAL_OK and fell < ref.FAIL_OK:
            assist = max(0.0, assist - args.fade_ok)
            max_speed = min(ref.MAX_SPEED, max_speed + 0.005)
            spread = min(1.0, spread + 0.003)
        elif n_ep and fell > 0.6:
            max_speed = max(args.speed0, max_speed - 0.01)
            spread = max(args.spread, spread - 0.003)
            if assist < ref.ASSIST_RECOVER:                  # how far the harness comes back is the family's
                assist = min(args.assist0, assist + 0.01)
        costs = " ".join(f"{k}={v:.3f}" for k, v in zip(cost_names, cm) if k not in ("heat", "breath"))
        line = (f"{it} sps={N / (time.time() - t0):.0f} ep_len={el:.0f} failed={fell:.2f} goal={goal:.3f} assist={assist:.2f} run_assist={run_assist:.2f} vmax={max_speed:.2f} spread={spread:.2f} "
                f"std={logstd.exp().mean().item():.3f} sym={sym_l:.4f} {costs}"
                + (f" style={amp_s:.3f} d_real={amp_d[0]:.2f} d_fake={amp_d[1]:.2f}" if disc is not None else "")
                + f" t={(time.time() - t_start) / 60:.1f}m")
        print(line, flush=True); log.write(line + "\n"); log.flush()
        if it % 10 == 9:
            save(it + 1)
        if (time.time() - t_start) / 3600 > args.hours:
            break
    save(it + 1)
    for c in conns: c.send(("stop",))


if __name__ == "__main__":
    main()
