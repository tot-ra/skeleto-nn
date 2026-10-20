"""Carry a trained policy over to a body whose actions changed (new muscles took over joint-motor axes).

    QUAD_MUSCLES=1 python -m motor.graft OLD.pt NEW.pt --family quadruped --new-muscles '^T(LEV|DEP|LAT)\\d'

Actions and observations are matched by name: joint motors as ``bone+axis``, muscles by name. The checkpoint was
trained without the muscles matching ``--new-muscles``; the axes those muscles cover were joint motors then. Kept
actions keep their output rows and their previous-command inputs; a new muscle starts silent (zero weights, so the
policy acts as before while it explores it) and a dropped motor's inputs and outputs are removed.
"""
from __future__ import annotations
import argparse, re
import numpy as np
import torch

from motor.ppo import make_env


def action_names(pc, skip_muscles=None, extra_axes=()):
    """Names of the actions: free joint-motor axes in degree-of-freedom order, then muscles."""
    free = sorted(set(pc.free) | set(extra_axes))
    names = [pc.sk.bones[pc.dof_list[k][0]].name + "XYZ"[pc.dof_list[k][1]] for k in free]
    return names + [n for n in pc.mnames if not (skip_muscles and skip_muscles.match(n))]


def obs_names(env, acts):
    out = []
    for key, sl in env.obs_layout().items():
        n = sl.stop - sl.start
        out += [f"prev:{a}" for a in acts] if key == "prev" else [f"{key}{i}" for i in range(n)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("--family", required=True)
    ap.add_argument("--new-muscles", required=True, help="regex of the muscle names the old checkpoint lacks")
    ap.add_argument("--logstd", type=float, default=float(np.log(0.2)), help="exploration of the new muscles")
    a = ap.parse_args()
    env = make_env(a.family, seed=0); pc = env.pc
    rx = re.compile(a.new_muscles)
    new_m = [n for n in pc.mnames if rx.match(n)]
    # the axes the new muscles cover were joint motors before
    idx = {(pc.sk.bones[i].name, "XYZ"[ax]): k for k, (i, ax) in enumerate(pc.dof_list)}
    was_free = [idx[c] for c in sorted(pc.ms.covered) if any(c[0] == b for b in _bones_of(pc, new_m))]
    old_acts = action_names(pc, rx, was_free)
    new_acts = action_names(pc)
    ck = torch.load(a.old, weights_only=False)
    assert ck["act_dim"] == len(old_acts), (ck["act_dim"], len(old_acts))
    old_obs, new_obs = obs_names(env, old_acts), obs_names(env, new_acts)
    assert ck["obs_dim"] == len(old_obs), (ck["obs_dim"], len(old_obs))
    oi = {n: i for i, n in enumerate(old_obs)}; ai = {n: i for i, n in enumerate(old_acts)}

    def cols(W):                                         # first layer: inputs by name, new ones unconnected
        out = torch.zeros(W.shape[0], len(new_obs), dtype=W.dtype)
        for j, n in enumerate(new_obs):
            if n in oi:
                out[:, j] = W[:, oi[n]]
        return out
    for net in ("pi", "vf"):
        ck[net]["0.weight"] = cols(ck[net]["0.weight"])
    last = max(int(k.split(".")[0]) for k in ck["pi"])
    W, b = ck["pi"][f"{last}.weight"], ck["pi"][f"{last}.bias"]
    W2 = torch.zeros(len(new_acts), W.shape[1], dtype=W.dtype); b2 = torch.zeros(len(new_acts), dtype=b.dtype)
    ls = torch.full((len(new_acts),), a.logstd, dtype=ck["logstd"].dtype)
    for j, n in enumerate(new_acts):
        if n in ai:
            W2[j], b2[j], ls[j] = W[ai[n]], b[ai[n]], ck["logstd"][ai[n]]
    ck["pi"][f"{last}.weight"], ck["pi"][f"{last}.bias"], ck["logstd"] = W2, b2, ls
    mean = np.zeros(len(new_obs), np.float32); var = np.ones(len(new_obs), np.float32)
    for j, n in enumerate(new_obs):
        if n in oi:
            mean[j], var[j] = ck["mean"][oi[n]], ck["var"][oi[n]]
    ck["mean"], ck["var"] = mean, var
    ck["obs_dim"], ck["act_dim"] = len(new_obs), len(new_acts)
    torch.save(ck, a.new)
    gone = [n for n in old_acts if n not in new_acts]
    print(f"actions {len(old_acts)} -> {len(new_acts)}: new {new_m}, motors dropped {gone}; obs {len(old_obs)} -> {len(new_obs)}")


def _bones_of(pc, muscles):
    """Bones whose axes the given muscles cover (every bone along their paths but the first)."""
    out = set()
    for n in muscles:
        path = pc.ms.muscles[pc.mnames.index(n)].path
        bones = [p[1] for p in path if p[0] == "site"]
        out |= set(bones[1:])
    return out


if __name__ == "__main__":
    main()
