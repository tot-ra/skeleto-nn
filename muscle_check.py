"""Check the leg muscles' attachments: moment arms (cm) over each joint's range, strength, fibre lengths.

    python muscle_check.py            a table per muscle: moment arm at the primary joint across its range

A negative arm in the primary column means the muscle pulls the wrong way somewhere in the range (a bad attachment).
"""
from __future__ import annotations
import numpy as np, mujoco
from bodies import human
from muscles import human_legs
from physics import PhysChar

COLS = [("thigh_L", "Y", "hip flex(-)/ext(+)"), ("thigh_L", "X", "hip abd(+)"), ("shank_L", "Y", "knee flex(+)"), ("foot_L", "Y", "ankle plantar(+)")]


def arms_at(pc, q):
    m, d = pc.model, mujoco.MjData(pc.model)
    def L(qq):
        d.qpos[:] = m.qpos0; d.qpos[pc.qadr] = qq; mujoco.mj_fwdPosition(m, d); return d.ten_length[pc.tid].copy()
    l0 = L(q); out = np.zeros((pc.nm, pc.nj))
    for k in range(pc.nj):
        qq = q.copy(); qq[k] += 1e-3; out[:, k] = -(L(qq) - l0) / 1e-3
    return out, l0


def main():
    c = human(); pc = PhysChar(c, torque=True, muscles=human_legs(c))
    idx = {(pc.sk.bones[i].name, "XYZ"[a]): k for k, (i, a) in enumerate(pc.dof_list)}
    print(f"{pc.nm} muscles, {len(pc.free)} joint motors left, {pc.n_act} actions")
    print("standing moment arms (cm):", "  ".join(c_[2] for c_ in COLS))
    for j, n in enumerate(pc.mnames[:pc.nm // 2]):
        print(f"  {n:6s} F0 {pc.F0[j]:6.0f} N  fibre {100 * pc.lopt[j]:4.1f} cm  ", " ".join(f"{100 * pc.arm0[j, idx[c_[:2]]]:6.2f}" for c_ in COLS))
    print("primary moment arm over the joint range (cm), and normalised fibre length at the ends:")
    for j, mu in enumerate(pc.ms.muscles[:pc.nm // 2]):
        bn, ax, sg = mu.primary; k = idx[(bn, ax)]
        row = []; lts = []
        for x in np.linspace(pc.lo[k], pc.hi[k], 7):
            q = np.zeros(pc.nj); q[k] = x; a, l = arms_at(pc, q)
            row.append(sg * a[j, k] * 100); lts.append((l[j] - pc.loff[j]) / pc.lopt[j])
        bad = "  <-- WRONG WAY" if min(row) < 0 else ""
        print(f"  {mu.name:6s} {bn} {ax}: " + " ".join(f"{r:5.1f}" for r in row) + f"   l {lts[0]:.2f}..{lts[-1]:.2f}{bad}")


if __name__ == "__main__":
    main()
