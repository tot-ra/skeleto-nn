"""Physical body of a creature in MuJoCo: the same skeleton, torque-limited PD joints, contact with the terrain.
A policy only adds small offsets to the PD targets that the planner's reference pose provides."""
from __future__ import annotations
import math
import numpy as np
import mujoco
from skeleton import Skeleton, AX

CTRL_HZ = 30
SUBSTEPS = 6
DT = 1.0 / (CTRL_HZ * SUBSTEPS)

def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q

def build_xml(creature, terrain=None, kp_scale=1.0, floor_friction=1.0, visual=False, terrain_friction=1.0):
    sk = creature.skel
    kids = sk.children
    foot_idx = {l.chain[-1] for l in creature.legs}
    hand_idx = {a.chain[-1] for a in creature.arms}
    lines = []; acts = []

    def body(i, ind):
        b = sk.bones[i]; sp = "  " * ind
        if sk.parent[i] < 0:
            pos = "0 0 0"
        else:
            a = sk.attach[i]; pos = f"{a[0]:.5f} {a[1]:.5f} {a[2]:.5f}"
        s = f'{sp}<body name="{b.name}" pos="{pos}">\n'
        if sk.parent[i] < 0:
            s += f'{sp}  <freejoint name="root"/>\n'
        mi = sk.mass[i]
        d = sk.dir[i] * sk.length[i]
        col = '' if not visual else ' rgba="0.6 0.62 0.68 1"'
        if i in foot_idx and b.shape == "box" and b.size:
            hx, hy, hz = b.size
            zc = next(l.heel[2] for l in creature.legs if l.chain[-1] == i) + hz
            s += f'{sp}  <geom name="g_{b.name}" type="box" pos="{d[0]/2:.4f} 0 {zc:.4f}" size="{hx:.4f} {hy:.4f} {hz:.4f}" mass="{mi:.4f}" friction="{floor_friction} 0.01 0.001" contype="2" conaffinity="0"{col}/>\n'
        else:
            gname = f"g_{b.name}"
            s += f'{sp}  <geom name="{gname}" type="capsule" fromto="0 0 0 {d[0]:.5f} {d[1]:.5f} {d[2]:.5f}" size="{max(b.radius, 0.008):.4f}" mass="{mi:.4f}" friction="{floor_friction} 0.01 0.001" contype="2" conaffinity="0"{col}/>\n'
        for (bn, t, off, kg, lab) in sk.extra:
            if sk.idx[bn] == i:
                p = d * t + np.array(off)
                if lab == "belly":                                # a carried belly can touch the ground: it gets a collision sphere, named so that its load is scored
                    s += f'{sp}  <geom name="g_belly" type="sphere" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="{0.09 + 0.012 * kg:.4f}" mass="{kg:.3f}" contype="2" conaffinity="0"/>\n'
                else:
                    s += f'{sp}  <geom type="sphere" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="0.04" mass="{kg:.3f}" contype="0" conaffinity="0"/>\n'
        # foot contact sites (heel, ball) so we can read their world position
        for leg in creature.legs:
            if leg.chain[-1] == i:
                for nm, pt in (("heel", leg.heel), ("ball", leg.ball)):
                    s += f'{sp}  <site name="s_{leg.name}_{nm}" pos="{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}" size="0.012"/>\n'
        if sk.parent[i] >= 0:
            order = sk.axes_order[i]
            for ax in order:
                if ax in b.lim:
                    lo, hi = b.lim[ax]
                    s += (f'{sp}  <joint name="j_{b.name}_{ax}" type="hinge" axis="{AX[ax][0]:.0f} {AX[ax][1]:.0f} {AX[ax][2]:.0f}" '
                          f'range="{lo:.2f} {hi:.2f}" limited="true" damping="{0.6 * mi ** 0.8 * 0.2:.4f}" armature="{0.002 * (1 + mi):.5f}" stiffness="0"/>\n')
                    kp = max(b.tau, 1.0) / 0.35 * kp_scale
                    kv = 0.08 * kp
                    acts.append(f'    <position name="a_{b.name}_{ax}" joint="j_{b.name}_{ax}" kp="{kp:.2f}" kv="{kv:.3f}" forcerange="{-b.tau:.1f} {b.tau:.1f}" ctrlrange="{math.radians(lo)*1.05:.4f} {math.radians(hi)*1.05:.4f}"/>\n')
        for c in kids[i]:
            s += body(c, ind + 1)
        s += f'{sp}</body>\n'
        return s

    root = body(0, 2)
    ter = ""
    ground = f'<geom name="floor" type="plane" size="300 300 0.1" contype="1" conaffinity="2" friction="{terrain_friction} 0.01 0.001"/>'
    if terrain is not None:
        ground = ground  # sloped ground is built from ramps
        for g in terrain.mjcf_geoms():
            ter += "    " + g.replace('contype="1" conaffinity="1"', f'contype="1" conaffinity="2" friction="{terrain_friction} 0.01 0.001"') + "\n"
    xml = f'''<mujoco model="{sk.name}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{DT:.6f}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <default><geom solref="0.02 1" solimp="0.9 0.95 0.001"/></default>
  <worldbody>
    {ground}
{ter}
{root}  </worldbody>
  <actuator>
{''.join(acts)}  </actuator>
</mujoco>'''
    return xml

class PhysChar:
    def __init__(self, creature, terrain=None, kp_scale=1.0, friction=1.0):
        self.c = creature; self.sk = creature.skel
        self.model = mujoco.MjModel.from_xml_string(build_xml(creature, terrain, kp_scale, floor_friction=friction, terrain_friction=friction))
        self.data = mujoco.MjData(self.model)
        m = self.model; sk = self.sk
        self.bid = [m.body(b.name).id for b in sk.bones]
        self.floor = m.geom("floor").id
        # joint tables
        self.jq = {}; self.jv = {}; self.act = {}
        self.dof_list = []     # (bone, axis index)
        for i, b in enumerate(sk.bones):
            if sk.parent[i] < 0: continue
            for ax in "XYZ":
                if ax in b.lim:
                    jid = m.joint(f"j_{b.name}_{ax}").id
                    self.jq[(i, "XYZ".index(ax))] = m.jnt_qposadr[jid]
                    self.jv[(i, "XYZ".index(ax))] = m.jnt_dofadr[jid]
                    self.act[(i, "XYZ".index(ax))] = m.actuator(f"a_{b.name}_{ax}").id
                    self.dof_list.append((i, "XYZ".index(ax)))
        self.nj = len(self.dof_list)
        self.dof_bone = np.array([d[0] for d in self.dof_list]); self.dof_ax = np.array([d[1] for d in self.dof_list])
        self.qadr = np.array([self.jq[d] for d in self.dof_list]); self.vadr = np.array([self.jv[d] for d in self.dof_list])
        self.aid = np.array([self.act[d] for d in self.dof_list])
        self.lo = np.array([math.radians(sk.bones[i].lim["XYZ"[a]][0]) for i, a in self.dof_list])
        self.hi = np.array([math.radians(sk.bones[i].lim["XYZ"[a]][1]) for i, a in self.dof_list])
        self.tau = np.array([sk.bones[i].tau for i, a in self.dof_list])
        self.foot_geoms = [m.geom("g_" + sk.bones[l.chain[-1]].name).id for l in creature.legs]
        self.hand_geoms = [m.geom("g_" + sk.bones[a.chain[-1]].name).id for a in creature.arms]
        self.leg_sites = [(m.site(f"s_{l.name}_heel").id, m.site(f"s_{l.name}_ball").id) for l in creature.legs]
        self.total_mass = float(m.body_subtreemass[self.bid[0]])
        self.mass_vec = m.body_mass.copy()

    # ---- state transfer ----------------------------------------------------------------
    def q_to_vec(self, q):
        return q[self.dof_bone, self.dof_ax]

    def vec_to_q(self, v):
        q = np.zeros((self.sk.n, 3)); q[self.dof_bone, self.dof_ax] = v; return q

    def set_state(self, root_pos, root_R, qvec, root_vel=None, root_angvel=None, qd=None):
        d = self.data; m = self.model
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = root_pos; d.qpos[3:7] = mat2quat(root_R)
        d.qpos[self.qadr] = qvec
        if root_vel is not None: d.qvel[0:3] = root_vel
        if root_angvel is not None: d.qvel[3:6] = root_R.T @ root_angvel   # free joint angular velocity is in the body frame
        if qd is not None: d.qvel[self.vadr] = qd
        d.ctrl[self.aid] = qvec
        mujoco.mj_forward(m, d)

    def step(self, target_vec, ext_force=None, ext_torque=None, wrench=None, probe=None, pre=None):
        d = self.data; m = self.model
        d.ctrl[self.aid] = np.clip(target_vec, self.lo * 1.05, self.hi * 1.05)
        d.xfrc_applied[:] = 0
        for _ in range(SUBSTEPS):
            if wrench is not None:
                d.xfrc_applied[self.bid[0], :] = wrench
            if ext_force is not None:
                d.xfrc_applied[self.bid[1 if len(self.bid) > 1 else 0], 0:3] = ext_force
            if pre is not None: pre()
            mujoco.mj_step(m, d)
            if probe is not None: probe()
        if ext_force is not None: d.xfrc_applied[:] = 0

    # ---- measurements ------------------------------------------------------------------
    def root_pose(self):
        d = self.data
        return d.qpos[0:3].copy(), d.xmat[self.bid[0]].reshape(3, 3).copy()

    def qvec(self): return self.data.qpos[self.qadr].copy()
    def qdvec(self): return self.data.qvel[self.vadr].copy()

    def foot_contacts(self):
        d = self.data; con = np.zeros(len(self.foot_geoms))
        for k in range(d.ncon):
            c = d.contact[k]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if b in self.foot_geoms and a != b and self.model.geom_contype[a] == 1:
                    con[self.foot_geoms.index(b)] = 1.0
        return con

    def body_impact(self):
        """Normal force (in body weights) through every part of the body except the feet, summed: a proxy for pain/bone load in training."""
        d = self.data; m = self.model; ok = set(self.foot_geoms); f6 = np.zeros(6); tot = 0.0
        for k in range(d.ncon):
            c = d.contact[k]; g1, g2 = c.geom1, c.geom2
            if m.geom_contype[g1] == 1 and m.geom_contype[g2] == 2 and g1 not in ok:
                mujoco.mj_contactForce(m, d, k, f6); tot += abs(f6[0])
            elif m.geom_contype[g2] == 1 and m.geom_contype[g1] == 2 and g2 not in ok:
                mujoco.mj_contactForce(m, d, k, f6); tot += abs(f6[0])
        return tot / (self.total_mass * 9.81)

    def bad_contact(self):
        """Any body part other than feet (and hands, which may brace) touching the terrain."""
        d = self.data; m = self.model
        ok = set(self.foot_geoms) | set(self.hand_geoms)
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = c.geom1, c.geom2
            if m.geom_contype[g1] == 1 and m.geom_contype[g2] == 2 and g2 not in ok: return True
            if m.geom_contype[g2] == 1 and m.geom_contype[g1] == 2 and g1 not in ok: return True
        return False

    def foot_points(self):
        d = self.data
        return np.array([[d.site_xpos[h].copy(), d.site_xpos[b].copy()] for h, b in self.leg_sites])

    def bone_ends(self):
        """World start and end of every bone (matches Skeleton.fk)."""
        sk = self.sk; d = self.data
        S = np.array([d.xpos[b] for b in self.bid])
        R = np.array([d.xmat[b].reshape(3, 3) for b in self.bid])
        E = S + np.einsum("nij,nj->ni", R, sk.dir * sk.length[:, None])
        return S, E, R

    def com(self):
        d = self.data
        return (d.xipos * self.mass_vec[:, None]).sum(0) / self.mass_vec.sum()
