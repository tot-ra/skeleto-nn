"""Physical body of a creature in MuJoCo: the same skeleton, torque-limited PD joints, contact with the terrain.
A policy only adds small offsets to the PD targets that the planner's reference pose provides."""
from __future__ import annotations
import math
import numpy as np
import mujoco
from skeleton import Skeleton, AX
from muscles import fl_active, fl_passive, FL_LO, FL_HI, V_MAX, TAU_ACT, TAU_DEACT

CTRL_HZ = 30
SUBSTEPS = 6
DT = 1.0 / (CTRL_HZ * SUBSTEPS)

def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q

VMAX = 20.0       # rad/s: joint speed at which a shortening muscle's force falls to nothing
# curvature of Hill's force-velocity relation (a / F0): at the joint, curves are flatter than a single fibre's 0.25;
# 0.8 keeps about 65 % of the strength at 4 rad/s and 40 % at 8 (a person's knee extension: roughly 60-70 and 40 %)
HILL_A = 0.8
ECC_MAX = 1.4     # a lengthening muscle resists up to 1.4 times its isometric strength


def hill(u):
    """Force share at joint speed ``u`` along the torque, as a share of VMAX (> 0: the muscle shortens, < 0: it is
    stretched): Hill's hyperbola when shortening, rising to ECC_MAX when lengthening."""
    c = np.clip(u, 0.0, 1.0)
    conc = (1.0 - c) / (1.0 + c / HILL_A)
    ecc = ECC_MAX - (ECC_MAX - 1.0) / (1.0 + 7.5 * np.maximum(-u, 0.0))
    return np.where(u >= 0, conc, ecc)


def soft_share(soft_range, b):
    """Share of a joint's range that is soft: by bone name without its side (``foot``), else by bone group."""
    sr = soft_range or {}
    return sr.get(b.name[:-2] if b.name.endswith(("_L", "_R")) else b.name, sr.get(b.group, 0.0))


def build_xml(creature, terrain=None, kp_scale=1.0, floor_friction=1.0, visual=False, terrain_friction=1.0, soft_range=None,
              passive=None, torque=False, muscles=None, fixed_root=None):
    sk = creature.skel
    ms = muscles
    kids = sk.children
    foot_idx = {l.chain[-1] for l in creature.legs}
    hand_idx = {a.chain[-1] for a in creature.arms}
    lines = []; acts = []

    def body(i, ind):
        b = sk.bones[i]; sp = "  " * ind
        if sk.parent[i] < 0:
            pos = "0 0 0" if fixed_root is None else " ".join(f"{x:.4f}" for x in fixed_root)
        else:
            a = sk.attach[i]; pos = f"{a[0]:.5f} {a[1]:.5f} {a[2]:.5f}"
        s = f'{sp}<body name="{b.name}" pos="{pos}">\n'
        if sk.parent[i] < 0 and fixed_root is None:         # (a fixed root: pinned in the air, for demonstrations)
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
        if ms is not None:                                  # muscle attachment sites and wrapping surfaces on this bone
            for w in ms.wraps:
                bn, nm, p, r, hl = w[:5]; kind = w[5] if len(w) > 5 else "cylinder"; za = w[6] if len(w) > 6 else (0, 1, 0)
                if bn == b.name:
                    geo = (f'type="sphere" size="{r:.4f}"' if kind == "sphere" else
                           f'type="cylinder" zaxis="{za[0]} {za[1]} {za[2]}" size="{r:.4f} {hl:.4f}"')
                    s += f'{sp}  <geom name="{nm}" {geo} pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" mass="0" contype="0" conaffinity="0" group="4"/>\n'
            for (bn, nm, p) in ms.sites:
                if bn == b.name:
                    s += f'{sp}  <site name="{nm}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="0.005" group="4"/>\n'
            for mu in ms.muscles:
                for k, it in enumerate(mu.path):
                    if it[0] == "site" and it[1] == b.name:
                        p = it[2]
                        s += f'{sp}  <site name="ms_{mu.name}_{k}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="0.006" group="4"/>\n'
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
                    # flight muscles beat at 3 to 10 Hz: a light armature and a fast response (time constant kv/kp of 20 ms,
                    # not the 80 ms of a limb, which passes only 40 % of a 4.5 Hz wingbeat)
                    wing = b.group == "wing"
                    kp = max(b.tau, 1.0) / 0.35 * kp_scale
                    # the spring of the muscle is explicit (only its damping is implicit), so a wing joint gets at least
                    # the armature that keeps kp dt^2 / inertia below 1: a heron's wing twist (a long bone turning about
                    # itself, very little inertia) was five times past that and blew the simulation up
                    arm = max(0.0003, kp * DT * DT) if wing else 0.002 * (1 + mi)
                    # soft end of range (``soft_range``: bone or bone group -> share of the range): the limit acts from that far
                    # inside it and stiffens toward the stop, as stretched ligaments and the opposing muscles do; a joint
                    # can no longer rest on its stop for free (the arms of a trained walker hung on their stops, held out
                    # backwards). Not for legs: a straight knee or an extended hip locked on its ligaments is how one stands
                    soft = ""
                    share = soft_share(soft_range, b)
                    if share > 0:
                        mg = math.radians(share * (hi - lo))           # margin and width are not converted from degrees
                        soft = f' margin="{mg:.4f}" solreflimit="0.04 1" solimplimit="0.05 0.95 {mg:.4f} 0.5 2"'
                    # passive elastic support (``passive``: bone group -> stiffness in joint strengths per radian), e.g. the
                    # nuchal ligament that carries a horse's head: free of energy, it pulls the joint back to its rest angle
                    stiff = (passive or {}).get(b.group, 0.0) * b.tau
                    s += (f'{sp}  <joint name="j_{b.name}_{ax}" type="hinge" axis="{AX[ax][0]:.0f} {AX[ax][1]:.0f} {AX[ax][2]:.0f}" '
                          f'range="{lo:.2f} {hi:.2f}" limited="true" damping="{0.6 * mi ** 0.8 * 0.2:.4f}" armature="{arm:.5f}" stiffness="{stiff:.3f}"{soft}/>\n')
                    kv = (0.02 if wing else 0.08) * kp
                    if ms is not None and (b.name, ax) in ms.covered:
                        continue                                # moved by line muscles (``muscles.py``), no joint motor
                    if torque:
                        # muscle torque (``torque``): the command is the activation, the force follows it (PhysChar.step
                        # applies the force-velocity relation); a relaxed joint is free, not held at a target angle
                        tn = b.tau * b.tau_dir.get(ax, (1.0, 1.0))[0]; tp = b.tau * b.tau_dir.get(ax, (1.0, 1.0))[1]
                        acts.append(f'    <motor name="a_{b.name}_{ax}" joint="j_{b.name}_{ax}" gear="1" '
                                    f'ctrlrange="{-tn * ECC_MAX:.1f} {tp * ECC_MAX:.1f}" forcerange="{-tn * ECC_MAX:.1f} {tp * ECC_MAX:.1f}"/>\n')
                        continue
                    acts.append(f'    <position name="a_{b.name}_{ax}" joint="j_{b.name}_{ax}" kp="{kp:.2f}" kv="{kv:.3f}" forcerange="{-b.tau * b.tau_dir.get(ax, (1.0, 1.0))[0]:.1f} {b.tau * b.tau_dir.get(ax, (1.0, 1.0))[1]:.1f}" ctrlrange="{math.radians(lo)*1.05:.4f} {math.radians(hi)*1.05:.4f}"/>\n')
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
    tendons = ""
    if ms is not None:
        tendons = "  <tendon>\n"
        for mu in ms.muscles:
            tendons += f'    <spatial name="t_{mu.name}" width="0.006">\n'
            for k, it in enumerate(mu.path):
                tendons += (f'      <site site="ms_{mu.name}_{k}"/>\n' if it[0] == "site" else f'      <geom geom="{it[1]}" sidesite="{it[2]}"/>\n')
            tendons += "    </spatial>\n"
            acts.append(f'    <motor name="m_{mu.name}" tendon="t_{mu.name}" gear="1"/>\n')
        tendons += "  </tendon>\n"
    xml = f'''<mujoco model="{sk.name}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{DT:.6f}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <default><geom solref="0.02 1" solimp="0.9 0.95 0.001"/></default>
  <worldbody>
    {ground}
{ter}
{root}  </worldbody>
{tendons}  <actuator>
{''.join(acts)}  </actuator>
</mujoco>'''
    return xml

class PhysChar:
    def __init__(self, creature, terrain=None, kp_scale=1.0, friction=1.0, soft_range=None, substeps=SUBSTEPS, passive=None,
                 springs=None, torque=False, muscles=None, fixed_root=None):
        self.c = creature; self.sk = creature.skel
        self.ms = muscles; self.strength = 1.0             # muscles: a ``muscles.MuscleSet``; strength: fatigue scaling
        self.substeps = substeps                             # physics steps per control step (the physics step DT is fixed)
        self.torque = torque                                 # step() takes activations in [-1, 1], not target angles
        self.model = mujoco.MjModel.from_xml_string(build_xml(creature, terrain, kp_scale, floor_friction=friction,
                                                              terrain_friction=friction, soft_range=soft_range, passive=passive,
                                                              torque=torque, muscles=muscles, fixed_root=fixed_root))
        self.fixed = fixed_root is not None
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
                    if muscles is None or (b.name, ax) not in muscles.covered:
                        self.act[(i, "XYZ".index(ax))] = m.actuator(f"a_{b.name}_{ax}").id
                    self.dof_list.append((i, "XYZ".index(ax)))
        self.nj = len(self.dof_list)
        self.dof_bone = np.array([d[0] for d in self.dof_list]); self.dof_ax = np.array([d[1] for d in self.dof_list])
        self.qadr = np.array([self.jq[d] for d in self.dof_list]); self.vadr = np.array([self.jv[d] for d in self.dof_list])
        # joint axes with a joint motor (all of them unless line muscles move some): ``aid`` lists their actuators
        self.free = np.array([k for k, d in enumerate(self.dof_list) if d in self.act], int)
        self.jmotor = np.zeros(self.nj, bool); self.jmotor[self.free] = True
        self.aid = np.array([self.act[self.dof_list[k]] for k in self.free], int)
        self.lo = np.array([math.radians(sk.bones[i].lim["XYZ"[a]][0]) for i, a in self.dof_list])
        self.hi = np.array([math.radians(sk.bones[i].lim["XYZ"[a]][1]) for i, a in self.dof_list])
        self.tau = np.array([sk.bones[i].tau for i, a in self.dof_list])
        # strength toward negative and positive angles (Bone.tau_dir; equal unless given)
        self.tau_neg = np.array([sk.bones[i].tau * sk.bones[i].tau_dir.get("XYZ"[a], (1.0, 1.0))[0] for i, a in self.dof_list])
        self.tau_pos = np.array([sk.bones[i].tau * sk.bones[i].tau_dir.get("XYZ"[a], (1.0, 1.0))[1] for i, a in self.dof_list])
        # joints with a soft end of range (their limit load is a ligament load, scored as pain)
        self.soft_mask = np.array([soft_share(soft_range, sk.bones[i]) > 0 for i, a in self.dof_list])
        self.foot_geoms = [m.geom("g_" + sk.bones[l.chain[-1]].name).id for l in creature.legs]
        self.hand_geoms = [m.geom("g_" + sk.bones[a.chain[-1]].name).id for a in creature.arms]
        self.touch_ok = set()                                # geoms whose ground contact is not a fall (a tail, a grazing head)
        self.leg_sites = [(m.site(f"s_{l.name}_heel").id, m.site(f"s_{l.name}_ball").id) for l in creature.legs]
        self.total_mass = float(m.body_subtreemass[self.bid[0]])
        self.mass_vec = m.body_mass.copy()
        # one-sided elastic springs (``springs``: bone name prefix -> (axis, stiffness in joint strengths per radian, slack
        # angle in radians)): taut only below the slack angle, e.g. the Achilles tendon stretched as the shank tips over the
        # planted foot, which gives the energy back in the push-off; free of energy, outside the muscles' strength
        sel = [(k, kk, sl) for k, (i, a) in enumerate(self.dof_list) for pre, (ax, kk, sl) in (springs or {}).items()
               if sk.bones[i].name.startswith(pre) and "XYZ"[a] == ax]
        self.sp_q = self.qadr[[s[0] for s in sel]]; self.sp_v = self.vadr[[s[0] for s in sel]]
        self.sp_k = np.array([s[1] * self.tau[s[0]] for s in sel]); self.sp_slack = np.array([s[2] for s in sel])
        self.nm = 0
        if muscles is not None:
            self._init_muscles()
        self.n_act = len(self.free) + self.nm                # step() takes joint activations, then muscle excitations

    def _init_muscles(self):
        """Muscle tables: tendon and actuator ids, fibre-length normalisation from the joint ranges each muscle spans,
        and the strength F0 that gives its share of the primary joint's strength standing (``muscles.py``)."""
        m = self.model; d = mujoco.MjData(m); sk = self.sk; ms = self.ms.muscles
        self.nm = len(ms); self.mnames = [mu.name for mu in ms]
        self.tid = np.array([m.tendon(f"t_{mu.name}").id for mu in ms]); self.mid = np.array([m.actuator(f"m_{mu.name}").id for mu in ms])
        self.ret_m = np.array([mu.tendon_return for mu in ms])
        index = {(sk.bones[i].name, "XYZ"[a]): k for k, (i, a) in enumerate(self.dof_list)}
        cov = [index[c] for c in sorted(self.ms.covered)]

        def lengths(q):
            d.qpos[:] = m.qpos0; d.qpos[self.qadr] = q; mujoco.mj_fwdPosition(m, d); return d.ten_length[self.tid].copy()
        q0 = np.zeros(self.nj); l0 = lengths(q0)
        lmin = l0.copy(); lmax = l0.copy(); arm = np.zeros((self.nm, self.nj))
        for k in cov:                                      # one joint axis at a time; lengths add up across joints
            lo_l = np.zeros(self.nm); hi_l = np.zeros(self.nm)
            for x in np.linspace(self.lo[k], self.hi[k], 13):
                q = q0.copy(); q[k] = x; dl = lengths(q) - l0
                lo_l = np.minimum(lo_l, dl); hi_l = np.maximum(hi_l, dl)
            lmin += lo_l; lmax += hi_l
            q = q0.copy(); q[k] = 1e-3; arm[:, k] = -(lengths(q) - l0) / 1e-3      # torque per newton of pull
        self.arm0 = arm                                    # moment arms standing (m), per muscle and joint axis
        self.lopt = np.maximum(lmax - lmin, 1e-3) / (FL_HI - FL_LO)
        self.loff = lmin - FL_LO * self.lopt
        self.F0 = np.zeros(self.nm); self.r0 = np.zeros(self.nm); self.m_bad = []
        for j, mu in enumerate(ms):
            bn, ax, sg = mu.primary; k = index[(bn, ax)]; b = sk.bones[sk.idx[bn]]
            tdir = b.tau * b.tau_dir.get(ax, (1.0, 1.0))[1 if sg > 0 else 0]
            r = sg * arm[j, k]
            if r < 1e-4:                                   # a bad attachment: muscle_check.py lists them, tests fail
                self.m_bad.append((mu.name, r)); r = 1e-3
            # strength is a joint's peak torque: F0 so that the best moment arm times force-length over the range
            # gives the muscle's share of it (a straight knee's extensors are weaker, as they are in a person)
            peak = 0.0
            for x in np.linspace(self.lo[k], self.hi[k], 13):
                q = q0.copy(); q[k] = x; ll = lengths(q)[j]
                qq = q.copy(); qq[k] = x + 1e-3; rr = -sg * (lengths(qq)[j] - ll) / 1e-3
                peak = max(peak, rr * fl_active((ll - self.loff[j]) / self.lopt[j]))
            self.F0[j] = mu.share * tdir / max(peak, 3e-4); self.r0[j] = r
        # summed strength per group of muscles (one side), when the set gives it: a leg's muscles in body weights
        for g, bw in (self.ms.targets or {}).items():
            sel = np.array([mu.group == g and mu.side == "L" for mu in ms])
            if sel.any():
                k = bw * self.total_mass * 9.81 / self.F0[sel].sum()
                self.F0[np.array([mu.group == g for mu in ms])] *= k
        self.m_act = np.zeros(self.nm); self.m_exc = np.zeros(self.nm); self.m_fact = np.zeros(self.nm)
        # drawn thickness: a muscle's strength goes with its cross-section, so its diameter with the square root of F0
        # (against the body weight, and in proportion to the body's size)
        self.m_thick = 0.007 * (self.c.leg_len() / 0.86) * np.sqrt(self.F0 / (self.total_mass * 9.81))

    def muscle_state(self):
        """Normalised fibre length and shortening speed (in V_MAX) of every muscle (rigid tendons)."""
        d = self.data
        lt = (d.ten_length[self.tid] - self.loff) / self.lopt
        u = -d.ten_velocity[self.tid] / (V_MAX * self.lopt)
        return lt, u

    def muscle_paths(self):
        """World points of every muscle's path (origin, wrap tangent points, via points, insertion)."""
        d = self.data; out = []
        for t in self.tid:
            a, n = int(d.ten_wrapadr[t]), int(d.ten_wrapnum[t])
            pts = d.wrap_xpos.reshape(-1, 3)[a:a + n].copy()
            pts = pts[np.linalg.norm(pts, axis=1) > 0]       # pulleys and unwrapped surfaces leave zeros
            out.append(pts)
        return out

    # ---- state transfer ----------------------------------------------------------------
    def q_to_vec(self, q):
        return q[self.dof_bone, self.dof_ax]

    def vec_to_q(self, v):
        q = np.zeros((self.sk.n, 3)); q[self.dof_bone, self.dof_ax] = v; return q

    def set_state(self, root_pos, root_R, qvec, root_vel=None, root_angvel=None, qd=None):
        d = self.data; m = self.model
        mujoco.mj_resetData(m, d)
        if not self.fixed:
            d.qpos[0:3] = root_pos; d.qpos[3:7] = mat2quat(root_R)
        d.qpos[self.qadr] = qvec
        if root_vel is not None and not self.fixed: d.qvel[0:3] = root_vel
        if root_angvel is not None and not self.fixed: d.qvel[3:6] = root_R.T @ root_angvel   # free joint angular velocity is in the body frame
        if qd is not None: d.qvel[self.vadr] = qd
        d.ctrl[self.aid] = 0.0 if self.torque else qvec[self.free]
        if self.nm:
            d.ctrl[self.mid] = 0.0; self.m_act[:] = 0.0; self.m_exc[:] = 0.0; self.m_fact[:] = 0.0
        mujoco.mj_forward(m, d)

    def step(self, target_vec, ext_force=None, ext_torque=None, wrench=None, probe=None, pre=None):
        d = self.data; m = self.model
        if self.nm:
            self.m_exc = np.clip(np.asarray(target_vec[len(self.free):], float), 0.0, 1.0)   # muscles only pull
            target_vec = target_vec[:len(self.free)]
            h = m.opt.timestep
            k_up, k_down = 1.0 - math.exp(-h / TAU_ACT), 1.0 - math.exp(-h / TAU_DEACT)
        if self.torque:
            a = np.clip(target_vec, -1.0, 1.0)
            # signed isometric torque the activation asks for; the force range (ECC_MAX times the strength) also carries
            # the environment's fatigue scaling
            fr = m.actuator_forcerange[self.aid]
            cap = np.where(a >= 0, fr[:, 1], -fr[:, 0]) / ECC_MAX * a
        else:
            d.ctrl[self.aid] = np.clip(target_vec, self.lo[self.free] * 1.05, self.hi[self.free] * 1.05)
        d.xfrc_applied[:] = 0
        for _ in range(self.substeps):
            if self.torque:                                  # force-velocity: weaker while shortening, stronger stretched
                d.ctrl[self.aid] = cap * hill(np.sign(cap) * d.qvel[self.vadr[self.free]] / VMAX)
            if self.nm:                                      # Hill muscles: activation, force-length, force-velocity, passive
                self.m_act += (self.m_exc - self.m_act) * np.where(self.m_exc > self.m_act, k_up, k_down)
                lt, u = self.muscle_state()
                self.m_fact = self.strength * self.F0 * self.m_act * fl_active(lt) * hill(u)
                d.ctrl[self.mid] = -(self.m_fact + self.F0 * fl_passive(lt))         # a tendon actuator pushes when > 0
            if wrench is not None:
                d.xfrc_applied[self.bid[0], :] = wrench
            if ext_force is not None:
                d.xfrc_applied[self.bid[1 if len(self.bid) > 1 else 0], 0:3] = ext_force
            if pre is not None: pre()
            if len(self.sp_k):
                if pre is None: d.qfrc_applied[self.sp_v] = 0.0        # a ``pre`` hook clears the applied forces itself
                d.qfrc_applied[self.sp_v] += self.sp_k * np.maximum(self.sp_slack - d.qpos[self.sp_q], 0.0)
            mujoco.mj_step(m, d)
            if probe is not None: probe()
        if ext_force is not None: d.xfrc_applied[:] = 0

    # ---- measurements ------------------------------------------------------------------
    def root_pose(self):
        d = self.data
        if self.fixed:
            return d.xpos[self.bid[0]].copy(), d.xmat[self.bid[0]].reshape(3, 3).copy()
        return d.qpos[0:3].copy(), d.xmat[self.bid[0]].reshape(3, 3).copy()

    def limit_torque(self):
        """Torque of the joint-limit constraints per joint axis (the load on ligaments at the end of the range)."""
        d = self.data; m = self.model; out = np.zeros(self.nj)
        if not hasattr(self, "_jdof"):
            self._jdof = {int(m.dof_jntid[a]): k for k, a in enumerate(self.vadr)}
        lim = np.nonzero(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)[0]
        for e in lim:
            k = self._jdof.get(int(d.efc_id[e]))
            if k is not None:
                out[k] += abs(float(d.efc_force[e]))
        return out

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
        ok = set(self.foot_geoms) | set(self.hand_geoms) | self.touch_ok
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
