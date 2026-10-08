"""Blender (headless): pose the shipped character rig from exported procedural motion and render PNG frames.

blender -b --python retarget_blender.py -- <character.glb> <clip.json> <out_dir> [every_n_frames] [width height]

Trunk, legs and feet take the world-rotation *delta* from the planner's skeleton (it has the rig's proportions);
the A-posed arms take only the direction of each segment. Everything else follows its parent rigidly.
"""
import bpy, sys, json, math, os
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb, clip_path, outdir = argv[0], argv[1], argv[2]
every = int(argv[3]) if len(argv) > 3 else 1
W, H = (int(argv[4]), int(argv[5])) if len(argv) > 5 else (480, 300)
os.makedirs(outdir, exist_ok=True)
clip = json.load(open(clip_path))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
for o in list(bpy.data.objects):
    if o.name.startswith("Icosphere"): bpy.data.objects.remove(o, do_unlink=True)
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
bpy.context.view_layer.objects.active = arm
arm.animation_data_clear()
bpy.ops.object.mode_set(mode="POSE")

Hg = Matrix.Rotation(-math.pi / 2, 3, "Z")        # creature +X (forward) -> rig -Y (forward)
rest = {}
for b in arm.data.bones:
    rest[b.name] = (b.matrix_local.to_3x3().copy(), b.head_local.copy(), b.parent.name if b.parent else None)
order = []
def visit(n):
    if n in order: return
    p = rest[n][2]
    if p: visit(p)
    order.append(n)
for n in rest: visit(n)

MAP = {"hips": ("pelvis", "delta"), "spine": ("lumbar", "delta"), "chest": ("chest", "delta"), "head": ("head", "delta"),
       "upperleg.l": ("thigh_L", "delta"), "lowerleg.l": ("shank_L", "delta"), "foot.l": ("foot_L", "delta"),
       "upperleg.r": ("thigh_R", "delta"), "lowerleg.r": ("shank_R", "delta"), "foot.r": ("foot_R", "delta"),
       "upperarm.l": ("uarm_L", "dir"), "lowerarm.l": ("farm_L", "dir"), "upperarm.r": ("uarm_R", "dir"), "lowerarm.r": ("farm_R", "dir")}

def M3(flat): return Matrix([flat[0:3], flat[3:6], flat[6:9]])

def align(a, b):
    a = a.normalized(); b = b.normalized()
    return a.rotation_difference(b).to_matrix()

for m in bpy.data.materials:
    if m.use_nodes and m.node_tree:
        for n in m.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED":
                bc = n.inputs["Base Color"].default_value
                m.diffuse_color = (bc[0], bc[1], bc[2], 1.0)
                break

# ---- scene -------------------------------------------------------------------
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.image_settings.file_format = "PNG"
sh = scene.display.shading
sh.light = "STUDIO"; sh.color_type = "TEXTURE"; sh.show_shadows = True; sh.shadow_intensity = 0.35; sh.show_object_outline = False
sh.show_cavity = False
try: scene.display.render_aa = "8"
except Exception: pass
world = bpy.data.worlds.new("w"); world.color = (0.66, 0.80, 0.94); scene.world = world
sh.background_type = "WORLD" if hasattr(sh, "background_type") else None

def make_mat(name, rgba):
    m = bpy.data.materials.new(name); m.diffuse_color = rgba; return m

def box(x0, x1, y0, y1, zb, zt, rgba, name="box"):
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.active_object; o.name = name
    # clip coords are creature frame (x forward, y left); the rig faces -Y, so rotate by Hg
    c = Hg @ Vector(((x0 + x1) / 2, (y0 + y1) / 2, (zb + zt) / 2))
    sx, sy = abs(x1 - x0), abs(y1 - y0)
    o.location = c; o.rotation_euler = (0, 0, -math.pi / 2); o.scale = (sx, sy, zt - zb)
    o.data.materials.append(make_mat(name, tuple(rgba)))
    return o

pits = []
for t in clip["terrain"]:
    a = t["a"]
    if t["kind"] == "box": box(a[0], a[1], a[2], a[3], a[4], a[5], t["rgba"], "stair")
    elif t["kind"] == "beam": box(a[0], a[1], a[2], a[3], a[4], a[5], t["rgba"], "beam")
    elif t["kind"] == "pit": pits.append(a); box(a[0], a[1], a[2], a[3], 0.0, 0.012, (0.12, 0.10, 0.09, 1), "pit")
    elif t["kind"] == "ramp":
        x0, x1, y0, y1, z0, z1 = a
        L = math.hypot(x1 - x0, z1 - z0); ang = math.atan2(z1 - z0, x1 - x0)
        bpy.ops.mesh.primitive_cube_add(size=1); o = bpy.context.active_object
        c = Hg @ Vector(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2 - 0.1))
        o.location = c; o.scale = (L, abs(y1 - y0), 0.2)
        o.rotation_euler = Matrix.Rotation(-math.pi / 2, 3, "Z").to_euler()
        # slope about the (rotated) local y axis
        o.rotation_mode = "XYZ"
        o.rotation_euler = (0, 0, 0)
        R = Hg @ Matrix.Rotation(-ang, 3, "Y")
        o.rotation_euler = R.to_euler()
        o.data.materials.append(make_mat("ramp", tuple(t["rgba"])))

door = clip.get("door")
door_obj = None
if door:
    bpy.ops.mesh.primitive_cube_add(size=1); door_obj = bpy.context.active_object; door_obj.name = "door"
    door_obj.scale = (door["thick"], door["width"], door["height"]); door_obj.data.materials.append(make_mat("door", (0.5, 0.33, 0.18, 1)))
# checkered floor
import bmesh
def floor(color, parity):
    me = bpy.data.meshes.new("f"); bm = bmesh.new()
    for i in range(-30, 30):
        for j in range(-30, 30):
            if (i + j) % 2 != parity: continue
            cx, cy = i * 2.0 + 1.0, j * 2.0 + 1.0
            inpit = False and any((Hg.inverted() @ Vector((cx, cy, 0)))[0] > p[0] - 1 and (Hg.inverted() @ Vector((cx, cy, 0)))[0] < p[1] + 1 and abs((Hg.inverted() @ Vector((cx, cy, 0)))[1]) < p[3] + 1 for p in pits) if pits else False
            if inpit: continue
            vs = [bm.verts.new((cx + dx, cy + dy, 0.0)) for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            bm.faces.new(vs)
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("floor", me); scene.collection.objects.link(o)
    o.data.materials.append(make_mat("floor", color))
floor((0.80, 0.78, 0.70, 1), 0); floor((0.70, 0.68, 0.60, 1), 1)
# pits: carve by dropping the floor under them (tiles skipped above)

cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam
cam_data.lens = 40
tgt = bpy.data.objects.new("tgt", None); scene.collection.objects.link(tgt)
con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"

z0 = clip["z0"]
hips_rest = rest["hips"][1]
dist = float(os.environ.get("CAM_DIST", "4.2")); az = math.radians(float(os.environ.get("CAM_AZ", "20")))
for fi, fr in enumerate(clip["frames"]):
    if fi % every: continue
    Rt = {}; Pt = {}
    root = Vector(fr["root"])
    for n in order:
        Rr, hr, par = rest[n]
        if par is None:
            R_new = Rr.copy(); H_new = hr.copy()
        else:
            Rp_new, Rp_rest, Hp_new, Hp_rest = Rt[par], rest[par][0], Pt[par], rest[par][1]
            H_new = Hp_new + Rp_new @ (Rp_rest.inverted() @ (hr - Hp_rest))
            R_new = Rp_new @ (Rp_rest.inverted() @ Rr)
        if n in MAP:
            on, mode = MAP[n]; Ro = M3(fr["R"][on])
            if mode == "delta":
                R_new = (Hg @ Ro @ Hg.inverted()) @ Rr
            else:
                d_t = Hg @ (Ro @ Vector((0, 0, -1)))
                d_g = Rr @ Vector((0, 1, 0))
                R_new = align(d_g, d_t) @ Rr
        if n == "hips":
            H_new = hr + Hg @ (root - Vector((0, 0, z0)))
        Rt[n] = R_new; Pt[n] = H_new
        pb = arm.pose.bones[n]
        pb.matrix = Matrix.Translation(H_new) @ R_new.to_4x4()
        bpy.context.view_layer.update()
    if door_obj is not None:
        phi = door["phi"][min(fi, len(door["phi"]) - 1)]
        hg = door["hinge"]; u = Vector((math.sin(phi), math.cos(phi), 0))
        ctr = Vector((hg[0], hg[1], 0)) + u * (door["width"] / 2) + Vector((0, 0, door["height"] / 2))
        door_obj.location = Hg @ ctr
        door_obj.rotation_euler = (Hg @ Matrix.Rotation(-phi, 3, "Z")).to_euler()
    hip = Pt["hips"]
    tgt.location = Vector((hip.x, hip.y, 0.75))
    cam.location = tgt.location + Vector((dist * math.cos(az), dist * math.sin(az), 0.25))
    scene.render.filepath = os.path.join(outdir, f"f{fi // every:05d}.png")
    bpy.ops.render.render(write_still=True)
print("rendered", len(clip["frames"]) // every, "frames")
