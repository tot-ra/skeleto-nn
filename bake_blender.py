"""Bake a procedural motion into an animation clip on a rigged character and export it as glTF (Godot, Unity, Blender, three.js read it).

python export_clip.py SHOT clip.json            # simulate the shot, write the world rotation of every mapped bone per frame
blender -b --python bake_blender.py -- <character.glb> <clip.json> <out.glb> [clip_name]

The retarget is the same as in retarget_blender.py (trunk, legs and feet take the world-rotation delta of the planner's skeleton, which has
the rig's proportions; the A-posed arms take only the direction of each segment), but instead of rendering it inserts a keyframe for every
bone at every frame, so the result is an ordinary skeletal animation that the engine plays with its own animation player."""
import bpy, sys, json, math, os
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb, clip_path, out = argv[0], argv[1], argv[2]
name = argv[3] if len(argv) > 3 else os.path.splitext(os.path.basename(out))[0]
clip = json.load(open(clip_path))
fps = int(clip["fps"])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
for o in list(bpy.data.objects):
    if o.name.startswith("Icosphere"): bpy.data.objects.remove(o, do_unlink=True)
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
bpy.context.view_layer.objects.active = arm
arm.animation_data_clear()
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones: pb.rotation_mode = "QUATERNION"

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
def align(a, b): return a.normalized().rotation_difference(b.normalized()).to_matrix()

scene = bpy.context.scene
scene.render.fps = fps
for a_ in list(bpy.data.actions): bpy.data.actions.remove(a_)      # only the baked clip is exported (the source rig may carry its own clips)
act = bpy.data.actions.new(name); arm.animation_data_create(); arm.animation_data.action = act
z0 = clip["z0"]
for fi, fr in enumerate(clip["frames"]):
    Rt = {}; Pt = {}; root = Vector(fr["root"])
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
            if mode == "delta": R_new = (Hg @ Ro @ Hg.inverted()) @ Rr
            else:
                d_t = Hg @ (Ro @ Vector((0, 0, -1))); d_g = Rr @ Vector((0, 1, 0))
                R_new = align(d_g, d_t) @ Rr
        if n == "hips": H_new = hr + Hg @ (root - Vector((0, 0, z0)))
        Rt[n] = R_new; Pt[n] = H_new
        pb = arm.pose.bones[n]
        pb.matrix = Matrix.Translation(H_new) @ R_new.to_4x4()
        bpy.context.view_layer.update()
    frame = fi + 1
    for pb in arm.pose.bones:
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        if pb.name == "hips": pb.keyframe_insert("location", frame=frame)
scene.frame_start, scene.frame_end = 1, len(clip["frames"])
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_animations=True, export_animation_mode="ACTIONS",
                          export_force_sampling=True, export_frame_range=True, export_apply=False)
print("baked", len(clip["frames"]), "frames at", fps, "fps ->", out)
