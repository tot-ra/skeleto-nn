"""Bake any catalog shot into a glTF animation clip for a rigged character.
python bake_clip.py SHOT rig.glb out.glb [clip_name]      (needs Blender; the rig's bone names are mapped in bake_blender.py)
Result: out.glb with the rig and one animation named after the shot, 20 fps, to import into Godot, Unity, Blender, three.js."""
import os, sys, subprocess, tempfile, shutil
here = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) < 4: sys.exit(__doc__)
shot, rig, out = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
name = sys.argv[4] if len(sys.argv) > 4 else shot
blender = os.environ.get("BLENDER", "/opt/homebrew/bin/blender")
tmp = tempfile.mkdtemp()
try:
    clip = os.path.join(tmp, "clip.json")
    subprocess.run([sys.executable, os.path.join(here, "export_clip.py"), shot, clip], check=True, cwd=here, stdout=subprocess.DEVNULL)
    subprocess.run([blender, "-b", "--python", os.path.join(here, "bake_blender.py"), "--", rig, clip, out, name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
