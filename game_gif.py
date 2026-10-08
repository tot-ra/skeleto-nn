"""Render a catalog shot on the shipped character mesh (Blender, Workbench) and write a GIF.
python game_gif.py SHOT out.gif [character.glb] [cam_dist] [cam_azimuth_deg]"""
import os, sys, subprocess, tempfile, shutil, glob
import numpy as np
from PIL import Image
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from render import save_gif

shot, out = sys.argv[1], sys.argv[2]
glb = sys.argv[3] if len(sys.argv) > 3 else sys.exit("usage: game_gif.py SHOT out.gif rig.glb [cam_dist] [cam_azimuth_deg]")
env = dict(os.environ)
if len(sys.argv) > 4: env["CAM_DIST"] = sys.argv[4]
if len(sys.argv) > 5: env["CAM_AZ"] = sys.argv[5]
tmp = tempfile.mkdtemp()
try:
    js = os.path.join(tmp, "clip.json")
    subprocess.run([sys.executable, os.path.join(here, "export_clip.py"), shot, js], check=True, cwd=here, stdout=subprocess.DEVNULL)
    frames_dir = os.path.join(tmp, "png")
    subprocess.run(["/opt/homebrew/bin/blender", "-b", "--python", os.path.join(here, "retarget_blender.py"), "--", glb, js, frames_dir, os.environ.get("EVERY", "1"), "480", "300"],
                   check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    import json
    fps = json.load(open(js))["fps"] // int(os.environ.get("EVERY", "1"))
    save_gif(frames, out, fps=fps)
    print("wrote", out, len(frames), "frames")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
