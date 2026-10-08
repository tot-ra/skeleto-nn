"""Re-encode GIFs smaller for a repository: python slim_gifs.py out_dir in1.gif in2.gif ... (width 400, 12 fps, 32 colours)."""
import os, subprocess, sys
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for f in sys.argv[2:]:
    dst = os.path.join(out, os.path.basename(f))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f, "-vf", "fps=12,scale=400:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=32:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle", dst], check=True)
    print(dst, os.path.getsize(dst) // 1024, "KB")
