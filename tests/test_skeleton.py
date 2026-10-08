"""Run: .venv/bin/python tests/test_skeleton.py  (plain asserts, no framework)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from skeleton import rot, euler_from_R, align

def test_euler_roundtrip():
    rng = np.random.default_rng(0); worst = 0.0
    for order in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
        for _ in range(300):
            a = rng.uniform(-1.2, 1.2, 3); d = {"X": a[0], "Y": a[1], "Z": a[2]}
            R = rot(order[0], d[order[0]]) @ rot(order[1], d[order[1]]) @ rot(order[2], d[order[2]])
            e = euler_from_R(R, order); d2 = {"X": e[0], "Y": e[1], "Z": e[2]}
            R2 = rot(order[0], d2[order[0]]) @ rot(order[1], d2[order[1]]) @ rot(order[2], d2[order[2]])
            worst = max(worst, abs(R - R2).max())
    assert worst < 1e-9, worst

def test_align():
    rng = np.random.default_rng(1)
    for _ in range(100):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = align(a, b)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)

if __name__ == "__main__":
    test_euler_roundtrip(); test_align(); print("ok")
