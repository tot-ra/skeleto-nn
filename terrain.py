"""Height-field terrain made of primitives, with ceilings. Only used to answer queries and to draw."""
from __future__ import annotations
import numpy as np

class Terrain:
    def __init__(self, x0=-6.0, x1=30.0, y0=-6.0, y1=6.0, res=0.04):
        self.x0, self.y0, self.res = x0, y0, res
        self.nx = int(round((x1 - x0) / res)) + 1; self.ny = int(round((y1 - y0) / res)) + 1
        self.H = np.zeros((self.nx, self.ny))
        self.C = np.full((self.nx, self.ny), np.inf)       # lowest ceiling above the floor
        self.pit = np.zeros((self.nx, self.ny), bool)      # cannot be stood on
        self.prims = []                                    # for drawing: (kind, args, rgba)
        self.slope = (0.0, 0.0)
        self.ground_rgba = (0.80, 0.77, 0.68, 1); self.ground_z = 0.0

    # --- building ---
    def _sl(self, x0, x1, y0, y1):
        i0 = max(0, int(round((x0 - self.x0) / self.res))); i1 = min(self.nx, int(round((x1 - self.x0) / self.res)) + 1)
        j0 = max(0, int(round((y0 - self.y0) / self.res))); j1 = min(self.ny, int(round((y1 - self.y0) / self.res)) + 1)
        return slice(i0, i1), slice(j0, j1)

    def set_slope(self, gx=0.0, gy=0.0, x_from=-1e9, x_to=1e9):
        """Ground plane z = gx * clamp(x) + gy * y between x_from and x_to, flat before and level after."""
        X = self.x0 + np.arange(self.nx)[:, None] * self.res; Y = self.y0 + np.arange(self.ny)[None, :] * self.res
        Xc = np.clip(X, x_from, x_to) - max(x_from, self.x0)
        self.H = np.maximum(0, Xc) * gx + Y * gy
        self.slope = (gx, gy)
        self.prims.append(("heightfield", None, self.ground_rgba))

    def add_box(self, x0, x1, y0, y1, top, rgba=(0.55, 0.50, 0.46, 1), bottom=None):
        sx, sy = self._sl(x0, x1, y0, y1)
        self.H[sx, sy] = np.maximum(self.H[sx, sy], top)
        self.prims.append(("box", (x0, x1, y0, y1, 0.0 if bottom is None else bottom, top), rgba))

    def add_stairs(self, x0, y0, n, rise, run, width, rgba=(0.62, 0.57, 0.50, 1), down=False):
        for k in range(n):
            top = rise * (k + 1) if not down else rise * (n - k - 1)
            if top < 0.01: continue
            self.add_box(x0 + k * run, x0 + (k + 1) * run, y0 - width / 2, y0 + width / 2, top, rgba, bottom=0.0)
        return x0 + n * run

    def add_ramp(self, x0, x1, y0, y1, z0, z1, rgba=(0.60, 0.56, 0.48, 1)):
        sx, sy = self._sl(x0, x1, y0, y1)
        X = self.x0 + np.arange(self.nx)[sx, None] * self.res
        z = z0 + (z1 - z0) * (X - x0) / (x1 - x0)
        self.H[sx, sy] = np.maximum(self.H[sx, sy], np.broadcast_to(z, (z.shape[0], self.H[sx, sy].shape[1])))
        self.prims.append(("ramp", (x0, x1, y0, y1, z0, z1), rgba))

    def add_pit(self, x0, x1, y0, y1, depth=1.5, rgba=(0.20, 0.18, 0.16, 1)):
        sx, sy = self._sl(x0, x1, y0, y1)
        self.H[sx, sy] = -depth; self.pit[sx, sy] = True
        self.prims.append(("pit", (x0, x1, y0, y1, depth), rgba))

    def add_beam(self, x0, x1, y0, y1, z_bottom, z_top=None, rgba=(0.45, 0.33, 0.22, 1)):
        sx, sy = self._sl(x0, x1, y0, y1)
        self.C[sx, sy] = np.minimum(self.C[sx, sy], z_bottom)
        self.prims.append(("beam", (x0, x1, y0, y1, z_bottom, z_bottom + 0.25 if z_top is None else z_top), rgba))

    # --- queries ---
    def _ij(self, x, y):
        i = np.clip(np.rint((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.nx - 1)
        j = np.clip(np.rint((np.asarray(y) - self.y0) / self.res).astype(int), 0, self.ny - 1)
        return i, j

    def h(self, x, y):
        i, j = self._ij(x, y); return self.H[i, j]

    def ceiling(self, x, y):
        i, j = self._ij(x, y); return self.C[i, j]

    def is_pit(self, x, y):
        i, j = self._ij(x, y); return self.pit[i, j]

    def max_along(self, p, q, n=24, half_width=0.0):
        """Largest ground height on the segment p -> q (xy), widened sideways by half_width."""
        p = np.asarray(p[:2], float); q = np.asarray(q[:2], float)
        t = np.linspace(0, 1, n)[:, None]; pts = p + (q - p) * t
        m = float(self.h(pts[:, 0], pts[:, 1]).max())
        if half_width > 0:
            d = q - p; nrm = np.array([-d[1], d[0]]); L = np.linalg.norm(nrm)
            if L > 1e-6:
                nrm = nrm / L * half_width
                for s in (-1, 1):
                    pp = pts + s * nrm; m = max(m, float(self.h(pp[:, 0], pp[:, 1]).max()))
        return m

    def normal(self, x, y, d=0.06):
        hx = (self.h(x + d, y) - self.h(x - d, y)) / (2 * d); hy = (self.h(x, y + d) - self.h(x, y - d)) / (2 * d)
        n = np.array([-hx, -hy, 1.0]); return n / np.linalg.norm(n)

    # --- drawing ---
    def mjcf_geoms(self):
        out = []
        for k, (kind, a, rgba) in enumerate(self.prims):
            c = " ".join(f"{v:.3f}" for v in rgba)
            if kind == "box":
                x0, x1, y0, y1, zb, zt = a
                out.append(f'<geom type="box" pos="{(x0+x1)/2:.4f} {(y0+y1)/2:.4f} {(zb+zt)/2:.4f}" size="{(x1-x0)/2:.4f} {(y1-y0)/2:.4f} {(zt-zb)/2:.4f}" rgba="{c}" contype="1" conaffinity="1"/>')
            elif kind == "ramp":
                x0, x1, y0, y1, z0, z1 = a
                L = np.hypot(x1 - x0, z1 - z0); ang = np.degrees(np.arctan2(z1 - z0, x1 - x0))
                th = 0.1
                cx = (x0 + x1) / 2 - np.sin(np.radians(ang)) * th; cz = (z0 + z1) / 2 - np.cos(np.radians(ang)) * th
                out.append(f'<geom type="box" pos="{cx:.4f} {(y0+y1)/2:.4f} {cz:.4f}" size="{L/2:.4f} {(y1-y0)/2:.4f} {th:.3f}" euler="0 {-ang:.3f} 0" rgba="{c}" contype="1" conaffinity="1"/>')
            elif kind == "beam":
                x0, x1, y0, y1, zb, zt = a
                out.append(f'<geom type="box" pos="{(x0+x1)/2:.4f} {(y0+y1)/2:.4f} {(zb+zt)/2:.4f}" size="{(x1-x0)/2:.4f} {(y1-y0)/2:.4f} {(zt-zb)/2:.4f}" rgba="{c}" contype="0" conaffinity="0"/>')
            elif kind == "pit":
                x0, x1, y0, y1, depth = a
                out.append(f'<geom type="box" pos="{(x0+x1)/2:.4f} {(y0+y1)/2:.4f} {-depth - 0.05:.4f}" size="{(x1-x0)/2:.4f} {(y1-y0)/2:.4f} 0.05" rgba="{c}" contype="1" conaffinity="1"/>')
        return out

    def ground_mjcf(self):
        gx, gy = self.slope
        if abs(gx) < 1e-9 and abs(gy) < 1e-9:
            return f'<geom name="floor" type="plane" pos="0 0 {self.ground_z}" size="200 200 0.1" material="grid" contype="1" conaffinity="1" friction="1.1 0.005 0.0001"/>'
        # sloped plane via a tilted box
        ang = np.degrees(np.arctan2(gx, 1.0))
        return f'<geom name="floor" type="box" pos="0 0 -0.1" size="200 200 0.1" euler="0 {-ang:.3f} 0" material="grid" contype="1" conaffinity="1" friction="1.1 0.005 0.0001"/>'
