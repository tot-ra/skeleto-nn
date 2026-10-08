"""A yielding stop must be safer than a rigid one, and fast loading must be worse than slow loading."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import injury as I
v = 3.0
rigid, soft = I.stroke_force(v, 0.01, 75), I.stroke_force(v, 0.30, 75)
assert abs(rigid / soft - 30) < 1e-6
assert I.risk(rigid, 2000, "arms") > 0.99 and I.risk(soft, 100, "arms") < 0.5, (I.risk(rigid, 2000, "arms"), I.risk(soft, 100, "arms"))
assert I.risk(4.0, 800, "legs") > I.risk(4.0, 20, "legs")          # same peak, faster load
assert I.risk(3.0, 100, "head") > I.risk(3.0, 100, "torso")        # the head is the most fragile
print("injury ok: rigid %.1f mg vs yielding %.1f mg" % (rigid, soft))
