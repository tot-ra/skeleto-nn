"""The leap network (adapt/leap.py) chooses by context: a waist-high wall is vaulted with the hands, a drop onto stone
is refused, deep water below a quay is jumped into, ice or a heavy load turns jumps into refusals; the numpy forward
pass agrees with the torch one."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
from adapt import leap as L
from adapt.conditions import channels_from

net = L.LeapNet(L.DEFAULT_OUT)
LEG = 0.86


def decide(feats, conds=()):
    z, w, thin = L.profile(LEG, *feats)
    return net.decide("biped", 58.0, LEG, 0.0, 1.0, 0.9, channels_from(list(conds)), z, w, ~thin)


wall = [("wall", 0.6, 0.85, 0.4, 0.0)]
d = decide(wall)
assert d["action"] == "jump" and d["label"] == "vault" and d["land_m"] > 1.0 and d["knobs"]["hands"] > 0.5, d
assert decide(wall, ["ice"])["action"] == "refuse"
assert decide([("drop", 0.6, 6.0, 0.0)])["action"] == "refuse"
assert decide([("drop", 0.6, 8.0, 0.4)])["action"] == "refuse", "shallow water below a cliff"
q = decide([("drop", 0.6, 2.0, 3.0)])
assert q["label"] == "jump into water" and q["water_m"] > 1.5, q
assert decide([("drop", 0.6, 2.0, 3.0)], ["heavy_load"])["action"] == "refuse", "a load in deep water"
assert decide([("step", 0.6, 1.4)])["label"] == "pull up onto"
assert decide([("wall", 0.6, 2.2, 0.4, 0.0)])["action"] == "refuse"

# numpy runtime == torch training network
tnet = L.load(L.DEFAULT_OUT)
z, w, thin = L.profile(LEG, *wall)
ch = channels_from([]); from adapt.schema import CHANNELS
shared = L.encode_shared_t(torch.tensor([0]), torch.tensor([58.0]), torch.tensor([LEG]), torch.tensor([0.0]), torch.tensor([1.0]),
                           torch.tensor([0.9]), torch.tensor([[ch[c.name] for c in CHANNELS]], dtype=torch.float32))
with torch.no_grad():
    logits, kn = tnet(shared, torch.tensor(L.cand_features(z[None] / LEG, w[None] / LEG)))
assert int(logits[0].masked_fill(torch.tensor(np.concatenate([[False], thin])), -1e4).argmax()) - 1 == d["cell"]
assert abs(float(kn[0, d["cell"], L.KI["hands"]]) - d["knobs"]["hands"]) < 1e-4
print("leap decisions ok")
