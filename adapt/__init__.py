"""Condition-adaptive motion: one small network turns a situation (as physical channels) into a motor style.

schema.py      channels (inputs), style knobs (outputs), body descriptor
conditions.py  named situations as data -> channels
physiology.py  body state over time (wetness, core temperature, blood loss, fatigue) from exposures
model.py       differentiable reduced-order biomechanics and physiology: the cost a style is trained on
net.py         MLP forward pass (numpy) and the portable JSON weight format
train.py       gradient training through the cost model, warm start, check against per-case optima
driver.py      applies a style to the planner (planner.Walker) every step
"""
