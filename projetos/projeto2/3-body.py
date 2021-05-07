# %%

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from projetos.projeto2.explicit_rk_vectorized import *

# %%
N_BODIES = 3
DIMENSION = 2


def Fij(ri, rj):
    rel_r = rj - ri
    return (1 / la.norm(rel_r, ord=2) ** 3) * rel_r


def F(t, y):
    _y = np.reshape(y, (N_BODIES, DIMENSION * 2))
    out = np.zeros_like(_y)
    print(_y)
    for i in range(0, N_BODIES, 2):
        for j in range(0, N_BODIES, 2):
            if i != j:
                out[i] = _y[i + 1]
                out[i + 1] += Fij(_y[i], _y[j])
                print(_y[i], _y[j])
    return out.flatten()


# initial_state = np.array(["x11", "x12", "v11", "v12", "x21", "x22", "v21", "v22"])
# initial_state = np.array(
#    [0, 0.5, -0.2, -0.2, 0.2, 0.5, -0.2, 0.2, 0.34, 0.38, -0.4, 0.0]
# )
# %%
# 0.322184765624991, 0.647989160156249

initial_state = np.array(
    [
        -1,
        0,
        0.322184765624991,
        0.647989160156249,
        1,
        0,
        0.322184765624991,
        0.647989160156249,
        0,
        0,
        -0.64436953,
        -1.29597832,
    ]
)

# %%


# %%
F(1, initial_state)
rk4 = RK4()
h = 0.01
t0 = 0
tf = 20
ts, ys = rk4.solve(F, t0, tf, initial_state, h)

_ys = ys.reshape((int(tf / h), N_BODIES, 4))
_ys
orbit_1 = _ys[:, 0, :2]
orbit_2 = _ys[:, 1, :2]
orbit_3 = _ys[:, 2, :2]
orbit_1
# %%

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(orbit_1[:, 0], orbit_1[:, 1], "o-", markersize=0.5)
ax.plot(orbit_2[:, 0], orbit_2[:, 1], "o-", markersize=0.5)
ax.plot(orbit_3[:, 0], orbit_3[:, 1], "o-", markersize=0.5)
fig
