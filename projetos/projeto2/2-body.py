# %%

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from projetos.projeto2.explicit_rk_vectorized import *

# %%
N_BODIES = 2
DIMENSION = 2


def Fij(ri, rj):
    rel_r = rj - ri
    return (1 / la.norm(rel_r, ord=2) ** 3) * rel_r


def F(t, y):
    _y = np.reshape(y, (N_BODIES, DIMENSION, 2))
    out = np.zeros_like(_y)
    # print(_y)
    for i, body_state_i in enumerate(_y):
        for j, body_state_j in enumerate(_y):
            if i != j:
                out[i][0] = _y[i][1]
                out[i][1] += Fij(_y[i][0], _y[j][0])
                # print(_y[i], _y[j])
    return out.flatten()


# initial_state = np.array(["x11", "x12", "v11", "v12", "x21", "x22", "v21", "v22"])
initial_state = np.array([0, 0.5, -0.2, -0.2, 0.2, -0.1, -0.1, 0.0])
# %%

initial_state
out = F(1, initial_state)
out

# %%


# %%

rk4 = RK4()

ts, ys = rk4.solve(F, 0, 1, initial_state, 0.001)

_ys = ys.reshape((1000, N_BODIES, 4))
_ys
orbit_1 = _ys[:, 0, :2]
orbit_2 = _ys[:, 1, :2]
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(orbit_1[:, 0], orbit_1[:, 1], "o", markersize=0.5)
ax.plot(orbit_2[:, 0], orbit_2[:, 1], "o", markersize=0.5)

fig
