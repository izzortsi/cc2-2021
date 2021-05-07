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
    _y = np.reshape(y, (N_BODIES, DIMENSION, 2))
    out = np.zeros_like(_y)

    for i, body_state_i in enumerate(_y):
        for j, body_state_j in enumerate(_y):
            if i != j:
                out[i][0] = _y[i][1]
                out[i][1] += Fij(_y[i][0], _y[j][0])
    return out.flatten()


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

rk4 = RK4()
h = 0.01
t0 = 0
tf = 20
ts, ys = rk4.solve(F, t0, tf, initial_state, h)


_ys = ys.reshape((int(tf / h), N_BODIES, 4))
# _ys
orbit_1 = _ys[:, 0, :2]
orbit_2 = _ys[:, 1, :2]
orbit_3 = _ys[:, 2, :2]
# orbit_1
# %%
orbits = [orbit_1, orbit_2, orbit_3]
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot()

ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))
# orbit_plot_1 = ax.plot(orbit_1[:, 0], orbit_1[:, 1], "o-", markersize=0.5)
# orbit_plot_2 = ax.plot(orbit_2[:, 0], orbit_2[:, 1], "o-", markersize=0.5)
# orbit_plot_3 = ax.plot(orbit_3[:, 0], orbit_3[:, 1], "o-", markersize=0.5)
# ax.lines
# mass1 = ax.scatter(orbit_1[0, 0], orbit_1[0, 1], c="red", s=15)
mass1 = ax.plot(
    orbit_1[0, 0],
    orbit_1[0, 1],
    c="red",
    marker="o",
    markersize=10,
)
trail1 = ax.plot(
    orbit_1[0, 0],
    orbit_1[0, 1],
    "--",
    c="red",
    lw=0.4,
    alpha=0.5,
)

# mass2 = ax.scatter(orbit_2[0, 0], orbit_2[0, 1], c="green", s=15)
mass2 = ax.plot(
    orbit_2[0, 0],
    orbit_2[0, 1],
    c="green",
    marker="o",
    markersize=10,
)
trail2 = ax.plot(
    orbit_2[0, 0],
    orbit_2[0, 1],
    "--",
    c="green",
    lw=0.4,
    alpha=0.5,
)

# mass3 = ax.scatter(orbit_3[0, 0], orbit_3[0, 1], c="blue", s=15)
mass3 = ax.plot(
    orbit_3[0, 0],
    orbit_3[0, 1],
    c="blue",
    marker="o",
    markersize=10,
)
trail3 = ax.plot(
    orbit_3[0, 0],
    orbit_3[0, 1],
    "--",
    c="blue",
    lw=0.4,
    alpha=0.5,
)

fig

from matplotlib.animation import FuncAnimation


def update(num, orbits, ax):

    for i, orbit in enumerate(orbits):
        mass = ax.lines[2 * i]
        trail = ax.lines[2 * i + 1]
        mass.set_data(orbit[num, 0], orbit[num, 1])
        trail.set_data(orbit[:num, 0], orbit[:num, 1])


num_frames = len(ts)
animation = FuncAnimation(
    fig, update, frames=num_frames, fargs=(orbits, ax), interval=12
)

animation.save("teste.mp4", dpi=240, bitrate=1500)
