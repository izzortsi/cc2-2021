# %%

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from explicit_rk import *

# %%


# %%


def F(t, y):
    return np.array([-np.exp(3 * t) * y[0]])


def F(t, y):
    return np.array([-3 * t])


y0 = np.array([10])
t0 = 0
tf = 30
h = 0.01
rk4 = RK4()
# %%

ts, ys = rk4.solve(F, t0, tf, y0, h)

# %%


sol = lambda t: -3 * (t ** 2) / 2 + 10
analytical_solution = sol(ts)

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(ts, ys[:, 0], color="C0", lw=6, ls="--", label="Position (rk4)", alpha=0.5)
ax[0].plot(ts, analytical_solution, color="r", label="Analytical Solution")
ax[1].plot(ts, ys[:, 1], color="C1", lw=6, alpha=0.5, ls="--", label="Velocity (rk4)")
ax[0].legend(loc="upper center")
ax[1].legend(loc="upper center")
ax[-1].set_xlabel("time")

fig


# %%
