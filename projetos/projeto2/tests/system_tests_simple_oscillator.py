# %%

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from explicit_rk import *

# %%
# this example was taken from here: https://prappleizer.github.io/Tutorials/RK4/RK4_Tutorial.html
# %%


def F(t, y):

    k = 1
    v = y[1]
    a = -(k ** 2) * y[0]
    return np.array([v, a])


y0 = np.array([-5, 0])
t0 = 0
tf = 10
h = 0.01
rk4 = RK4()
# %%

ts, ys = rk4.solve(F, t0, tf, y0, h)

# %%


analytical_solution = -5 * np.cos(ts)
analytical_velocity = 5 * np.sin(ts)

# %%

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(ts, ys[:, 0], color="C0", lw=6, ls="--", label="Position (rk4)", alpha=0.5)
ax[0].plot(ts, analytical_solution, color="r", label="Analytical Solution")
ax[1].plot(ts, ys[:, 1], color="C1", lw=6, alpha=0.5, ls="--", label="Velocity (rk4)")
ax[1].plot(ts, analytical_velocity, "C2", label="Analytical Solution")
ax[0].legend(loc="upper center")
ax[1].legend(loc="upper center")
ax[-1].set_xlabel("time")
# %%

fig


# %%
