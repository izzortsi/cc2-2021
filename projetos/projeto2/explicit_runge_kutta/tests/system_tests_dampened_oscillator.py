# %%

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from explicit_rk import *

# %%


def F(t, y):

    x, v = y
    β, ω = 3, 5
    dx = v
    dv = -β * v - x * ω ** 2
    return np.array([dx, dv])


y0 = np.array([1, 0])
t0 = 0
tf = 30
h = 0.01
rk4 = RK4()
# %%

ts, ys = rk4.solve(F, t0, tf, y0, h)
# %%
# computing initial conditions for the analytical solution
β, ω = 3, 5
ϕ = np.arctan(-β / ω)
A0 = 1 / np.cos(ϕ)
# %%

x_t = lambda t, β, ω: A0 * np.exp(-β * t / 2) * np.cos(ω * t + ϕ)
v_t = (
    lambda t, β, ω: A0
    * np.exp(-β * t / 2)
    * (np.cos(ω * t + ϕ) - ω * np.sin(ω * t + ϕ))
)
analytical_solution = x_t(ts, β, ω)
analytical_velocity = v_t(ts, β, ω)
# %%

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(ts, ys[:, 0], color="C0", lw=6, ls="--", label="Position (rk4)", alpha=0.5)
ax[0].plot(ts, analytical_solution, color="r", label="Analytical Solution")
ax[1].plot(ts, ys[:, 1], color="C1", lw=6, alpha=0.5, ls="--", label="Velocity (rk4)")
ax[1].plot(ts, analytical_velocity, "C2", label="Analytical Velocity")
ax[0].legend(loc="upper center")
ax[1].legend(loc="upper center")
ax[-1].set_xlabel("time")

fig
