# %%
from projetos.projeto2.explicit_rk import ExplicitRungeKutta
import numpy as np

# %%

rk4_A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
rk4_c = np.array([0, 0.5, 0.5, 1])
rk4_b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
erk4 = ExplicitRungeKutta(4, rk4_A, rk4_c, rk4_b)

# %%
f = lambda t, y: 10 * np.exp(-((t - 2) ** 2) / 2 * (0.075) ** 2) - 0.6 * y

# %%

from matplotlib import pyplot

# erk4.yield_ks(f, 0, 0.5, 0.00001)
ts, ys = erk4.solve(f, 0, 4, 0.5, 0.00001)
ys = np.array(ys)

pyplot.plot(ts, ys[1:])


# %%
def f2(x, y):
    return (x - y) / 2


x0 = 0
y = 1
x = 2
h = 0.2

# %%

ts, ys = erk4.solve(f2, x0, x, y, h)
