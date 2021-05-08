# %%
from projetos.projeto2.explicit_rk import ExplicitRungeKutta, RK4, RK4r38, RK3, RK3G
import numpy as np
from matplotlib import pyplot


# %%

from projetos.projeto2.explicit_rk import *

# %%
rk4 = RK4()
rk38 = RK4r38()
rk3 = RK3()
rk3g = RK3G(1 / 4)
# %%
rk4.c
rk38.c
rk3.c
rk3g.c
# %%


f = lambda t, y: 10 * np.exp(-((t - 2) ** 2) / 2 * (0.075) ** 2) - 0.6 * y

ts4, ys4 = rk4.solve(f, 0, 4, 0.5, 0.00001)
ts38, ys38 = rk38.solve(f, 0, 4, 0.5, 0.00001)
ts3, ys3 = rk3.solve(f, 0, 4, 0.5, 0.00001)
ts3g, ys3g = rk3g.solve(f, 0, 4, 0.5, 0.00001)
# %%
pyplot.plot(ts38, ys38)
pyplot.plot(ts4, ys4)
pyplot.plot(ts3, ys3)
pyplot.plot(ts3g, ys3g)
# %%
np.alltrue(ts38 == ts4)
np.alltrue(ys38 == ys4)
np.alltrue(ys38 == ys3)
np.alltrue(ys4 == ys3)
np.alltrue(ys3 == ys3g)
# %%


# %%
def f2(x, y):
    return (x - y) / 2


x0 = 0
y = 1
x = 2
h = 0.2

# %%

ts, ys = erk4.solve(f2, x0, x, y, h)


# %%
f1 = lambda t, x: x[1]
f2 = lambda t, x: -x[1] - np.sin(x[0]) + np.sin(t)
f = lambda t, x: np.array([f1(t, x), f2(t, x)])
x10 = 0
x20 = 1


# %%
rk4.solve(f, 0, 2, np.array([x10, x20]), 0.001)
