# %%
import os
import numpy as np
import numpy.linalg as la
import matplotlib.animation as animation
import matplotlib as mpl
from IPython.display import HTML
from matplotlib import pyplot as plt
from explicit_rk import ExplicitRungeKutta, integrators

# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 80
N = n ** 2
K = (n / np.sqrt(n) * 2 * np.pi) ** np.sqrt(np.pi)
ω = np.random.rand(N) * 2 * np.pi
_ω = ω.reshape(n, n)
θ = np.random.rand(N) * 2 * np.pi
80 ** 2
positions = np.array([np.array([i, j]) for i in range(n) for j in range(n)]).reshape(
    n, n, 2
)
positions.shape


# y = positions[-i, -j]
# la.norm(x-y)
# la.norm(positions-x, axis=2)
# %%


def topology_check(i, j):

    _θ = θ.reshape(n, n)
    x = positions[i, j]
    distances = la.norm(positions - x, axis=2)
    distances[i, j] = 1
    distances = distances ** (-0.5)
    # norms = la.norm(positions, axis=2)
    # mindist = np.min(distances)
    # maxdist = np.max(distances)
    # mindist
    # maxdist
    # distances = (distances - mindist) / maxdist
    # distances[i, j] /= np.sqrt(K)
    lconv = distances * np.sin(_θ - _θ[i, j])
    coupling_term = K * lconv
    # lconv[i, j] = 1
    # print(lconv)
    # dθ[i, j] = _ω[i, j] + coupling_term
    output = _ω[i, j] + coupling_term
    return output


# out = topology_check(i, j)


def plot_topology(gsize):
    f, ax = plt.subplots(
        gsize,
        gsize,
        figsize=(2 * gsize, 2 * gsize),
    )

    for i, pi in zip(
        range(gsize, 19 * gsize, (n - 1) // gsize),
        range(gsize),
    ):
        for j, pj in zip(
            range(gsize, 19 * gsize, (n - 1) // gsize),
            range(gsize),
        ):
            out = topology_check(i, j)
            ax[pi, pj].imshow(out)
    return f, ax


gsize = n // 20
f, ax = plot_topology(gsize)
# %%
