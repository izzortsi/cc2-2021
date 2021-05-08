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
n = 25
N = n ** 2
K = 8 * np.sqrt(np.pi)
ω = np.random.rand(N) * 2 * np.pi
_ω = ω.reshape(n, n)
θ = np.random.rand(N) * 2 * np.pi

positions = np.array([np.array([i, j]) for i in range(n) for j in range(n)]).reshape(
    n, n, 2
)


# %%

# %%
# _ω = ω.reshape(n, n)
# _θ = θ.reshape(n, n)
# _θ
# %%
# kernel = np.full((5, 5), 1 / 9) + np.diag([i for i in range(5)])
k_dim = n
kernel = np.zeros((k_dim, k_dim))
# kernel
# %%
i, j = 10, 20

# %%
def local_convolution(A, f, i, j, kernel):
    k_dim, _ = kernel.shape
    idx_var = k_dim // 2
    index_bounds = lambda k: (max(k - idx_var, 0), min(k + idx_var + 1, n))

    min_i, max_i = index_bounds(i)
    min_j, max_j = index_bounds(j)

    slicex = np.s_[min_i:max_i]
    slicey = np.s_[min_j:max_j]
    A_slice = A[slicex, slicey]
    sx, sy = A[slicex, slicey].shape

    slicex_ker = np.s_[0:sx]
    slicey_ker = np.s_[0:sy]
    if i > k_dim and n - i < k_dim:
        slicex_ker = np.s_[k_dim + i - n - idx_var :]
    if j > k_dim and n - j < k_dim:
        slicey_ker = np.s_[k_dim + j - n - idx_var :]
    kernel_section = kernel[slicex_ker, slicey_ker]
    phase_difference = f(A[i, j], A)
    distances = la.norm(positions - np.array([i, j]), axis=2)
    distances[i, j] = 1
    distances = distances ** 2
    summand = phase_difference / distances
    # print(phase_difference.shape)

    return np.sum(summand)


# %%


def F(t, θ):

    # dθ = dθ.reshape(n, n)
    _θ = θ.reshape(n, n)
    dθ = np.zeros_like(_θ)
    f = lambda θ_i, θ_j: np.sin(θ_j - θ_i)
    for i in range(n):
        for j in range(n):
            # print(_θ[i, j])
            # dθ[i] = ω[i] + (K / N) * np.sum(np.sin(θ - θ_i))
            lconv = local_convolution(_θ, f, i, j, kernel)
            # print(lconv)
            dθ[i, j] = _ω[i, j] + K * lconv
    return dθ.flatten()


# %%


rk4 = integrators["RK4"]()
# %%

ts, θs = rk4.solve(F, 0, 250, θ, 1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%


# %%
fig, ax = plt.subplots(figsize=(n // 10, n // 10))
ax.set_axis_off()
ax.imshow(θs[0])
fig


def init_plot():
    return ax.images


def update(num, θs, ax):
    ax.images[0].set_data(θs[num])
    return ax.images


anim = animation.FuncAnimation(
    fig,
    update,
    frames=NUM_TS,
    fargs=(θs, ax),
    interval=5,
    repeat=True,
)
# %%

anim.save("bonus/euclidean_kuramoto.mp4", fps=6)
