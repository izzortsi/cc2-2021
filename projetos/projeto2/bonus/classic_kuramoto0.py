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
n = 10
N = n ** 2
K = 1
ω = np.random.rand(N) * 2 * np.pi
θ = np.random.rand(N) * 2 * np.pi
# %%
_θ = θ.reshape(n, n)
_θ
# %%
kernel = np.full((5, 5), 1 / 9) + np.diag([i for i in range(5)])
kernel
# %%


def convolution(A, kernel):
    k_dim, _ = kernel.shape
    idx_var = k_dim // 2
    index_bounds = lambda k: (max(k - idx_var, 0), min(k + idx_var + 1, n))
    for (i, j) in np.ndindex(A.shape):

        min_i, max_i = index_bounds(i)
        min_j, max_j = index_bounds(j)

        slicex = np.s_[min_i:max_i]
        slicey = np.s_[min_j:max_j]
        sx, sy = _θ[slicex, slicey].shape
        print(i, j)
        print(sx, sy)
        # print(_θ[slicex, slicey])
        if (sx * sy > k_dim ** 2) and (i * j > (n ** 2) / 2):
            kernel_section = kernel[
                sx - idx_var : sx + idx_var, sy - idx_var : sy + idx_var
            ]
            print(kernel_section.shape)
        else:
            kernel_section = kernel[0:sx, 0:sy]
            print(kernel_section.shape)
        print(kernel_section)


convolution(_θ, kernel)

# %%


def F(t, θ):
    dθ = np.zeros_like(θ)
    dθ = dθ.reshape(n, n)
    _θ = θ.reshape(n, n)
    for i in range(n):
        for j in range(n):
            print(_θ[i, j])
            # dθ[i] = ω[i] + (K / N) * np.sum(np.sin(θ - θ_i))
    return dθ


F(0, θ)
# %%


rk4 = integrators["RK4"]()
# %%

ts, θs = rk4.solve(F, 0, 120, θ, 1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%
def init_plot():
    fig, ax = plt.subplots(figsize=(n // 10, n // 10))
    ax.set_axis_off()
    ax.imshow(θs[0])
    return ax.images


def update(num, θs, ax):
    ax.images[0].set_data(θs[num])
    return ax.images


anim = animation.FuncAnimation(
    fig,
    update,
    init_func=init_plot,
    frames=NUM_TS,
    fargs=(θs, ax),
    interval=5,
    blit=True,
    repeat=True,
)

anim.save("bonus/kuramoto.mp4", fps=6)
# %%
Ar = np.array([[1, 2, 3], [5, 4, 3]])
Ar.shape
i, j = np.indices(Ar.shape, sparse=True)
i
j
xv, yv = np.meshgrid(Ar[0], Ar[1], sparse=False, indexing="ij")
xv
yv
