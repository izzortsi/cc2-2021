# %%
from imports import *
import matplotlib as mpl
from scipy.signal import convolve2d


# %%


# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 50
N = n ** 2
K = np.sqrt(np.pi / 2)
ω = np.random.rand(N) * 2 * np.pi
_ω = ω.reshape(n, n)
θ = np.random.rand(N) * 2 * np.pi
# %%
# _ω = ω.reshape(n, n)
# _θ = θ.reshape(n, n)
# _θ
# %%
# kernel = np.full((5, 5), 1 / 9) + np.diag([i for i in range(5)])
k_dim = 7
kernel = np.full((k_dim, k_dim), 1 / k_dim ** 2)

# %%


def F(t, θ):

    # dθ = dθ.reshape(n, n)
    _θ = θ.reshape(n, n)
    dθ = np.zeros_like(_θ)

    f = lambda θ_i, θ_j: np.sin(θ_j - θ_i)

    for i in range(n):
        for j in range(n):
            # print(_θ[i, j])
            phase_differences = np.sin(_θ - _θ[i, j])
            # _θ[i, j] = 1
            dθ[i, j] = _ω[i, j] + K * np.sum(convolve2d(phase_differences, kernel))
    return dθ.flatten()


# %%


integrator = Integrators["ForwardEuler"]()
# %%
# precompile functions
# integrator.solve(F, 0, 2, θ, 1)
# %%

ts, θs = integrator.solve(F, 0, 60, θ, 1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%
np.min(θs)
np.max(θs)


# %%
fig, ax = plt.subplots(figsize=(n // 10, n // 10))
ax.set_axis_off()
im = ax.imshow(θs[0], vmin=0, vmax=2 * np.pi)
fig.colorbar(im)


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
    blit=True,
)
# %%

file_path = os.path.join(KURAMOTO_OUTS, "nonglobal_kuramoto.mp4")
anim.save(file_path, fps=6)
