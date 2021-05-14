# %%
from imports import *
import matplotlib as mpl

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

positions = np.array([np.array([i, j]) for i in range(n) for j in range(n)]).reshape(
    n, n, 2
)

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
            phase_difference = f(_θ[i, j], _θ)
            distances = la.norm(positions - np.array([i, j]), axis=2)
            distances[i, j] = n
            distances = distances ** -2
            lconv = phase_difference * distances
            dθ[i, j] = _ω[i, j] + K * np.sum(lconv)
    return dθ.flatten()


# %%


integrator = Integrators["ForwardEuler"]()
# %%

ts, θs = integrator.solve(F, 0, 80, θ, 1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%


# %%
fig, ax = plt.subplots(figsize=(n // 10, n // 10))
ax.set_axis_off()
im = ax.imshow(θs[0], vmin=0, vmax=2 * np.pi)
fig.colorbar(im)
# fig


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
)
# %%
file_path = os.path.join(KURAMOTO_OUTS, "euclidean_kuramoto.mp4")
anim.save(file_path, fps=6)
