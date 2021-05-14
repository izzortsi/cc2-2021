# %%
from imports import *
import matplotlib as mpl


# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 25
N = n ** 2
K = 1
ω = np.random.rand(N) * 2 * np.pi
θ = np.random.rand(N) * 2 * np.pi
α = np.random.randn(N)


def F(t, θ):
    dθ = np.zeros_like(θ)
    for i, θ_i in enumerate(θ):
        dθ[i] = ω[i] * np.cos(α[i] * t) - (
            (K / N) * np.sum(np.sin(θ_i - θ) + α[i] * np.sin(t * (θ_i - θ)))
        )
    #        print((K / N) * np.sum(np.sin(θ_i - θ)), dθ[i])
    return dθ


integrator = Integrators["ForwardEuler"]()
# %%

ts, θs = integrator.solve(F, 0, 50, θ, 0.1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%
# θs[10]
np.min(θs[10])
np.max(θs[10])


# %%


fig, ax = plt.subplots(figsize=(n // 10, n // 10))
ax.set_axis_off()
im = ax.imshow(θs[0])  # , vmin=0, vmax=2 * np.pi)
# fig.colorbar(im)


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
file_path = os.path.join(KURAMOTO_OUTS, "global_nonautonomous_kuramoto.mp4")
anim.save(file_path, fps=6)
