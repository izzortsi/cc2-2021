# %%
from imports import *
import matplotlib as mpl


# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 40
N = n ** 2
K = 1
ω = np.random.rand(N) * 2 * np.pi
θ = np.random.rand(N) * 2 * np.pi


def F(t, θ):
    dθ = np.zeros_like(θ)
    for i, θ_i in enumerate(θ):
        dθ[i] = ω[i] + (K / N) * np.sum(np.sin(θ - θ_i))
    return dθ


rk4 = Integrators["RK4"]()
# %%

ts, θs = rk4.solve(F, 0, 120, θ, 1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
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
file_path = os.path.join(KURAMOTO_OUTS, "classic_kuramoto.mp4")
anim.save(file_path, fps=6)
