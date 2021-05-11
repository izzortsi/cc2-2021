# %%
from imports import *
import matplotlib as mpl
from astropy.convolution import RickerWavelet2DKernel, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft

# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 50
N = n ** 2
K = np.sqrt(np.pi)
ω = np.random.rand(N) * 2 * np.pi
_ω = ω.reshape(n, n)
θ = np.random.rand(N) * 2 * np.pi
# %%
# _ω = ω.reshape(n, n)
_θ = θ.reshape(n, n)
# _θ
# %%
# kernel = np.full((5, 5), 1 / 9) + np.diag([i for i in range(5)])
k_dim = 7
# kernel = np.full((k_dim, k_dim), 1 / k_dim ** 2)
kernel = np.array(
    Gaussian2DKernel(
        x_stddev=K,
        y_stddev=K,
        # theta=np.sqrt(np.pi) / n,
        x_size=k_dim,
        y_size=k_dim,
        factor=10,
    )
)
# kernel = np.array(RickerWavelet2DKernel(1, x_size=k_dim, y_size=k_dim))
kernel = np.array(kernel)

# %%
plt.imshow(kernel)


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
            conv = convolve_fft(phase_difference, kernel, boundary="fill")
            dθ[i, j] = _ω[i, j] + K * np.sum(conv)
    return dθ.flatten()


# %%


rk4 = Integrators["RK4"]()
# %%

ts, θs = rk4.solve(F, 0, 12, θ, 1)
NUM_TS = len(ts)
θs = θs.reshape(NUM_TS, n, n)
# %%


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

file_path = os.path.join(KURAMOTO_OUTS, "nonglobal_kuramoto_filters.mp4")
anim.save(file_path, fps=6)
