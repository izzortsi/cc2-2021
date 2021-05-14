# %%
from imports import *
import matplotlib as mpl

# from numpy.fft import
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian

# from numpy.fft import fft2, ifft2


def np_fftconvolve(A, B):
    return np.real(np.fft.ifft2(np.fft.fft2(A) * np.fft.fft2(B, s=A.shape)))


# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 30
N = n ** 2
K = np.sqrt(np.pi / 4)
ω = np.random.rand(N) * 2 * np.pi
_ω = ω.reshape(n, n)
θ = np.random.rand(N) * 2 * np.pi
# %%
# _ω = ω.reshape(n, n)
# _θ = θ.reshape(n, n)
# _θ
# %%
# kernel = np.full((5, 5), 1 / 9) + np.diag([i for i in range(5)])
k_dim = 5
# kernel = np.full((k_dim, k_dim), 1 / k_dim ** 2)

window = gaussian(k_dim, std=np.sqrt(2) * k_dim / np.sqrt(n))
# %%
plt.plot(window)

# %%
kernel = np.atleast_2d(window).T * window
# %%
plt.imshow(kernel)

# %%


def F(t, θ):

    # dθ = dθ.reshape(n, n)
    _θ = θ.reshape(n, n)
    dθ = np.zeros_like(_θ)

    def f(θ_i, θ_j):
        return np.sin(θ_j - θ_i)

    for i in range(n):
        for j in range(n):
            # print(_θ[i, j])
            phase_differences = f(_θ[i, j], _θ)
            # _θ[i, j] = 1
            dθ[i, j] = _ω[i, j] * np.cos(_ω[i, j] * t) + (
                K
                * np.sin(t)
                * np.sum(convolve2d(phase_differences, kernel))
                / k_dim ** 2
            )
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

file_path = os.path.join(KURAMOTO_OUTS, "nonglobal_nonautonomous_kuramoto.mp4")
anim.save(file_path, fps=6)
