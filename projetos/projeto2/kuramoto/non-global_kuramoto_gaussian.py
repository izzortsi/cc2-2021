# %%
from imports import *
import matplotlib as mpl
from scipy.signal import convolve2d
from scipy.signal.signaltools import wiener


def kernel_core(s, r):
    rm = np.minimum(r, 1)
    if s.kernel_type == 0:
        return (4 * rm * (1 - rm)) ** 4
    else:
        return np.exp(4 - 1 / (rm * (1 - rm)))


Lenia.kernel_core = kernel_core


def kernel_shell(s, r):
    k = len(s.peaks)
    kr = k * r
    peak = s.peaks[np.minimum(np.floor(kr).astype(int), k - 1)]
    return (r < 1) * s.kernel_core(kr % 1) * peak


Lenia.kernel_shell = kernel_shell


def calc_kernel(s):
    I = np.array(
        [
            np.arange(SIZE),
        ]
        * SIZE
    )
    X = (I - MID) / s.R
    Y = X.T
    D = np.sqrt(X ** 2 + Y ** 2)

    s.kernel = s.kernel_shell(D)
    s.kernel_sum = np.sum(s.kernel)
    kernel_norm = s.kernel / s.kernel_sum
    s.kernel_FFT = np.fft.fft2(kernel_norm)


Lenia.calc_kernel = calc_kernel
# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 25
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
k_dim = 5


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
            conv = wiener(phase_difference, k_dim)
            dθ[i, j] = _ω[i, j] + K * np.sum(conv)
    return dθ.flatten()


# %%


rk4 = Integrators["ForwardEuler"]()
# %%

ts, θs = rk4.solve(F, 0, 40, θ, 1)
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

file_path = os.path.join(KURAMOTO_OUTS, "nonglobal_kuramoto_gaussian.mp4")
anim.save(file_path, fps=6)
