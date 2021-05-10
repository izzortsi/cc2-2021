# %%

from imports import *
import matplotlib as mpl

# %%
# https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
mpl.rcParams["image.interpolation"] = "none"
# %%

np.random.seed(0)
n = 30
N = n ** 2
K = (n / np.sqrt(n) * 2 * np.pi) ** np.sqrt(np.pi)
ω = np.random.rand(N) * 2 * np.pi
_ω = ω.reshape(n, n)
θ = np.random.rand(N) * 2 * np.pi
μ = -0.5
positions = np.array([np.array([i, j]) for i in range(n) for j in range(n)]).reshape(
    n, n, 2
)
positions.shape


# %%

# in case one also wants to use a convolution kernel
k_dim = n
kernel = np.zeros((k_dim, k_dim))

# %%


# %%
def convolution(A, f, i, j, kernel):
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
    distances = distances ** μ
    summand = phase_difference * distances
    summand[i, j] = A[i, j]

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
            lconv = convolution(_θ, f, i, j, kernel)

            coupling_term = K * lconv
            # print(lconv)
            dθ[i, j] = _ω[i, j] + coupling_term
    return dθ.flatten()


# %%


rk4 = Integrators["RK4"]()
# %%

ts, θs = rk4.solve(F, 0, 30, θ, 1)
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

file_path = os.path.join(KURAMOTO_OUTS, "tweaked_euclidean_kuramoto.mp4")
anim.save(file_path, fps=6)
