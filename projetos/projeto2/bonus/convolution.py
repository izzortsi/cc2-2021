import numpy as np


def convolution(A, kernel):
    k_dim, _ = kernel.shape
    idx_var = k_dim // 2
    index_bounds = lambda k: (max(k - idx_var, 0), min(k + idx_var + 1, n))
    for (i, j) in np.ndindex(A.shape):

        min_i, max_i = index_bounds(i)
        min_j, max_j = index_bounds(j)

        slicex = np.s_[min_i:max_i]
        slicey = np.s_[min_j:max_j]
        _θ_slice = _θ[slicex, slicey]
        sx, sy = _θ[slicex, slicey].shape

        slicex_ker = np.s_[0:sx]
        slicey_ker = np.s_[0:sy]
        if i > k_dim and n - i < k_dim:
            slicex_ker = np.s_[k_dim + i - n - idx_var :]
        if j > k_dim and n - j < k_dim:
            slicey_ker = np.s_[k_dim + j - n - idx_var :]
        kernel_section = kernel[slicex_ker, slicey_ker]
        product = _θ_slice * kernel_section
        A[i, j] = np.sum(product)
    return A
