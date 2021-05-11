import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import RickerWavelet2DKernel

ricker_2d_kernel = RickerWavelet2DKernel(10)
plt.imshow(ricker_2d_kernel, interpolation="none", origin="lower")
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()
arr = np.random.randn(9, 9)
plt.imshow(arr)
kern = RickerWavelet2DKernel(1, x_size=3, y_size=3)
kernarr = np.array(kern)

prod = kernarr * arr

plt.imshow(prod)
