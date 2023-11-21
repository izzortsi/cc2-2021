import numpy as np
def newtons_method(f, df, x0, tol=1e-4, maxiter=100):
    x = x0
    for _ in range(maxiter):
        x_new = x - f(x)/df(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError(f"no convergence after {maxiter} iterations")