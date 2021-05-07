import numpy as np
import numpy.linalg as la


class ExplicitRungeKutta:
    """
    Initialize this class by giving the number of stages s, the Runge-Kutta matrix,
    which must be a s by s numpy array, with zeros filling the unneeded entries,
    and two arrays of coefficients, c and b, of length s. Defaults to the classic
    RK4 method.
    It yields an object capable of solving an IVP using an RK method with the
    given parameters. To solve such an IVP, use the method `solve`.
    """

    def __init__(self, s: int, A: np.array, c: np.array, b: np.array):
        self.s = s
        self.A = A
        self.c = c
        self.b = b
        # self.ks = lambda f, t, y, h: self.yield_ks(f, t, y, h)

    def yield_ks(self, f, t_n, y_n, h):
        n_bodies, dim_state = y_n.shape
        ks = np.zeros((dim, self.s))
        for s in range(self.s):
            # print(ks[:, :s].shape, self.A[s][:s].shape)
            ks[:, s] = f(
                t_n + self.c[s] * h,
                y_n + h * np.sum(ks[:, :s] * self.A[s][:s], axis=1),
            )
        return ks

    def y(self, f, t_n, y_n, h):
        ks = self.yield_ks(f, t_n, y_n, h)
        return y_n + h * np.sum(self.b * ks)

    def solve(self, f, t0, tf, y0, h):
        """
        Use this method to solve an IVP.

        Args:
            f (function): the function f(t, y)
            t0 (float): the initial point of the solution's interval
            tf (float): the endpoint of the solution's interval
            y0 (float): the value of y(t) at t0
            h (float): the size of the step to be used

        Returns:
            (interval, ys) (tuple): returns a tuple of numpy arrays,
            the first entry being the array of t_n's and the second
            the array of the corresponding y_n's.

        """
        ys = [y0]
        interval = np.arange(t0, tf, h)

        y = lambda t_n, y_n: self.y(f, t_n, y_n, h)

        for t_n in interval[1:]:
            ys.append(y(t_n, ys[-1]))

        return interval, np.array(ys)


class RK4(ExplicitRungeKutta):
    """
    Initializes the general class `ExplicitRungeKutta` with the parameters for
    the classic RK4 method. No arguments are needed.
    """

    def __init__(self):
        super().__init__(
            s=4,
            A=np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]]),
            c=np.array([0, 0.5, 0.5, 1]),
            b=np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        )


# %%

# %%

rk4 = RK4()
# %%

init_state = np.array([[1, 1, -0.2, 1], [0, 0.5, 0, 0.7]])


def Fij(t, X):
    np.zeros_like(X)
    for i, body_i in enumerate(X):
        x_i, y_i, dx_i, dy_i = body_i
        for j, body_j in enumerate(X):
            if j > i:
                x_j, y_j, dx_j, dy_j = body_j
