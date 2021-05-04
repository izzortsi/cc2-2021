# %%


import numpy as np


class ExplicitRungeKutta:
    """
    Initialize by giving the number of stages s, the Runge-Kutta matrix,\n
    which must be a s by s numpy array, with zeros filling the unneeded entries,\n
    and two arrays of coefficients, c and b, of length s.\n
    To solve an IVP, use the method `solve`.
    """

    def __init__(self, s: int, A: np.array, c: np.array, b: np.array):
        self.s = s
        self.A = A
        self.c = c
        self.b = b
        # self.ks = lambda f, t, y, h: self.yield_ks(f, t, y, h)

    def yield_ks(self, f, t_n, y_n, h):

        # ks = [f(t_n, y_n)]
        ks = []
        for i in range(self.s):

            ks.append(
                f(
                    t_n + self.c[i] * h,
                    y_n + h * (np.sum(np.array(ks) * self.A[i][:i])),
                )
            )

        return ks

    def y(self, f, t_n, y_n, h):
        ks = self.yield_ks(f, t_n, y_n, h)
        return y_n + h * np.sum(self.b * ks)

    def solve(self, f, t0, tf, y0, h):
        """
        A short description.

        A bit longer description.

        Args:
            variable (type): description

        Returns:
            type: description

        Raises:
            Exception: description

        """
        ys = [y0]
        interval = np.arange(t0, tf, h)

        y = lambda t_n, y_n: self.y(f, t_n, y_n, h)

        for t_n in interval[1:]:
            ys.append(y(t_n, ys[-1]))

        return interval, np.array(ys)


# %%


class RK4(ExplicitRungeKutta):
    def __init__(self):
        rk4_A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        rk4_c = np.array([0, 0.5, 0.5, 1])
        rk4_b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        super().__init__(rk4_A, rk4_c, rk4_b)
        erk4 = ExplicitRungeKutta(4, rk4_A, rk4_c, rk4_b)
