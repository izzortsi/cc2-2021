# %%


import numpy as np


class ExplicitRungeKutta:
    """
    Initialize this class by giving the number of stages s, the Runge-Kutta matrix,
    which must be a s by s numpy array, with zeros filling the unneeded entries,
    and two arrays of coefficients, c and b, of length s.
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


# %%


class RK4(ExplicitRungeKutta):
    """
    Initializes the general class `ExplicitRungeKutta` with the parameters for
    the usual RK4 method. No arguments are needed.
    """

    def __init__(self):
        s = 4
        rk4_A = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
        rk4_c = np.array([0, 0.5, 0.5, 1])
        rk4_b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        super().__init__(s, rk4_A, rk4_c, rk4_b)
