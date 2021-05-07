# %%

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt

# %%


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
        dim = len(y_n)
        # print(dim)
        ks = np.zeros((dim, self.s))
        # print(ks[:, 0])
        ks[:, 0] = f(
            t_n + self.c[0] * h, y_n + h * np.sum(ks[:, :0] * self.A[0][:1], axis=1)
        )
        for s in range(1, self.s):
            # print(ks[:, :s], self.A[s][:s])
            ks[:, s] = f(
                t_n + self.c[s] * h,
                y_n + h * np.sum(ks[:, :s] * self.A[s][:s], axis=1),
            )
        return ks

    def y(self, f, t_n, y_n, h):
        ks = self.yield_ks(f, t_n, y_n, h)
        return y_n + h * np.sum(self.b * ks, axis=1)

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
            (ts, ys) (tuple): returns a tuple of numpy arrays,
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


def F(t, y):
    return np.array([-np.exp(3 * t) * y[0]])

    # Running a small integration


def F(t, y):
    return np.array([-3 * t])


y0 = np.array([10])  # initialize oscillator at x = -5, with 0 velocity.
t0 = 0
tf = 30
h = 0.01
rk4 = RK4()
# %%
rk4.A
ts, ys = rk4.solve(F, t0, tf, y0, h)
# Running a small integration
# %%


sol = lambda t: -3 * (t ** 2) / 2 + 10
analytical_solution = sol(ts)
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(ts, ys[:, 0], color="C0", lw=6, ls="--", label="Position (rk4)", alpha=0.5)
ax[0].plot(ts, analytical_solution, color="r", label="Analytical Solution")
ax[1].plot(ts, ys[:, 1], color="C1", lw=6, alpha=0.5, ls="--", label="Velocity (rk4)")
ax[1].plot(ts, analytical_velocity, "C2", label="Analytical Solution")
ax[0].legend(loc="upper center")
ax[1].legend(loc="upper center")
ax[-1].set_xlabel("time")

fig


# %%

a = np.array([1, 2, 3])
a.shape
b = np.array([[1, 2], [3, 4]])
b.shape
b[0]
