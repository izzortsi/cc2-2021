# %%


import numpy as np


class ExplicitRungeKutta:
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
        ys = [y0]
        interval = np.arange(t0, tf, h)

        y = lambda t_n, y_n: self.y(f, t_n, y_n, h)

        for t_n in interval[1:]:
            ys.append(y(t_n, ys[-1]))

        return interval, np.array(ys)


# %%
