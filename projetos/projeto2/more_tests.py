import numpy as np

def runge_kutta(f, t0, tf, y0, h):
    '''
    Classical Runge-Kutta method for dy/dt = f(t, y), y(t0) = y0,
    with step h, and the specified tolerance and max_steps.
    This function is a generator which can give infinitely many points
    of the estimated solution, in pairs (t[n], y[n]).
    To get only finitely many values of
    the solution we can for example do,
        >>> from itertools import islice
        >>> list(islice(runge_kutta(f, t0, h), n))
        [(t[0], y[0]), (t[1], y[1]), ..., (t[n], y[n])]
    and infact, we could define another function to do this like,
        >>> runge_kutta_N = lambda f, t0, y0, h, N: list(islice(
        ...     runge_kutta(f, t0, y0, h), N))
    It would also be easy to change this function to take an extra
    parameter N and then return a list of the first N, (t_n, y_n),
    directly (by replacing the while loop with for n in range(N)).
    Note also that whilst the question asks for a solution, this
    function only returns an approximation of the solution at
    certain points. We can turn use this to generate a continuous
    function passing through the points specified using either of
    interpolation methods specified lower down the file.
    '''
    # y and t represent y[n] and t[n] respectively at each stage
    y = y0
    t = t0

    # Whilst it would be more elegant to write this recursively,
    # in Python this would be very inefficient, and lead to errors when
    # many iterations are required, as the language does not perform
    # tail call optimisations as would be the case in languages such
    # as C, Lisp, or Haskell.
    #
    # Instead we use a simple infinite loop, which will yield more values
    # of the function indefinitely.
    while t < tf:
        # Generate the next values of the solution y
        yield [t, y]

        # Values for weighted average (compare with Wikipedia)
        k1 = f(t, y)
        k2 = f(t + h/2, y + (h/2)*k1)
        k3 = f(t + h/2, y + (h/2)*k2)
        k4 = f(t + h/2, y + h*k3)

        # Calculate the new value of y and t as per the Runge-Kutta method
        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

# %%

f = lambda t, y: 10*np.exp(-((t-2)**2)/2*(0.075)**2) -0.6*y

# %%

from matplotlib import pyplot

rk4 = runge_kutta(f, 0, 4, 0.5, 0.00001)

points = list(rk4)
points = np.array(points)
points
ts = points[:, 0]
ys = points[:, 1]
pyplot.plot(ts, ys)
