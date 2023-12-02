
import numpy as np
import matplotlib.pyplot as plt

class Solver(object):
    # def __init__(self, 
    #              f= lambda x: np.sin(np.log(x)) + np.log(x/np.exp(x)), 
    #              df = lambda x: np.cos(np.log(x))*(1/x) + (np.exp(x)/x)*(1/np.exp(x) - x/np.exp(x)), 
    #              y0 = 1.5, 
    #              t0=0, 
    #              tf = 4, 
    #              h=0.1):
    #     self.f = f
    #     self.df = df
    #     self.y0 = y0
    #     self.t0 = t0
    #     self.tf = tf
    #     self.h = h
    def __init__(self, figsize=(10, 6), method = "Some method", analytical_solution = None):
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.method = method
        self.title = f"{method}, h from 0.1 to 0.00625"
        self.analytical_solution = analytical_solution
        self.ax.set_title(self.title)
    
    def plot(self, label = 'Approximate Solution'):
        # fig, ax = plt.subplots(figsize=(10, 6))
        self.ax.plot(self.ts, self.ys, label=label)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('y(t)')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.show()
        # self.fig, self.ax = fig, ax

    def multiple_plots(self, method, h, f, *args):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.title = f"{self.method}, h from 0.1 to 0.00625"
        self.ax.set_title(self.title)
        for h_i in h:
            self.ts, self.ys = method(f, *args, h_i)
            self.plot(label = f"{self.method}, h = {h_i:.4f}")        

    def newtons_method(self, g, dg, initial_guess, tolerance=1e-6, max_iterations=100):
        """
        Newton's method for finding roots of a real-valued function.

        Args:
        - g (function): Function for which to find the root.
        - dg (function): Derivative of g.
        - initial_guess (float): Initial guess for the root.
        - tolerance (float): Tolerance for convergence.
        - max_iterations (int): Maximum number of iterations.

        Returns:
        - root (float): Approximated root of the function.
        """
        root = initial_guess
        for _ in range(max_iterations):
            g_value = g(root)
            dg_value = dg(root)
            if abs(g_value) < tolerance:
                return root
            root -= g_value / dg_value
        return root

    def forward_euler(self, f, y0, t0, tf, h):
        """
        Forward Euler method for solving ODE y' = f(t, y) with initial condition y(t0) = y0
        from t0 to tf with step size h.

        Args:
        - f (function): The function f(t, y) defining the ODE.
        - y0 (float): Initial value y(t0).
        - t0 (float): Initial time.
        - tf (float): Final time.
        - h (float): Step size.

        Returns:
        - ts (numpy array): Array of time values.
        - ys (numpy array): Array of approximate solution values at the time values.
        """
        # Number of steps
        n_steps = int(np.ceil((tf - t0) / h))

        # Time points
        ts = np.linspace(t0, t0 + n_steps * h, n_steps + 1)

        # Solution array
        ys = np.zeros(n_steps + 1)
        ys[0] = y0

        # Iterative Forward Euler Method
        for i in range(n_steps):
            ys[i + 1] = ys[i] + h * f(ts[i], ys[i])
            if ts[i] == 0.2:
                try:
                    print(f"error at t_i = {ts[i]:.3f} for h = {h:.4f}: {abs(self.analytical_solution(ts[i], y0) - ys[i]):.4f}")
                except Exception as e:
                    print(f"print {e}")
        self.ts, self.ys = ts, ys

        return ts, ys
    
    def backward_euler_newton(self, f, df, y0, t0, tf, h):
        """
        Backward Euler method using Newton's method for solving nonlinear ODE y' = f(t, y) with initial condition y(t0) = y0
        from t0 to tf with step size h.

        Args:
        - f (function): The function f(t, y) defining the ODE.
        - df (function): The derivative of f with respect to y.
        - y0 (float): Initial value y(t0).
        - t0 (float): Initial time.
        - tf (float): Final time.
        - h (float): Step size.

        Returns:
        - ts (numpy array): Array of time values.
        - ys (numpy array): Array of approximate solution values at the time values.
        """
        n_steps = int(np.ceil((tf - t0) / h))
        ts = np.linspace(t0, t0 + n_steps * h, n_steps + 1)
        ys = np.zeros(n_steps + 1)
        ys[0] = y0

        for i in range(n_steps):
            g = lambda y: y - ys[i] - h * f(ts[i+1], y)
            dg = lambda y: 1 - h * df(ts[i+1], y)
            ys[i + 1] = self.newtons_method(g, dg, ys[i])
            if ts[i] == 0.2:
                try:
                    print(f"error at t_i = {ts[i]:.3f} for h = {h:.4f}: {abs(self.analytical_solution(ts[i], y0) - ys[i]):.4f}")
                except Exception as e:
                    print(f"print {e}")
        self.ts, self.ys = ts, ys

        return ts, ys
    
    def crank_nicolson_newton(self, f, df, u0, t0, tf, h):
        """
        Crank-Nicolson method using Newton's method for solving ODE u' = f(t, u) with initial condition u(t0) = u0
        from t0 to tf with step size h.

        Args:
        - f (function): The function f(t, u) defining the ODE.
        - df (function): The derivative of f with respect to u.
        - u0 (float): Initial value u(t0).
        - t0 (float): Initial time.
        - tf (float): Final time.
        - h (float): Step size.

        Returns:
        - ts (numpy array): Array of time values.
        - us (numpy array): Array of approximate solution values at the time values.
        """
        n_steps = int(np.ceil((tf - t0) / h))
        ts = np.linspace(t0, t0 + n_steps * h, n_steps + 1)
        us = np.zeros(n_steps + 1)
        us[0] = u0

        for i in range(n_steps):
            g = lambda u: u - us[i] - (h/2) * (f(ts[i], us[i]) + f(ts[i+1], u))
            dg = lambda u: 1 - (h/2) * df(ts[i+1], u)
            us[i + 1] = self.newtons_method(g, dg, us[i])
            if ts[i] == 0.2:
                try:
                    print(f"error at t_i = {ts[i]:.3f} for h = {h:.4f}: {abs(self.analytical_solution(ts[i], u0) - us[i]):.4f}")
                except Exception as e:
                    print(f"print {e}")
        self.ts, self.ys = ts, us

        return ts, us

    def heun_method(self, f, u0, t0, tf, h):
        """
        Heun's method (Improved Euler method) for solving ODE u' = f(t, u) with initial condition u(t0) = u0
        from t0 to tf with step size h.

        Args:
        - f (function): The function f(t, u) defining the ODE.
        - u0 (float): Initial value u(t0).
        - t0 (float): Initial time.
        - tf (float): Final time.
        - h (float): Step size.

        Returns:
        - ts (numpy array): Array of time values.
        - us (numpy array): Array of approximate solution values at the time values.
        """
        n_steps = int(np.ceil((tf - t0) / h))
        ts = np.linspace(t0, t0 + n_steps * h, n_steps + 1)
        us = np.zeros(n_steps + 1)
        us[0] = u0

        for i in range(n_steps):
            # Predictor step
            u_predict = us[i] + h * f(ts[i], us[i])
            # Corrector step
            us[i + 1] = us[i] + (h/2) * (f(ts[i], us[i]) + f(ts[i+1], u_predict))
            if ts[i] == 0.2:
                try:
                    print(f"error at t_i = {ts[i]:.3f} for h = {h:.4f}: {abs(self.analytical_solution(ts[i], u0) - us[i]):.4f}")
                except Exception as e:
                    print(f"print {e}")
        self.ts, self.ys = ts, us

        return ts, us    
