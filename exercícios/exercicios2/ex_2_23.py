
import numpy as np
import matplotlib.pyplot as plt

def heun_method(f, u0, t0, t_end, h):
    """Solve du/dt = f(u, t) by Heun's method.

    Parameters:
    f: the right-hand side function of the ODE
    u0: initial value of u at time t0
    t0: initial time
    t_end: final time
    h: step size

    Returns:
    t: array of time points
    u: array of approximate solution at each time point
    """
    # Number of time steps: we add 1 to include the initial condition
    n_steps = int(np.ceil((t_end - t0) / h)) + 1
    
    # Initialize arrays to hold time points and approximate solution
    t = np.linspace(t0, t_end, n_steps)
    u = np.zeros(n_steps)
    
    # Set initial condition
    u[0] = u0
    
    # Time-stepping loop
    for i in range(n_steps - 1):
        u_pred = u[i] + h * f(u[i], t[i])                   # Predictor step
        u_corr = f(u_pred, t[i+1])                           # Evaluate f at the predicted value
        u[i+1] = u[i] + (h / 2) * (f(u[i], t[i]) + u_corr)  # Corrector step
    
    return t, u

# Define the ODE as a Python function
def f(u, t):
    return -5 * u

# Define the analytical solution
def u_analytical(t, u0):
    return u0 * np.exp(-5 * t)

# Set initial conditions and parameters
u0 = 1     # The initial value of u
t0 = 0     # The initial time
t_end = 2  # The final time we want to solve to
h = 0.1    # Step size

# Solve the ODE
t, u = heun_method(f, u0, t0, t_end, h)

# Calculate the analytical solution
u_true = u_analytical(t, u0)

# Plot the results
plt.plot(t, u, 'b-', label='Heun Method')
plt.plot(t, u_true, 'r--', label='Analytical Solution')
plt.xlabel('Time t')
plt.ylabel('Solution u')
plt.title('Comparison of Heun Method with Analytical Solution')
plt.legend()
plt.show()
