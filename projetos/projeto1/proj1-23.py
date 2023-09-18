import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import argparse

# Step 0: Parse the arguments
parser = argparse.ArgumentParser(
    description="Solves the ODE system Du = Au for a vector of dependent variables u, given the matrix A and a vector of initial conditions u0"
)
parser.add_argument("-dim", "--dimension", type=int, help="The dimension of the matrix A and vector u0", default = 0)
parser.add_argument("--seed", type=int, help="Random seed to be used if randomly sampling the coefficients of A", default = -1)
args = parser.parse_args()

DIM = args.dimension if args.dimension > 0 else np.random.randint(3, 6)
SEED = args.seed if args.seed >= 0 else np.random.randint(0, 100)
np.random.seed(SEED)

# Step 1: Analytical solution
def analytical_solution(A, u0, t):
    eigvals, eigvecs = np.linalg.eig(A)
    
    # Check if the matrix is defective
    defective = np.linalg.matrix_rank(eigvecs) < len(eigvals)
    
    if defective:
        raise NotImplementedError("The matrix is defective, analytical solution is not implemented for defective matrices")
    else:
        # Use the exponential of the matrix to find the solution
        u = np.array([sp.linalg.expm(A * t_i) @ u0 for t_i in t]).T
        return u        

# Step 2: Numerical solution
def numerical_solution(A, u0, t):
    def du_dt(t, u): return A @ u
    sol = solve_ivp(du_dt, [t[0], t[-1]], u0, t_eval=t)
    return sol.y

# Define the matrix A and initial condition u0
# A = np.array([[0, 1], [-1, 0]])
# u0 = np.array([1, 0])

# Uniformly sample the coefficients of A and the coordinates of u0 within [-1, 1]

A = np.random.uniform(-1, 1, (DIM, DIM))
u0 = np.random.uniform(-1, 1, DIM)

# Define the time points where solution is computed
t = np.linspace(0, 10, 100)

# Get the analytical and numerical solutions
u_analytical = analytical_solution(A, u0, t)
u_numerical = numerical_solution(A, u0, t)

# Step 3: Plot the solutions
fig, axs = plt.subplots(DIM + 1, 1, figsize=(8, 2 * DIM))
for i in range(DIM):
    axs[i].plot(t, u_analytical[i], label=f'Analytical solution (u{i+1})')
    axs[i].plot(t, u_numerical[i], '--', label=f'Numerical solution (u{i+1})')
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel(f'u{i+1}')
    axs[i].legend()

# Step 4: Plot the error between the solutions
error = np.linalg.norm(u_analytical - u_numerical, axis=0)
axs[-1].plot(t, error)
axs[-1].set_xlabel('Time')
axs[-1].set_ylabel('Error')
axs[-1].set_title('Error between analytical and numerical solutions')

plt.tight_layout()
plt.show()
