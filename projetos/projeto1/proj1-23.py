import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import argparse

def analytical_solution(A, u0, t):
    eigvals, eigvecs = np.linalg.eig(A)
    # print("evals: \n", eigvals)
    # print("evecs: \n", eigvecs)
    # print(np.linalg.matrix_rank(eigvecs))
    # print(len(eigvals))

    # Check if the matrix is defective
    defective = np.linalg.matrix_rank(eigvecs) < len(eigvals)
    
    if defective:
        raise NotImplementedError("The matrix is defective, analytical solution is not implemented for defective matrices")
    else:
        # Use the exponential of the matrix to find the solution
        u = np.array([sp.linalg.expm(A * t_i) @ u0 for t_i in t]).T
        return u        

def numerical_solution(A, u0, t):
    def du_dt(t, u): return A @ u
    sol = solve_ivp(du_dt, [t[0], t[-1]], u0, t_eval=t)
    return sol.y
def main():

    # Uniformly sample the coefficients of A and the coordinates of u0 within [-COEFS_RANGE, COEFS_RANGE]

    A = np.random.uniform(-COEFS_RANGE, COEFS_RANGE, (DIM, DIM))
    u0 = np.random.uniform(-COEFS_RANGE, COEFS_RANGE, DIM)

    # Define the time points where solution is computed
    t = np.arange(0, FINALTIME, TIMESTEP)

    # Get the analytical and numerical solutions
    u_analytical = analytical_solution(A, u0, t)
    u_numerical = numerical_solution(A, u0, t)

    fig, axs = plt.subplots(DIM + 1, 1, figsize=(8, 2 * DIM))
    
    errors = []
    for i in range(DIM):
        error_i = np.abs(u_analytical[i] - u_numerical[i])
        errors.append(error_i)
        axs[i].plot(t, u_analytical[i], label=f'Analytical solution (u{i+1})')
        axs[i].plot(t, u_numerical[i], '--', label=f'Numerical solution (u{i+1})')
        
        if i == DIM - 1: 
            axs[i].set_xlabel('Time')

        axs[i].set_ylabel(f'u{i+1}')
        axs[i].legend()

        axs[-1].plot(t, error_i, label=f'(u{i+1})')
        axs[-1].set_xlabel('Time')
        axs[-1].set_ylabel('Absolute error (log scale)')
        axs[-1].set_title('Errors between analytical and numerical solutions')
        axs[-1].set_yscale('log')
        axs[-1].legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Solves the ODE system Du = Au for a vector of dependent variables u, given the matrix A and a vector of initial conditions u0"
    )
    parser.add_argument("-dim", "--dimension", type=int, help="The dimension of the matrix A and vector u0", default = 0)
    parser.add_argument("--seed", type=int, help="Random seed to be used if randomly sampling the coefficients of A", default = -1)
    parser.add_argument("--time_step", type=float, help="", default = 1e-4)
    parser.add_argument("--final_time", type=float, help="", default = 5.0)
    parser.add_argument("--coefs_range", type=float, help="", default = 4.0)
    args = parser.parse_args()

    DIM = args.dimension if args.dimension > 0 else np.random.randint(3, 6)
    SEED = args.seed if args.seed >= 0 else np.random.randint(0, 100)
    TIMESTEP = args.time_step
    FINALTIME = args.final_time
    COEFS_RANGE = args.coefs_range
    np.random.seed(SEED)

    main()