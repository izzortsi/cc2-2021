import argparse
import numpy as np
from sympy import *
from sympy.abc import t

init_printing(use_unicode=True)

parser = argparse.ArgumentParser(
    description="Solves the ODE system Du = Au for a vector of dependent variables u, given the matrix A and a vector of initial conditions u0"
)
parser.add_argument(
    "-Ar",
    action="append",
    type=np.float64,
    nargs="+",
    help="Adds a row to the matrix A. To input, for instance, a 2 by 2 matrix, use the following syntax: -Ar 1 2 -Ar 3 4",
)
parser.add_argument(
    "-u0",
    type=np.float64,
    nargs="+",
    help="The initial condition vector. Usage: -u0 1 2 3",
)
parser.add_argument(
    "-n",
    type=int,
    help="The number of terms used in the taylor expansion, if needed. Defaults to 10.",
)
parser.add_argument("-tf", type=np.float64, help="The final value for t.")
value = parser.parse_args()

A = Matrix(value.Ar)
u0 = Matrix(value.u0)
(N, N) = A.shape

solution = None


def taylor_expand(A, t, n=10):
    global N
    eAt = eye(N)
    for i in range(n + 1):
        eAt += 1 / (factorial(i)) * (A * t) ** i
    return eAt


def __main__():
    global solution
    if A.is_diagonalizable(reals_only=True):
        eAt = (A * t).exp() * u0
        leAt = lambdify(t, eAt, "numpy")
        if value.tf != None:
            num_sol = leAt(value.tf)
            print(f"Numerical solution for t from t=0 to t={value.tf}: {num_sol}")
        solution = leAt

    else:
        if value.n != None:
            eAt = taylor_expand(A, t, n=value.n)
        else:
            eAt = taylor_expand(A, t)
        leAt = lambdify(t, eAt, "numpy")
        if value.tf != None:
            num_sol = leAt(value.tf)
            print(f"Numerical solution for t from t=0 to t={value.tf}: {num_sol}")
        solution = leAt


__main__()
