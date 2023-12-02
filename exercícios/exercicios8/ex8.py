import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation

# Parameters
Nx, Ny = 50, 50  # Grid size
dx, dy = 2.0 / (Nx - 1), 1.0 / (Ny - 1)  # Spatial resolution
dt = 1e-4  # Time step
alpha = 0.01  # Thermal diffusivity
t_end = 0.05  # End time for the simulation
T_bound = 200  # Boundary condition temperature

# Initialize temperature array
T = T_bound * np.ones((Ny, Nx))

# Initialize A and B matrices
def setup_matrices_2D(Nx, Ny, alpha, dt, dx, dy):
    # Number of internal points in each direction
    N = (Nx - 2) * (Ny - 2)

    # Initialize the diagonals
    main_diag = (1 + 2 * alpha * dt / dx**2 + 2 * alpha * dt / dy**2) * np.ones(N)
    lower_diag = -alpha * dt / dx**2 * np.ones(N - 1)
    upper_diag = -alpha * dt / dx**2 * np.ones(N - 1)
    # Set the diagonals for the boundaries (Neumann conditions)
    for i in range(Ny - 2):
        lower_diag[(i + 1) * (Nx - 2) - 1] = 0
        upper_diag[i * (Nx - 2)] = 0
    # The super and sub-diagonal terms
    super_diag = -alpha * dt / dy**2 * np.ones(N - (Nx - 2))
    sub_diag = -alpha * dt / dy**2 * np.ones(N - (Nx - 2))

    # Assemble the diagonals into the A matrix
    diagonals_A = [main_diag, lower_diag, upper_diag, super_diag, sub_diag]
    offsets_A = [0, -1, 1, -(Nx - 2), Nx - 2]

    # The A matrix will be used in the linear system: A x = b
    A = diags(diagonals=diagonals_A, offsets=offsets_A, shape=(N, N))
    A = A.tocsc()  # Convert to CSC format for efficient arithmetic and solvers

    # Define B matrix as the identity matrix because Crank-Nicolson is implicit
    # This may need to be adjusted depending on your specific BC handling
    B = diags(diagonals=[np.ones(N)], offsets=[0], shape=(N, N))
    B = B.tocsc()  # Convert to CSC format

    return A, B

A, B = setup_matrices_2D(Nx, Ny, alpha, dt, dx, dy)

# Function to update the temperature field
def update_temperature(T, A, B, Nx, Ny, T_bound):
    # Extract the interior points
    T_interior = T[1:-1, 1:-1].ravel()

    # Compute the right-hand side, considering only the interior points
    b = B.dot(T_interior)

    # Apply the boundary conditions to the right-hand side 'b'
    # Left and Right Boundary
    for j in range(1, Ny - 1):
        b[j * (Nx - 2)] += alpha * dt / dx**2 * T_bound  # Left boundary
        b[j * (Nx - 2) + Nx - 3] += alpha * dt / dx**2 * T_bound  # Right boundary

    # Top and Bottom Boundary
    b[:Nx - 2] += alpha * dt / dy**2 * T_bound  # Bottom boundary
    b[-(Nx - 2):] += alpha * dt / dy**2 * T_bound  # Top boundary

    # Solve the linear system for the interior points
    T_interior = spsolve(A, b)

    # Update the full temperature array
    T[1:-1, 1:-1] = T_interior.reshape((Ny - 2, Nx - 2))

    return T


# Set up the figure for animation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.linspace(0, 2, Nx), np.linspace(0, 1, Ny))
ax.set_zlim(T_bound - 5, T_bound + 5)

# Initial plot
surf = ax.plot_surface(X, Y, T, cmap=cm.viridis)

# Animation update function
def animate(t):
    global T
    T = update_temperature(T, A, B, Nx, Ny, T_bound)
    ax.clear()
    ax.set_zlim(T_bound - 5, T_bound + 5)
    surf = ax.plot_surface(X, Y, T, cmap=cm.viridis)
    return surf,

# Create the animation
ani = FuncAnimation(fig, animate, frames=int(t_end / dt), interval=50)

# Show the animation
plt.show()
