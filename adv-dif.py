import torch
import pygame
import numpy as np

# Parameters
D = 0.01  # Diffusion coefficient
nu = 0.01  # Kinematic viscosity
dx = 0.1  # Spatial resolution
dt = 0.1  # Time step
size = 512  # Size of the grid
scale = 1   # Scaling factor for visualization

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((size*scale, size*scale))
clock = pygame.time.Clock()

# Initialize the density and velocity field
rho = torch.zeros(size, size, dtype=torch.float32)
u = torch.zeros(size, size, 2, dtype=torch.float32)  # 2D velocity field
rho[size // 2, size // 2] = 1.0  # Single density peak
u[:, :, 0] = 1.0  # Uniform velocity in x-direction

# Define the Laplacian and advection functions (same as before)
def laplacian(f):
    """Compute the Laplacian of a 2D field `f`."""
    return f.roll(1, dims=0) + f.roll(-1, dims=0) + f.roll(1, dims=1) + f.roll(-1, dims=1) - 4 * f

def advect(f, v):
    """Advect field `f` by velocity `v`."""
    f_x = torch.arange(size)
    f_y = torch.arange(size)
    f_xx, f_yy = torch.meshgrid(f_x, f_y, indexing='ij')

    indices_x = (f_xx - v[..., 0] * dt / dx).round().long() % size
    indices_y = (f_yy - v[..., 1] * dt / dx).round().long() % size

    return f[indices_x, indices_y]

# Simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update density
    rho += dt * (D * laplacian(rho) - advect(rho, u))

    # Update velocity
    u += dt * nu * laplacian(u)

    # Convert density to a Pygame surface
# Convert density to a Pygame surface
    rho_np = rho.cpu().numpy()  # Convert to NumPy array
    rho_np = np.clip(rho_np, 0, 1)  # Clip values to [0, 1] for valid grayscale
    rho_np_scaled = np.uint8(rho_np * 255)  # Scale to 0-255 for grayscale
    surface = pygame.surfarray.make_surface(rho_np_scaled.repeat(3, axis=0).reshape(size, size, 3))
    surface = pygame.transform.scale(surface, (size*scale, size*scale))


    # Draw and update
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # Time control (e.g., 60 frames per second)
    clock.tick(60)

pygame.quit()
