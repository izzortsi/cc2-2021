import torch
import pygame
import numpy as np

# Simulation parameters
D = 0.1  # Diffusion coefficient
nu = 0.1  # Kinematic viscosity
dx = 0.001  # Spatial resolution
dt = 0.001  # Time step
size = 100  # Grid size
scale = 5  # Scale factor for visualization

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((size * scale, size * scale))

# Initialize fields
rho = torch.rand(size, size, dtype=torch.float32) * 0.5  # Density field
u = torch.randn(size, size, 2, dtype=torch.float32)  # Velocity field (2D)
# u[:, :, 0] = 1.0  # Initialize a constant velocity in x-direction

# Define the Laplacian operator using a five-point stencil
def laplacian(f):
    return (f.roll(1, dims=0) + f.roll(-1, dims=0) +
            f.roll(1, dims=1) + f.roll(-1, dims=1) -
            4 * f) / (dx**2)

# Time-stepping function
def step(rho, u, D, nu):
    # Update density field
    rho_new = rho + dt * (D * laplacian(rho) - (u[:, :, 0] * rho).roll(-1, dims=0) - (u[:, :, 1] * rho).roll(-1, dims=1))

    # Update velocity field
    u_new = u + dt * nu * laplacian(u)
    
    return rho_new, u_new

# Simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Perform a simulation step
    rho, u = step(rho, u, D, nu)

    # Visualize the density field
    rho_np = rho.cpu().numpy()
    print(rho_np.min(), rho_np.max())
    surface = pygame.surfarray.make_surface(np.uint8(rho_np * 255))
    surface = pygame.transform.scale(surface, (size * scale, size * scale))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

pygame.quit()
