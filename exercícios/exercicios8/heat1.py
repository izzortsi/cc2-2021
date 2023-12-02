import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

# Create a directory to save frames
frames_dir = "heat_eq_frames"
os.makedirs(frames_dir, exist_ok=True)

# Parameters
Nx, Ny = 50, 50
dx, dy = 2.0 / (Nx - 1), 1.0 / (Ny - 1)
dt = 0.001
alpha = 1e-2  # Thermal diffusivity
steps = 2000  # Number of steps to animate

# Parameters for the Gaussians
x_center_1, y_center_1 = 0.5, 0.5  # Center of the 1st Gaussian
x_center_2, y_center_2 = 1.5, 0.5  # Center of the 2nd Gaussian

sigma_1 = 0.10  # Standard deviation of the 1st Gaussian
sigma_2 = 0.20  # Standard deviation of the 2nd Gaussian

magnitude = 200  # Magnitude of the temperature increase

# Initialize the temperature array with two Gaussian distributions

X, Y = np.meshgrid(np.linspace(0, 2, Nx), np.linspace(0, 1, Ny))

gaussian_field_1 = magnitude * np.exp(-((X - x_center_1)**2 + (Y - y_center_1)**2) / (2 * sigma_1**2))

gaussian_field_2 = magnitude * np.exp(-((X - x_center_2)**2 + (Y - y_center_2)**2) / (2 * sigma_2**2))
# plt.matshow(gaussian_field_1
#             + gaussian_field_2)
# plt.show()
T = 200 + gaussian_field_1 + gaussian_field_2

# Set up the figure and axis
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(190, 410)

# Initial plot
surf = ax.plot_surface(X, Y, T, cmap='hot')

speed_anim = 2
gframe = 0

def update(frame):
    global T, surf, gframe
    Tn = T.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            T[j, i] = Tn[j, i] + alpha * dt * (
                (Tn[j, i+1] - 2*Tn[j, i] + Tn[j, i-1]) / dx**2 +
                (Tn[j+1, i] - 2*Tn[j, i] + Tn[j-1, i]) / dy**2
            )

    # Apply boundary conditions
    T[:, 0] = 200
    T[:, -1] = 200
    T[0, :] = 200
    T[-1, :] = 200
    # Update plot

    
    # Save the current frame
    
    if frame % speed_anim == 0:
        print(frame)
        ax.clear()
        ax.set_zlim(190, 410)
        surf = ax.plot_surface(X, Y, T, cmap='hot')        
        frame_path = os.path.join(frames_dir, f"frame_{gframe:04d}.png")
        plt.savefig(frame_path)    
        gframe += 1

    return surf,

# Create animation
ani = FuncAnimation(fig, update, frames=steps, repeat=False)
# ani.save("heateq.mp4", fps=60, 
#         progress_callback=lambda frame, max_frames: 
#             print(f"{frame}/{max_frames}" if frame < max_frames else print(f"Animation saved at {os.path.join(os.getcwd(), 'heateq.mp4')}")))
plt.show()
