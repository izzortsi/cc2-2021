import torch
import pygame
import numpy as np
import math



# Parameters
size = 100  # Size of the 2D grid
K = 0.04  # Coupling strength
dt = 0.01  # Time step
scale = 4  # Scaling factor for visualization
import cv2
# Desired FPS
fps = 12
# Set up video writer
video_filename = 'simulation_video.avi'
frame_size = (size*scale, size*scale)
video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

# Check if GPU is available and use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((size*scale, size*scale))
clock = pygame.time.Clock()
pygame.font.init()  # Initialize the font module
font_size = 20  # Size of the font
font = pygame.font.Font(None, font_size)  # Default font with specified size

# Initialize phases and frequencies on GPU
phases = torch.rand(size, size, device=device) * 2 * math.pi
omega = torch.randn(size, size, device=device)

# Initialize the topology matrix (adjacency matrix) on GPU
topology = torch.rand(size * size, size * size, device=device)

# Function to calculate the phase difference influence using batch operations
def phase_influence(phases, K, topology):
    N = phases.shape[0]
    phases_flat = phases.reshape(N * N, 1)
    sin_diff = torch.sin(phases_flat - phases_flat.T)

    # Reshape sin_diff to be a matrix of size (N^2, N^2)
    sin_diff_matrix = sin_diff.reshape(N * N, N * N)

    # Perform the influence computation
    influence_flat = torch.matmul(topology, sin_diff_matrix)
    
    # Sum the influences for each oscillator and reshape
    influence_sum = torch.sum(influence_flat, dim=1).reshape(N, N)
    return K * influence_sum






# Function to update the network topology
def update_topology():
    # Randomly update the topology matrix
    return torch.rand(size * size, size * size, device=device)

def increase_coupling_at(topology, x, y, radius, increase_amount, size):
    for i in range(size):
        for j in range(size):
            dx, dy = i - x, j - y
            distance = math.sqrt(dx**2 + dy**2)
            if distance <= radius:
                topology[i * size + j] += increase_amount
                topology[j * size + i] += increase_amount
    return topology

# Simulation loop
running = True
RADIUS = 5
INCREASE_AMOUNT = 0.1

mouse_held_down = False
last_mouse_pos = (0, 0)

show_message = False
message_timer = 0
message_duration = 1.5  # Duration to display the message in seconds

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:  # 'T' key pressed
                topology = update_topology()  # Update the network topology
                show_message = True
                message_timer = pygame.time.get_ticks()  # Get current time in milliseconds
                print("Updating topology")
        
            if event.key == pygame.K_q:  # 'T' key pressed
                RADIUS += 1
                print(f"New radius: {RADIUS}")               
        
            if event.key == pygame.K_a:  # 'T' key pressed
                if RADIUS <= 1:
                    RADIUS = 1
                else:
                    RADIUS -= 1
                print(f"New radius: {RADIUS}")                    
            if event.key == pygame.K_w:  # 'T' key pressed
                INCREASE_AMOUNT += 0.1
                print(f"New increase amount: {INCREASE_AMOUNT}")               
        
            if event.key == pygame.K_s:  # 'T' key pressed
                INCREASE_AMOUNT -= 0.1                    
                print(f"New increase amount: {INCREASE_AMOUNT}")        

            if event.key == pygame.K_e:  # 'T' key pressed
                K += 0.01
                print(f"New coupling strenght: {K}")               
        
            if event.key == pygame.K_d:  # 'T' key pressed
                K -= 0.01                    
                print(f"New increase amount: {INCREASE_AMOUNT}")                            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # mouse_presses = pygame.mouse.get_pressed()
            # if mouse_presses[0]:            
            topology_old = topology.clone()
            if event.button == 3:  # Right mouse button
                # p = pygame.mouse.get_pos()
                # dp = pygame.mouse.get_rel()
                mouse_held_down = True
                last_mouse_pos = pygame.mouse.get_pos()
                grid_x, grid_y = last_mouse_pos[0] // scale, last_mouse_pos[1] // scale
                topology = increase_coupling_at(topology, grid_x, grid_y, radius=RADIUS, increase_amount=INCREASE_AMOUNT, size=size)
                
        elif event.type == pygame.MOUSEMOTION:
            mouse_presses = pygame.mouse.get_pressed()
            if mouse_held_down:
                last_mouse_pos = pygame.mouse.get_pos()
                grid_x, grid_y = last_mouse_pos[0] // scale, last_mouse_pos[1] // scale
                topology = increase_coupling_at(topology, grid_x, grid_y, radius=RADIUS, increase_amount=INCREASE_AMOUNT, size=size)
                print(f"Difference between couplings: {torch.norm(topology - topology_old)}")
                topology_old = topology.clone()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:  # Right mouse button
                mouse_held_down = False
                print(f"Difference between couplings: {torch.norm(topology - topology_old)}")
                topology_old = topology.clone()

    # Update phases
    influences = phase_influence(phases, K, topology)
    phases += (omega + influences) * dt
    phases = phases % (2 * math.pi)

    # Visualization (transfer data to CPU)
    visualization = (phases.cpu() / (2 * math.pi)).numpy()  # Normalize phases to [0, 1]
    surface = pygame.surfarray.make_surface(np.uint8(visualization * 255).repeat(3, axis=0).reshape(size, size, 3))
    surface = pygame.transform.scale(surface, (size*scale, size*scale))
    screen.blit(surface, (0, 0))
    if mouse_held_down:
        pygame.draw.circle(screen, (255, 0, 0), last_mouse_pos, RADIUS * scale, 1)  # Red circle
    # Show message if required
    parameters_surface = font.render(f'Coupling Strength = {K:.3f}', True, (255, 255, 255))  # White color
    screen.blit(parameters_surface, (5, 5))
    if show_message:
        current_time = pygame.time.get_ticks()
        if current_time - message_timer > message_duration * 1000:
            show_message = False
        else:
            message_surface = font.render('Topology Updated!', True, (255, 255, 255))  # White color
            screen.blit(message_surface, (220, 5))  # Position the message at the top-left corner

    
    # Capture the frame
    pygame.display.flip()
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = frame.transpose([1, 0, 2])  # Transpose it into the correct format for OpenCV
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR for OpenCV


    # Time control
    clock.tick(60)
# Release the VideoWriter
video_writer.release()

pygame.quit()

