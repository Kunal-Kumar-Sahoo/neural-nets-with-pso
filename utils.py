import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid()
    plt.show()

def plot_swarm_movement(pso_optimizer):
    num_particles = len(pso_optimizer.particles)
    particles_positions = np.zeros((num_particles, 2))
    for i, particle in enumerate(pso_optimizer.particles):
        particles_positions[i] = particle.best_position.detach().cpu().numpy()

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 + Y ** 2

    plt.figure(figsize=(10, 6))
    plt.contour(X, Y, Z, levels=20)
    plt.scatter(particles_positions[:, 0], particles_positions[:, 1], color='r', label='Particles')
    plt.scatter(pso_optimizer.global_best_position[0].item(), pso_optimizer.global_best_position[1].item(), 
                color='blue', label='Global Best', marker='x', s=100)
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Particle Swarm Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()