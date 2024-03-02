import torch
import numpy as np


class Particle:
    def __init__(self, model_params):
        self.position = torch.tensor(model_params.clone().detach().requires_grad_(True))
        self.velocity = torch.randn_like(model_params)
        self.best_position = self.position.clone().detach().requires_grad_(False)
        self.best_loss = float('inf')


class PSO:
    def __init__(self, model, criterion, num_particles, max_iters, lr, inertia_weight, cognitive_weight, social_weight):
        self.model = model
        self.criterion = criterion
        self.num_particles = num_particles
        self.max_iters = max_iters
        self.lr = lr
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        self.particles = []
        self.global_best_position = None
        self.global_best_loss = float('inf')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for param in model.parameters():
            self.particles.append(Particle(param))

    def optimize(self, train_loader, val_loader):
        for _ in range(self.max_iters):
            for particle in self.particles:
                self._evaluate(particle, train_loader)
                if particle.best_loss < self.global_best_loss:
                    self.global_best_loss = particle.best_loss
                    self.global_best_position = particle.best_position.clone().detach().requires_grad_(False)

                intertia_term = self.inertia_weight * particle.velocity
                cognitive_term = self.cognitive_weight * (particle.best_position - particle.position)
                social_term = self.social_weight * (self.global_best_position - particle.position)

                particle.velocity = intertia_term + cognitive_term + social_term
                particle.position += self.lr * particle.velocity

    def _evaluate(self, particle, dataloader):
        total_loss, total_samples = 0.0, 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples

        if avg_loss < particle.best_loss:
            particle.best_loss = avg_loss
            particle.best_position = particle.position.clone().detach().requires_grad_(False)