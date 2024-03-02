import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from resnet import ResNet18
from pso import PSO
from train import train, train_val_split
from utils import plot_metrics, plot_swarm_movement

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_particles = 10
max_iters = 100
lr = 0.01
inertia_weight = 0.9
cognitive_weight = 0.5
social_weight = 0.5
max_epochs = 10
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
pso_optimizer = PSO(model, criterion, num_particles, max_iters, lr, inertia_weight, cognitive_weight, social_weight)

print('Optimizing model parameters using Particle Swarm Optimization')
pso_optimizer.optimize(train_loader, test_loader)

print('Training the model using optimized parameters')
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
train_losses, val_losses = train(model, criterion, optimizer, train_loader, test_loader, max_epochs)

plot_metrics(train_losses, val_losses)

plot_swarm_movement(pso_optimizer)