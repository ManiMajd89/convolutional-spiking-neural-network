import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate, functional as SF, utils
import matplotlib.pyplot as plt
import numpy as np

# Device and dtype setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

dtype = torch.float

# Parameters
batch_size = 128
beta = 0.5
num_steps = 100
num_epochs = 1
lr = 1e-2

# Surrogate gradient and neuron model
spike_grad = surrogate.fast_sigmoid(slope=25)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

# Dataset and loaders
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Model definition
from snntorch import Leaky  

net = nn.Sequential(

    # Conv Block 1
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 28x28 → 28x28
    nn.MaxPool2d(2),                                       # 28x28 → 14x14
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    # Conv Block 2
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 14x14 → 14x14
    nn.MaxPool2d(2),                                       # 14x14 → 7x7
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    # Flatten for FC
    nn.Flatten(),

    # Fully Connected Hidden Layer
    nn.Linear(64 * 7 * 7, 512),                            # 64×7×7 = 3136
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

    # Output Layer
    nn.Linear(512, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
).to(device)



# Forward pass over time
def forward_pass(net, num_steps, data):
    utils.reset(net)
    spk_rec, mem_rec = [], []
    for _ in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)

# Accuracy calculation
def batch_accuracy(loader, net, num_steps):
    with torch.no_grad():
        net.eval()
        acc, total = 0, 0
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)
            acc += SF.accuracy_rate(spk_rec, targets) * data.size(0)
            total += data.size(0)
    return acc / total

# Training setup
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_fn = SF.ce_rate_loss()
loss_hist, test_acc_hist = [], []

# Training loop
counter = 0
for epoch in range(num_epochs):
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)

        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)
        loss = loss_fn(spk_rec, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())

        if counter % 50 == 0:
            acc = batch_accuracy(test_loader, net, num_steps)
            test_acc_hist.append(acc.item())
            print(f"Iter {counter}, Test Acc: {acc * 100:.2f}%")

        counter += 1

# Final evaluation
final_acc = batch_accuracy(test_loader, net, num_steps)
print(f"Final Test Accuracy: {final_acc * 100:.2f}%")



# Create a new matplotlib figure with white background
fig = plt.figure(facecolor="w")

# Plot the test accuracy history that we recorded during training
# Each entry in `test_acc_hist` corresponds to the accuracy at a particular point in training
plt.plot(test_acc_hist)

# Add a title and axis labels
plt.title("Test Set Accuracy")         # Graph title
plt.xlabel("Iteration (every 50 steps)")  # X-axis = checkpoints during training
plt.ylabel("Accuracy")                # Y-axis = accuracy value (0 to 1)

# Display the plot
plt.show()

