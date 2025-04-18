# For reading data
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For visualizing
import plotly.express as px

# For model building
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load training and test datasets
train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

# Wrap in DataLoader for batching
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

class FirstNet(nn.Module):
    def __init__(self):
      # We define the components of our model here
      super(FirstNet, self).__init__()
      # Function to flatten our image
      self.flatten = nn.Flatten()
      # Create the sequence of our network
      self.linear_relu_model = nn.Sequential(
            # Add a linear output layer w/ 10 perceptrons
            nn.LazyLinear(10),
        )
    def forward(self, x):
      # We construct the sequencing of our model here
      x = self.flatten(x)
      # Pass flattened images through our sequence
      output = self.linear_relu_model(x)

      # Return the evaluations of our ten
      #   classes as a 10-dimensional vector
      return output

# Create an instance of our model
model = FirstNet()

# Define some training parameters
learning_rate = 1e-2
batch_size = 64
epochs = 5

# Define our loss function
#   This one works for multiclass problems
loss_fn = nn.CrossEntropyLoss()

# Build our optimizer with the parameters from
#   the model we defined, and the learning rate
#   that we picked
optimizer = torch.optim.SGD(model.parameters(),
     lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    # important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    # Loop over batches via the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation and looking for improved gradients
        loss.backward()
        optimizer.step()
        # Zeroing out the gradient (otherwise they are summed)
        #   in preparation for next round
        optimizer.zero_grad()

        # Print progress update every few loops
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    # important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures
    # that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations
    # and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Printing some output after a testing round
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Need to repeat the training process for each epoch.
#   In each epoch, the model will eventually see EVERY
#   observations in the data
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

model.eval()

# Make predictions
pred = model(test_data.__getitem__(1)[0]).argmax()
truth = test_data.__getitem__(1)[1]
print(f"This image is predicted to be a {pred}, and is labeled as {truth}")

# Save our model for later, so we can train more or make predictions

EPOCH = epochs
# We use the .pt file extension by convention for saving
#    pytorch models
PATH = "model.pt"

# The save function creates a binary storing all our data for us
torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
