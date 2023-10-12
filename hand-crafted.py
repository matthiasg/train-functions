# Output from GPT-4. Does not work of course

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 2, bias=False) # Input to hidden layer
        self.fc2 = nn.Linear(2, 1, bias=False) # Hidden layer to output

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x) # ReLU activation
        x = self.fc2(x)
        return x

# Create a dataset
x_data = np.linspace(-10, 10, 1000).reshape(-1, 1)
x_tensor = torch.tensor(x_data, dtype=torch.float32)

# Initialize the model
model = SimpleNet()

# Manually set the weights
with torch.no_grad():
    model.fc1.weight.copy_(torch.tensor([[0.], [2.]]))
    model.fc2.weight.copy_(torch.tensor([[0., 2.]]))

# Get the predictions
y_pred = model(x_tensor)

# Plot the results
plt.plot(x_data, 2 * x_data**2, label='True function')
plt.plot(x_data, y_pred.detach().numpy(), label='Neural network', linestyle='dashed')
plt.legend()

# Save the plot as an SVG file
plt.savefig('hand crafted.svg', format='svg')

plt.show()
