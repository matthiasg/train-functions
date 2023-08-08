# Example of a NN trying to learn a binomial
#
# Also see https://stackoverflow.com/questions/55170460/neural-network-for-square-x2-approximation for other
# discussion on this topic

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print('TORCH', torch.__version__)

TRAINING_EPOCHS = 5_000
TRAINING_SAMPLES_OVER_FUNCTION = 50
TEST_SAMPLES_OVER_FUNCTION = 100
HIDDEN_SIZE=100

loss_values = []

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth = 8):
        super(Net, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.activations = nn.ModuleList([nn.ReLU()])
         
         # Hidden layers
        for _ in range(depth - 2):  # Subtract 2 to account for input and output layers
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.activations.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))

        # The final layer doesn't have an activation function
        x = self.layers[-1](x)
    
        return x

# Instantiate the network
net = Net(input_size=1, hidden_size=HIDDEN_SIZE, output_size=2).to(device)

# Define a loss function and optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

# Generate some training data
x_train = np.random.uniform(-55, 40, TRAINING_SAMPLES_OVER_FUNCTION).reshape(-1, 1).astype('float32')
y_train = (3*x_train**2 + 2*x_train + 1).astype('float32')  # 2nd degree polynomial
y_train_prime = (6*x_train + 2).astype('float32')  # derivative of 2nd degree polynomial

x_train_tensor = torch.from_numpy(x_train).to(device)
y_train_tensor = torch.from_numpy(np.concatenate((y_train, y_train_prime), axis=1)).to(device)

# Generate some test data
x_test = np.linspace(-70, 60, TEST_SAMPLES_OVER_FUNCTION).reshape(-1, 1).astype('float32')
y_test = (3*x_test**2 + 2*x_test + 1).astype('float32')  # 2nd degree polynomial
y_test_prime = (6*x_test + 2).astype('float32')  # 3rd degree polynomial

x_test_tensor = torch.from_numpy(x_test).to(device)
y_test_tensor = torch.from_numpy(np.concatenate((y_test, y_test_prime), axis=1)).to(device)

# Train the network
for epoch in range(TRAINING_EPOCHS):  # loop over the dataset multiple times
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    if torch.isnan(loss):
        print(f'NaN loss on epoch {epoch}')
        print(f'Outputs: {outputs}')
        break

    loss_values.append(loss.item())  # save the current loss value
    
    if epoch % 100 == 0:  # print loss every 100 epochs
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    if loss < 0.0000000000001:
        print(f'Epoch {epoch}, Loss: {loss.item()}. Abort')
        break

    loss.backward()
    optimizer.step()

print('Finished Training for function')

# create the plot
plt.plot(loss_values)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# save the plot as an SVG file
plt.savefig('two-loss_plot.svg', format='svg')


# Test the network and create plot
# Make predictions on the test data
y_pred = net(x_test_tensor).detach().cpu().numpy()

# Test the network and create plot
# Make predictions on the test data
y_pred = net(x_test_tensor).detach().cpu().numpy()

# Find the minimum and maximum values of x and y for the data and predictions
x_min = min(x_train_tensor.cpu().min(), x_test_tensor.cpu().min())
x_max = max(x_train_tensor.cpu().max(), x_test_tensor.cpu().max())
y_min = min(y_train_tensor.cpu().min(), y_test_tensor.cpu().min(), y_pred.min())
y_max = max(y_train_tensor.cpu().max(), y_test_tensor.cpu().max(), y_pred.max())

# Create a figure and a set of subplots
fig, axs = plt.subplots(2)

# Plot the training data
axs[0].plot(x_train_tensor.cpu(), y_train_tensor.cpu()[:, 0], 'b.', label='Training data (f)')
axs[0].plot(x_train_tensor.cpu(), y_train_tensor.cpu()[:, 1], 'r.', label='Training data (f\')')
axs[0].set_title('Training Data')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(y_min, y_max)

# Plot the test data and the network's predictions
axs[1].plot(x_test_tensor.cpu(), y_test_tensor.cpu()[:, 0], 'r.', label='Test data (f)')
axs[1].plot(x_test_tensor.cpu(), y_pred[:, 0], 'g-', label='Predicted (f)')

axs2 = axs[1].twinx()  # Instantiate a second y-axis that shares the same x-axis
axs2.plot(x_test_tensor.cpu(), y_test_tensor.cpu()[:, 1], 'c.', label='Test data (f\')')
axs2.plot(x_test_tensor.cpu(), y_pred[:, 1], 'm-', label='Predicted (f\')')

axs[1].set_title('Test Data and Predictions')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y (f)')
axs2.set_ylabel('y (f\')')

# Align the y-axis labels
fig.align_ylabels([axs[1], axs2])

# Add the legends to the plot
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs[1].transAxes)

axs[1].set_xlim(x_min, x_max)
axs2.set_ylim(y_min, y_max)

# Adjust the space between plots
plt.subplots_adjust(hspace=0.5)

# Save the plot as an SVG file
plt.savefig('two-data_and_predictions.svg', format='svg')