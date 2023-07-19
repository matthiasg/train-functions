import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print('TORCH', torch.__version__)

TRAINING_EPOCHS = 25_000
TRAINING_SAMPLES_OVER_FUNCTION = 15
TEST_SAMPLES_OVER_FUNCTION = 100

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
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        # out = self.relu3(self.fc3(out))
        # out = self.relu4(self.fc4(out))
        # out = self.relu5(self.fc5(out))
        out = self.fc6(out)
        return out

# Instantiate the network
net = Net(input_size=1, hidden_size=70, output_size=1).to(device)

# Define a loss function and optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Generate some training data
x_train = np.linspace(-15, 30, TRAINING_SAMPLES_OVER_FUNCTION).reshape(-1, 1).astype('float32')
# y_train = (2*x_train + 1).astype('float32')  # Linear function
y_train = (3*x_train**2 + 2*x_train + 1).astype('float32')  # 2nd degree polynomial
# y_train = (9*x_train**3 + -3*x_train**2 + 2*x_train + 1).astype('float32')  # 3rd degree polynomial
# y_train = (y_train - y_train.mean()) / y_train.std()

train_data = [(x_train, y_train)] #, (x_train, y_train_poly2), (x_train, y_train_poly3)]

# Generate some test data
x_test = np.linspace(-30, 20, TEST_SAMPLES_OVER_FUNCTION).reshape(-1, 1).astype('float32')
# y_test = (2*x_test + 1).astype('float32')  # Linear function
y_test = (3*x_test**2 + 2*x_test + 1).astype('float32')  # 2nd degree polynomial
# y_test = (9*x_test**3 + -3*x_test**2 + 2*x_test + 1).astype('float32')  # 3rd degree polynomial
# y_test = (y_test - y_test.mean()) / y_test.std()

# print(f'x_train: {x_train.shape}', x_train)

for x_train, y_train in train_data:
    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)

    # Train the network
    for epoch in range(TRAINING_EPOCHS):  # loop over the dataset multiple times
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(x_train)
      loss = criterion(outputs, y_train)
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
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# save the plot as an SVG file
plt.savefig('loss_plot.svg', format='svg')


# y_test = (4*x_test**3 + 3*x_test**2 + 2*x_test + 1).astype('float32')  # 3rd degree polynomial
# print(f'x_test: {x_test.shape}', x_test)
# print(f'y_test: {y_test.shape}')
# print(f'y_test_poly2: {y_test_poly2.shape}')
# print(f'y_test_poly3: {y_test_poly3.shape}')

test_data = [(x_test, y_test)]

for x_test, y_test in test_data:
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    # Test the network
    net.eval()  # set the network to evaluation mode
    with torch.no_grad():  # we don't need gradients for testing
        predictions = net(x_test)
        
    # Calculate the mean squared error of the predictions
    mse = criterion(predictions, y_test)
    print(f'Test MSE: {mse.item()}', x_test[8], 'expected', y_test[8], 'real', predictions[8])


# make predictions on the test data
y_pred = net(x_test).detach().cpu().numpy()

# find the minimum and maximum values of x and y for the data and predictions
x_min = min(x_train.cpu().min(), x_test.cpu().min())
x_max = max(x_train.cpu().max(), x_test.cpu().max())
y_min = min(y_train.cpu().min(), y_test.cpu().min(), y_pred.min())
y_max = max(y_train.cpu().max(), y_test.cpu().max(), y_pred.max())

# create a figure and a set of subplots
fig, axs = plt.subplots(2)

# plot the training data
axs[0].plot(x_train.cpu(), y_train.cpu(), 'b.', label='Training data')
axs[0].set_title('Training Data')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(y_min, y_max)

# plot the test data and the network's predictions
axs[1].plot(x_test.cpu(), y_test.cpu(), 'r.', label='Test data')
axs[1].plot(x_test.cpu(), y_pred, 'g-', label='Predicted')
axs[1].set_title('Test Data and Predictions')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend()
axs[1].set_xlim(x_min, x_max)
axs[1].set_ylim(y_min, y_max)

# adjust the space between plots
plt.subplots_adjust(hspace=0.5)

# save the plot as an SVG file
plt.savefig('data_and_predictions.svg', format='svg')
