# Example of a NN trying to learn a binomial
#
# Also see https://stackoverflow.com/questions/55170460/neural-network-for-square-x2-approximation for other
# discussion on this topic

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

print('TORCH', torch.__version__)
torch.manual_seed(42)

TRAINING_EPOCHS = 2_000
SAMPLES_OVER_FUNCTION = 5
OUTSIDE_OF_DISTRO_SAMPLES = 100

HIDDEN_SIZE=3
NUMBER_OF_SPLITS=2

loss_values = []

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

# Define the neural network
class NetForPolynomals(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(NetForPolynomals, self).__init__()
        self.fc1 = nn.Linear(2, hidden_size, bias=False)  
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fcfinal = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_squared = x ** 2
        x_combined = torch.cat((x_squared, x), dim=1)  # Combines x^2
        
        out = self.relu1(self.fc1(x_combined))
        # out = self.relu2(self.fc2(out))
        # out = self.relu3(self.fc3(out))
        out = self.fcfinal(out)
        return out

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

# Instantiate the network
net = NetForPolynomals(hidden_size=HIDDEN_SIZE, output_size=1).to(device)
# net = Polynomial3().to(device)
net.apply(weights_init)

# Define a loss function and optimizer
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)

# Generate some training data
# x_data = np.linspace(-15, 30, TRAINING_SAMPLES_OVER_FUNCTION).reshape(-1, 1).astype('float32')
x_data = np.random.uniform(-55, 40, SAMPLES_OVER_FUNCTION).reshape(-1, 1).astype('float32')
# y_data = (2*x_data + 1).astype('float32')  # Linear function
# y_data = (x_data**2).astype('float32')  # 2nd degree polynomial
y_data = (3*x_data**2 + 6*x_data + 7).astype('float32')  # 2nd degree polynomial
# y_data = (9*x_data**3 + -3*x_data**2 + 2*x_data + 1).astype('float32')  # 3rd degree polynomial
# y_data = (y_data - y_data.mean()) / y_data.std()

x_outside_of_distro_samples = np.random.uniform(-105, 80, OUTSIDE_OF_DISTRO_SAMPLES).reshape(-1, 1).astype('float32')
# y_outside_of_distro_samples = (x_outside_of_distro_samples**2).astype('float32')  # 2nd degree polynomial
y_outside_of_distro_samples = (3*x_outside_of_distro_samples**2 + 6*x_outside_of_distro_samples + 7).astype('float32')  # 2nd degree polynomial

# train_data = [(x_data, y_data)] #, (x_train, y_data_poly2), (x_train, y_data_poly3)]
sort_indices = np.argsort(x_outside_of_distro_samples.flatten())

x_outside_of_distro_samples = x_outside_of_distro_samples[sort_indices].reshape(x_outside_of_distro_samples.shape)
y_outside_of_distro_samples = y_outside_of_distro_samples[sort_indices].reshape(y_outside_of_distro_samples.shape)

print(f'x_train: {x_data.shape} y_data: {y_data.shape}')
print("TRAINING_SAMPLES_OVER_FUNCTION:", SAMPLES_OVER_FUNCTION)
    
kf = KFold(n_splits=NUMBER_OF_SPLITS, shuffle=True)
fold = 0
# Splitting the training data into k folds
for train_index, val_index in kf.split(x_data):
    fold += 1
    print('fold', fold)

    x_train_fold = x_data[train_index]
    y_train_fold = y_data[train_index]

    x_val_fold = x_data[val_index]
    y_val_fold = y_data[val_index]

    # print(x_train_fold, y_train_fold, x_val_fold, y_val_fold)

    # Convert to PyTorch tensors
    x_training_data = torch.tensor(x_train_fold, dtype=torch.float32).to(device)
    y_training_data = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
    x_validation_data = torch.tensor(x_val_fold, dtype=torch.float32).to(device)
    y_validation_data = torch.tensor(y_val_fold, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    
    # Train the network
    for epoch in range(TRAINING_EPOCHS):  # loop over the dataset multiple times

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x_training_data)
        loss = criterion(outputs, y_training_data)

        if torch.isnan(loss):
            print(f'NaN loss on epoch {epoch}')
            print(f'Outputs: {outputs}')
            break

        loss_values.append(loss.item())  # save the current loss value
        
        if epoch % 1000 == 0:  # print loss every 100 epochs
            print(f'Epoch {epoch}, Loss: {loss.item()}')

        if loss < 0.0000001:
            print(f'Epoch {epoch}, Loss: {loss.item()}. Abort')
            break

        loss.backward()
        optimizer.step()

    
       # Validation for each fold
    with torch.no_grad():
        val_predictions = net(x_validation_data)  # Forward pass on validation data
        val_loss = criterion(val_predictions, y_validation_data) # Compute the loss on validation data
        print("Validation loss for fold:", val_loss.item())

# create the plot
plt.plot(loss_values)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# save the plot as an SVG file
plt.savefig('cross-loss_plot.svg', format='svg')


# y_test = (4*x_test**3 + 3*x_test**2 + 2*x_test + 1).astype('float32')  # 3rd degree polynomial
# print(f'x_test: {x_test.shape}', x_test)
# print(f'y_test: {y_test.shape}')
# print(f'y_test_poly2: {y_test_poly2.shape}')
# print(f'y_test_poly3: {y_test_poly3.shape}')

test_data = [(x_data, y_data)]

for x, y in test_data:
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    # Test the network
    net.eval()  # set the network to evaluation mode
    with torch.no_grad():  # we don't need gradients for testing
        predictions = net(x)
        
    # Calculate the mean squared error of the predictions
    mse = criterion(predictions, y)
    print(f'Test MSE: {mse.item()}')

# make predictions on the test data
y_pred = net(torch.from_numpy(x_outside_of_distro_samples).to(device)).detach().cpu().numpy()

# find the minimum and maximum values of x and y for the data and predictions
x_min = min(x_data.min(), x_outside_of_distro_samples.min())
x_max = max(x_data.max(), x_outside_of_distro_samples.max())
y_min = min(y_data.min(), y_outside_of_distro_samples.min(), y_pred.min())
y_max = max(y_data.max(), y_outside_of_distro_samples.max(), y_pred.max())

# create a figure and a set of subplots
fig, axs = plt.subplots(2)

# plot the training data
axs[0].plot(x_data, y_data, 'b.', label='Training data')
axs[0].set_title('Training Data')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(y_min, y_max)

# plot the test data and the network's predictions
axs[1].plot(x_outside_of_distro_samples, y_outside_of_distro_samples, 'r.', label='Test data')
axs[1].plot(x_outside_of_distro_samples, y_pred, 'g-', label='Predicted')
axs[1].set_title('Test Data and Predictions')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend()
axs[1].set_xlim(x_min, x_max)
axs[1].set_ylim(y_min, y_max)

# adjust the space between plots
plt.subplots_adjust(hspace=0.5)

# save the plot as an SVG file
plt.savefig('cross-data_and_predictions.svg', format='svg')
