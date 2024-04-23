import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import io
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import LambdaLR


class density_NN(nn.Module):
    def __init__(self, num_neuron=120):
        super(density_NN,self).__init__()
        self.ifb = nn.Sequential(
            nn.Linear(4, num_neuron),
            nn.BatchNorm1d(num_neuron)
        )
        self.fb = nn.Sequential(
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron)
        )
        self.lfb = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron)
        )
        self.fblfb = nn.Sequential(
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron)
        )
        self.fbl = nn.Sequential(
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.ff = nn.Sequential(
            nn.Linear(num_neuron, num_neuron),
            nn.Linear(num_neuron, num_neuron),
            nn.Linear(num_neuron, 5)
        )

    def forward(self,x):
        x = self.ifb(x)
        save_x = x
        x = self.lfb(x) + save_x
        x = F.leaky_relu(x, negative_slope=0.1)
        save_x = x
        x = self.fblfb(x) + save_x
        x = F.leaky_relu(x, negative_slope=0.1)
        save_x = x
        x = self.fb(x) + save_x
        x = F.leaky_relu(x, negative_slope=0.1)
        save_x = x
        x = self.fbl(x) + save_x
        x = self.ff(x)
        return x


class f_NN(nn.Module):
    def __init__(self, num_neuron=120):
        super(f_NN,self).__init__()
        self.ifb = nn.Sequential(
            nn.Linear(4, num_neuron),
            nn.BatchNorm1d(num_neuron)
        )
        self.fb = nn.Sequential(
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron)
        )
        self.lfb = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron)
        )
        self.fblfb = nn.Sequential(
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron)
        )
        self.fbl = nn.Sequential(
            nn.Linear(num_neuron, num_neuron),
            nn.BatchNorm1d(num_neuron),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.ff = nn.Sequential(
            nn.Linear(num_neuron, num_neuron),
            nn.Linear(num_neuron, num_neuron),
            nn.Linear(num_neuron, 5)
        )

    def forward(self,x):
        x = self.ifb(x)
        save_x = x
        x = self.lfb(x) + save_x
        x = F.leaky_relu(x, negative_slope=0.1)
        save_x = x
        x = self.fblfb(x) + save_x
        x = F.leaky_relu(x, negative_slope=0.1)
        save_x = x
        x = self.fb(x) + save_x
        x = F.leaky_relu(x, negative_slope=0.1)
        save_x = x
        x = self.fbl(x) + save_x
        x = self.ff(x)
        return x


# get data from files
def get_data(data_dir, split=[0.8, 0.1, 0.1]):
    data = io.loadmat(data_dir)
    x_data = data['pointsMatrix'][:, :].astype(np.float32)
    y_data = data['propertyMatrix'][:, :].astype(np.float32)

    x_data = np.reshape(x_data, (-1,4), order='F')
    y_data = np.reshape(y_data, (-1,5), order='F')

    # spliting data into train, validate and test
    dim = y_data.shape
    num_train= int(split[0]*dim[0])
    num_val  = int(split[1]*dim[0])
    num_test = int(split[2]*dim[0])

    x_train = x_data[:num_train, :]
    y_train = y_data[:num_train, :]
    x_val = x_data[num_train:(num_train+num_val), :]
    y_val = y_data[num_train:(num_train+num_val), :]
    x_test = x_data[(num_train+num_val):, :]
    y_test = y_data[(num_train+num_val):, :]

    return x_train, y_train, x_val, y_val, x_test, y_test


# numpy to tensor
def numpy2tensor(x, device, grad=False):
    x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=grad)
    return x


# Define the learning rate schedule function
def lr_lambda(epoch):
    lr_decay = 0.1
    num_epochs_per_decay = (epoch // 3) + 1
    lr = lr_decay ** (epoch // num_epochs_per_decay)
    # if lr < 1e-6:
    #     lr = 1e-6

    return lr


# training function
def train_model(model, train_loader, val_loader, num_epochs, log_file='data/training_f_NN_log.txt'):
    history = {'train_loss': [], 'val_loss': []}
    best_train_loss = 1.
    with open(log_file, 'w') as f:
        for epoch in range(num_epochs):
            # training phase
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                # forward pass
                y_pred = model(inputs)

                # calculate loss
                loss = criterion(y_pred, targets)

                # zero gradients, backward pass, and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    y_pred = model(inputs)
                    val_loss += criterion(y_pred, targets)

            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            # Step the learning rate scheduler after each epoch
            scheduler.step()

            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss: .4e}, Val Loss: {val_loss: .4e}, best_train_loss: {best_train_loss: .4e}")
                f.write(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}\n")

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                # Save the trained model
                torch.save(model.state_dict(), "data/f_NN_large.pth")
                print('Save model')

    return history


# main function
if __name__=='__main__':
    # select devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, y_train, x_val, y_val, x_test, y_test = get_data('data/shape_property_large.mat')
    X_train, y_train = numpy2tensor(x_train, device), numpy2tensor(y_train, device)
    X_val, y_val   = numpy2tensor(x_val, device), numpy2tensor(y_val, device)

    # training data
    train_dataset = TensorDataset(X_train, y_train)
    train_loader  = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # validate data
    val_dataset = TensorDataset(X_val, y_val)
    val_loader  = DataLoader(val_dataset, batch_size=512, shuffle=True)

    # create an instance of the model
    model = f_NN(num_neuron=50).to(device)

    # define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create the learning rate scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=300)


