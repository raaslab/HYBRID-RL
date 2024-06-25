import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load the data from multiple datasets
def load_data(filepaths):
    inputs_images = []
    inputs_props = []
    waypoint_labels = []

    for filepath in filepaths:
        with h5py.File(filepath, 'r') as file:
            for i in range(5):  # Adjust based on the number of demonstrations in each dataset
                demo_key = f'data/demo_{i}'
                inputs_images.append(file[demo_key + '/obs/corner2_image'][:])
                inputs_props.append(file[demo_key + '/obs/prop'][:])
                waypoint_labels.append(file[demo_key + '/waypoint1'][:])  # Use the dedicated waypoint key

    inputs_images = np.concatenate(inputs_images, axis=0)
    inputs_props = np.concatenate(inputs_props, axis=0)
    waypoint_labels = np.concatenate(waypoint_labels, axis=0)

    return inputs_images, inputs_props, waypoint_labels

class WaypointPredictor(nn.Module):
    def __init__(self):
        super(WaypointPredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 24 * 24 + 1, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 3)
        )

    def forward(self, img, prop):
        img = self.conv_layers(img)
        img = img.view(img.size(0), -1)
        combined = torch.cat((img, prop), dim=1)
        return self.fc_layers(combined)
    
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, props, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images, props[:, -1].unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)  # Store epoch loss for plotting
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
    return train_losses

def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, props, labels in test_loader:
            outputs = model(images, props[:, -1].unsqueeze(1))
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss

def plot_losses(train_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Per Epoch')
    plt.legend()
    plt.show()

def main():
    filepaths = ['/home/amisha/ibrl/release/data/metaworld/mw12/assembly_part1.hdf5', 
                     '/home/amisha/ibrl/release/data/metaworld/mw12/boxclose_part1.hdf5',
                     '/home/amisha/ibrl/release/data/metaworld/mw12/stickpull_part1.hdf5',
                     '/home/amisha/ibrl/release/data/metaworld/mw12/coffeepush_part1.hdf5']

    images, props, waypoints = load_data(filepaths)
    
    images_t = torch.tensor(images, dtype=torch.float32)
    props_t = torch.tensor(props, dtype=torch.float32)
    waypoints_t = torch.tensor(waypoints, dtype=torch.float32)
    
    X_img_train, X_img_test, X_prop_train, X_prop_test, y_train, y_test = train_test_split(
        images_t, props_t, waypoints_t, test_size=0.2, random_state=42
    )

    train_data = TensorDataset(X_img_train, X_prop_train, y_train)
    test_data = TensorDataset(X_img_test, X_prop_test, y_test)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

    model = WaypointPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = train_model(model, train_loader, criterion, optimizer)
    test_loss = test_model(model, test_loader, criterion)
    
    plot_losses(train_losses)  # Plot training losses after training

    torch.save(model.state_dict(), 'waypoint_test.pth')
    print("Model Saved!")

if __name__ == '__main__':
    main()
