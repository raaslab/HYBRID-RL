import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import h5py

# Load the data from multiple datasets
def load_data(filepaths):
    inputs_corner_images = []
    inputs_eye_images = []
    inputs_props = []
    waypoint_labels = []

    for filepath in filepaths:
        with h5py.File(filepath, 'r') as file:
            for i in range(len(file['data'].keys())):  # Adjust based on the number of demonstrations in each dataset
                demo_key = f'data/demo_{i}'
                # --- Metaworld environment dataset ---
                inputs_corner_images.append(file[demo_key + '/obs/corner2_image'][:])
                inputs_eye_images.append(file[demo_key + '/obs/eye_in_hand_image'][:])
                inputs_props.append(file[demo_key + '/obs/prop'][:])
                waypoint_labels.append(file[demo_key + '/waypoint'][:, :3])  # Use the dedicated waypoint key

    inputs_corner_images = np.concatenate(inputs_corner_images, axis=0)
    inputs_eye_images = np.concatenate(inputs_eye_images, axis=0)
    inputs_props = np.concatenate(inputs_props, axis=0)
    waypoint_labels = np.concatenate(waypoint_labels, axis=0)

    return inputs_corner_images, inputs_eye_images, inputs_props, waypoint_labels

class WaypointPredictor(nn.Module):
    def __init__(self, input_channels=3, input_height=96, input_width=96):
        super(WaypointPredictor, self).__init__()
        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layers2 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        conv_output_size1 = self._get_conv_output_size(self.conv_layers1, input_channels, input_height, input_width)
        conv_output_size2 = self._get_conv_output_size(self.conv_layers2, input_channels, input_height, input_width)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size1 + conv_output_size2 + 1, 120),  # Adjust input size accordingly
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 3)  # Output size for waypoints
        )


    def forward(self, img1, img2, prop):
        img1 = self.conv_layers1(img1)
        img1 = img1.view(img1.size(0), -1)
        img2 = self.conv_layers2(img2)
        img2 = img2.view(img2.size(0), -1)
        combined = torch.cat((img1, img2, prop), dim=1)
        return self.fc_layers(combined)
    
    def _get_conv_output_size(self, conv_layers, channels, height, width):
        dummy_input = torch.zeros(1, channels, height, width)
        conv_output = conv_layers(dummy_input)
        return int(torch.prod(torch.tensor(conv_output.size()[1:])))


# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs=40):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images1, images2, props, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images1, images2, props[:, -1].unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images1.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images1, images2, props, labels in test_loader:
            outputs = model(images1, images2, props[:, -1].unsqueeze(1))
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images1.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}')

def main():
    filepaths = ['release/data/real_robot_2S/data_2S_hyrl_SDSD.hdf5']
    corner_images, eye_images, props, waypoints = load_data(filepaths)
    
    corner_images_t = torch.tensor(corner_images, dtype=torch.float32)
    eye_images_t = torch.tensor(eye_images, dtype=torch.float32)
    props_t = torch.tensor(props, dtype=torch.float32)
    waypoints_t = torch.tensor(waypoints, dtype=torch.float32)
    
    X_corner_img_train, X_corner_img_test, X_eye_img_train, X_eye_img_test, X_prop_train, X_prop_test, y_train, y_test = train_test_split(
        corner_images_t, eye_images_t, props_t, waypoints_t, test_size=0.2, random_state=42
    )

    train_data = TensorDataset(X_corner_img_train, X_eye_img_train, X_prop_train, y_train)
    test_data = TensorDataset(X_corner_img_test, X_eye_img_test, X_prop_test, y_test)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

    model = WaypointPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer)
    torch.save(model.state_dict(), 'waypoint_predictor2S_dual_SDSD.pth')
    print("Model Saved!")

    test_model(model, test_loader, criterion)

if __name__ == '__main__':
    main()