import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import h5py

# Load the data from multiple datasets
def load_data(filepaths):
    inputs_images = []
    inputs_props = []
    waypoint_labels = []

    for filepath in filepaths:
        with h5py.File(filepath, 'r') as file:
            for i in range(len(file['data'].keys())):  # Adjust based on the number of demonstrations in each dataset
                demo_key = f'data/demo_{i}'
                # --- Metaworld environment dataset ---
                # inputs_images.append(file[demo_key + '/obs/corner2_image'][:])
                # inputs_props.append(file[demo_key + '/obs/prop'][:])
                # inputs_props.append(file[demo_key + '/obs/prop'][:])
                # waypoint_labels.append(file[demo_key + '/waypoint1'][:])  # Use the dedicated waypoint key

                # --- Robosuite environment dataset ---
                inputs_images.append(file[demo_key + '/obs/agentview_image'][:])
                pose = file[demo_key + '/obs/robot0_eef_pos'][:]
                quat = file[demo_key + '/obs/robot0_eef_quat'][:]
                prop = np.concatenate([pose, quat], axis=1)
                inputs_props.append(prop)
                waypoint_labels.append(file[demo_key + '/actions'][:])  # Use the dedicated waypoint key

    inputs_images = np.concatenate(inputs_images, axis=0)
    inputs_props = np.concatenate(inputs_props, axis=0)
    waypoint_labels = np.concatenate(waypoint_labels, axis=0)

    return inputs_images, inputs_props, waypoint_labels

class WaypointPredictor(nn.Module):
    def __init__(self, input_channels=3, input_height=96, input_width=96):
        
        super(WaypointPredictor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        conv_output_size = self._get_conv_output_size(input_channels, input_height, input_width)
        self.fc_layers = nn.Sequential(
            # nn.Linear(32 * 24 * 24 + 1, 120),         # Metaworld
            nn.Linear(conv_output_size+7, 120),        # Robosuite
            nn.ReLU(),  
            nn.Linear(120, 60),
            nn.ReLU(),
            # nn.Linear(60, 3)      # Metaworld - Output [waypoint x3]
            nn.Linear(60, 7)        # Robosuite - Output [actions x7]
        )


    def forward(self, img, prop):
        img = self.conv_layers(img)
        img = img.view(img.size(0), -1)
        prop = prop.view(prop.size(0), -1)  # Robosuite - Ensure prop is also flattened
        combined = torch.cat((img, prop), dim=1)
        return self.fc_layers(combined)
    
    def _get_conv_output_size(self, channels, height, width): 
        dummy_input = torch.zeros(1, channels, height, width) 
        conv_output = self.conv_layers(dummy_input)        
        return int(torch.prod(torch.tensor(conv_output.size()[1:])))






    
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, props, labels in train_loader:
            optimizer.zero_grad()
            # outputs = model(images, props[:, -1].unsqueeze(1))
            outputs = model(images, props)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, props, labels in test_loader:
            # outputs = model(images, props[:, -1].unsqueeze(1))
            outputs = model(images, props)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}')

def main():
    # filepaths = ['/home/amisha/ibrl/release/data/metaworld/mw12/assembly_part1.hdf5', 
    #                  '/home/amisha/ibrl/release/data/metaworld/mw12/boxclose_part1.hdf5',
    #                  '/home/amisha/ibrl/release/data/metaworld/mw12/stickpull_part1.hdf5',
    #                  '/home/amisha/ibrl/release/data/metaworld/mw12/coffeepush_part1.hdf5']
    filepaths = ['/home/amisha/ibrl/release/data/robomimic/square/processed_data96withmode.hdf5']
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

    train_model(model, train_loader, criterion, optimizer)
    torch.save(model.state_dict(), 'robosuite_waypoint.pth')
    print("Model Saved!")

    test_model(model, test_loader, criterion)


if __name__ == '__main__':
    main()
