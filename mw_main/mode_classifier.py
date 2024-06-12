import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.init() 
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SparseDenseDataset(Dataset):
    def __init__(self, hdf5_paths, transform=None):
        self.transform = transform
        self.images = []
        self.props = []
        self.modes = []
        
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as file:
                for demo_key in file['data'].keys():
                    demo_group = file['data'][demo_key]
                    images = demo_group['obs']['corner2_image'][:]
                    images = images.transpose(0, 2, 3, 1)  # Change from [N, H, W, C] to [N, C, H, W]
                    self.images.append(images)
                    
                    # Extract only the last column of the prop data
                    props = demo_group['obs']['prop'][:, -1]  # Assuming the last column is what you need
                    self.props.append(props)
                    
                    self.modes.append(demo_group['mode'][:])
        
        self.images = np.concatenate(self.images, axis=0)
        self.props = np.concatenate(self.props, axis=0)
        self.modes = np.concatenate(self.modes, axis=0)
    
    def __len__(self):
        return len(self.modes)

    def __getitem__(self, idx):
        image = self.images[idx]
        prop = self.props[idx]  # prop now contains only the last column value
        mode = self.modes[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # Normalize the single prop data value if needed
        prop = (prop - np.mean(self.props)) / np.std(self.props)
        
        return {'image': image, 'prop': prop, 'mode': mode}


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])




class HybridResNet(nn.Module):
    def __init__(self, num_prop_features=1):  # Changed number of prop features to 1
        super(HybridResNet, self).__init__()
        base_model = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.prop_processor = nn.Sequential(
            nn.Linear(num_prop_features, 50), nn.ReLU(),
            nn.Linear(50, 20), nn.ReLU()
        )
        self.classifier = nn.Linear(512 + 20, 2)

    def forward(self, images, props):
        img_features = self.features(images)
        img_features = img_features.view(img_features.size(0), -1)
        props = props.view(props.size(0), -1)  # Ensure props is correctly shaped for processing
        prop_features = self.prop_processor(props)
        combined_features = torch.cat((img_features, prop_features), dim=1)
        outputs = self.classifier(combined_features)
        return outputs


def train(model, train_loader, val_loader, device, num_epochs=50):
    device = get_device()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = data['image'].to(device)
            props = data['prop'].to(device).float()
            modes = data['mode'].to(device).long()
            optimizer.zero_grad()
            outputs = model(images, props)
            loss = criterion(outputs, modes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Training Loss: {running_loss / len(train_loader)}')
        val_loss = validate(model, val_loader, device)

        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'mode.pth')
            print(f'Model saved: Epoch {epoch+1}, Validation Loss: {val_loss}')

def validate(model, val_loader, device):
    device = get_device()
    model.eval()
    total_loss = 0.0
    criterion = CrossEntropyLoss()
    with torch.no_grad():
        for data in val_loader:
            images = data['image'].to(device)
            props = data['prop'].to(device).float()
            modes = data['mode'].to(device).long()
            outputs = model(images, props)
            loss = criterion(outputs, modes)
            total_loss += loss.item()
    
    average_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {average_loss}')
    return average_loss


def main():
    device = get_device()
    dataset_paths = ['/home/amisha/Downloads/mode/assembly_mode.hdf5', 
                     '/home/amisha/Downloads/mode/boxclose_mode.hdf5',
                     '/home/amisha/Downloads/mode/coffeepush_mode.hdf5',
                     '/home/amisha/Downloads/mode/stickpull_mode.hdf5']
    transform = get_transform()
    dataset = SparseDenseDataset(dataset_paths, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = HybridResNet(num_prop_features=1)
    model.to(device)
    train(model, train_loader, val_loader, device)

if __name__ == '__main__':
    main()
