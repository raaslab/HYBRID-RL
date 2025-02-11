import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparseDenseDataset(Dataset):
    def __init__(self, hdf5_paths, transform=None, use_states=False):
        self.transform = transform
        self.images = []
        self.modes = []
        self.states = [] if use_states else None
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as file:
                for demo_key in file['data'].keys():
                    demo_group = file['data'][demo_key]
                    images = demo_group['obs']['corner2_image'][:]
                    images = images.transpose(0, 2, 3, 1)  # Correcting shape to [N, H, W, C]
                    self.images.append(images)
                    self.modes.append(demo_group['mode1'][:])
                    if use_states:
                        # Assuming the state information is stored under 'state' and relevant indices are 4:7
                        states = demo_group['obs']['state'][:, 4:7]
                        self.states.append(states)
        self.images = np.concatenate(self.images, axis=0)
        self.modes = np.concatenate(self.modes, axis=0)
        if use_states:
            self.states = np.concatenate(self.states, axis=0)

    def __len__(self):
        return len(self.modes)

    def __getitem__(self, idx):
        image = self.images[idx]
        mode = self.modes[idx]
        if self.transform:
            image = self.transform(image)
        item = {'image': image, 'mode1': mode}
        if self.states is not None:
            item['state'] = self.states[idx]
        return item

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class SegmentedHybridResNet(nn.Module):
    def __init__(self, model_type='resnet50', use_states=False):
        super(SegmentedHybridResNet, self).__init__()
        self.use_states = use_states
        self.segmentation = deeplabv3_resnet50(pretrained=True)
        for param in self.segmentation.parameters():
            param.requires_grad = False
        if model_type == 'resnet50':
            base_model = models.resnet50(pretrained=True)
        else:
            base_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Linear(2048 if model_type == 'resnet50' else 512, 2)
        if use_states:
            self.state_processor = nn.Sequential(
                nn.Linear(3, 128),  # Input size 3 for object position
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU()
            )
            self.classifier = nn.Linear((2048 if model_type == 'resnet50' else 512) + 128, 2)

    def forward(self, images, states=None):
        seg_masks = self.segmentation(images)['out']
        seg_masks = torch.argmax(seg_masks, dim=1, keepdim=True)
        images = images * seg_masks.float()  # Apply mask to images
        img_features = self.features(images)
        img_features = img_features.view(img_features.size(0), -1)
        if self.use_states and states is not None:
            state_features = self.state_processor(states)
            img_features = torch.cat((img_features, state_features), dim=1)
        outputs = self.classifier(img_features)
        return outputs

def train_and_validate(model, train_loader, val_loader, num_epochs=50):
    device = get_device()
    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = data['image'].to(device)
            modes = data['mode1'].to(device).long()
            states = data.get('state', None)
            if states is not None:
                states = states.to(device).float()
            optimizer.zero_grad()
            outputs = model(images, states)
            loss = criterion(outputs, modes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_loss = validate(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'mode_classifier_with_segmentation_50.pth')
            print(f'Model saved: Epoch {epoch+1}, Validation Loss: {val_loss}')

def validate(model, val_loader):
    device = get_device()
    model.eval()
    criterion = CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            images = data['image'].to(device)
            modes = data['mode1'].to(device).long()
            states = data.get('state', None)
            if states is not None:
                states = states.to(device).float()
            outputs = model(images, states if states is not None else None)
            loss = criterion(outputs, modes)
            total_loss += loss.item()
    average_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {average_loss}')
    return average_loss

# Main routine
def main():
    dataset_paths = ['/home/amisha/ibrl/release/data/metaworld/mw12/assembly_mw12.hdf5',
                     '/home/amisha/ibrl/release/data/metaworld/mw12/boxclose_mw12.hdf5',
                     '/home/amisha/ibrl/release/data/metaworld/mw12/stickpull_mw12.hdf5',
                     '/home/amisha/ibrl/release/data/metaworld/mw12/coffeepush_mw12.hdf5']
    transform = get_transform()
    dataset = SparseDenseDataset(dataset_paths, transform=transform, use_states=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = SegmentedHybridResNet(use_states=True)
    train_and_validate(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
