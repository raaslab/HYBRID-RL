import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, resnet101, resnet18
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparseDenseDataset(Dataset):
    def __init__(self, hdf5_paths, transform=None):
        self.transform = transform
        self.images = []
        self.modes = []
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as file:
                for demo_key in file['data'].keys():
                    demo_group = file['data'][demo_key]
                    images = demo_group['obs']['corner2_image'][:]
                    images = images.transpose(0, 2, 3, 1)
                    self.images.append(images)
                    self.modes.append(demo_group['mode1'][:])
        self.images = np.concatenate(self.images, axis=0)
        self.modes = np.concatenate(self.modes, axis=0)

    def __len__(self):
        return len(self.modes)

    def __getitem__(self, idx):
        image = self.images[idx]
        mode = self.modes[idx]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'mode1': mode}

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class HybridResNet(nn.Module):
    def __init__(self, model_type='resnet50'):
        super(HybridResNet, self).__init__()
        if model_type == 'resnet50':
            base_model = resnet50(pretrained=True)
        elif model_type == 'resnet101':
            base_model = resnet101(pretrained=True)
        else:
            base_model = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        if model_type in ['resnet50', 'resnet101']:
            self.classifier = nn.Linear(2048, 2)
        else:
            self.classifier = nn.Linear(512, 2)

    def forward(self, images):
        img_features = self.features(images)
        img_features = img_features.view(img_features.size(0), -1)
        outputs = self.classifier(img_features)
        return outputs

def train_and_validate(model, train_loader, val_loader, device, num_epochs=50):
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model.train()
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for epoch in range(num_epochs):
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = data['image'].to(device)
            modes = data['mode1'].to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, modes)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds = []
        all_modes = []
        with torch.no_grad():
            for data in val_loader:
                images = data['image'].to(device)
                modes = data['mode1'].to(device).long()
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_modes.extend(modes.cpu().numpy())


                # Compute metrics
        accuracy = accuracy_score(all_modes, all_preds)
        precision = precision_score(all_modes, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_modes, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_modes, all_preds, average='macro')

        all_metrics['accuracy'].append(accuracy)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['f1'].append(f1)
    return all_metrics

def plot_metrics(metrics):
    epochs = list(range(1, len(metrics['accuracy']) + 1))
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, metrics['accuracy'], label='Accuracy')
    plt.plot(epochs, metrics['precision'], label='Precision')
    plt.plot(epochs, metrics['recall'], label='Recall')
    plt.plot(epochs, metrics['f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Performance Metrics')
    plt.legend()
    plt.show()

def main():
    # dataset_paths = ['/home/amisha/ibrl/release/data/metaworld/mw12/assembly_mw12.hdf5', 
    #                  '/home/amisha/ibrl/release/data/metaworld/mw12/boxclose_mw12.hdf5',
    #                  '/home/amisha/ibrl/release/data/metaworld/mw12/stickpull_mw12.hdf5',
    #                  '/home/amisha/ibrl/release/data/metaworld/mw12/coffeepush_mw12.hdf5']
    dataset_paths = ['/home/amisha/ibrl/release/data/metaworld/mw12/boxclose_mw12.hdf5']
    transform = get_transform()
    dataset = SparseDenseDataset(dataset_paths, transform=transform)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_metrics = []

    for train_index, test_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, test_index)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model = HybridResNet().to(get_device())
        metrics = train_and_validate(model, train_loader, val_loader, get_device())
        all_fold_metrics.append(metrics)

    # Average metrics across folds
    avg_metrics = {k: np.mean([m[k] for m in all_fold_metrics], axis=0) for k in all_fold_metrics[0]}
    plot_metrics(avg_metrics)

if __name__ == '__main__':
    main()
