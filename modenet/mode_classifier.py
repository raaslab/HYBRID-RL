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
import time 
import cv2
import argparse



# import sys
# sys.path.append('/home/amisha/.local/lib/python3.7/site-packages/efficientnet_pytorch')
# from efficientnet_pytorch import EfficientNet
from torchvision.transforms import RandomHorizontalFlip, ColorJitter, RandomRotation
from torch.optim.lr_scheduler import StepLR
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
                    # images = demo_group['obs']['agentview_image'][:]
                    images = images.transpose(0, 2, 3, 1)
                    self.images.append(images)
                    self.modes.append(demo_group['mode1'][:])
                    # self.modes.append(demo_group['mode'][:])
        self.images = np.concatenate(self.images, axis=0)
        self.modes = np.concatenate(self.modes, axis=0)

    def __len__(self):
        return len(self.modes)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Convert image to uint8 if it is in int64
        if image.dtype == np.int64:
            image = image.astype(np.uint8)
        
        mode = self.modes[idx]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'mode1': mode}

def get_advanced_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Adjust size for EfficientNet
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



class HybridResNet(nn.Module):
    def __init__(self, model_type='resnet101'):
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

class EnhancedHybridResNet(nn.Module):
    def __init__(self, model_type='resnet50'):
        super(EnhancedHybridResNet, self).__init__()
        if model_type == 'resnet50':
            base_model = resnet50(pretrained=True)
            num_features = 2048
        elif model_type == 'resnet101':
            base_model = resnet101(pretrained=True)
            num_features = 2048
        else:
            base_model = resnet18(pretrained=True)
            num_features = 512
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Enhanced MLP
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Assuming binary classification
        )

    def forward(self, images):
        img_features = self.features(images)
        img_features = img_features.view(img_features.size(0), -1)
        outputs = self.classifier(img_features)
        return outputs

# def train_and_validate(model, train_loader, val_loader, device, num_epochs=100):
#     criterion = CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=0.001)
#     scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
def train_and_validate(model, train_loader, val_loader, device, num_epochs=100):
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = data['image'].to(device)
            modes = data['mode1'].to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, modes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_loss = validate(model, val_loader, device)

        # Save the model if the validation loss is the lowest
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch+1}: New best model saved with validation loss {val_loss}")

    return best_model_state

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    criterion = CrossEntropyLoss()
    with torch.no_grad():
        for data in val_loader:
            images = data['image'].to(device)
            modes = data['mode1'].to(device).long()
            outputs = model(images)
            loss = criterion(outputs, modes)
            total_loss += loss.item()
    return total_loss / len(val_loader)




def plot_metrics(metrics):
    epochs = list(range(1, len(metrics['accuracy']) + 1))
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, metrics['accuracy'], label='Accuracy')
    plt.plot(epochs, metrics['precision'], label='Precision')
    plt.plot(epochs, metrics['recall'], label='Recall')
    plt.plot(epochs, metrics['f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title(model_name)
    plt.legend()
    plt.savefig(f"mode_{model_name}.png")  # Save the plot to a file
    # plt.show()

def main(args):
    dataset_paths = {
        'coffeepush': 'release/data/metaworld/mw12/coffeepush_mw12.hdf5',
        'assembly': 'release/data/metaworld/mw12/assembly_mw12.hdf5',
        'boxclose': 'release/data/metaworld/mw12/boxclose_mw12.hdf5',
        'stickpull': 'release/data/metaworld/mw12/stickpull_mw12.hdf5'
    }
    
    paths = [dataset_paths[name] for name in args.datasets if name in dataset_paths]
    print("Using Datasets: ", paths)
    print("Saving model name: ", model_name)
    # dataset_paths = ['/home/amisha/ibrl/release/data/robomimic/square/processed_data96withmode.hdf5']
    transform = get_transform()
    # transform=get_advanced_transform()
    dataset = SparseDenseDataset(paths, transform=transform)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_metrics = []
    best_val_loss = float('inf')
    best_acc = 0.0
    best_model_overall = None

    for train_index, test_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, test_index)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model = HybridResNet().to(get_device())
        # model = EnhancedHybridResNet().to(get_device())
        # model = EfficientHybridNet().to(get_device())
        metrics, best_model_state = train_and_validate(model, train_loader, val_loader, get_device())
        all_fold_metrics.append(metrics)

        # Check if the current fold produced a better model
        current_fold_best_loss = min(metrics['val_loss'])
        if current_fold_best_loss < best_val_loss:
            best_acc = metrics['accuracy'][metrics['val_loss'].index(current_fold_best_loss)]
            print(f"New Best model: val_loss: {current_fold_best_loss}; Accuracy: {best_acc}")
            best_val_loss = current_fold_best_loss
            best_model_overall = best_model_state

    # Write the final validation loss to a log file
    log_file_path = f"{model_name}.txt"
    with open(log_file_path, "w") as log_file:
        log_file.write(f"New Best model: val_loss: {best_val_loss};  Accuracy: {best_acc}")

    # Save the best model found across all folds
    torch.save(best_model_overall, f"{model_name}.pth")

    # Average metrics across folds
    avg_metrics = {k: np.mean([m[k] for m in all_fold_metrics], axis=0) for k in all_fold_metrics[0]}
    plot_metrics(avg_metrics)






if __name__ == '__main__':
    # train3_test1()
    parser = argparse.ArgumentParser(description="Train a model on specified datasets")
    parser.add_argument('--datasets', nargs='+', type=str, choices=['coffeepush', 'assembly', 'boxclose', 'stickpull'], help='Names of datasets to use')
    args = parser.parse_args()
    global model_name
    model_name = "mode_" + "_".join(args.datasets)
    main(args)
