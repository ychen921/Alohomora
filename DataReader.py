import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CIFAR10Dataset(Dataset):
    def __init__(self, data_folder, label_file, transform=None):
        self.data_folder = data_folder
        self.labels = self.load_labels(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Assuming image filenames are 1.png, 2.png, etc.
        img_name = os.path.join(self.data_folder, f"{idx+1}.png") 
        image = Image.open(img_name).convert('RGB')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return (image, label)
    
    def load_labels(self, label_file):
        with open(label_file, 'r') as file:
            labels = file.readlines()
        return [int(label.strip()) for label in labels]