import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class LiverDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        label_filename = image_filename.replace(".png", "_mask.png")

        image_path = os.path.join(self.image_dir, image_filename)
        label_path = os.path.join(self.label_dir, label_filename)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
        if label is None:
            raise FileNotFoundError(f"Label not found or cannot be read: {label_path}")

        image = image / 255.0
        label = label / 255.0

        if self.transform:
            image = self.transform(image).float()  # Ensure type is float32
            label = self.transform(label).float()  # Ensure type is float32

        return image, label

def get_data_loaders(image_dir, label_dir, batch_size=8):
    dataset = LiverDataset(image_dir, label_dir, transform=transforms.ToTensor())
    
    # Ensure the dataset has at least 400 images
    if len(dataset) < 400:
        raise ValueError("The dataset must have at least 400 images for this split.")

    # Split into 360 for training and 40 for validation
    train_size = 360
    val_size = 40
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader