import torch
import numpy as np
from sklearn.metrics import jaccard_score, accuracy_score
from unet_model import UNet
from preprocess import get_data_loaders

# Set paths to your data directories
image_dir = '/content/AI_Final_Project/data/Images'
label_dir = '/content/AI_Final_Project/data/Labels'

# Get the validation data loader
_, val_loader = get_data_loaders(image_dir, label_dir, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('/content/AI_Final_Project/models/unet_model.pth'))
model.eval()

def calculate_metrics(pred, label):
    pred = pred > 0.5  # Binarize predictions
    pred = pred.flatten()
    label = label.flatten()
    mIoU = jaccard_score(label, pred, average='binary')
    mPA = accuracy_score(label, pred)
    return mIoU, mPA

# Run evaluation
with torch.no_grad():
    mIoU_list = []
    mPA_list = []
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs).squeeze().cpu().numpy() > 0.5

        for i in range(len(images)):
            pred = outputs[i]
            label = labels[i].cpu().numpy()
            mIoU, mPA = calculate_metrics(pred, label)
            mIoU_list.append(mIoU)
            mPA_list.append(mPA)

mean_mIoU = np.mean(mIoU_list)
mean_mPA = np.mean(mPA_list)

print(f"Mean mIoU: {mean_mIoU:.4f}")
print(f"Mean mPA: {mean_mPA:.4f}")