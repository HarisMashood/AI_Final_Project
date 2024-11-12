import torch
import cv2
import matplotlib.pyplot as plt
from unet_model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('/content/AI_Final_Project/models/unet_model.pth'))
model.eval()

def segment_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
    image = image / 255.0  # Normalize the image
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output).squeeze().cpu().numpy() > 0.5  # Binarize the output

    return output

def display_result(original_image_path, segmented_image):
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError(f"Original image not found: {original_image_path}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(segmented_image, cmap='gray')
    plt.show()  # Ensure the plot displays

if __name__ == "__main__":
    image_path = '/content/AI_Final_Project/data/Images/000.png'  # Replace '000.png' as needed
    segmented_result = segment_image(image_path)
    display_result(image_path, segmented_result)