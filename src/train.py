import torch
import torch.optim as optim
import torch.nn as nn
from unet_model import UNet
from preprocess import get_data_loaders
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

train_loader, val_loader = get_data_loaders('../data/Images', '../data/Labels')

num_epochs = 50
best_val_loss = float('inf')

# Lists to store loss values for plotting
train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    model.train()
    running_train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed, Training Loss: {avg_train_loss:.4f}")

    # Validation step
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Save the model if the validation loss is the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), '../models/unet_model.pth')
        print(f"Model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")

# Plot the loss progression
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_loss_list, label='Training Loss')
plt.plot(epochs, val_loss_list, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Progression During Training')
plt.legend()
plt.savefig('../reports/loss_progression.png')  # Save the plot as a file
plt.show()  # Display the plot