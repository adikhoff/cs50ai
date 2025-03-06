import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Constants
EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
BATCH_SIZE = 32

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.pth]")

    # Load image arrays and labels
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE
    )

    # Convert to NumPy arrays and normalize pixel values to [0, 1]
    x_train = np.stack([img.astype(np.float32) / 255.0 for img in x_train], axis=0)
    x_test = np.stack([img.astype(np.float32) / 255.0 for img in x_test], axis=0)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Convert to Torch tensors
    # Permute dimensions to (batch_size, channels, height, width) for PyTorch
    x_train = torch.from_numpy(x_train).permute(0, 3, 1, 2)
    x_test = torch.from_numpy(x_test).permute(0, 3, 1, 2)
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    # Create datasets and data loaders
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model and move to device
    model = get_model().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss /= len(test_loader)
        accuracy = correct / total
        print(f"Test Loss: {test_loss}, Accuracy: {accuracy}")

    # Save model if filename is provided
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir, save_dir=None):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` is a list of all images as numpy
    ndarrays with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` is a list of
    integer labels corresponding to each image's category.
    """
    labels = []
    images = []
    for i in range(0, NUM_CATEGORIES):
        dir = f"{data_dir}{os.sep}{i}"
        print(f"reading dir {dir}")
        for file_name in os.listdir(dir):
            image = cv2.imread(f"{dir}{os.sep}{file_name}")
            height, width = image.shape[:2]
            if height != IMG_HEIGHT or width != IMG_WIDTH:
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            if save_dir is not None:
                save_image(save_dir, file_name, image)
            images.append(image)
            labels.append(i)
    return (images, labels)

def save_image(save_dir, file_name, image):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name_out = save_dir + os.sep + file_name.split('.')[0] + ".png"
    print(f"Writing {file_name_out}")
    cv2.imwrite(file_name_out, image)

def get_model():
    """
    Returns a convolutional neural network model. The input shape is assumed to be
    (batch_size, 3, IMG_HEIGHT, IMG_WIDTH), and the output layer has NUM_CATEGORIES units.
    """
    class TrafficSignNet(nn.Module):
        def __init__(self):
            super(TrafficSignNet, self).__init__()
            # 3 input channels (RGB), 256 output channels, 3x3 kernel, padding to maintain size
            self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            # After conv (30x30) and pool (15x15), 256 channels
            self.fc1 = nn.Linear(256 * (IMG_HEIGHT // 2) * (IMG_WIDTH // 2), 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc4 = nn.Linear(256, NUM_CATEGORIES)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.dropout(x)
            x = self.fc4(x)  # Output logits (no softmax)
            return x
    return TrafficSignNet()

if __name__ == "__main__":
    main()
    
