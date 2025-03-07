import sys

import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from torch_helpers import LabelDirImageDataset
from torch_helpers import TrainingRunner
from torch_helpers import TestRunner

from networks import ImageRecognitionNetwork
from networks import ImageRecognitionNetworkGrok
from networks import ImageRecognitionNetworkChatGPT

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
BATCH_SIZE = 64

# Config GPUs
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    dataset = load_data(sys.argv[1])
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1 - TEST_SIZE, TEST_SIZE])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Get a compiled neural network
    model = get_model().to(device)
    
    print(f"model: {model}")

    config = {
        "device": device,
        "model": model,
        "criterion": nn.CrossEntropyLoss(),
        "optimizer": optim.Adam(model.parameters()),
    }

    # Training loop
    training_runner = TrainingRunner(config)
    for epoch in range(EPOCHS):
        loss = training_runner.run_model(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")
        
    # Evaluation
    test_runner = TestRunner(config)
    with torch.no_grad():
        (test_loss, accuracy) = test_runner.run_model(test_loader)
        print(f"Test Loss: {test_loss}, Accuracy: {accuracy}")

    # Save model if filename is provided
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}.")
    

def load_data(data_dir):
    """
    Create dataset for image data from directory `data_dir`.
    """
    def convert_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Change from BGR to RGB
        height, width = image.shape[:2]
        if height != IMG_HEIGHT or width != IMG_WIDTH:  # Make uniform size
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        return image
    
    return LabelDirImageDataset(data_dir, convert_image)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
       
    return ImageRecognitionNetworkChatGPT(IMG_WIDTH, IMG_HEIGHT, NUM_CATEGORIES)


if __name__ == "__main__":
    main()
