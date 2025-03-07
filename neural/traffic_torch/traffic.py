import os
import sys

import matplotlib.pyplot as plt
import cv2

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize
import torchvision.transforms.functional as tf

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
BATCH_SIZE = 32

# Config GPUs
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def show_image(img, index, label):
    print(f"Showing image {index} with label: {label}")
    plt.imshow(img.permute(1, 2, 0), cmap="gray")
    plt.show()

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    dataset = load_data(sys.argv[1])
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1 - TEST_SIZE, TEST_SIZE])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Display image and label.
    images, labels = next(iter(train_loader))
    for i in range(0, 5):
        img = images[i]
        label = labels[i]
        show_image(img, i, label)

    # # Get a compiled neural network
    # model = get_model()

    # # Fit model on training data
    # model.fit(x_train, y_train, epochs=EPOCHS)

    # # Evaluate neural network performance
    # model.evaluate(x_test,  y_test, verbose=2)

    # # Save model to file
    # if len(sys.argv) == 3:
    #     filename = sys.argv[2]
    #     model.save(filename)
    #     print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Create dataset for image data from directory `data_dir`.
    """
    class LabelDirImageDataset(Dataset):
        def __init__(self, data_dir):
            print(f"Investigating {data_dir}...")
            self.img_labels = []
            for i in range(0, NUM_CATEGORIES):
                dir = os.path.join(data_dir, str(i))
                for file_name in os.listdir(dir):
                    self.img_labels.append((os.path.join(str(i), file_name), i))
            print(f"Found {len(self.img_labels)} images in {NUM_CATEGORIES} label directories")
            self.img_dir = data_dir

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
            label = self.img_labels[idx][1]
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # Change from BGR to RGB
            height, width = image.shape[:2]
            if height != IMG_HEIGHT or width != IMG_WIDTH: # Make uniform size
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)

            tensor = tf.to_tensor(image)
            return (tensor, label)

    return LabelDirImageDataset(data_dir)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(
    #         256, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    #     ),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     tf.keras.layers.Dense(128, activation="relu"),
    #     # tf.keras.layers.Dense(256, activation="relu"),
    #     tf.keras.layers.Dropout(0.50),
    #     tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    # ])
    # model.compile(
    #     optimizer="adam",
    #     loss="categorical_crossentropy",
    #     metrics=["accuracy"]
    # )
    # return model


if __name__ == "__main__":
    main()
