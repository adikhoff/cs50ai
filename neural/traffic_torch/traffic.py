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


def show_image(img, label):
    img = img.permute(1, 2, 0)
    print(f"Label: {label}")
    plt.imshow(img, cmap="gray")
    plt.show()

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    loader = load_data(sys.argv[1])
    
    # Display image and label.
    images, labels = next(iter(loader))
    print(f"Feature batch shape: {images.size()}")
    print(f"Labels batch shape: {labels.size()}")
    for i in range(0, 5):
        img = images[i]
        label = labels[i]
        show_image(img, label)

    # Split data into training and testing sets
    # labels = tf.keras.utils.to_categorical(labels)
    # x_train, x_test, y_train, y_test = train_test_split(
    #     np.array(images), np.array(labels), test_size=TEST_SIZE
    # )

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


class LabelDirImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_labels = []
        for i in range(0, NUM_CATEGORIES):
            dir = os.path.join(data_dir, str(i))
            for file_name in os.listdir(dir):
                self.img_labels.append((os.path.join(str(i), file_name), i))
        
        self.img_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = cv2.imread(img_path)
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        return (tf.to_tensor(image), label)


def load_data(data_dir):
    """
    Create loader for image data from directory `data_dir`.
    """
    def resize_cv2(image):
        height, width = image.shape[:2]
        if height != IMG_HEIGHT or width != IMG_WIDTH:
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        return image
    
    dataset = LabelDirImageDataset(data_dir, resize_cv2)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


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
