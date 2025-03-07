from torch import nn

"""
Network that I created myself, based on the original
TensorFlow network, which was based on the lecture.
"""


class ImageRecognitionNetwork(nn.Module):
    def __init__(self, img_height, img_width, num_categories):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256 * (img_height // 2) * (img_width // 2), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_categories),
        )

    def forward(self, x):
        x = self.convolution(x)
        return self.linear_relu_stack(x)


"""
Network after I asked Grok to improve it. Not sure what
all the steps do, but a couple of key points:
* Most of the work is done by the convolution sequence.
* Convolution sequence has dropout configured.
* Added BatchNorm steps, which (I think) helps keep the
  network optimized for backward propagation.
"""


class ImageRecognitionNetworkAI(nn.Module):
    def __init__(self, img_height, img_width, num_categories):
        super().__init__()
        self.convolution = nn.Sequential(
            # First Conv Block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Second Conv Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Flatten(),
        )
        
        # Calculate the size after convolutions and pooling
        # IMG_HEIGHT and IMG_WIDTH are divided by 4 due to two MaxPool layers (2x2 each)
        flattened_size = 128 * (img_height // 4) * (img_width // 4)
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_categories),
        )

    def forward(self, x):
        x = self.convolution(x)
        return self.linear_relu_stack(x)
