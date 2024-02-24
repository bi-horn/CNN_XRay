# -*- coding: utf-8 -*-
"""
CNN with 2 convolutional layers and 3 fully connected layers

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#self-defined convolutional network
class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        #Two convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5, 1)  # Input channels: 3, Output channels: 6, Kernel size: 5x5
        self.conv2 = nn.Conv2d(6, 16, 5, 1)  # Input channels: 6, Output channels: 16, Kernel size: 5x5
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Input size: 16*53*53, Output size: 120
        #53 * 53 because the input image size is 224x224 pixels, and each max-pooling operation with a 2x2 kernel and stride 2 reduces the dimensions by a factor of 2. After passing through two max-pooling layers, the height and width dimensions are reduced to approximately one-fourth of the original size, resulting in a feature map size of approximately 53x53.
        self.fc2 = nn.Linear(120, 84)  # Input size: 120, Output size: 84
        self.fc3 = nn.Linear(84, 2)  # Input size: 84, Output size: 2 (2 classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)  # Max pooling with 2x2 kernel and stride 2; dimensions of the feature map are reduced by a factor of 2 in both height and width
        # Second convolutional layer
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)  # Max pooling with 2x2 kernel and stride 2

        # Flatten the feature map
        X = X.view(-1, 16 * 53 * 53)  # Calculate the new size based on the dimensions of the feature map after the second convolutional layer

        # Fully connected layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)