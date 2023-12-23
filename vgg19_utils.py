import torch.nn as nn

import torchvision.models as models

from typing import Optional

class ModifiedVGG19(nn.Module):

    def __init__(self, num_classes, *, pretrained : Optional[ bool ] = True):
        
        super(ModifiedVGG19, self).__init__()

        # Load pre-trained VGG19 model with batch normalization
        self.features = models.vgg19_bn(pretrained = pretrained).features

        # Replace the classifier with global average pooling and a new linear layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, num_classes),  # Adjust the input size based on your specific vgg19_bn configuration
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        # Forward pass through the modified VGG19 model
        x = self.features(x)
        x = self.classifier(x)
        return x
