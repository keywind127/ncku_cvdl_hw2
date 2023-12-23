import torch.nn as nn

import torchvision.models as models

class ModifiedResNet50(nn.Module):

    def __init__(self):

        super(ModifiedResNet50, self).__init__()

        self.features = nn.Sequential(*list(models.resnet50(pretrained = True).children())[:-2])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x
