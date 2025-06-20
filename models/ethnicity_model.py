import torch.nn as nn
from torchvision import models

def get_model(num_classes, pretrained=True):
    # Load ResNet18
    model = models.resnet18(pretrained=pretrained)

    # Get the number of input features to the final layer
    in_features = model.fc.in_features

    # Replace the final classification layer with a new one
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model
