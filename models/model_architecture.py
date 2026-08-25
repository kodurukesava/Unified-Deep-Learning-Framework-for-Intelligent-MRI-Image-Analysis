import torch
import torch.nn as nn

from torchvision.models import resnet50


def create_resnet50(
    num_classes
):

    model = resnet50(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model