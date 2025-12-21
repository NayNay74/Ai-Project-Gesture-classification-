import torch
import torch.nn as nn
from torchvision import models
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names(path="data/class_names.json"):
    with open(path, "r") as f:
        class_names = json.load(f)
    return class_names


def load_model(weights_path, num_classes):
    model = models.inception_v3(
        weights=None,
        aux_logits=True
    )

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model