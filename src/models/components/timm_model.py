import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.layers.gem import GeM


class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True):
        super(ISICModel, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")

        self.pooling = GeM()
        self.linear = nn.Linear(self.model.num_features, num_classes)

    def forward(self, images):
        features = self.model.forward_features(images)
        if "swinv2" in self.model_name:
            features = features.permute(0, 3, 1, 2)
        elif "eva" in self.model_name:
            features = features[:, self.model.num_prefix_tokens :].transpose(1, 2)
            features = features.reshape(
                features.shape[0],
                features.shape[1],
                int(math.sqrt(features.shape[2])),
                int(math.sqrt(features.shape[2])),
            )
        pooled_features = self.pooling(features).flatten(1)
        output = self.linear(pooled_features)
        return output
