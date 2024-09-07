import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.isic_utils.utils import MAX_CATEGORY_DIM


class ISICModel(nn.Module):
    def __init__(
        self,
        num_inputs_num,
        num_inputs_cat,
        cat_dims=4,
        hidden_dims=512,
        feature_dims=128,
        num_classes=2,
        dropout=0.3,
        input_dropout=0,
        use_cat=True,
    ):
        super(ISICModel, self).__init__()
        self.embeddings = nn.ModuleList()
        for i in range(num_inputs_cat):
            self.embeddings.append(nn.Embedding(MAX_CATEGORY_DIM, cat_dims))
        if use_cat:
            input_dim = num_inputs_num + cat_dims * num_inputs_cat
        else:
            input_dim = num_inputs_num

        self.features = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, feature_dims),
            nn.BatchNorm1d(feature_dims),
            nn.GELU(),
        )
        self.head = nn.Linear(feature_dims, num_classes)
        self.num_features = feature_dims
        self.use_cat = use_cat

    def forward_features(self, metadata_num, metadata_cat):
        x = [metadata_num]
        if self.use_cat:
            for i, embedding in enumerate(self.embeddings):
                x.append(embedding(metadata_cat[:, i]))
        x = torch.concat(x, dim=-1)
        return self.features(x)

    def forward_head(self, x):
        return self.head(x)

    def forward(self, metadata_num, metadata_cat):
        x = self.forward_features(metadata_num, metadata_cat)
        x = self.forward_head(x)

        return x
