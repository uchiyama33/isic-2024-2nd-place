import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from glob import glob
import rootutils
from torch_geometric.nn.models import GAT

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.layers.gem import GeM


def create_fully_connected_edge_index(N, M):
    row = []
    col = []
    for i in range(N):
        for j in range(M):
            for k in range(M):
                row.append(i * M + j)
                col.append(i * M + k)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index


class ISICModel(nn.Module):
    def __init__(
        self,
        model_name,
        batch_size,
        n_data_per_patient,
        num_classes=2,
        pretrained=True,
        dropout_encoder=None,
        projection_dim=256,
        num_layers_gat=1,
        num_heads_gat=4,
        dropout_gat=0.0,
        my_pretrain_path=None,
        target_meta=False,
        num_meta_feature=68,
    ):
        super(ISICModel, self).__init__()
        self.model_name = model_name
        self.target_meta = target_meta
        self.batch_size = batch_size
        self.n_data_per_patient = n_data_per_patient

        self.register_buffer("edge_index", create_fully_connected_edge_index(batch_size, n_data_per_patient))

        self.image_encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if projection_dim is None:
            self.projection = nn.Identity()
            self.num_features = self.image_encoder.num_features
        else:
            self.projection = nn.Linear(self.image_encoder.num_features, projection_dim)
            self.num_features = projection_dim

        self.gat = GAT(
            in_channels=self.num_features,
            hidden_channels=self.num_features,
            num_layers=num_layers_gat,
            out_channels=self.num_features,
            dropout=dropout_gat,
            v2=True,
        )

        self.head = nn.Linear(self.num_features, num_classes)
        if target_meta:
            self.head_meta = nn.Linear(self.num_features, num_meta_feature)

        if my_pretrain_path:
            ckpt_path = glob(my_pretrain_path)[0]
            state_dict = torch.load(ckpt_path)["state_dict"]
            state_dict = {key.replace("net.", ""): value for key, value in state_dict.items()}
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
            state_dict = {key.replace("model.", "image_encoder."): value for key, value in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

        if dropout_encoder:
            if "swinv2" in model_name:
                encoder = self.image_encoder.layers
            elif "eva02" in model_name:
                encoder = self.image_encoder.blocks
            elif "maxvit" in model_name:
                encoder = self.image_encoder.stages
            elif "mobilenetv4" in model_name:
                encoder = self.image_encoder.blocks
            else:
                assert False
            for name, module in encoder.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_encoder

        if target_meta:
            assert NotImplementedError

    def forward_features(self, x, pool=False):
        x = self.image_encoder.forward_features(x)
        if pool:
            if "swinv2" in self.model_name:
                x = self.image_encoder.head.global_pool(x)
            elif "efficientnet" in self.model_name:
                x = self.image_encoder.global_pool(x)
            elif "convnextv2" in self.model_name:
                x = self.image_encoder.head.global_pool(x)
                x = self.image_encoder.head.norm(x)
                x = self.image_encoder.head.flatten(x)
            elif "eva" in self.model_name:
                x = x[:, self.image_encoder.num_prefix_tokens :].mean(dim=1)
                x = self.image_encoder.fc_norm(x)
                x = self.image_encoder.head_drop(x)
            elif "maxvit" in self.model_name:
                x = self.image_encoder.head.global_pool(x)
                x = self.image_encoder.head.norm(x)
                x = self.image_encoder.head.flatten(x)
            elif "mobilenetv4" in self.model_name:
                x = self.image_encoder.global_pool(x)
                x = self.image_encoder.conv_head(x)
                x = self.image_encoder.norm_head(x)
                x = self.image_encoder.act2(x)
                x = self.image_encoder.flatten(x)
            else:
                assert False
        return x

    def forward_gat(self, x):
        x = self.projection(x)
        x = self.gat(x, self.edge_index)
        return x

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x, pool=True)
        x = self.forward_gat(x)
        self.features = x
        x = self.forward_head(x)

        return x

    def forward_meta(self):
        return self.head_meta(self.features)
