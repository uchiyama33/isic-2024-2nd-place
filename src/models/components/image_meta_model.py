import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from glob import glob
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class ISICModel(nn.Module):
    def __init__(
        self,
        image_model,
        meta_model,
        num_classes=2,
        dropout=0.5,
        head_mlp=False,
        hidden_dims=256,
        image_pretrain_path=None,
        meta_pretrain_path=None,
        use_bn=False,
        inter_ce=False,
    ):
        super(ISICModel, self).__init__()
        self.image_model = image_model
        self.meta_model = meta_model
        self.inter_ce = inter_ce

        if head_mlp:
            self.head = nn.Sequential(
                (
                    nn.BatchNorm1d(image_model.num_features + meta_model.num_features)
                    if use_bn
                    else nn.Identity()
                ),
                nn.Dropout(dropout),
                nn.Linear(image_model.num_features + meta_model.num_features, hidden_dims),
                nn.BatchNorm1d(hidden_dims),
                nn.GELU(),
                nn.Linear(hidden_dims, num_classes),
            )
        else:
            self.head = nn.Sequential(
                (
                    nn.BatchNorm1d(image_model.num_features + meta_model.num_features)
                    if use_bn
                    else nn.Identity()
                ),
                nn.Dropout(dropout),
                nn.Linear(image_model.num_features + meta_model.num_features, num_classes),
            )

        if image_pretrain_path is not None:
            ckpt_path = glob(image_pretrain_path)[0]
            state_dict = torch.load(ckpt_path)["state_dict"]
            state_dict = {key.replace("net.", ""): value for key, value in state_dict.items()}
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
            self.image_model.load_state_dict(state_dict)
        if meta_pretrain_path is not None and meta_pretrain_path != "":
            ckpt_path = glob(meta_pretrain_path)[0]
            state_dict = torch.load(ckpt_path)["state_dict"]
            state_dict = {key.replace("net.", ""): value for key, value in state_dict.items()}
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
            self.meta_model.load_state_dict(state_dict)

        if inter_ce:
            self.inter_ce_image = InterCEResModule(image_model.num_features, num_classes)
            self.inter_ce_meta = InterCEResModule(meta_model.num_features, num_classes)

    def forward(self, image, metadata_num, metadata_cat):
        image_features = self.image_model.forward_features(image, pool=True)
        meta_features = self.meta_model.forward_features(metadata_num, metadata_cat)

        if self.inter_ce:
            image_features, self.image_logits = self.inter_ce_image(image_features)
            meta_features, self.meta_logits = self.inter_ce_meta(meta_features)

        features = torch.concat([image_features, meta_features], dim=-1)
        output = self.head(features)

        return output

    def get_inter_ce_logits(self):
        return self.image_logits, self.meta_logits


class InterCEResModule(nn.Module):

    def __init__(self, dim_model, num_classes):
        super(InterCEResModule, self).__init__()

        self.proj_1 = nn.Linear(dim_model, num_classes)
        self.proj_2 = nn.Linear(num_classes, dim_model)

    def forward(self, x):

        logits = self.proj_1(x)
        x = x + self.proj_2(logits.softmax(dim=-1))

        return x, logits
