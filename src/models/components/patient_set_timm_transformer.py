import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from glob import glob
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.layers.gem import GeM


class PreNormSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, use_layer_scale=False, layer_scale_init_value=1e-5):
        super(PreNormSelfAttention, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.full((d_model,), layer_scale_init_value))

    def forward(self, src):
        src2 = self.norm(src)
        src2, _ = self.self_attn(src2, src2, src2, need_weights=False)
        if self.use_layer_scale:
            src2 = self.layer_scale.unsqueeze(0).unsqueeze(0) * src2
        src = src + self.dropout(src2)
        return src


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0, use_layer_scale=False, layer_scale_init_value=1e-5):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.full((d_model,), layer_scale_init_value))

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if self.use_layer_scale:
            x = self.layer_scale.unsqueeze(0).unsqueeze(0) * x
        return x


class PreNormTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.0, use_layer_scale=False, layer_scale_init_value=1e-5):
        super(PreNormTransformerEncoderLayer, self).__init__()
        self.self_attn = PreNormSelfAttention(
            d_model, nhead, dropout, use_layer_scale, layer_scale_init_value
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout, use_layer_scale, layer_scale_init_value)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.self_attn(src)
        src2 = self.norm(src)
        src = src + self.dropout(self.ff(src2))
        return src


class PreNormTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ff,
        dropout=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ):
        super(PreNormTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                PreNormTransformerEncoderLayer(
                    d_model, nhead, d_ff, dropout, use_layer_scale, layer_scale_init_value
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return self.norm(src)


class ISICModel(nn.Module):
    def __init__(
        self,
        model_name,
        batch_size,
        num_classes=2,
        pretrained=True,
        dropout_encoder=None,
        projection_dim=256,
        num_layers_transformer=2,
        num_heads_transformer=4,
        dropout_transformer=0.0,
        my_pretrain_path=None,
        target_meta=False,
        num_meta_feature=68,
    ):
        super(ISICModel, self).__init__()
        self.model_name = model_name
        self.target_meta = target_meta
        self.batch_size = batch_size
        self.image_encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if projection_dim is None:
            self.projection = nn.Identity()
            self.num_features = self.image_encoder.num_features
        else:
            self.projection = nn.Linear(self.image_encoder.num_features, projection_dim)
            self.num_features = projection_dim

        self.transformer = PreNormTransformerEncoder(
            num_layers_transformer,
            self.num_features,
            num_heads_transformer,
            self.num_features * 2,
            dropout_transformer,
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

    def forward_transformer(self, x):
        x = self.projection(x)
        x = x.reshape(self.batch_size, -1, self.num_features).permute(1, 0, 2)
        x = self.transformer(x).permute(1, 0, 2).reshape(-1, self.num_features)
        return x

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x, pool=True)
        x = self.forward_transformer(x)
        self.features = x
        x = self.forward_head(x)

        return x

    def forward_meta(self):
        return self.head_meta(self.features)
