import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from glob import glob
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.layers.gem import GeM


class ISICModel(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=2,
        pretrained=True,
        dropout_encoder=None,
        dropout_head=None,
        my_pretrain_path=None,
        target_meta=False,
        num_meta_feature_num=34,
        num_classes_meta_feature_cat=[3, 2, 21, 8],
        separate_head=False,
        use_n_clusters=None,
    ):
        super(ISICModel, self).__init__()
        self.model_name = model_name
        self.target_meta = target_meta
        self.separate_head = separate_head
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        if "mobilenetv4_conv_medium" in model_name:
            self.num_features = 1280
        else:
            self.num_features = self.model.num_features

        # define head
        if separate_head:
            if "eva02" in model_name:
                tmp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
                self.head = nn.Sequential(
                    tmp_model.fc_norm,
                    tmp_model.head_drop,
                    tmp_model.head,
                )
            elif "efficientnet" in model_name:
                tmp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
                self.head = tmp_model.classifier
            elif "convnext" in model_name:
                tmp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
                self.head = tmp_model.head
            elif "beit" in model_name:
                tmp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
                self.head = nn.Sequential(
                    tmp_model.fc_norm,
                    tmp_model.head_drop,
                    tmp_model.head,
                )
            elif "cait" in model_name:
                tmp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
                self.head = nn.Sequential(
                    tmp_model.head_drop,
                    tmp_model.head,
                )
            elif "vit" in model_name:
                tmp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
                self.head = nn.Sequential(
                    tmp_model.fc_norm,
                    tmp_model.head_drop,
                    tmp_model.head,
                )
            else:
                assert False

        if my_pretrain_path:
            ckpt_path = glob(my_pretrain_path)[0]
            state_dict = torch.load(ckpt_path)["state_dict"]
            state_dict = {key.replace("net.", ""): value for key, value in state_dict.items()}
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}
            self.load_state_dict(state_dict)

        if dropout_encoder:
            if "swinv2" in model_name:
                encoder = self.model.layers
            elif "eva02" in model_name:
                encoder = self.model.blocks
            elif "maxvit" in model_name:
                encoder = self.model.stages
            elif "mobilenetv4" in model_name:
                encoder = self.model.blocks
            else:
                assert False
            for name, module in encoder.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_encoder

        if dropout_head:
            if "swinv2" in model_name:
                head = self.model.head
            elif "eva02" in model_name:
                head = self.model.head_drop
            elif "maxvit" in model_name:
                head = self.model.head
            elif "mobilenetv4" in model_name:
                assert False, "mobilenetv4 does not have a dropout layer"
            else:
                assert False
            for name, module in head.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_head

        if target_meta:
            if "convnext" in model_name:
                self.head_meta_num = nn.Sequential(
                    timm.create_model(model_name, pretrained=pretrained, num_classes=2048).head,
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Linear(2048, num_meta_feature_num),
                )
                self.head_meta_cat_list = nn.ModuleList(
                    [
                        nn.Sequential(
                            timm.create_model(model_name, pretrained=pretrained, num_classes=1024).head,
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Linear(1024, n),
                        )
                        for n in num_classes_meta_feature_cat
                    ]
                )
            elif "efficientnet" in model_name:
                self.head_meta_num = nn.Sequential(
                    timm.create_model(model_name, pretrained=pretrained, num_classes=2048).global_pool,
                    timm.create_model(model_name, pretrained=pretrained, num_classes=2048).classifier,
                    nn.BatchNorm1d(2048),
                    nn.ReLU(),
                    nn.Linear(2048, num_meta_feature_num),
                )
                self.head_meta_cat_list = nn.ModuleList(
                    [
                        nn.Sequential(
                            timm.create_model(
                                model_name, pretrained=pretrained, num_classes=1024
                            ).global_pool,
                            timm.create_model(model_name, pretrained=pretrained, num_classes=1024).classifier,
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Linear(1024, n),
                        )
                        for n in num_classes_meta_feature_cat
                    ]
                )
            # elif "eva02" in model_name:
            #     tmp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_meta_feature)
            #     self.head_meta = nn.Sequential(
            #         tmp_model.fc_norm,
            #         tmp_model.head_drop,
            #         tmp_model.head,
            #     )
            else:
                assert False

        if use_n_clusters is not None:
            if "eva02" in model_name:
                tmp_model = timm.create_model(model_name, pretrained=pretrained, num_classes=use_n_clusters)
                self.head_clusters = nn.Sequential(
                    tmp_model.fc_norm,
                    tmp_model.head_drop,
                    tmp_model.head,
                )
            elif "swin" in model_name:
                self.head_clusters = timm.create_model(
                    model_name, pretrained=pretrained, num_classes=use_n_clusters
                ).head
            else:
                self.head_clusters = timm.create_model(
                    model_name, pretrained=pretrained, num_classes=use_n_clusters
                )

    def forward_features(self, x, pool=False):
        x = self.model.forward_features(x)
        if pool:
            x = self.pool(x)
        return x

    def pool(self, x):
        if "swinv2" in self.model_name:
            x = self.model.head.global_pool(x)
        elif "efficientnet" in self.model_name:
            x = self.model.global_pool(x)
        elif "convnextv2" in self.model_name:
            x = self.model.head.global_pool(x)
            x = self.model.head.norm(x)
            x = self.model.head.flatten(x)
        elif "maxvit" in self.model_name:
            x = self.model.head.global_pool(x)
            x = self.model.head.norm(x)
            x = self.model.head.flatten(x)
        elif "mobilenetv4" in self.model_name:
            x = self.model.global_pool(x)
            x = self.model.conv_head(x)
            x = self.model.norm_head(x)
            x = self.model.act2(x)
            x = self.model.flatten(x)
        elif "eva02" in self.model_name:
            x = x[:, self.model.num_prefix_tokens :].mean(dim=1)
        elif "beit" in self.model_name:
            x = x[:, self.model.num_prefix_tokens :].mean(dim=1)
        elif "cait" in self.model_name:
            x = x[:, 0]
        elif "vit" in self.model_name:
            x = self.model.pool(x)
        else:
            assert False

        return x

    def forward_head(self, x):
        if self.separate_head:
            if (
                "eva02" in self.model_name
                or "efficientnet" in self.model_name
                or "beit" in self.model_name
                or "cait" in self.model_name
                or "vit" in self.model_name
            ):
                x = self.pool(x)
            x = self.head(x)
        else:
            x = self.model.forward_head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        self.features = x
        x = self.forward_head(x)

        return x

    def forward_meta(self):
        if "eva02" in self.model_name:
            self.features = self.pool(self.features)
        logits_num = self.head_meta_num(self.features)
        logits_cat_list = [head(self.features) for head in self.head_meta_cat_list]
        return logits_num, logits_cat_list

    def forward_cluster(self):
        if "eva02" in self.model_name or "swin" in self.model_name:
            if "eva02" in self.model_name:
                self.features = self.pool(self.features)
            logits = self.head_clusters(self.features)
        else:
            logits = self.head_clusters.forward_head(self.features)
        return logits
