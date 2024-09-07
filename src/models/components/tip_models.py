import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from glob import glob
from timm.models.layers import DropPath, trunc_normal_
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.num_features = self.model.num_features

    def forward_features(self, x):
        x = self.model.forward_features(x)
        return x

    def pool(self, x):
        if "swin" in self.model_name:
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
            x = self.model.fc_norm(x)
            x = self.model.head_drop(x)
        else:
            assert False

        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        expansion_ratio=2,
        dropout=0.1,
        use_cross_attention=False,
        use_layer_scale=False,
        norm_type="post",
    ):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = (
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            if use_cross_attention
            else None
        )
        self.use_cross_attention = use_cross_attention
        self.linear1 = nn.Linear(d_model, d_model * expansion_ratio)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_ratio, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) if use_cross_attention else None
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout) if use_cross_attention else None

        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.scale1 = nn.Parameter(torch.ones(d_model))
            self.scale2 = nn.Parameter(torch.ones(d_model))
            self.scale3 = nn.Parameter(torch.ones(d_model)) if use_cross_attention else None

        self.norm_type = norm_type
        if self.norm_type not in ["pre", "post"]:
            raise ValueError("norm_type must be either 'pre' or 'post'")

    def forward(self, src, tgt=None, attn_mask=None, key_padding_mask=None):
        if self.norm_type == "pre":
            # Pre-norm Self-attention block
            src2 = self.norm1(src)
            src2 = self.self_attn(
                src2, src2, src2, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
            )[0]
            if self.use_layer_scale:
                src2 = self.scale1 * src2
            src = src + self.dropout1(src2)

            if self.use_cross_attention:
                if tgt is None:
                    raise ValueError("tgt must be provided for cross attention")
                # Pre-norm Cross-attention block
                src2 = self.norm3(src)
                src2 = self.cross_attn(
                    src2, tgt, tgt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
                )[0]
                if self.use_layer_scale:
                    src2 = self.scale3 * src2
                src = src + self.dropout3(src2)

            # Pre-norm Feedforward block
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
            if self.use_layer_scale:
                src2 = self.scale2 * src2
            src = src + self.dropout2(src2)
        else:
            # Post-norm Self-attention block
            src2 = self.self_attn(
                src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
            )[0]
            if self.use_layer_scale:
                src2 = self.scale1 * src2
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            if self.use_cross_attention:
                if tgt is None:
                    raise ValueError("tgt must be provided for cross attention")
                # Post-norm Cross-attention block
                src2 = self.cross_attn(
                    src, tgt, tgt, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
                )[0]
                if self.use_layer_scale:
                    src2 = self.scale3 * src2
                src = src + self.dropout3(src2)
                src = self.norm3(src)

            # Post-norm Feedforward block
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            if self.use_layer_scale:
                src2 = self.scale2 * src2
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class TabularTransformerEncoder(nn.Module):
    """
    Tabular Transformer Encoder based on BERT
    cat_lengths_tabular: categorical feature length list, e.g., [5,4,2]
    con_lengths_tabular: continuous feature length list, e.g., [1,1]
    """

    def __init__(
        self,
        tabular_embedding_dim,
        num_layers,
        num_heads,
        cat_lengths_tabular,
        con_lengths_tabular,
        embedding_dropout=0.1,
        use_layer_scale=False,
        norm_type="post",
        dropout=0,
    ) -> None:
        super(TabularTransformerEncoder, self).__init__()

        self.embedding_dim = tabular_embedding_dim
        self.num_heads = num_heads
        self.num_cat = len(cat_lengths_tabular)
        self.num_con = len(con_lengths_tabular)
        self.num_unique_cat = sum(cat_lengths_tabular)
        print("TabularTransformerEncoder uses Mask Attention")
        # print('TabularTransformerEncoder No Mask Attention')

        # categorical embedding
        cat_offsets = torch.tensor([0] + cat_lengths_tabular[:-1]).cumsum(0)
        self.register_buffer("cat_offsets", cat_offsets, persistent=False)
        self.cat_embedding = nn.Embedding(self.num_unique_cat, tabular_embedding_dim)
        # continuous embedding
        self.con_proj = nn.Linear(1, tabular_embedding_dim)
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, tabular_embedding_dim))
        self.mask_special_token = nn.Parameter(torch.zeros(1, 1, tabular_embedding_dim))
        pos_ids = torch.arange(self.num_cat + self.num_con + 1).expand(1, -1)
        self.register_buffer("pos_ids", pos_ids, persistent=False)
        # print('TabularTransformerEncoder No Column Embedding')
        self.column_embedding = nn.Embedding(self.num_cat + self.num_con + 1, tabular_embedding_dim)

        self.norm = nn.LayerNorm(tabular_embedding_dim)
        self.dropout = nn.Dropout(embedding_dropout) if embedding_dropout > 0.0 else nn.Identity()

        # transformer
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerLayer(
                    tabular_embedding_dim,
                    num_heads,
                    dropout=dropout,
                    use_cross_attention=False,
                    use_layer_scale=use_layer_scale,
                    norm_type=norm_type,
                )
                for i in range(num_layers)
            ]
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_special_token, std=0.02)

    def embedding(self, x, mask_special=None):
        # categorical embedding
        cat_x = self.cat_embedding(x[:, : self.num_cat].long() + self.cat_offsets)
        # continuous embedding
        con_x = self.con_proj(x[:, self.num_cat :].unsqueeze(-1))
        x = torch.cat([cat_x, con_x], dim=1)
        # mask special token
        if mask_special is not None:
            mask_special = mask_special.unsqueeze(-1)
            mask_special_tokens = self.mask_special_token.expand(x.shape[0], x.shape[1], -1)
            x = mask_special * mask_special_tokens + (~mask_special) * x
        # concat
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        column_embed = self.column_embedding(self.pos_ids)
        x = x + column_embed
        x = self.norm(x)
        x = self.dropout(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor = None, mask_special: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.embedding(x, mask_special=mask_special)
        # create attention mask
        if mask is not None:
            B, N = mask.shape
            cls_mask = torch.zeros(B, 1).bool().to(mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            mask = mask[:, None, :].repeat(1, N + 1, 1)
            mask_eye = ~torch.eye(N + 1).bool().to(mask.device)
            mask_eye = mask_eye[None, :, :]
            mask = mask * mask_eye
            mask = mask[:, None, :, :]
            mask = mask * (-1e9)
            assert x.shape[1] == mask.shape[2]
            mask = mask.expand(B, self.num_heads, N + 1, N + 1).reshape(-1, N + 1, N + 1)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask=mask)
            # x = transformer_block(x)
        return x


class MultimodalTransformerEncoder(nn.Module):
    """
    Tabular Transformer Encoder based on BERT
    """

    def __init__(
        self,
        multimodal_embedding_dim,
        image_model_name,
        tabular_embedding_dim,
        num_layers,
        num_heads,
        dropout=0,
        norm_type="post",
        use_layer_scale=False,
    ) -> None:
        super(MultimodalTransformerEncoder, self).__init__()

        self.image_model_name = image_model_name
        self.embedding_dim = multimodal_embedding_dim
        image_embedding_dim = timm.create_model(image_model_name).num_features
        self.image_proj = nn.Linear(image_embedding_dim, multimodal_embedding_dim)
        self.image_norm = nn.LayerNorm(multimodal_embedding_dim)
        self.tabular_proj = (
            nn.Linear(tabular_embedding_dim, multimodal_embedding_dim)
            if tabular_embedding_dim != multimodal_embedding_dim
            else nn.Identity()
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerLayer(
                    multimodal_embedding_dim,
                    num_heads,
                    dropout=dropout,
                    use_cross_attention=True,
                    use_layer_scale=use_layer_scale,
                    norm_type=norm_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(multimodal_embedding_dim)

    def forward(self, x: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        if "swin" in self.image_model_name:
            B, H, W, C = image_features.shape
            image_features = image_features.reshape(B, H * W, C)
        if len(image_features.shape) == 4:
            B, C, H, W = image_features.shape
            image_features = image_features.reshape(B, C, H * W).permute(0, 2, 1)
        image_features = self.image_proj(image_features)
        image_features = self.image_norm(image_features)
        x = self.tabular_proj(x)

        for i, transformer_block in enumerate(self.transformer_blocks):
            x = transformer_block(x, tgt=image_features)
        x = self.norm(x)
        return x


class TabularPredictor(nn.Module):
    """Masked Tabular Reconstruction"""

    def __init__(
        self, tabular_embedding_dim, cat_lengths_tabular, con_lengths_tabular, num_unique_cat: int = None
    ) -> None:
        super(TabularPredictor, self).__init__()
        self.num_cat = len(cat_lengths_tabular)
        self.num_con = len(con_lengths_tabular)
        if num_unique_cat is None:
            self.num_unique_cat = sum(cat_lengths_tabular)
        else:
            self.num_unique_cat = num_unique_cat
        # categorical classifier
        self.cat_classifier = nn.Linear(tabular_embedding_dim, self.num_unique_cat, bias=True)
        # continuous regessor
        self.con_regressor = nn.Linear(tabular_embedding_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # remove clstoken
        x = x[:, 1:, :]
        # categorical classifier
        cat_x = self.cat_classifier(x[:, : self.num_cat])
        # continuous regessor
        con_x = self.con_regressor(x[:, self.num_cat :])
        return (cat_x, con_x)


class TIPBackboneEnsemble(nn.Module):
    def __init__(
        self,
        encoder_image,
        encoder_tabular,
        encoder_multimodal,
        ckpt_path=None,
        finetune_strategy_image="trainable",
        finetune_strategy_tabular="trainable",
        finetune_strategy_multimodal="trainable",
        num_classes=2,
        dropout_classifier=0.0,
        use_n_clusters: int = None,
        use_image=True,
        use_tabular=True,
        use_multimodal=True,
    ) -> None:
        super(TIPBackboneEnsemble, self).__init__()
        self.use_image = use_image
        self.use_tabular = use_tabular
        self.use_multimodal = use_multimodal

        if ckpt_path is not None:
            ckpt_path = glob(ckpt_path)[0]
            print(f"Checkpoint name: {ckpt_path}")
            # Load weights
            checkpoint = torch.load(ckpt_path)
            state_dict = checkpoint["state_dict"]
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

        self.encoder_image = encoder_image
        self.encoder_tabular = encoder_tabular
        self.encoder_multimodal = encoder_multimodal

        if ckpt_path is not None:
            for module, module_name, finetune_strategy in zip(
                [self.encoder_image, self.encoder_tabular, self.encoder_multimodal],
                ["encoder_image", "encoder_tabular", "encoder_multimodal"],
                [finetune_strategy_image, finetune_strategy_tabular, finetune_strategy_multimodal],
            ):
                self.load_weights(module, module_name, state_dict)
                if finetune_strategy == "frozen":
                    for _, param in module.named_parameters():
                        param.requires_grad = False
                    parameters = list(filter(lambda p: p.requires_grad, module.parameters()))
                    assert len(parameters) == 0
                    print(f"Freeze {module_name}")
                elif finetune_strategy == "trainable":
                    print(f"Full finetune {module_name}")
                else:
                    assert False, f"Unknown finetune strategy {finetune_strategy}"

        self.classifier_multimodal = nn.Sequential(
            nn.Dropout(dropout_classifier), nn.Linear(self.encoder_multimodal.embedding_dim, num_classes)
        )
        self.classifier_imaging = nn.Sequential(
            nn.Dropout(dropout_classifier), nn.Linear(self.encoder_image.num_features, num_classes)
        )
        self.classifier_tabular = nn.Sequential(
            nn.Dropout(dropout_classifier), nn.Linear(self.encoder_tabular.embedding_dim, num_classes)
        )

        if use_n_clusters is not None:
            self.classifier_clusters = nn.Sequential(
                nn.Dropout(dropout_classifier),
                nn.Linear(self.encoder_multimodal.embedding_dim, use_n_clusters),
            )

    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not "projection_head" in k and not "prototypes" in k:
                state_dict_module[k[len(module_name) + 1 :]] = state_dict[k]
        print(f"Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}")
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0

    def forward_image(self, x_i):
        x_i = self.encoder_image.forward_features(x_i)
        out_i = self.classifier_imaging(self.encoder_image.pool(x_i))

        return x_i, out_i

    def forward_tabular(self, x_t):
        x_t = self.encoder_tabular(x_t)  # (B,N_t,C)
        out_t = self.classifier_tabular(x_t[:, 0, :])

        return x_t, out_t

    def forward_multimodal(self, x_i, x_t):
        x_m = self.encoder_multimodal(x=x_t, image_features=x_i)
        out_m = self.classifier_multimodal(x_m[:, 0, :])

        return x_m, out_m

    def forward(self, x_i: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        x_i, out_i = self.forward_image(x_i)
        x_t, out_t = self.forward_tabular(x_t)
        self.x_m, out_m = self.forward_multimodal(x_i, x_t)

        out = []
        if self.use_image:
            out.append(out_i)
        if self.use_tabular:
            out.append(out_t)
        if self.use_multimodal:
            out.append(out_m)
        x = torch.stack(out).mean(0)

        return x

    def forward_cluster(self):
        x = self.classifier_clusters(self.x_m[:, 0, :])
        return x


class TIPBackboneOnlyTabular(nn.Module):
    def __init__(
        self,
        encoder_tabular,
        ckpt_path=None,
        finetune_strategy_tabular="trainable",
        num_classes=2,
        dropout_classifier=0.0,
        use_n_clusters: int = None,
    ) -> None:
        super().__init__()

        if ckpt_path is not None:
            ckpt_path = glob(ckpt_path)[0]
            print(f"Checkpoint name: {ckpt_path}")
            # Load weights
            checkpoint = torch.load(ckpt_path)
            state_dict = checkpoint["state_dict"]
            state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

        self.encoder_tabular = encoder_tabular

        if ckpt_path is not None:
            for module, module_name, finetune_strategy in zip(
                [self.encoder_tabular],
                ["encoder_tabular"],
                [finetune_strategy_tabular],
            ):
                self.load_weights(module, module_name, state_dict)
                if finetune_strategy == "frozen":
                    for _, param in module.named_parameters():
                        param.requires_grad = False
                    parameters = list(filter(lambda p: p.requires_grad, module.parameters()))
                    assert len(parameters) == 0
                    print(f"Freeze {module_name}")
                elif finetune_strategy == "trainable":
                    print(f"Full finetune {module_name}")
                else:
                    assert False, f"Unknown finetune strategy {finetune_strategy}"

        self.classifier_tabular = nn.Sequential(
            nn.Dropout(dropout_classifier), nn.Linear(self.encoder_tabular.embedding_dim, num_classes)
        )

        if use_n_clusters is not None:
            self.classifier_clusters = nn.Sequential(
                nn.Dropout(dropout_classifier),
                nn.Linear(self.encoder_multimodal.embedding_dim, use_n_clusters),
            )

    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not "projection_head" in k and not "prototypes" in k:
                state_dict_module[k[len(module_name) + 1 :]] = state_dict[k]
        print(f"Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}")
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0

    def forward_tabular(self, x_t):
        x_t = self.encoder_tabular(x_t)  # (B,N_t,C)
        out_t = self.classifier_tabular(x_t[:, 0, :])

        return x_t, out_t

    def forward(self, x_i: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        self.x_t, out = self.forward_tabular(x_t)

        return out

    def forward_cluster(self):
        x = self.classifier_clusters(self.x_t[:, 0, :])
        return x
