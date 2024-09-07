from typing import List, Tuple, Dict, Any

import torch
from lightning import LightningModule
import torchmetrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import rootutils
from lightly.models.modules import SimCLRProjectionHead
from sklearn.linear_model import LogisticRegression

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.utils.reconstruct_loss import ReconstructionLoss
from src.models.utils.clip_loss import CLIPLoss


class ISIC2024LitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        encoder_image,
        encoder_tabular,
        encoder_multimodal,
        predictor_tabular,
        cat_lengths_tabular,
        con_lengths_tabular,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        batch_size,
        multimodal_embedding_dim,
        mlp_image_dim=2048,
        projection_dim=128,
        temperature=0.1,
        lambda_0=0.5,
        classifier_freq=5,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.encoder_image = encoder_image
        self.encoder_tabular = encoder_tabular
        self.cat_lengths_tabular = cat_lengths_tabular
        self.con_lengths_tabular = con_lengths_tabular

        self.projector_image = SimCLRProjectionHead(encoder_image.num_features, mlp_image_dim, projection_dim)
        self.projector_tabular = SimCLRProjectionHead(
            encoder_tabular.embedding_dim, encoder_tabular.embedding_dim, projection_dim
        )

        self.encoder_multimodal = encoder_multimodal
        self.predictor_tabular = predictor_tabular

        # image tabular matching
        self.itm_head = nn.Linear(multimodal_embedding_dim, 2)

        # loss
        nclasses = batch_size
        self.criterion_val_itc = CLIPLoss(temperature=temperature, lambda_0=lambda_0)
        self.criterion_train_itc = self.criterion_val_itc
        self.criterion_tr = ReconstructionLoss(
            num_cat=len(cat_lengths_tabular),
            cat_offsets=self.encoder_tabular.cat_offsets,
            num_con=len(con_lengths_tabular),
        )
        self.criterion_itm = nn.CrossEntropyLoss(reduction="mean")

        self.initialize_classifier_and_metrics(nclasses, nclasses)

    def initialize_classifier_and_metrics(self, nclasses_train, nclasses_val):
        """
        Initializes classifier and metrics. Takes care to set correct number of classes for embedding similarity metric depending on loss.
        """
        # Classifier
        self.estimator = None

        # Accuracy calculated against all others in batch of same view except for self (i.e. -1) and all of the other view
        self.top1_acc_train = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=nclasses_train)
        self.top1_acc_val = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=nclasses_val)

        self.top5_acc_train = torchmetrics.Accuracy(task="multiclass", top_k=5, num_classes=nclasses_train)
        self.top5_acc_val = torchmetrics.Accuracy(task="multiclass", top_k=5, num_classes=nclasses_val)

        n_classes_cat = self.encoder_tabular.num_unique_cat
        self.top1_acc_train_cat = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=n_classes_cat)
        self.top5_acc_train_cat = torchmetrics.Accuracy(task="multiclass", top_k=5, num_classes=n_classes_cat)
        self.top1_acc_val_cat = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=n_classes_cat)
        self.top5_acc_val_cat = torchmetrics.Accuracy(task="multiclass", top_k=5, num_classes=n_classes_cat)
        self.auc_train_cat = torchmetrics.AUROC(task="multiclass", num_classes=n_classes_cat)
        self.auc_val_cal = torchmetrics.AUROC(task="multiclass", num_classes=n_classes_cat)

        self.acc_train_itm = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.acc_val_itm = torchmetrics.Accuracy(task="binary", num_classes=2)

        task = "binary"

        self.classifier_acc_train = torchmetrics.Accuracy(task=task, num_classes=2)
        self.classifier_acc_val = torchmetrics.Accuracy(task=task, num_classes=2)

        self.classifier_auc_train = torchmetrics.AUROC(task=task, num_classes=2)
        self.classifier_auc_val = torchmetrics.AUROC(task=task, num_classes=2)

    def forward(self, x: torch.Tensor, tabular: torch.tensor) -> torch.Tensor:
        """
        Generates encoding of multimodal data. Pick Clstoken
        """
        _, image_features = self.forward_imaging(x)
        _, tabular_features = self.forward_tabular(tabular)
        multimodal_features = self.encoder_multimodal(x=tabular_features, image_features=image_features)
        return multimodal_features[:, 0, :]

    def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates projection and encoding of imaging data.
        """
        y = self.encoder_image.forward_features(x)
        z = self.encoder_image.pool(y)
        z = self.projector_image(z)
        return z, y

    def forward_tabular(
        self, x: torch.Tensor, mask: torch.Tensor = None, mask_special: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Generates projection and encoding of tabular data.
        """
        # (B,N,D)
        y = self.encoder_tabular(x, mask=mask, mask_special=mask_special)
        # (B,N1,C) and (B,N2,1)
        z = self.projector_tabular(y[:, 0, :])
        # projected feature, original feature
        return z, y

    def forward_multimodal(
        self, tabular_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates prediction of tabular data.
        """
        y = self.encoder_multimodal(x=tabular_features, image_features=image_features)
        z = self.predictor_tabular(y)
        return z, y[:, 0, :]

    def forward_multimodal_feature(
        self, tabular_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates feature of tabular data.
        """
        y = self.encoder_multimodal(x=tabular_features, image_features=image_features)
        return y[:, 0, :]

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.top1_acc_train.reset()
        self.top1_acc_val.reset()

        self.top5_acc_train.reset()
        self.top5_acc_val.reset()

        self.top1_acc_train_cat.reset()
        self.top5_acc_train_cat.reset()
        self.top1_acc_val_cat.reset()
        self.top5_acc_val_cat.reset()
        self.auc_train_cat.reset()
        self.auc_val_cal.reset()

        self.acc_train_itm.reset()
        self.acc_val_itm.reset()

        self.classifier_acc_train.reset()
        self.classifier_acc_val.reset()

        self.classifier_auc_train.reset()
        self.classifier_auc_val.reset()

    def cal_image_tabular_matching_loss(
        self, image_embeddings: torch.Tensor, tabular_embeddings: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        current_device = image_embeddings.device
        output_pos = self.forward_multimodal_feature(
            tabular_features=tabular_embeddings, image_features=image_embeddings
        )
        B = image_embeddings.shape[0]
        # get negative pairs
        with torch.no_grad():
            weights_i2t = F.softmax(logits, dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)
            weights_t2i = F.softmax(logits.T, dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)

        tabular_embeddings_neg = torch.zeros_like(tabular_embeddings).to(current_device)
        for b in range(B):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            tabular_embeddings_neg[b] = tabular_embeddings[neg_idx]

        image_embeddings_neg = torch.zeros_like(image_embeddings).to(current_device)
        for b in range(B):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeddings_neg[b] = image_embeddings[neg_idx]

        tabular_embeddings_all = torch.cat([tabular_embeddings, tabular_embeddings_neg], dim=0)
        image_embeddings_all = torch.cat([image_embeddings_neg, image_embeddings], dim=0)
        output_neg = self.forward_multimodal_feature(
            tabular_features=tabular_embeddings_all, image_features=image_embeddings_all
        )
        z = self.itm_head(torch.cat([output_pos, output_neg], dim=0))
        itm_labels = torch.cat([torch.ones(B), torch.zeros(2 * B)], dim=0).long().to(logits.device)
        loss_itm = self.criterion_itm(z, itm_labels)
        return loss_itm, z, itm_labels

    # def on_train_epoch_start(self) -> None:
    #     self.output_list_train = []

    def training_step(self, batch, _) -> torch.Tensor:
        """
        Train
        Tabular-imaging contrastive learning
        Tabular reconstruction learning
        """
        im_views, tab_views, y, _, original_tab = batch

        # =======================================  itc    =======================================================================
        # Augmented image and unagumented tabular
        z0, image_embeddings = self.forward_imaging(im_views[1])
        z1, tabular_embeddings = self.forward_tabular(tab_views[0])
        loss_itc, logits, labels = self.criterion_train_itc(z0, z1, y)
        self.log("train/multimodal/ITCloss", loss_itc, on_epoch=True, on_step=False)

        # =======================================  itm  =======================================================================
        loss_itm, logits_itm, labels_itm = self.cal_image_tabular_matching_loss(
            image_embeddings, tabular_embeddings, logits
        )
        self.log("train/multimodal/ITMloss", loss_itm, on_epoch=True, on_step=False)

        # =======================================  tr    =======================================================================
        # masked tabular
        mask, mask_special = tab_views[2], tab_views[3]
        _, tabular_embeddings = self.forward_tabular(tab_views[1], mask=mask, mask_special=mask_special)
        z2, multimodal_embeddings = self.forward_multimodal(
            tabular_features=tabular_embeddings, image_features=image_embeddings
        )
        loss_tr, pred_cat, target_cat, mask_cat = self.criterion_tr(z2, original_tab, mask=mask)
        self.log("train/multimodal/TRloss", loss_tr, on_epoch=True, on_step=False)

        if len(im_views[0]) == self.hparams.batch_size:
            self.calc_and_log_train_embedding_acc(logits=logits, labels=labels, modality="multimodal")
            self.calc_and_log_train_cat_embedding_acc(
                logits=pred_cat, labels=target_cat, mask=mask_cat, modality="multimodal"
            )
            self.calc_and_log_train_itm_acc(logits=logits_itm, labels=labels_itm, modality="multimodal")

        loss = (loss_itc + loss_tr + loss_itm) / 3.0
        self.log("train/multimodal/loss", loss, on_epoch=True, on_step=False)

        # self.output_list_train.append(
        #     {
        #         "loss": loss,
        #         "embeddings": multimodal_embeddings.detach().cpu(),
        #         "labels": y.cpu(),
        #     }
        # )
        return loss

    # def on_validation_epoch_start(self) -> None:
    #     self.output_list_val = []

    def validation_step(self, batch, _) -> torch.Tensor:
        """
        Validate
        Tabular-imaging contrastive learning
        Tabular reconstruction learning
        """
        im_views, tab_views, y, original_im, original_tab = batch

        # =======================================  itc    =======================================================================
        # Unaugmented views
        z0, image_embeddings = self.forward_imaging(original_im)
        z1, tabular_embeddings = self.forward_tabular(original_tab)
        loss_itc, logits, labels = self.criterion_val_itc(z0, z1, y)
        self.log("val/multimodal/ITCloss", loss_itc, on_epoch=True, on_step=False)

        # =======================================  itm  =======================================================================
        loss_itm, logits_itm, labels_itm = self.cal_image_tabular_matching_loss(
            image_embeddings, tabular_embeddings, logits
        )
        self.log("val/multimodal/ITMloss", loss_itm, on_epoch=True, on_step=False)

        # =======================================  tr    =======================================================================
        # masked tabular
        mask, mask_special = tab_views[2], tab_views[3]
        _, tabular_embeddings = self.forward_tabular(tab_views[1], mask=mask, mask_special=mask_special)
        z2, multimodal_embeddings = self.forward_multimodal(
            tabular_features=tabular_embeddings, image_features=image_embeddings
        )
        loss_tr, pred_cat, target_cat, mask_cat = self.criterion_tr(z2, original_tab, mask=mask)
        self.log("val/multimodal/TRloss", loss_tr, on_epoch=True, on_step=False)

        if len(im_views[0]) == self.hparams.batch_size:
            self.calc_and_log_val_embedding_acc(logits=logits, labels=labels, modality="multimodal")
            self.calc_and_log_val_cat_embedding_acc(
                logits=pred_cat, labels=target_cat, mask=mask_cat, modality="multimodal"
            )
            self.calc_and_log_val_itm_acc(logits=logits_itm, labels=labels_itm, modality="multimodal")

        loss = (loss_itc + loss_tr + loss_itm) / 3.0
        # loss = (loss_itc + loss_tr)/3.0
        self.log("val/multimodal/loss", loss, on_epoch=True, on_step=False)

        # self.output_list_val.append(
        #     {
        #         "sample_augmentation": im_views[1],
        #         "embeddings": multimodal_embeddings.detach().cpu(),
        #         "labels": y.cpu(),
        #     }
        # )

    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """
        Define and return optimizer and scheduler for contrastive model.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())

        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}  # Contrastive

    def calc_and_log_train_embedding_acc(self, logits, labels, modality: str) -> None:
        self.top1_acc_train(logits, labels)
        self.top5_acc_train(logits, labels)

        self.log(f"train/{modality}/top1", self.top1_acc_train, on_epoch=True, on_step=False)
        self.log(f"train/{modality}/top5", self.top5_acc_train, on_epoch=True, on_step=False)

    def calc_and_log_val_embedding_acc(self, logits, labels, modality: str) -> None:
        self.top1_acc_val(logits, labels)
        self.top5_acc_val(logits, labels)

        self.log(f"val/{modality}/top1", self.top1_acc_val, on_epoch=True, on_step=False)
        self.log(f"val/{modality}/top5", self.top5_acc_val, on_epoch=True, on_step=False)

    def calc_and_log_train_cat_embedding_acc(self, logits, labels, mask, modality: str) -> None:
        logits, labels = logits[mask].detach(), labels[mask].detach()
        # print(logits.shape, labels.shape)
        self.top1_acc_train_cat(logits, labels)
        self.top5_acc_train_cat(logits, labels)
        self.auc_train_cat(logits, labels)
        self.log(f"train/{modality}/categorical.top1", self.top1_acc_train_cat, on_epoch=True, on_step=False)
        self.log(f"train/{modality}/categorical.top5", self.top5_acc_train_cat, on_epoch=True, on_step=False)
        self.log(f"train/{modality}/categorical.auc", self.auc_train_cat, on_epoch=True, on_step=False)

    def calc_and_log_val_cat_embedding_acc(self, logits, labels, mask, modality: str) -> None:
        logits, labels = logits[mask].detach(), labels[mask].detach()
        self.top1_acc_val_cat(logits, labels)
        self.top5_acc_val_cat(logits, labels)
        self.auc_val_cal(logits, labels)
        self.log(f"val/{modality}/categorical.top1", self.top1_acc_val_cat, on_epoch=True, on_step=False)
        self.log(f"val/{modality}/categorical.top5", self.top5_acc_val_cat, on_epoch=True, on_step=False)
        self.log(f"val/{modality}/categorical.auc", self.auc_val_cal, on_epoch=True, on_step=False)

    def calc_and_log_train_itm_acc(self, logits, labels, modality: str) -> None:
        logits, labels = logits.detach(), labels.detach()
        self.acc_train_itm(logits, torch.nn.functional.one_hot(labels, num_classes=2))
        self.log(f"train/{modality}/ITMacc", self.acc_train_itm, on_epoch=True, on_step=False)

    def calc_and_log_val_itm_acc(self, logits, labels, modality: str) -> None:
        logits, labels = logits.detach(), labels.detach()
        self.acc_val_itm(logits, torch.nn.functional.one_hot(labels, num_classes=2))
        self.log(f"val/{modality}/ITMacc", self.acc_val_itm, on_epoch=True, on_step=False)

    # def on_train_epoch_end(self) -> None:
    #     """
    #     Train and log classifier
    #     """
    #     if self.current_epoch != 0 and self.current_epoch % self.hparams.classifier_freq == 0:
    #         embeddings, labels = self.stack_outputs(self.output_list_train)

    #         self.estimator = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
    #             embeddings, labels
    #         )
    #         preds, probs = self.predict_live_estimator(embeddings)

    #         self.classifier_acc_train(preds, labels)
    #         self.classifier_auc_train(probs, labels)

    #         self.log("train/classifier/accuracy", self.classifier_acc_train, on_epoch=True, on_step=False)
    #         self.log("train/classifier/auc", self.classifier_auc_train, on_epoch=True, on_step=False)

    # def on_validation_epoch_end(self) -> None:
    #     super().on_train_epoch_end
    #     """
    #     Log an image from each validation step and calc validation classifier performance
    #     """
    #     # Validate classifier
    #     if not self.estimator is None and self.current_epoch % self.hparams.classifier_freq == 0:
    #         embeddings, labels = self.stack_outputs(self.output_list_val)

    #         preds, probs = self.predict_live_estimator(embeddings)

    #         self.classifier_acc_val(preds, labels)
    #         self.classifier_auc_val(probs, labels)

    #         self.log("val/classifier/accuracy", self.classifier_acc_val, on_epoch=True, on_step=False)
    #         self.log("val/classifier/auc", self.classifier_auc_val, on_epoch=True, on_step=False)

    # def stack_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
    #     """
    #     Stack outputs from multiple steps
    #     """
    #     labels = outputs[0]["labels"]
    #     embeddings = outputs[0]["embeddings"]
    #     for i in range(1, len(outputs)):
    #         labels = torch.cat((labels, outputs[i]["labels"]), dim=0)
    #         embeddings = torch.cat((embeddings, outputs[i]["embeddings"]), dim=0)

    #     embeddings = embeddings
    #     labels = labels

    #     return embeddings, labels

    # def predict_live_estimator(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Predict using live estimator
    #     """
    #     preds = self.estimator.predict(embeddings)
    #     probs = self.estimator.predict_proba(embeddings)

    #     preds = torch.tensor(preds)
    #     probs = torch.tensor(probs)

    #     probs = probs[:, 1]

    #     return preds, probs

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage in ["fit", "predict"]:
            self.encoder_image = torch.compile(self.encoder_image)
            self.encoder_tabular = torch.compile(self.encoder_tabular)
            self.encoder_multimodal = torch.compile(self.encoder_multimodal)
            self.predictor_tabular = torch.compile(self.predictor_tabular)
