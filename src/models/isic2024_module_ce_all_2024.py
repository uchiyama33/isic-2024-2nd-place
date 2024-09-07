from typing import Any, Dict, Tuple

import os
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.auroc import BinaryAUROC
import numpy as np
from sklearn.metrics import roc_curve, auc
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.utils.ema import MyModelEma


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
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        optimize_config,
        use_ema: bool = False,
        preds_save_path: str = None,
        sample_weight_24: float = 1.0,
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

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # metric objects for calculating and averaging accuracy across batches
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.val_2024_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.test_2024_auroc = BinaryAUROC()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_pauc_best = MaxMetric()
        self.val_2024_pauc_best = MaxMetric()

        if use_ema:
            self.net_ema = MyModelEma(
                self.net,
                decay=0.99,
                use_warmup=True,
                warmup_power=4 / 5,
            )

    def forward(self, batch, net) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        image = batch["image"]
        return net(image)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_auroc.reset()
        self.val_pauc_best.reset()

    def model_step(
        self,
        batch,
        pred=False,
        tta=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        if self.hparams.use_ema:
            if self.training:
                net = self.net
            else:
                net = self.net_ema
        else:
            net = self.net

        id = batch["isic_id"]
        y = batch["target"]

        logits = self.forward(batch, net)
        preds = torch.softmax(logits, dim=1)[:, 1]
        if pred:
            return logits[:, 1], preds, id
        else:
            losses = {}
            y_onehot = F.one_hot(y, num_classes=2).to(logits.dtype)
            loss_list = self.criterion(logits, y_onehot)

            source = np.array(batch["source"])

            sample_weights = torch.ones(len(y)).to(loss_list)
            sample_weights[source == "isic-2024"] = (
                sample_weights[source == "isic-2024"] * self.hparams.sample_weight_24
            )

            loss = (loss_list * sample_weights).mean()

            losses["loss_main"] = loss
            losses["loss_total"] = losses["loss_main"]

            return losses, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(losses["loss_main"])
        self.train_auroc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return losses["loss_total"]

    def on_after_backward(self) -> None:
        if self.hparams.use_ema:
            self.net_ema.update(self.net, self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pauc = self.pAUC(torch.concat(self.train_auroc.preds), torch.concat(self.train_auroc.target))
        self.log("train/pauc", pauc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.no_grad():
            losses, preds, targets = self.model_step(batch)

            # update and log metrics
            self.val_loss(losses["loss_main"])
            self.val_auroc(preds, targets)
            self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

            source = np.array(batch["source"])
            idx_2024 = source == "isic-2024"
            if idx_2024.sum() > 1:
                self.val_2024_auroc(preds[idx_2024], targets[idx_2024])
                self.log("val/auroc_2024", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if torch.allclose(torch.concat(self.val_auroc.target), torch.tensor(0)):
            # sanity checkのときは、全てのターゲットが0になることがあるのでエラー回避
            pauc = 0
        else:
            pauc = self.pAUC(torch.concat(self.val_auroc.preds), torch.concat(self.val_auroc.target))
        self.log("val/pauc", pauc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_pauc_best(pauc)  # update best so far val auroc
        # log `val_pauc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/pauc_best", self.val_pauc_best.compute(), sync_dist=True, prog_bar=True)

        if torch.allclose(torch.concat(self.val_2024_auroc.target), torch.tensor(0)):
            # sanity checkのときは、全てのターゲットが0になることがあるのでエラー回避
            pauc = 0
        else:
            pauc = self.pAUC(
                torch.concat(self.val_2024_auroc.preds), torch.concat(self.val_2024_auroc.target)
            )
        self.log("val/pauc_2024", pauc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_2024_pauc_best(pauc)  # update best so far val auroc
        # log `val_pauc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/pauc_2024_best", self.val_2024_pauc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.preds_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.inference_mode():
            losses, preds, targets = self.model_step(batch, tta=self.hparams.tta)

            # update and log metrics
            self.test_loss(losses["loss_main"])
            self.test_auroc(preds, targets)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

            source = np.array(batch["source"])
            idx_2024 = source == "isic-2024"
            if idx_2024.sum() > 1:
                self.test_2024_auroc(preds[idx_2024], targets[idx_2024])
                self.log("test/auroc_2024", self.test_2024_auroc, on_step=False, on_epoch=True, prog_bar=True)

        self.preds_list.append(preds.cpu())

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pauc = self.pAUC(torch.concat(self.test_auroc.preds), torch.concat(self.test_auroc.target))
        self.log("test/pauc", pauc, on_step=False, on_epoch=True, prog_bar=True)

        pauc = self.pAUC(torch.concat(self.test_2024_auroc.preds), torch.concat(self.test_2024_auroc.target))
        self.log("test/pauc_2024", pauc, on_step=False, on_epoch=True, prog_bar=True)

    def pAUC(self, pred, gt, min_tpr=0.8):
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().float().cpu().numpy()

        v_gt = abs(gt - 1)

        # flip the submissions to their compliments
        v_pred = -1.0 * np.asarray(pred)

        max_fpr = abs(1 - min_tpr)

        # using sklearn.metric functions: (1) roc_curve and (2) auc
        fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
        if max_fpr is None or max_fpr == 1:
            return auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        partial_auc = auc(fpr, tpr)

        return partial_auc

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        with torch.inference_mode():
            preds = self.model_step(batch, pred=True, tta=self.hparams.tta)
        return preds

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a predict epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage in ["fit", "predict"]:
            self.net = torch.compile(self.net)
            if self.hparams.use_ema:
                self.net_ema = torch.compile(self.net_ema)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        if self.hparams.optimize_config.mode == "normal":
            optimizer = self.hparams.optimizer(params=self.parameters())

        elif self.hparams.optimize_config.mode == "finetuning":
            head_layer_list = self.get_target_param(self.hparams.optimize_config.head_name)
            params = list(
                map(
                    lambda x: x[1], list(filter(lambda kv: kv[0] in head_layer_list, self.named_parameters()))
                )
            )
            base_params = list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: kv[0] not in head_layer_list, self.named_parameters())),
                )
            )

            optimizer = self.hparams.optimizer(
                params=[
                    {
                        "params": base_params,
                        "lr": self.hparams.optimize_config.encoder_lr
                        * self.hparams.optimize_config.encoder_lr_coef,
                    },
                    {"params": params},
                ],
            )

        elif self.hparams.optimize_config.mode == "target_decay":
            target_layer_list = self.get_target_param(self.hparams.optimize_config.target_name)
            params = list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: kv[0] in target_layer_list, self.named_parameters())),
                )
            )
            base_params = list(
                map(
                    lambda x: x[1],
                    list(filter(lambda kv: kv[0] not in target_layer_list, self.named_parameters())),
                )
            )

            optimizer = self.hparams.optimizer(
                params=[
                    {
                        "params": base_params,
                        "lr": self.hparams.optimize_config.lr_base
                        * self.hparams.optimize_config.lr_decay_coef,
                    },
                    {"params": params},
                ],
            )

        else:
            assert False, "optimize_mode"

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def get_target_param(self, target_name):
        layer_list = []
        for name, param in self.named_parameters():
            if target_name in name:
                # print(name, param.requires_grad)
                layer_list.append(name)

        assert len(layer_list) > 0

        return layer_list


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
