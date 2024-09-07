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
from src.models.isic2024_module_ce import ISIC2024LitModule


class ISIC2024LitModuleTIPFinetune(ISIC2024LitModule):
    def forward(self, batch, net) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return net(batch["image"], batch["tabular"])
