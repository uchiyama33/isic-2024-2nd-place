from typing import Any, Dict, List, Optional, Tuple

import os
import pandas as pd
from glob import glob
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import wandb
import numpy as np
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)
torch.set_float32_matmul_precision("high")


def predict(cfg: DictConfig, ckpt_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    outputs = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    logits = torch.concat([o[0] for o in outputs]).float().numpy()
    preds = torch.concat([o[1] for o in outputs]).float().numpy()
    ids = np.concatenate([o[2] for o in outputs])
    return pd.DataFrame({"logits": logits, "probabilities": preds, "isic_id": ids})


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    cfg.logger = None
    cfg.data.hdf5_test_name = cfg.data.hdf5_train_name
    cfg.data.meta_csv_test_name = cfg.data.meta_csv_train_name
    extras(cfg)

    save_dir = os.path.join(cfg.log_dir, "train_predictions")
    os.makedirs(save_dir, exist_ok=True)

    for fold in range(cfg.data.n_fold):
        cfg.data.fold = fold
        cfg.seed = fold
        if fold == 0:
            ckpt_name = "last.ckpt"
        else:
            ckpt_name = f"last-v{fold}.ckpt"
        ckpt_path = glob(os.path.join(cfg.callbacks.model_checkpoint.dirpath, ckpt_name))[0]
        df_predictions = predict(cfg, ckpt_path)

        df_predictions.to_parquet(os.path.join(save_dir, f"fold{fold}.parquet"))


if __name__ == "__main__":
    main()
