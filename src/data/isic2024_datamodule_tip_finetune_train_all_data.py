from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.isic2024_dataset_tip_finetune import ISIC2024Dataset
from src.data.components.transforms import get_transforms
from src.data.components.sampler import UnderSampler, PatientBatchSampler
from src.data.isic2024_datamodule_tip_finetune import ISIC2024DataModule


class ISIC2024DataModuleTrainAllData(ISIC2024DataModule):
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if stage in ["fit", "validate", "test"]:
            if not self.data_train and not self.data_val and not self.data_test:
                self.data_train = ISIC2024Dataset(
                    self.hparams.data_dir,
                    "train",
                    self.hparams.hdf5_train_name,
                    self.hparams.meta_csv_train_name,
                    # neg_sampling_ratio=self.hparams.neg_sampling_ratio,
                    kfold_df_name=self.hparams.kfold_df_name,
                    fold=self.hparams.fold,
                    n_fold=self.hparams.n_fold,
                    kfold_method=self.hparams.kfold_method,
                    transforms=self.transforms_train,
                    transforms_type=self.transforms_type,
                    corruption_rate=self.hparams.corruption_rate,
                    use_n_clusters=self.hparams.use_n_clusters,
                    tabular_data_version=self.hparams.tabular_data_version,
                )
                self.data_val = ISIC2024Dataset(
                    self.hparams.data_dir,
                    "val",
                    self.hparams.hdf5_train_name,
                    self.hparams.meta_csv_train_name,
                    # neg_sampling_ratio=self.hparams.neg_sampling_ratio,
                    transforms=self.transforms_test,
                    transforms_type=self.transforms_type,
                    use_n_clusters=self.hparams.use_n_clusters,
                    tabular_data_version=self.hparams.tabular_data_version,
                )
                # valと同じデータでneg_sampling_ratioを設定しないのをtestとする
                self.data_test = ISIC2024Dataset(
                    self.hparams.data_dir,
                    "val",
                    self.hparams.hdf5_train_name,
                    self.hparams.meta_csv_train_name,
                    # neg_sampling_ratio=None,
                    transforms=self.transforms_test,
                    transforms_type=self.transforms_type,
                    tabular_data_version=self.hparams.tabular_data_version,
                )
