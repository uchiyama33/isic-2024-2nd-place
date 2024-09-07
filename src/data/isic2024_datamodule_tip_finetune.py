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


class ISIC2024DataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir,
        hdf5_train_name,
        hdf5_test_name,
        meta_csv_train_name,
        meta_csv_test_name,
        cat_lengths_tabular,
        con_lengths_tabular,
        neg_sampling_ratio=None,
        kfold_df_name=None,
        fold=None,
        n_fold=5,
        kfold_method="sgkf",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        img_size: int = 224,
        transforms_version: int = 1,
        patient_set: bool = False,
        n_data_per_patient: int = 10,
        corruption_rate: float = 0.3,
        use_n_clusters: int = None,
        tabular_data_version: int = 1,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms_train, self.transforms_test, self.transforms_type = get_transforms(
            transforms_version, img_size
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_pred: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

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
                    kfold_df_name=self.hparams.kfold_df_name,
                    fold=self.hparams.fold,
                    n_fold=self.hparams.n_fold,
                    kfold_method=self.hparams.kfold_method,
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
                    kfold_df_name=self.hparams.kfold_df_name,
                    fold=self.hparams.fold,
                    n_fold=self.hparams.n_fold,
                    kfold_method=self.hparams.kfold_method,
                    transforms=self.transforms_test,
                    transforms_type=self.transforms_type,
                    tabular_data_version=self.hparams.tabular_data_version,
                )
        else:
            if not self.data_pred:
                self.data_pred = ISIC2024Dataset(
                    self.hparams.data_dir,
                    "test",
                    self.hparams.hdf5_test_name,
                    self.hparams.meta_csv_test_name,
                    transforms=self.transforms_test,
                    transforms_type=self.transforms_type,
                    tabular_data_version=self.hparams.tabular_data_version,
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.hparams.patient_set:
            sampler = PatientBatchSampler(
                self.data_train,
                self.batch_size_per_device,
                self.hparams.n_data_per_patient,
                neg_sampling_ratio=self.hparams.neg_sampling_ratio,
            )
            return DataLoader(
                dataset=self.data_train,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                batch_sampler=sampler,
            )
        else:
            sampler = UnderSampler(
                self.data_train, self.hparams.neg_sampling_ratio, random_sampling=True, shuffle=True
            )
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                drop_last=True,
                sampler=sampler,
            )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.hparams.patient_set:
            sampler = PatientBatchSampler(
                self.data_val,
                self.batch_size_per_device,
                self.hparams.n_data_per_patient,
                mode="val",
                neg_sampling_ratio=self.hparams.neg_sampling_ratio,
            )
            return DataLoader(
                dataset=self.data_val,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                batch_sampler=sampler,
            )
        else:
            sampler = UnderSampler(
                self.data_val, self.hparams.neg_sampling_ratio, random_sampling=False, shuffle=False
            )
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                drop_last=True,
                sampler=sampler,
            )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.hparams.patient_set:
            sampler = PatientBatchSampler(
                self.data_test, self.batch_size_per_device, self.hparams.n_data_per_patient, mode="test"
            )
            return DataLoader(
                dataset=self.data_test,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                batch_sampler=sampler,
            )
        else:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )

    def predict_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.hparams.patient_set:
            assert NotImplementedError
            sampler = PatientSampler(self.data_pred, 10, "test")
            return DataLoader(
                dataset=self.data_pred,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                drop_last=True,
                sampler=sampler,
            )
        else:
            return DataLoader(
                dataset=self.data_pred,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ISIC2024DataModule()
