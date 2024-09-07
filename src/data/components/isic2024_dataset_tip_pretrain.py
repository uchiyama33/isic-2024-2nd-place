from typing import List, Tuple
import os
import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import h5py

from PIL import Image
from io import BytesIO
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.isic_utils.utils import prepare_df_for_dnn


class ISIC2024Dataset(Dataset):
    def __init__(
        self,
        root,
        split,
        hdf5_name,
        meta_csv_name,
        # neg_sampling_ratio=None,
        kfold_df_name=None,
        n_fold=5,
        fold=None,
        kfold_method="sgkf",
        transforms=None,
        transforms_default=None,
        transforms_type="albumentations",
        train_meta_csv_name="train-metadata.csv",
        augmentation_rate=0.95,
        corruption_rate=0.3,
        replace_special_rate=0.5,
        replace_random_rate=0.0,
        tabular_data_version=1,
    ):
        assert transforms_type in ["albumentations", "torchvision"]
        self.split = split
        self.transforms_type = transforms_type
        self.transforms = transforms
        self.transforms_default = transforms_default
        self.augmentation_rate = augmentation_rate
        self.c = corruption_rate
        self.augmentation_rate = augmentation_rate
        self.replace_random_rate = replace_random_rate
        self.replace_special_rate = replace_special_rate

        df = pd.read_csv(os.path.join(root, meta_csv_name))
        df_train_all = pd.read_csv(os.path.join(root, train_meta_csv_name))
        if fold is not None:
            assert split in ["train", "val"]
            df_fold = pd.read_parquet(os.path.join(root, kfold_df_name))
            df = df.merge(df_fold, how="left", on="isic_id")
            if kfold_method == "sgkf":
                df = df[df[f"StratifiedGroupKFold_{n_fold}_{fold}"] == split]
            elif kfold_method == "sgkf":
                df = df[df[f"StratifiedGroupKFold_{n_fold}_{fold}"] == split]
            elif kfold_method == "tsgkf":
                df = df[df[f"TSGKF_{n_fold}_{fold}"] == split]
            else:
                assert False, "kfold_method"

        # Handle neg_sampling_ratio being None
        # if neg_sampling_ratio is not None:
        #     # Separate positive and negative samples
        #     df_positive = df[df["target"] == 1].reset_index()
        #     df_negative = df[df["target"] == 0].reset_index()

        #     num_pos_samples = len(df_positive)
        #     num_neg_samples = int(neg_sampling_ratio * num_pos_samples)

        #     sampled_indices = np.random.choice(len(df_negative), size=num_neg_samples, replace=False)
        #     df_sampled_negative = df_negative.iloc[sampled_indices]

        #     df = pd.concat([df_positive, df_sampled_negative]).reset_index(drop=True)

        metadata_num, metadata_cat = prepare_df_for_dnn(df, df_train_all, tabular_data_version)
        self.data_tabular = np.concatenate([metadata_cat, metadata_num], axis=1)
        self.generate_marginal_distributions()
        self.fp_hdf = h5py.File(os.path.join(root, hdf5_name), mode="r")
        self.isic_ids = df["isic_id"].values
        self.patient_ids = df["patient_id"].values
        if split in ["train", "val"]:
            self.targets = df["target"].values

    def generate_marginal_distributions(self) -> None:
        """
        Generates empirical marginal distribution by transposing data
        """
        data = np.array(self.data_tabular)
        self.marginal_distributions = np.transpose(data)

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        imaging_views, unaugmented_image = self.generate_imaging_views(index)

        if self.c > 0:
            tabular_views = [torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
        else:
            tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float)]
        masked_view, mask, mask_special, mask_random = self.mask(self.data_tabular[index])
        tabular_views.append(torch.from_numpy(masked_view).float())
        tabular_views = tabular_views + [torch.from_numpy(mask), torch.from_numpy(mask_special)]
        unaugmented_tabular = torch.tensor(self.data_tabular[index], dtype=torch.float)

        if self.split in ["train", "val"]:
            target = self.targets[index]
        else:
            target = -1

        return imaging_views, tabular_views, target, unaugmented_image, unaugmented_tabular

    def corrupt(self, subject: List[float]) -> List[float]:
        """
        Creates a copy of a subject, selects the indices
        to be corrupted (determined by hyperparam corruption_rate)
        and replaces their values with ones sampled from marginal distribution
        """
        subject = copy.deepcopy(subject)
        subject = np.array(subject)

        indices = random.sample(list(range(len(subject))), int(len(subject) * self.c))
        pick_value_positions = np.random.choice(self.marginal_distributions.shape[1], size=len(indices))
        subject[indices] = self.marginal_distributions[indices, pick_value_positions]
        return subject

    def mask(self, subject: List[float]) -> List[float]:
        """
        Create a copy of a subject, selects
        some indices keeping the same
        some indices replacing their values with
        """
        subject = copy.deepcopy(subject)
        subject = np.array(subject)

        indices = random.sample(
            list(range(len(subject))),
            round(len(subject) * (self.replace_random_rate + self.replace_special_rate)),
        )
        num_random = int(
            len(indices) * self.replace_random_rate / (self.replace_random_rate + self.replace_special_rate)
        )
        num_special = len(indices) - num_random
        # replace some positions with random sample from marginal distribution
        pick_value_positions = np.random.choice(self.marginal_distributions.shape[1], size=num_random)
        subject[indices[:num_random]] = self.marginal_distributions[
            indices[:num_random], pick_value_positions
        ]

        mask, mask_random, mask_special = (
            np.zeros_like(subject, dtype=bool),
            np.zeros_like(subject, dtype=bool),
            np.zeros_like(subject, dtype=bool),
        )
        mask[indices] = True
        mask_random[indices[:num_random]] = True
        mask_special[indices[num_random:]] = True
        assert np.sum(mask) == np.sum(mask_special)

        return subject, mask, mask_special, mask_random

    def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
        """
        Generates two views of a subjects image. Also returns original image resized to required dimensions.
        The first is always augmented. The second has {augmentation_rate} chance to be augmented.
        """
        isic_id = self.isic_ids[index]
        im = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        ims = [self.transforms(image=im)["image"]]
        if random.random() < self.augmentation_rate:
            ims.append(self.transforms(image=im)["image"])
        else:
            ims.append(self.transforms_default(image=im)["image"])

        orig_im = self.transforms_default(image=im)["image"]

        return ims, orig_im
