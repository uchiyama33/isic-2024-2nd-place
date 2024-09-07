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
from skimage.exposure import match_histograms

from PIL import Image
from io import BytesIO
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.isic_utils.utils import prepare_df_for_dnn
from src.data.components.histogram_matching import match_histograms_by_region


class ISIC2024Dataset(Dataset):
    def __init__(
        self,
        root,
        split,
        df_name,
        hdf5_name,
        # neg_sampling_ratio=None,
        n_fold=5,
        fold=None,
        kfold_method="sgkf",
        transforms=None,
        transforms_type="albumentations",
        match_histograms=False,
    ):
        assert transforms_type in ["albumentations", "torchvision"]
        assert split in ["train", "val"]

        self.root = root
        self.split = split
        self.transforms_type = transforms_type
        self.match_histograms = match_histograms
        df = pd.read_parquet(os.path.join(root, df_name))
        if fold is not None:
            assert split in ["train", "val"]
            if kfold_method == "sgkf":
                df = df[df[f"StratifiedGroupKFold_{n_fold}_{fold}"] == split]
            elif kfold_method == "sgkf":
                df = df[df[f"StratifiedGroupKFold_{n_fold}_{fold}"] == split]
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

        self.fp_hdf = h5py.File(os.path.join(root, hdf5_name), mode="r")
        self.source = df["source"].values
        self.targets = df["target"].values
        self.index_2024_pos = np.where((self.source == "isic-2024") & (self.targets == 1))[0]
        self.index_2024_neg = np.where((self.source == "isic-2024") & (self.targets == 0))[0]
        self.isic_ids = df["isic_id"].values
        self.patient_ids = df["patient_id"].values
        self.transforms = transforms
        self.transforms_for_external = A.Compose(
            [
                A.OneOf(
                    [
                        A.Posterize(num_bits=1),
                        A.ImageCompression(quality_lower=50, quality_upper=80),
                        A.Downscale(scale_min=0.5, scale_max=0.75),
                    ],
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5),
                        A.MedianBlur(blur_limit=5),
                        A.GaussianBlur(blur_limit=5),
                        A.GaussNoise(var_limit=(5.0, 30.0)),
                    ],
                    p=0.7,
                ),
            ]
        )

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        patient_id = self.patient_ids[index]
        source = self.source[index]
        target = self.targets[index]
        image = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))

        if source != "isic-2024" and self.match_histograms:
            if target == 1:
                index_24_rand_samp = random.choice(self.index_2024_pos)
            else:
                index_24_rand_samp = random.choice(self.index_2024_neg)
            isic_id_24_rand_samp = self.isic_ids[index_24_rand_samp]
            image_24_rand_samp = np.array(Image.open(BytesIO(self.fp_hdf[isic_id_24_rand_samp][()])))
            alpha = random.uniform(0.6, 0.9)
            image = match_histograms_by_region(image, image_24_rand_samp, alpha)
            image = self.transforms_for_external(image=np.array(image))["image"]

        if self.transforms_type == "albumentations":
            image = self.transforms(image=np.array(image))["image"]
        elif self.transforms_type == "torchvision":
            image = self.transforms(image)

        return {
            "image": image,
            "isic_id": isic_id,
            "patient_id": patient_id,
            "target": target,
            "source": source,
        }
