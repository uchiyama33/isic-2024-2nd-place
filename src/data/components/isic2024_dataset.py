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
        transforms_type="albumentations",
        train_meta_csv_name="train-metadata.csv",
        use_n_clusters: int = None,
        iddx_cluster_name: str = None,
        metadata_version=1,
    ):
        assert transforms_type in ["albumentations", "torchvision"]
        self.split = split
        self.transforms_type = transforms_type
        df = pd.read_csv(os.path.join(root, meta_csv_name))
        df_train_all = pd.read_csv(os.path.join(root, train_meta_csv_name))
        if fold is not None:
            assert split in ["train", "val"]
            df_fold = pd.read_parquet(os.path.join(root, kfold_df_name))
            df = df.merge(df_fold, how="left", on="isic_id")
            if kfold_method == "sgkf":
                df = df[df[f"StratifiedGroupKFold_{n_fold}_{fold}"] == split]
            elif kfold_method == "gkf":
                df = df[df[f"GroupKFold_{n_fold}_{fold}"] == split]
            elif kfold_method == "tsgkf":
                df = df[df[f"TSGKF_{n_fold}_{fold}"] == split]
            else:
                assert False, "kfold_method"

        if use_n_clusters is not None:
            df_cluster = pd.read_parquet(os.path.join(root, iddx_cluster_name))
            label_cols = df_cluster.columns.drop("isic_id")
            df = df.merge(df_cluster, on="isic_id", how="left")
            self.clusters = df[label_cols].to_numpy()
        else:
            self.clusters = [-1] * len(df)

        self.metadata_num, self.metadata_cat = prepare_df_for_dnn(df, df_train_all, metadata_version)
        self.fp_hdf = h5py.File(os.path.join(root, hdf5_name), mode="r")
        self.isic_ids = df["isic_id"].values
        self.patient_ids = df["patient_id"].values
        if split in ["train", "val"]:
            self.targets = df["target"].values
        self.transforms = transforms

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        patient_id = self.patient_ids[index]
        metadata_num = self.metadata_num[index]
        metadata_cat = self.metadata_cat[index]
        cluster = self.clusters[index]
        image = Image.open(BytesIO(self.fp_hdf[isic_id][()]))
        if self.split in ["train", "val"]:
            target = self.targets[index]
        else:
            target = -1

        if self.transforms_type == "albumentations":
            image = self.transforms(image=np.array(image))["image"]
        elif self.transforms_type == "torchvision":
            image = self.transforms(image)

        return {
            "metadata_num": metadata_num,
            "metadata_cat": metadata_cat,
            "image": image,
            "isic_id": isic_id,
            "patient_id": patient_id,
            "target": target,
            "cluster": cluster,
        }
