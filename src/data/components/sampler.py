import torch
from torch.utils.data import Sampler, BatchSampler
import numpy as np
from collections import defaultdict


class UnderSampler(Sampler):
    def __init__(self, data_source, pos_neg_ratio=1, random_sampling=False, shuffle=False):
        self.data_source = data_source
        self.pos_neg_ratio = pos_neg_ratio
        self.random_sampling = random_sampling
        self.shuffle = shuffle

        self.positive_indices = np.where(self.data_source.targets == 1)[0]
        self.negative_indices = np.where(self.data_source.targets == 0)[0]
        self.n_positives = len(self.positive_indices)
        self.n_negatives = self.n_positives * self.pos_neg_ratio

        if not random_sampling:
            selected_negative_indices = np.random.choice(
                self.negative_indices, self.n_negatives, replace=False
            )
            self.balanced_indices = np.concatenate([self.positive_indices, selected_negative_indices])

    def __iter__(self):
        if self.random_sampling:
            selected_negative_indices = np.random.choice(
                self.negative_indices, self.n_negatives, replace=False
            )
            self.balanced_indices = np.concatenate([self.positive_indices, selected_negative_indices])

        if self.shuffle:
            np.random.shuffle(self.balanced_indices)

        return iter(self.balanced_indices)

    def __len__(self):
        return self.n_positives + self.n_negatives


class UnderSamplerForWideDeep(Sampler):
    def __init__(self, targets, pos_neg_ratio=1, random_sampling=False, shuffle=False):
        self.pos_neg_ratio = pos_neg_ratio
        self.random_sampling = random_sampling
        self.shuffle = shuffle

        self.positive_indices = np.where(targets == 1)[0]
        self.negative_indices = np.where(targets == 0)[0]
        self.n_positives = len(self.positive_indices)
        self.n_negatives = self.n_positives * self.pos_neg_ratio

        if not random_sampling:
            selected_negative_indices = np.random.choice(
                self.negative_indices, self.n_negatives, replace=False
            )
            self.balanced_indices = np.concatenate([self.positive_indices, selected_negative_indices])

    def __iter__(self):
        if self.random_sampling:
            selected_negative_indices = np.random.choice(
                self.negative_indices, self.n_negatives, replace=False
            )
            self.balanced_indices = np.concatenate([self.positive_indices, selected_negative_indices])

        if self.shuffle:
            np.random.shuffle(self.balanced_indices)

        return iter(self.balanced_indices)

    def __len__(self):
        return self.n_positives + self.n_negatives


class PatientBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, n_samples_per_patient, mode="train", neg_sampling_ratio=None):
        self.patient_ids = dataset.patient_ids
        self.batch_size = batch_size
        self.n_samples_per_patient = n_samples_per_patient
        self.mode = mode

        if neg_sampling_ratio:
            n_pos = (dataset.targets == 1).sum()
            n_neg = n_pos * neg_sampling_ratio
            sampled_indices_pos = np.where(dataset.targets == 1)[0]
            sampled_indices_neg = np.random.choice(
                np.where(dataset.targets == 0)[0], size=n_neg, replace=False
            )
            self.patient_ids = self.patient_ids[np.concatenate([sampled_indices_pos, sampled_indices_neg])]

        # 各患者IDに対応するインデックスを格納
        self.patient_to_indices = defaultdict(list)
        for idx, patient_id in enumerate(self.patient_ids):
            self.patient_to_indices[patient_id].append(idx)

        if self.mode == "train":
            # 一度全バッチを作ってみてバッチ数を計算
            all_indices_tmp = []
            for patient_id, indices in self.patient_to_indices.items():
                np.random.shuffle(indices)
                for i in range(0, len(indices), self.n_samples_per_patient):
                    chunk = indices[i : i + self.n_samples_per_patient]
                    while len(chunk) < self.n_samples_per_patient:
                        # リストの末尾に達した場合、不足分を先頭から補完
                        remaining = self.n_samples_per_patient - len(chunk)
                        chunk += indices[:remaining]
                    assert len(chunk) == self.n_samples_per_patient
                    all_indices_tmp.append(chunk)

            np.random.shuffle(all_indices_tmp)
            all_indices_train = np.concatenate(all_indices_tmp).tolist()

            self.n_batches = len(all_indices_train) // (self.batch_size * self.n_samples_per_patient) + 1
        else:
            # 検証/テストモード: 全サンプルを使用
            self.all_indices = []
            for patient_id, indices in self.patient_to_indices.items():
                for i in range(0, len(indices), self.n_samples_per_patient):
                    chunk = indices[i : i + self.n_samples_per_patient]
                    while len(chunk) < self.n_samples_per_patient:
                        # リストの末尾に達した場合、不足分を先頭から補完
                        remaining = self.n_samples_per_patient - len(chunk)
                        chunk += indices[:remaining]
                    assert len(chunk) == self.n_samples_per_patient
                    self.all_indices.extend(chunk)

            self.n_batches = len(self.all_indices) // (self.batch_size * self.n_samples_per_patient) + 1
            # 固定サイズになるようにする
            self.all_indices = (self.all_indices + self.all_indices)[
                : self.n_batches * (self.batch_size * self.n_samples_per_patient)
            ]

    def __iter__(self):
        if self.mode == "train":
            # train時は毎回シャッフルする
            all_indices_tmp = []
            for patient_id, indices in self.patient_to_indices.items():
                np.random.shuffle(indices)
                for i in range(0, len(indices), self.n_samples_per_patient):
                    chunk = indices[i : i + self.n_samples_per_patient]
                    while len(chunk) < self.n_samples_per_patient:
                        # リストの末尾に達した場合、不足分を先頭から補完
                        remaining = self.n_samples_per_patient - len(chunk)
                        chunk += indices[:remaining]
                    assert len(chunk) == self.n_samples_per_patient
                    all_indices_tmp.append(chunk)

            np.random.shuffle(all_indices_tmp)
            all_indices_train = np.concatenate(all_indices_tmp).tolist()

            # 固定サイズになるようにする
            all_indices_train = (all_indices_train + all_indices_train)[
                : self.n_batches * (self.batch_size * self.n_samples_per_patient)
            ]

            for i in range(0, len(all_indices_train), self.batch_size * self.n_samples_per_patient):
                yield all_indices_train[i : i + self.batch_size * self.n_samples_per_patient]

        else:
            for i in range(0, len(self.all_indices), self.batch_size * self.n_samples_per_patient):
                yield self.all_indices[i : i + self.batch_size * self.n_samples_per_patient]

    def __len__(self):
        return self.n_batches
