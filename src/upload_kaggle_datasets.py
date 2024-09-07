# %%
import os
import zipfile
import kaggle
from pathlib import Path
from kaggle import KaggleApi

api = KaggleApi()
api.authenticate()

# %%
# api.dataset_initialize("/workspace/src")
# api.dataset_initialize("/workspace/configs")
# api.dataset_initialize("/workspace/pip_packages")

# %%
# api.dataset_create_new("/workspace/src", dir_mode="zip")
# api.dataset_create_new("/workspace/configs", dir_mode="zip")
# api.dataset_create_new("/workspace/pip_packages", dir_mode="zip")

# %%
version_notes = "0904"
api.dataset_create_version("/workspace/src", version_notes, dir_mode="zip")
api.dataset_create_version("/workspace/configs", version_notes, dir_mode="zip")
# api.dataset_create_version("/workspace/pip_packages", version_notes, dir_mode="zip")

# %% CNN
experiment_list = ["0713-efficientnet_b0"]
# %%
for e in experiment_list:
    api.dataset_initialize(os.path.join("/workspace/logs/train/runs", e))

# %%
for e in experiment_list:
    api.dataset_create_new(os.path.join("/workspace/logs/train/runs", e), dir_mode="zip")

# %%
version_notes = "0713"
for e in experiment_list:
    api.dataset_create_version(os.path.join("/workspace/logs/train/runs", e), version_notes, dir_mode="zip")

# %% lgb
experiment_list = ["0714-optuna_best"]
# %%
for e in experiment_list:
    api.dataset_initialize(os.path.join("/workspace/logs/lgb/runs", e))

# %%
for e in experiment_list:
    api.dataset_create_new(os.path.join("/workspace/logs/lgb/runs", e), dir_mode="zip")

# %%
version_notes = "0714"
for e in experiment_list:
    api.dataset_create_version(os.path.join("/workspace/logs/lgb/runs", e), version_notes, dir_mode="zip")
