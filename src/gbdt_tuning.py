# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from hydra import compose, initialize
import os
from sklearn.preprocessing import OrdinalEncoder
import rootutils
import wandb
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import copy
from tqdm import tqdm

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.isic_utils.utils import (
    prepare_df_for_gbdt,
    comp_score,
    custom_metric,
    preprocess_df,
    SelectColumns,
    DNNFeatureEngineering,
)
from src.isic_utils.feature_engineering import feature_engineering_new
from src.isic_utils.gbdt_models import GBDTModels

# %%
gbdt_params = "0906-9NNs-18types-feV7-s5-tuning_weights"

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
        config_name="gbdt",
        overrides=[f"gbdt_params={gbdt_params}"],
        return_hydra_config=True,
    )
    cfg.paths.output_dir = "${hydra.runtime.output_dir}"
    cfg.paths.work_dir = "${hydra.runtime.cwd}"
    cfg.hydra.run.dir = cfg.log_dir
    cfg.hydra.runtime.output_dir = cfg.hydra.run.dir

os.makedirs(cfg.log_dir, exist_ok=True)

# %%
df_train = pd.read_csv(os.path.join(cfg.data.data_dir, cfg.data.meta_csv_train_name))
df_test = pd.read_csv(os.path.join(cfg.data.data_dir, cfg.data.meta_csv_test_name))
df_fold = pd.read_parquet(os.path.join(cfg.data.data_dir, cfg.data.kfold_df_name))
assert (df_train["isic_id"] == df_fold["isic_id"]).all()

folds = []
for k in range(cfg.data.n_fold):
    if cfg.data.kfold_method == "sgkf":
        col_name = f"StratifiedGroupKFold_{cfg.data.n_fold}_{k}"
    if cfg.data.kfold_method == "tsgkf":
        col_name = f"TSGKF_{cfg.data.n_fold}_{k}"
    else:
        assert False
    train_idx = np.where(df_fold[col_name] == "train")[0]
    val_idx = np.where(df_fold[col_name] == "val")[0]
    folds.append([train_idx, val_idx])


# %%
if cfg.gbdt_params.use_logits:
    dnn_col_name = "logits"
else:
    dnn_col_name = "probabilities"

dnn_run_name_list = []
df_dnn_preds_list = []
for dnn_run in cfg.gbdt_params.get("dnn_predictions", []):
    df_list = []
    for k in range(cfg.data.n_fold):
        df_tmp = pd.read_parquet(dnn_run.dir + f"/fold{k}.parquet")
        df_tmp = df_tmp[["isic_id", dnn_col_name]].rename({dnn_col_name: "predictions"}, axis="columns")
        if cfg.gbdt_params.dnn_binning:
            assert not cfg.gbdt_params.use_logits
            bins = np.linspace(0, 1, cfg.gbdt_params.dnn_binning)
            df_tmp["predictions"] = pd.cut(df_tmp["predictions"], bins=bins, labels=False)
        df_list.append(df_tmp)
    df_preds = df_list[0]
    for k, _df_preds in enumerate(df_list[1:], start=1):
        df_preds = df_preds.merge(_df_preds, how="left", on="isic_id", suffixes=("", f"_{k}"))
    df_preds = df_preds.rename({"predictions": "predictions_0"}, axis="columns")
    df_dnn_preds_list.append(df_preds)
    dnn_run_name_list.append(dnn_run.name)

# stackingのために、val予測を新たなtrainとする
df_train_2nd = []
for fold in range(cfg.data.n_fold):
    for run_name, df_preds in zip(dnn_run_name_list, df_dnn_preds_list):
        assert (df_train["isic_id"] == df_preds["isic_id"]).all()
        df_train[run_name] = df_preds[f"predictions_{fold}"]

    _df_valid = df_train.iloc[folds[fold][1]]
    df_train_2nd.append(_df_valid)

df_train = pd.concat(df_train_2nd).sort_index()

# dummy
df_test[dnn_run_name_list] = 0

# %%
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    df_train[dnn_run_name_list].corr(), square=True, vmax=1, vmin=-1, center=0, cmap="coolwarm", annot=True
)

# %%
df_train, feature_cols, cat_cols = feature_engineering_new(df_train, version=cfg.gbdt_params.version_fe)
df_test, _, _ = feature_engineering_new(df_test, version=cfg.gbdt_params.version_fe)

df_train, df_test, feature_cols, cat_cols = preprocess_df(df_train, df_test, feature_cols, cat_cols)
target_col = "target"

feature_cols_without_dnn = copy.copy(feature_cols)
feature_cols += dnn_run_name_list

# %%
use_dnn_tuning = [
    # "0821-convnextv2_tiny-meta_target03-transV2-lr1e-3-target_decay001-bs128_2-ep100-neg5-tsgkf",
    # "0821-eva02_small-sep_head-transV6-lr1e-3-target_decay001-warmup50-wd1e-2-drop01-bs32_8-ep80-neg3-tsgkf",
    # "0821-beitv2_base-sep_head-transV8-lr1e-3-target_decay001-warmup50-wd1e-2-bs32_8-mixup-ep100-neg3-tsgkf",
    # "0824-swinv2_small-transV2-lr1e-3-target_decay001-bs32_8-drop01-ep200-neg3-cluster7t5-tsgkf",
    # "0824-eva02_small-sep_head-transV8-lr1e-3-target_decay0008-warmup50-wd1e-2-drop01-bs32_8-ep80-neg3-cluster7t5-tsgkf",
    # "0825-tip_finetune_onlyImage-convnextv2_nano_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs64_2-transV2-ep80",
    # "0827-deit3_small-transV2-lr1e-3-target_decay001-bs32_8-ep200-neg3-tsgkf",
    # "0828-resnext50-transV2-lr1e-3-target_decay001-bs32_8-ep200-neg3-cluster7t5-tsgkf",
    # "0827-tip_finetune_onlyImage-swin_tiny_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs_64_2-transV8-ep80",
    # "edgenext0822",
    # "edgenext0822_128",
]
n_trials = 200
null_imp_setting = None
version_dnn_fe = None
not_use_col_keywards = ["patient_loc_norm", "patient_loc_sim_norm", "patient_site_general_norm"]

# folds = folds[1:5]
# %% optuna lightgbm
import optuna


def lgb_objective(trial):
    n_iter = 200  # trial.suggest_int("n_iter", 100, 1000),
    params = {
        "objective": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 2e-2, 1e-1, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 1.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 0.8),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.8),
        "bagging_freq": trial.suggest_int("bagging_freq", 3, 7),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 40, 200),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 2.0),
    }

    estimator = Pipeline(
        [
            (
                "sampler_1",
                RandomOverSampler(sampling_strategy=cfg.gbdt_params.over_sampling_ratio),
            ),
            (
                "sampler_2",
                RandomUnderSampler(
                    sampling_strategy=cfg.gbdt_params.under_sampling_ratio,
                ),
            ),
            (
                "filter",
                SelectColumns(
                    feature_cols_without_dnn,
                    use_dnn_tuning,
                    null_imp_setting=null_imp_setting,
                    not_use_col_keywards=not_use_col_keywards,
                ),
            ),
            ("dnn_fe", DNNFeatureEngineering(use_dnn_tuning, feature_cols_without_dnn, version_dnn_fe)),
            ("classifier", lgb.LGBMClassifier(**params, n_estimators=n_iter)),
        ]
    )

    X = df_train
    y = df_train[target_col]

    val_score = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        cv=folds,
        scoring=custom_metric,
    )

    print(np.mean(val_score), np.std(val_score))

    return np.mean(val_score)  # - np.std(val_score)


study = optuna.create_study(direction="maximize")
study.optimize(
    lgb_objective,
    n_trials=n_trials,
    # timeout=600,
)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %% optuna catboost
import optuna


def cb_objective(trial):
    params = {
        "loss_function": "Logloss",
        "verbose": False,
        "iterations": 200,  # trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 3e-2, 1e-1, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 2.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 0.8),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 1.3, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 0.8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "random_strength": trial.suggest_float("random_strength", 0.5, 1.5),
    }

    params["cat_features"] = cat_cols
    estimator = Pipeline(
        [
            (
                "sampler_1",
                RandomOverSampler(sampling_strategy=cfg.gbdt_params.over_sampling_ratio),
            ),
            (
                "sampler_2",
                RandomUnderSampler(
                    sampling_strategy=cfg.gbdt_params.under_sampling_ratio,
                ),
            ),
            (
                "filter",
                SelectColumns(
                    feature_cols_without_dnn,
                    use_dnn_tuning,
                    null_imp_setting=null_imp_setting,
                    not_use_col_keywards=not_use_col_keywards,
                ),
            ),
            ("dnn_fe", DNNFeatureEngineering(use_dnn_tuning, feature_cols_without_dnn, version_dnn_fe)),
            ("classifier", cb.CatBoostClassifier(**params)),
        ]
    )

    X = df_train
    y = df_train[target_col]

    val_score = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        cv=folds,
        scoring=custom_metric,
    )

    print(np.mean(val_score), np.std(val_score))

    return np.mean(val_score)  # - np.std(val_score)


study = optuna.create_study(direction="maximize")
study.optimize(
    cb_objective,
    n_trials=n_trials,
    # timeout=600,
)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %% optuna xgboost
import optuna


def xgb_objective(trial):
    params = {
        "enable_categorical": True,
        "device": "cuda",
        "tree_method": "hist",
        "n_estimators": 200,  # trial.suggest_int("nn_estimators_iter", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 2e-2, 1e-1, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 1e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 5),
        "subsample": trial.suggest_float("subsample", 0.2, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 0.8),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.2, 0.8),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 2.0),
    }

    estimator = Pipeline(
        [
            (
                "sampler_1",
                RandomOverSampler(sampling_strategy=cfg.gbdt_params.over_sampling_ratio),
            ),
            (
                "sampler_2",
                RandomUnderSampler(
                    sampling_strategy=cfg.gbdt_params.under_sampling_ratio,
                ),
            ),
            (
                "filter",
                SelectColumns(
                    feature_cols_without_dnn,
                    use_dnn_tuning,
                    null_imp_setting=null_imp_setting,
                    not_use_col_keywards=not_use_col_keywards,
                ),
            ),
            ("dnn_fe", DNNFeatureEngineering(use_dnn_tuning, feature_cols_without_dnn, version_dnn_fe)),
            ("classifier", xgb.XGBClassifier(**params)),
        ]
    )

    X = df_train
    y = df_train[target_col]

    val_score = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,
        cv=folds,
        scoring=custom_metric,
    )

    print(np.mean(val_score), np.std(val_score))

    return np.mean(val_score)  # - np.std(val_score)


study = optuna.create_study(direction="maximize")
study.optimize(
    xgb_objective,
    n_trials=n_trials,
    # timeout=600,
)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
