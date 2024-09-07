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

from src.isic_utils.utils import prepare_df_for_gbdt, comp_score, custom_metric, preprocess_df, SelectColumns
from src.isic_utils.feature_engineering import feature_engineering_new
from src.isic_utils.gbdt_models import GBDTModels

# %%
gbdt_params = "0831-feV8-for_null_importance-s3"

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
df_train, feature_cols, cat_cols = feature_engineering_new(df_train, version=cfg.gbdt_params.version_fe)
df_test, _, _ = feature_engineering_new(df_test, version=cfg.gbdt_params.version_fe)

df_train, df_test, feature_cols, cat_cols = preprocess_df(df_train, df_test, feature_cols, cat_cols)
target_col = "target"

feature_cols_without_dnn = copy.copy(feature_cols)

# %%
result_dict = {}
for fold in range(cfg.data.n_fold):
    _df_train = df_train.iloc[folds[fold][0]].reset_index(drop=True)
    _df_valid = df_train.iloc[folds[fold][1]].reset_index(drop=True)

    X, y = _df_train[feature_cols], _df_train[target_col]

    gbdt_models = GBDTModels(cfg.gbdt_params, feature_cols_without_dnn, cat_cols)
    gbdt_models.fit(X, y)

    save_file_path = os.path.join(cfg.log_dir, f"model_{fold}.joblib")
    joblib.dump(gbdt_models, save_file_path)

    gbdt_models = joblib.load(save_file_path)

    preds = gbdt_models.predict(_df_valid[feature_cols])
    score = comp_score(_df_valid[[target_col]], pd.DataFrame(preds, columns=["prediction"]), "")
    print(f"fold: {fold} - Partial AUC Score: {score:.5f}")

    # save predictions for stacking
    preds_all = gbdt_models.predict(df_train[feature_cols])
    df_preds = pd.DataFrame({"isic_id": df_train["isic_id"], "predictions": preds_all})
    df_preds.to_parquet(os.path.join(cfg.log_dir, f"fold{fold}.parquet"))

    result_dict[f"fold_{fold}"] = score

result_dict["cv_score"] = np.array([result_dict[f"fold_{fold}"] for fold in range(cfg.data.n_fold)]).mean()

result_dict


# %% feature importance
fi = []
for fold in range(cfg.data.n_fold):
    save_file_path = os.path.join(cfg.log_dir, f"model_{fold}.joblib")
    gbdt_models = joblib.load(save_file_path)

    fi.append(gbdt_models.get_feature_importance())

for fold in range(cfg.data.n_fold):
    for model_idx in range(len(fi[fold][0])):
        df_imp = (
            pd.DataFrame({"feature": fi[fold][1][model_idx], "importance": fi[fold][0][model_idx]})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        df_imp.to_csv(os.path.join(cfg.log_dir, f"feat_imp-fold{fold}-{model_idx}.csv"), index=False)

        # plt.figure(figsize=(16, 12))
        # plt.barh(df_imp["feature"], df_imp["importance"])
        # plt.show()

# %%


# %%
N = 100

gbdt_params = "0831-feV8-for_null_importance-s1"


for i in range(N):
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

    cfg.log_dir = cfg.log_dir + f"_null_importance_{i}"
    os.makedirs(cfg.log_dir, exist_ok=True)
    result_dict = {}
    for fold in range(cfg.data.n_fold):
        _df_train = df_train.iloc[folds[fold][0]].reset_index(drop=True)
        _df_valid = df_train.iloc[folds[fold][1]].reset_index(drop=True)

        X, y = _df_train[feature_cols], _df_train[target_col]

        y = np.random.permutation(y)

        gbdt_models = GBDTModels(cfg.gbdt_params, feature_cols_without_dnn, cat_cols)
        gbdt_models.fit(X, y, seed=(i + 1) * 100)

        save_file_path = os.path.join(cfg.log_dir, f"model_{fold}.joblib")
        joblib.dump(gbdt_models, save_file_path)

    # feature importance
    fi = []
    for fold in range(cfg.data.n_fold):
        save_file_path = os.path.join(cfg.log_dir, f"model_{fold}.joblib")
        gbdt_models = joblib.load(save_file_path)

        fi.append(gbdt_models.get_feature_importance())

    for fold in range(cfg.data.n_fold):
        for model_idx in range(len(fi[fold][0])):
            df_imp = (
                pd.DataFrame({"feature": fi[fold][1][model_idx], "importance": fi[fold][0][model_idx]})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
            df_imp.to_csv(os.path.join(cfg.log_dir, f"feat_imp-fold{fold}-{model_idx}.csv"), index=False)

            # plt.figure(figsize=(16, 12))
            # plt.barh(df_imp["feature"], df_imp["importance"])
            # plt.show()

# %%
select_model_type = 2

actual_imp_df = pd.DataFrame()
null_imp_df = pd.DataFrame()

for fold in range(cfg.data.n_fold):
    imp_df = pd.read_csv(
        f"/workspace/logs/gbdt/runs/0831-feV8-for_null_importance-s3/feat_imp-fold{fold}-{select_model_type}.csv"
    )
    imp_df["fold"] = fold
    actual_imp_df = pd.concat([actual_imp_df, imp_df])

    for i in range(N):
        imp_df = pd.read_csv(
            f"/workspace/logs/gbdt/runs/0831-feV8-for_null_importance-s1_null_importance_{i}/feat_imp-fold{fold}-{select_model_type}.csv"
        )
        imp_df["fold"] = fold
        imp_df["run"] = i
        null_imp_df = pd.concat([null_imp_df, imp_df])


def display_distributions(actual_imp_df, null_imp_df, feature):
    # ある特徴量に対する重要度を取得
    actual_imp = actual_imp_df.query(f"feature == '{feature}'")["importance"].mean()
    null_imp = null_imp_df.query(f"feature == '{feature}'")["importance"]

    # 可視化
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    a = ax.hist(null_imp, label="Null importances")
    ax.vlines(x=actual_imp, ymin=0, ymax=np.max(a[0]), color="r", linewidth=10, label="Real Target")
    ax.legend(loc="upper right")
    ax.set_title(f"Importance of {feature.upper()}", fontweight="bold")
    plt.xlabel(f"Null Importance Distribution for {feature.upper()}")
    plt.ylabel("Importance")
    plt.show()


features_import_order = (
    actual_imp_df.groupby("feature").mean()["importance"].sort_values(ascending=False).index
)
for feature in features_import_order[-50:]:
    display_distributions(actual_imp_df, null_imp_df, feature)

# %%
# 閾値を設定
THRESHOLD = 80

imp_features_sets = []
for select_model_type in range(3):
    actual_imp_df = pd.DataFrame()
    null_imp_df = pd.DataFrame()

    for fold in range(cfg.data.n_fold):
        imp_df = pd.read_csv(
            f"/workspace/logs/gbdt/runs/0831-feV8-for_null_importance-s3/feat_imp-fold{fold}-{select_model_type}.csv"
        )
        imp_df["fold"] = fold
        actual_imp_df = pd.concat([actual_imp_df, imp_df])

        for i in range(N):
            imp_df = pd.read_csv(
                f"/workspace/logs/gbdt/runs/0831-feV8-for_null_importance-s1_null_importance_{i}/feat_imp-fold{fold}-{select_model_type}.csv"
            )
            imp_df["fold"] = fold
            imp_df["run"] = i
            null_imp_df = pd.concat([null_imp_df, imp_df])
    # 閾値を超える特徴量を取得
    imp_features = []
    for feature in features_import_order:
        actual_value = actual_imp_df.query(f"feature=='{feature}'")["importance"].mean()
        null_value = null_imp_df.query(f"feature=='{feature}'")["importance"].values
        percentage = (null_value < actual_value).sum() / null_value.size * 100
        if percentage >= THRESHOLD:
            imp_features.append(feature)

    print(imp_features)
    print(len(imp_features))

    imp_features_sets.append(set(imp_features))

imp_features_intersection = list(set.intersection(*imp_features_sets))
imp_features_union = list(set.union(*imp_features_sets))
# %%
