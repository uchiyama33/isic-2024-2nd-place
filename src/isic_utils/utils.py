import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.utils.null_importance import null_imp_features_dict

MAX_CATEGORY_DIM = 30


def feature_engineering(df):
    # New features to try...
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(
        df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2
    )
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    # df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2)
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]

    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (
        df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]
    ) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = (
        df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    )
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt(
        (df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3
    )
    df["color_shape_composite_index"] = (
        df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]
    ) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (
        df["tbp_lv_area_perim_ratio"]
        + df["tbp_lv_eccentricity"]
        + df["tbp_lv_norm_color"]
        + df["tbp_lv_symm_2axis"]
    ) / 4

    new_num_cols = [
        "lesion_size_ratio",
        "lesion_shape_index",
        "hue_contrast",
        "luminance_contrast",
        "lesion_color_difference",
        "border_complexity",
        # "color_uniformity",
        "3d_position_distance",
        "perimeter_to_area_ratio",
        "lesion_visibility_score",
        "symmetry_border_consistency",
        "color_consistency",
        "size_age_interaction",
        "hue_color_std_interaction",
        "lesion_severity_index",
        "shape_complexity_index",
        "color_contrast_index",
        "log_lesion_area",
        "normalized_lesion_size",
        "mean_hue_difference",
        "std_dev_contrast",
        "color_shape_composite_index",
        "3d_lesion_orientation",
        "overall_color_difference",
        "symmetry_perimeter_interaction",
        "comprehensive_lesion_index",
    ]
    new_cat_cols = ["combined_anatomical_site"]
    return df, new_num_cols, new_cat_cols


def prepare_df_for_gbdt(df_train, df_test=None):
    df_train, new_num_cols, new_cat_cols = feature_engineering(df_train.copy())
    if df_test is not None:
        df_test, _, _ = feature_engineering(df_test.copy())

    num_cols = [
        "age_approx",
        "clin_size_long_diam_mm",
        "tbp_lv_A",
        "tbp_lv_Aext",
        "tbp_lv_B",
        "tbp_lv_Bext",
        "tbp_lv_C",
        "tbp_lv_Cext",
        "tbp_lv_H",
        "tbp_lv_Hext",
        "tbp_lv_L",
        "tbp_lv_Lext",
        "tbp_lv_areaMM2",
        "tbp_lv_area_perim_ratio",
        "tbp_lv_color_std_mean",
        "tbp_lv_deltaA",
        "tbp_lv_deltaB",
        "tbp_lv_deltaL",
        "tbp_lv_deltaLB",
        "tbp_lv_deltaLBnorm",
        "tbp_lv_eccentricity",
        "tbp_lv_minorAxisMM",
        "tbp_lv_nevi_confidence",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_perimeterMM",
        "tbp_lv_radial_color_std_max",
        "tbp_lv_stdL",
        "tbp_lv_stdLExt",
        "tbp_lv_symm_2axis",
        "tbp_lv_symm_2axis_angle",
        "tbp_lv_x",
        "tbp_lv_y",
        "tbp_lv_z",
    ] + new_num_cols
    # anatom_site_general
    cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"] + new_cat_cols

    category_encoder = OrdinalEncoder(
        categories="auto",
        dtype=int,
        handle_unknown="use_encoded_value",
        unknown_value=-2,
        encoded_missing_value=-1,
    )

    X_cat = category_encoder.fit_transform(df_train[cat_cols])
    for c, cat_col in enumerate(cat_cols):
        df_train[cat_col] = X_cat[:, c]

    if df_test is not None:
        X_cat = category_encoder.transform(df_test[cat_cols])
        for c, cat_col in enumerate(cat_cols):
            df_test[cat_col] = X_cat[:, c]

    train_cols = num_cols + cat_cols
    target_col = "target"

    return df_train, df_test, train_cols, target_col, cat_cols


def comp_score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float = 0.80
):
    v_gt = abs(np.asarray(solution.values) - 1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc


def custom_metric(estimator, X, y_true):
    y_hat = estimator.predict_proba(X)[:, 1]
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)

    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])

    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc


def prepare_df_for_dnn(df_target, df_all, version):
    if version == 1:
        # 全特徴入り [3, 2, 21, 8]
        num_cols = [
            "age_approx",
            "clin_size_long_diam_mm",
            "tbp_lv_A",
            "tbp_lv_Aext",
            "tbp_lv_B",
            "tbp_lv_Bext",
            "tbp_lv_C",
            "tbp_lv_Cext",
            "tbp_lv_H",
            "tbp_lv_Hext",
            "tbp_lv_L",
            "tbp_lv_Lext",
            "tbp_lv_areaMM2",
            "tbp_lv_area_perim_ratio",
            "tbp_lv_color_std_mean",
            "tbp_lv_deltaA",
            "tbp_lv_deltaB",
            "tbp_lv_deltaL",
            "tbp_lv_deltaLB",
            "tbp_lv_deltaLBnorm",
            "tbp_lv_eccentricity",
            "tbp_lv_minorAxisMM",
            "tbp_lv_nevi_confidence",
            "tbp_lv_norm_border",
            "tbp_lv_norm_color",
            "tbp_lv_perimeterMM",
            "tbp_lv_radial_color_std_max",
            "tbp_lv_stdL",
            "tbp_lv_stdLExt",
            "tbp_lv_symm_2axis",
            "tbp_lv_symm_2axis_angle",
            "tbp_lv_x",
            "tbp_lv_y",
            "tbp_lv_z",
        ]

        # nanがあるのはage_approxのみ
        df_all[num_cols] = df_all[num_cols].fillna(df_all[num_cols].mean(numeric_only=True))
        df_target[num_cols] = df_target[num_cols].fillna(df_all[num_cols].mean(numeric_only=True))

        scaler = StandardScaler()
        df_all[num_cols] = scaler.fit_transform(df_all[num_cols])
        df_target[num_cols] = scaler.transform(df_target[num_cols])

        # anatom_site_general
        cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]

        # nanがあるのはsexのみ
        category_encoder = OrdinalEncoder(
            categories="auto",
            dtype=int,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=2,
        )
        X_cat = category_encoder.fit_transform(df_all[cat_cols])
        for c, cat_col in enumerate(cat_cols):
            df_all[cat_col] = X_cat[:, c]

        X_cat = category_encoder.transform(df_target[cat_cols])
        for c, cat_col in enumerate(cat_cols):
            df_target[cat_col] = X_cat[:, c]

    if version == 2:
        # target用、v1からlocationを削除しattributionを追加 [3, 2, 7]
        num_cols = [
            "age_approx",
            "clin_size_long_diam_mm",
            "tbp_lv_A",
            "tbp_lv_Aext",
            "tbp_lv_B",
            "tbp_lv_Bext",
            "tbp_lv_C",
            "tbp_lv_Cext",
            "tbp_lv_H",
            "tbp_lv_Hext",
            "tbp_lv_L",
            "tbp_lv_Lext",
            "tbp_lv_areaMM2",
            "tbp_lv_area_perim_ratio",
            "tbp_lv_color_std_mean",
            "tbp_lv_deltaA",
            "tbp_lv_deltaB",
            "tbp_lv_deltaL",
            "tbp_lv_deltaLB",
            "tbp_lv_deltaLBnorm",
            "tbp_lv_eccentricity",
            "tbp_lv_minorAxisMM",
            "tbp_lv_nevi_confidence",
            "tbp_lv_norm_border",
            "tbp_lv_norm_color",
            "tbp_lv_perimeterMM",
            "tbp_lv_radial_color_std_max",
            "tbp_lv_stdL",
            "tbp_lv_stdLExt",
            "tbp_lv_symm_2axis",
            "tbp_lv_symm_2axis_angle",
            "tbp_lv_x",
            "tbp_lv_y",
            "tbp_lv_z",
        ]

        # nanがあるのはage_approxのみ
        df_all[num_cols] = df_all[num_cols].fillna(df_all[num_cols].mean(numeric_only=True))
        df_target[num_cols] = df_target[num_cols].fillna(df_all[num_cols].mean(numeric_only=True))

        scaler = StandardScaler()
        df_all[num_cols] = scaler.fit_transform(df_all[num_cols])
        df_target[num_cols] = scaler.transform(df_target[num_cols])

        # anatom_site_general
        cat_cols = ["sex", "tbp_tile_type", "attribution"]

        # nanがあるのはsexのみ
        category_encoder = OrdinalEncoder(
            categories="auto",
            dtype=int,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=2,
        )
        X_cat = category_encoder.fit_transform(df_all[cat_cols])
        for c, cat_col in enumerate(cat_cols):
            df_all[cat_col] = X_cat[:, c]

        X_cat = category_encoder.transform(df_target[cat_cols])
        for c, cat_col in enumerate(cat_cols):
            df_target[cat_col] = X_cat[:, c]

    if version == 3:
        # 0.178のノートブック参考、TIP向け [3, 2, 21, 8, 6]
        num_cols = [
            "age_approx",
            "clin_size_long_diam_mm",
            "tbp_lv_A",
            "tbp_lv_Aext",
            "tbp_lv_B",
            "tbp_lv_Bext",
            "tbp_lv_C",
            "tbp_lv_Cext",
            "tbp_lv_H",
            "tbp_lv_Hext",
            "tbp_lv_L",
            "tbp_lv_Lext",
            "tbp_lv_areaMM2",
            "tbp_lv_area_perim_ratio",
            "tbp_lv_color_std_mean",
            "tbp_lv_deltaA",
            "tbp_lv_deltaB",
            "tbp_lv_deltaL",
            "tbp_lv_deltaLB",
            "tbp_lv_deltaLBnorm",
            "tbp_lv_eccentricity",
            "tbp_lv_minorAxisMM",
            "tbp_lv_nevi_confidence",
            "tbp_lv_norm_border",
            "tbp_lv_norm_color",
            "tbp_lv_perimeterMM",
            "tbp_lv_radial_color_std_max",
            "tbp_lv_stdL",
            "tbp_lv_stdLExt",
            "tbp_lv_symm_2axis",
            "tbp_lv_symm_2axis_angle",
            "tbp_lv_x",
            "tbp_lv_y",
            "tbp_lv_z",
        ]

        df_all["n_data"] = df_all.patient_id.map(df_all.groupby(["patient_id"]).isic_id.count())
        df_target["n_data"] = df_target.patient_id.map(df_target.groupby(["patient_id"]).isic_id.count())
        num_cols += ["n_data"]

        # nanがあるのはage_approxのみ
        df_all[num_cols] = df_all[num_cols].fillna(df_all[num_cols].mean(numeric_only=True))
        df_target[num_cols] = df_target[num_cols].fillna(df_all[num_cols].mean(numeric_only=True))

        scaler = StandardScaler()
        df_all[num_cols] = scaler.fit_transform(df_all[num_cols])
        df_target[num_cols] = scaler.transform(df_target[num_cols])

        # 患者で値が共通する列
        not_calc_patient_norm_cols = ["age_approx", "n_data"]

        # 新しい列を作成し、リストに格納 df_all
        new_columns = []
        err = 1e-5
        for col in num_cols:
            if col in not_calc_patient_norm_cols:
                continue
            norm_col_name = f"{col}_patient_norm"
            norm_col = (
                df_all.groupby("patient_id")[col]
                .transform(lambda x: (x - x.mean()) / (x.std() + err))
                .rename(norm_col_name)
            )
            # 一つしかデーがない患者はNaNになるので置換
            norm_col = norm_col.fillna(0.0)
            new_columns.extend([norm_col])
        # pd.concatを使って新しい列を一度に追加
        df_new_columns = pd.concat(new_columns, axis=1)
        df_all = pd.concat([df_all, df_new_columns], axis=1)

        # 新しい列を作成し、リストに格納 df_target
        new_columns = []
        err = 1e-5
        for col in num_cols:
            if col in not_calc_patient_norm_cols:
                continue
            norm_col_name = f"{col}_patient_norm"
            norm_col = (
                df_target.groupby("patient_id")[col]
                .transform(lambda x: (x - x.mean()) / (x.std() + err))
                .rename(norm_col_name)
            )
            # 一つしかデーがない患者はNaNになるので置換
            norm_col = norm_col.fillna(0.0)
            new_columns.extend([norm_col])
        # pd.concatを使って新しい列を一度に追加
        df_new_columns = pd.concat(new_columns, axis=1)
        df_target = pd.concat([df_target, df_new_columns], axis=1)

        new_num_cols = []
        for col in num_cols:
            if col in not_calc_patient_norm_cols:
                continue
            new_num_cols += [f"{col}_patient_norm"]
        num_cols += new_num_cols

        # meta_mlpではclipの有無で変化なし
        # df_target[num_cols] = df_target[num_cols].clip(-5, 5)

        # log transformation
        # for col in num_cols:
        #     # df_allから最小値を決定
        #     min_value = df_all[col].min()
        #     df_target[col] = np.clip(df_target[col], min_value, None)
        #     # 必要な定数を決定（最小値が負の場合、その絶対値に1を足す）
        #     constant = 1 - min_value if min_value <= 0 else 0
        #     # 小さな定数を追加して対数変換を適用
        #     df_target[col] = df_target[col].apply(lambda x: np.log(x + constant + 1e-5)) / 4

        # anatom_site_general
        cat_cols = [
            "sex",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "anatom_site_general",
        ]

        # 欠損は新たなカテゴリを割り当ててしまう
        df_all["sex"] = df_all["sex"].fillna("Missing")
        df_all["anatom_site_general"] = df_all["anatom_site_general"].fillna("Missing")
        df_target["sex"] = df_target["sex"].fillna("Missing")
        df_target["anatom_site_general"] = df_target["anatom_site_general"].fillna("Missing")

        category_encoder = OrdinalEncoder(
            categories="auto",
            dtype=int,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=0,
        )
        X_cat = category_encoder.fit_transform(df_all[cat_cols])
        for c, cat_col in enumerate(cat_cols):
            df_all[cat_col] = X_cat[:, c]

        X_cat = category_encoder.transform(df_target[cat_cols])
        for c, cat_col in enumerate(cat_cols):
            df_target[cat_col] = X_cat[:, c]

    return df_target[num_cols].to_numpy().astype(np.float32), df_target[cat_cols].to_numpy().astype(np.int32)


def preprocess_df(df_train, df_test, feature_cols, cat_cols):
    # replace nan, inf
    for col in list(set(feature_cols) - set(cat_cols)):
        median_value = df_train[col].median()
        df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
        df_train[col] = df_train[col].fillna(median_value)
        df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)
        df_test[col] = df_test[col].fillna(median_value)

    # encode category
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown="ignore")
    encoder.fit(df_train[cat_cols])

    new_cat_cols = [f"onehot_{i}" for i in range(len(encoder.get_feature_names_out()))]

    new_train_cols = pd.DataFrame(encoder.transform(df_train[cat_cols]), columns=new_cat_cols)
    new_test_cols = pd.DataFrame(encoder.transform(df_test[cat_cols]), columns=new_cat_cols)

    df_train = pd.concat([df_train, new_train_cols], axis=1)
    df_test = pd.concat([df_test, new_test_cols], axis=1)

    df_train[new_cat_cols] = df_train[new_cat_cols].astype("category")
    df_test[new_cat_cols] = df_test[new_cat_cols].astype("category")

    for col in cat_cols:
        feature_cols.remove(col)

    feature_cols.extend(new_cat_cols)
    cat_cols = new_cat_cols

    # DNN特徴量の集約で使うため
    # feature_cols.append("patient_id")

    return df_train, df_test, feature_cols, cat_cols


def preprocess_df_for_nn(df_train, df_test, feature_cols):
    # replace nan, inf
    for col in feature_cols:
        median_value = df_train[col].mode()[0]
        df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
        df_train[col] = df_train[col].fillna(median_value)
        df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)
        df_test[col] = df_test[col].fillna(median_value)

    return df_train, df_test


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_cols_without_dnn,
        use_dnn,
        not_use_col_specific_names=[],
        not_use_col_keywards=[],
        null_imp_setting=None,
    ):
        self.feature_cols_without_dnn = feature_cols_without_dnn
        self.use_dnn = use_dnn
        self.not_use_col_specific_names = not_use_col_specific_names
        self.not_use_col_keywards = not_use_col_keywards
        self.null_imp_setting = null_imp_setting

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        use_cols = self.feature_cols_without_dnn

        if self.null_imp_setting is not None:
            null_imp_features = null_imp_features_dict[self.null_imp_setting]
            use_cols = [item for item in use_cols if item not in null_imp_features]

        use_cols = [item for item in use_cols if item not in self.not_use_col_specific_names]
        for keyward in self.not_use_col_keywards:
            use_cols = [item for item in use_cols if keyward not in item]

        # patient_idはDNNFeatureEngineeringのため
        use_cols = use_cols + self.use_dnn + ["patient_id"]
        return X[use_cols]


class AddNoiseDnnPreds(BaseEstimator, TransformerMixin):
    def __init__(self, dnn_columns, noise_std):
        self.dnn_columns = dnn_columns
        self.noise_std = noise_std

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.noise_std:
            for col in self.dnn_columns:
                noise = np.random.normal(loc=0, scale=self.noise_std, size=X[col].shape)
                X[col] = X[col] + noise
                X[col] = X[col].clip(0, 1)

        return X


class DNNFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, dnn_cols, other_cols, version=None):
        self.dnn_cols = dnn_cols
        self.other_cols = other_cols
        self.version = version

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(self.dnn_cols) == 0:
            X = X.drop(["patient_id"], axis=1)
            return X

        if self.version == 1:
            # 予測値の統計量を計算
            predictions_min = X[self.dnn_cols].min(axis=1)
            predictions_max = X[self.dnn_cols].max(axis=1)
            predictions_mean = X[self.dnn_cols].mean(axis=1)
            predictions_std = X[self.dnn_cols].std(axis=1)

            # 新しい列を一度に追加
            X = pd.concat(
                [
                    X,
                    predictions_min.rename("predictions_min"),
                    predictions_max.rename("predictions_max"),
                    predictions_mean.rename("predictions_mean"),
                    predictions_std.rename("predictions_std"),
                ],
                axis=1,
            )

            new_cols = [
                "predictions_min",
                "predictions_max",
                "predictions_mean",
                "predictions_std",
            ]

            tmp_rename_dict = {}
            for i in range(len(self.dnn_cols)):
                tmp = f"tmp_dnn_{i}"
                tmp_rename_dict[self.dnn_cols[i]] = tmp
                new_cols.append(tmp)
            X = X.rename(tmp_rename_dict, axis=1)

            # 各patient_idごとに標準化
            grouped = X.groupby("patient_id")[new_cols].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))

            # 新しい列を一度に追加
            X = pd.concat([X, grouped.add_suffix("_patient_norm")], axis=1)

            tmp_rename_dict = {}
            for i in range(len(self.dnn_cols)):
                tmp = f"tmp_dnn_{i}_patient_norm"
                tmp_rename_dict[self.dnn_cols[i] + "_patient_norm"] = tmp

            X = X.rename({v: k for k, v in tmp_rename_dict.items()}, axis=1)

        elif self.version == 2:
            age_normalized_mydnn_nevi_confidence_list = []
            new_cols = []
            for col in self.dnn_cols:
                new_col = (1 - X[col]) / X["age_approx"]
                new_col = new_col.rename(f"age_normalized_mydnn_nevi_conf_{col}")
                new_cols.append(f"age_normalized_mydnn_nevi_conf_{col}")
                age_normalized_mydnn_nevi_confidence_list.append(new_col)
            X = pd.concat(
                [X] + age_normalized_mydnn_nevi_confidence_list,
                axis=1,
            )

            new_cols += self.dnn_cols

            tmp_new_cols = []
            tmp_rename_dict = {}
            for i in range(len(new_cols)):
                tmp = f"tmp_dnn_{i}"
                tmp_rename_dict[new_cols[i]] = tmp
                tmp_new_cols.append(tmp)
            X = X.rename(tmp_rename_dict, axis=1)

            # 各patient_idごとに標準化
            grouped = X.groupby("patient_id")[tmp_new_cols].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-5)
            )

            # 新しい列を一度に追加
            X = pd.concat([X, grouped.add_suffix("_patient_norm")], axis=1)

            for i in range(len(new_cols)):
                tmp = f"tmp_dnn_{i}_patient_norm"
                tmp_rename_dict[new_cols[i] + "_patient_norm"] = tmp

            X = X.rename({v: k for k, v in tmp_rename_dict.items()}, axis=1)

        X = X.drop(["patient_id"], axis=1)

        return X
