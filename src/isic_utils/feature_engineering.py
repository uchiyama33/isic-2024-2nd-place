import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from hydra import compose, initialize
import os
from sklearn.preprocessing import OrdinalEncoder
import rootutils
import wandb
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import polars as pl
from scipy.stats import zscore
from numpy import nanmean, nanstd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=os.cpu_count())


def feature_engineering(df, train_cols, dnn_run_name_list, version=1):
    if version == 1:
        if len(dnn_run_name_list) > 0:
            df["predictions_min"] = df[dnn_run_name_list].min(1)
            df["predictions_max"] = df[dnn_run_name_list].max(1)
            df["predictions_mean"] = df[dnn_run_name_list].mean(1)
            df["predictions_std"] = df[dnn_run_name_list].std(1)
            train_cols += ["predictions_min", "predictions_max", "predictions_mean", "predictions_std"]

        patient_mean_diff_cols = [
            "clin_size_long_diam_mm",
            "tbp_lv_areaMM2",
            "tbp_lv_minorAxisMM",
            "tbp_lv_perimeterMM",
            "tbp_lv_area_perim_ratio",
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
            "tbp_lv_deltaA",
            "tbp_lv_deltaB",
            "tbp_lv_deltaL",
            "tbp_lv_deltaLBnorm",
            "tbp_lv_color_std_mean",
            "tbp_lv_radial_color_std_max",
            "tbp_lv_stdL",
            "tbp_lv_stdLExt",
            "tbp_lv_eccentricity",
            "tbp_lv_symm_2axis",
            "tbp_lv_symm_2axis_angle",
            "tbp_lv_norm_border",
        ]
        if len(dnn_run_name_list) > 0:
            patient_mean_diff_cols += [
                "predictions_min",
                "predictions_max",
                "predictions_mean",
                "predictions_std",
            ]

        # 患者ごとの特徴量の平均を計算
        patient_mean = df.groupby("patient_id")[patient_mean_diff_cols].mean().reset_index()
        patient_mean = patient_mean.rename(columns={col: f"mean_{col}" for col in patient_mean_diff_cols})

        # 元のデータフレームに患者ごとの平均をマージ
        df = df.merge(patient_mean, on="patient_id")

        # 各病変の特徴量と患者平均との差を計算
        for col in patient_mean_diff_cols:
            df[f"diff_{col}"] = df[col] - df[f"mean_{col}"]
            train_cols += [f"mean_{col}", f"diff_{col}"]

    if version == 2:
        if len(dnn_run_name_list) > 0:
            df["predictions_min"] = df[dnn_run_name_list].min(1)
            df["predictions_max"] = df[dnn_run_name_list].max(1)
            df["predictions_mean"] = df[dnn_run_name_list].mean(1)
            df["predictions_std"] = df[dnn_run_name_list].std(1)
            train_cols += ["predictions_min", "predictions_max", "predictions_mean", "predictions_std"]

        patient_mean_diff_cols = [
            "lesion_size_ratio",
            "lesion_shape_index",
            "hue_contrast",
            "luminance_contrast",
            "lesion_color_difference",
            "border_complexity",
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
        if len(dnn_run_name_list) > 0:
            patient_mean_diff_cols += [
                "predictions_min",
                "predictions_max",
                "predictions_mean",
                "predictions_std",
            ]

        # 患者ごとの特徴量の平均を計算
        patient_mean = df.groupby("patient_id")[patient_mean_diff_cols].mean().reset_index()
        patient_mean = patient_mean.rename(columns={col: f"mean_{col}" for col in patient_mean_diff_cols})

        # 元のデータフレームに患者ごとの平均をマージ
        df = df.merge(patient_mean, on="patient_id")

        # 各病変の特徴量と患者平均との差を計算
        for col in patient_mean_diff_cols:
            df[f"diff_{col}"] = df[col] - df[f"mean_{col}"]
            train_cols += [f"mean_{col}", f"diff_{col}"]
    else:
        assert False, "version"

    return df, train_cols


def feature_engineering_new(df, version=1, additional_num_cols=[]):
    if version == 1:
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "attribution",
        ]
        id_col = "isic_id"
        norm_cols = [f"{col}_patient_norm" for col in num_cols + new_num_cols + additional_num_cols]
        special_cols = ["count_per_patient"]
        feature_cols = num_cols + new_num_cols + additional_num_cols + cat_cols + norm_cols + special_cols

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over("patient_id"))
                    / (pl.col(col).std().over("patient_id") + err)
                ).alias(f"{col}_patient_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
            .to_pandas()
        )
        # .set_index(
        #     id_col
        # )

        # for col in df.select_dtypes(include=[np.float64]).columns:
        #     median_value = df[col].median()
        #     df[col] = df[col].fillna(median_value)

        return df, feature_cols, cat_cols

    if version == 2:
        # v1から、patient_normをpatient_anatom_site_normに変更
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "attribution",
        ]
        id_col = "isic_id"
        # norm_cols = [f"{col}_patient_norm" for col in num_cols + new_num_cols]
        patient_anatom_site_norm_cols = [f"{col}_patient_anatom_site_norm" for col in num_cols + new_num_cols]
        special_cols = ["count_per_patient"]
        feature_cols = (
            num_cols + new_num_cols + cat_cols + special_cols + patient_anatom_site_norm_cols  # + norm_cols
        )

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            # .with_columns(
            #     (
            #         (pl.col(col) - pl.col(col).mean().over("patient_id"))
            #         / (pl.col(col).std().over("patient_id") + err)
            #     ).alias(f"{col}_patient_norm")
            #     for col in (num_cols + new_num_cols)
            # )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "anatom_site_general"]))
                    / (pl.col(col).std().over(["patient_id", "anatom_site_general"]) + err)
                ).alias(f"{col}_patient_anatom_site_norm")
                for col in (num_cols + new_num_cols)
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
            .to_pandas()
        )
        # .set_index(
        #     id_col
        # )

    elif version == 3:
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
            # add from v1
            "asymmetry_ratio",
            "asymmetry_area_ratio",
            "color_variation_intensity",
            "color_contrast_ratio",
            "shape_irregularity",
            "border_density",
            "size_age_ratio",
            "area_diameter_ratio",
            "position_norm_3d",
            "position_angle_3d_xz",
            "lab_chroma",
            "lab_hue",
            "texture_contrast",
            "texture_uniformity",
            "color_difference_AB",
            "color_difference_total",
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "attribution",
        ]

        id_col = "isic_id"
        norm_cols = [f"{col}_patient_norm" for col in num_cols + new_num_cols + additional_num_cols]
        special_cols = ["count_per_patient"]

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            #  add from v1
            .with_columns(
                asymmetry_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_perimeterMM") + err),
                asymmetry_area_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_areaMM2") + err),
                color_variation_intensity=pl.col("tbp_lv_norm_color") * pl.col("tbp_lv_deltaLBnorm"),
                color_contrast_ratio=pl.col("tbp_lv_deltaLBnorm") / (pl.col("tbp_lv_L") + err),
                shape_irregularity=pl.col("tbp_lv_perimeterMM")
                / (2 * np.sqrt(np.pi * pl.col("tbp_lv_areaMM2") + err)),
                border_density=pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_perimeterMM") + err),
                size_age_ratio=pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + err),
                area_diameter_ratio=pl.col("tbp_lv_areaMM2") / (pl.col("clin_size_long_diam_mm") ** 2 + err),
                position_norm_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                position_angle_3d_xz=pl.arctan2(pl.col("tbp_lv_z"), pl.col("tbp_lv_x")),
                lab_chroma=(pl.col("tbp_lv_A") ** 2 + pl.col("tbp_lv_B") ** 2).sqrt(),
                lab_hue=pl.arctan2(pl.col("tbp_lv_B"), pl.col("tbp_lv_A")),
                texture_contrast=(pl.col("tbp_lv_stdL") / (pl.col("tbp_lv_L") + 1e-5)),
                texture_uniformity=(1 / (1 + pl.col("tbp_lv_color_std_mean"))),
                color_difference_AB=(pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2).sqrt(),
                color_difference_total=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over("patient_id"))
                    / (pl.col(col).std().over("patient_id") + err)
                ).alias(f"{col}_patient_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
            .to_pandas()
        )
        # .set_index(
        #     id_col
        # )

        # for col in df.select_dtypes(include=[np.float64]).columns:
        #     median_value = df[col].median()
        #     df[col] = df[col].fillna(median_value)

        df, ud_num_cols = ugly_duckling_processing(df.copy(), num_cols + new_num_cols + additional_num_cols)

        feature_cols = (
            num_cols + new_num_cols + additional_num_cols + cat_cols + norm_cols + special_cols + ud_num_cols
        )

    elif version == 4:
        # v3にinclude_patient_wide_udを入れる
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
            # add from v1
            "asymmetry_ratio",
            "asymmetry_area_ratio",
            "color_variation_intensity",
            "color_contrast_ratio",
            "shape_irregularity",
            "border_density",
            "size_age_ratio",
            "area_diameter_ratio",
            "position_norm_3d",
            "position_angle_3d_xz",
            "lab_chroma",
            "lab_hue",
            "texture_contrast",
            "texture_uniformity",
            "color_difference_AB",
            "color_difference_total",
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "attribution",
        ]

        id_col = "isic_id"
        norm_cols = [f"{col}_patient_norm" for col in num_cols + new_num_cols + additional_num_cols]
        special_cols = ["count_per_patient"]

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            #  add from v1
            .with_columns(
                asymmetry_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_perimeterMM") + err),
                asymmetry_area_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_areaMM2") + err),
                color_variation_intensity=pl.col("tbp_lv_norm_color") * pl.col("tbp_lv_deltaLBnorm"),
                color_contrast_ratio=pl.col("tbp_lv_deltaLBnorm") / (pl.col("tbp_lv_L") + err),
                shape_irregularity=pl.col("tbp_lv_perimeterMM")
                / (2 * np.sqrt(np.pi * pl.col("tbp_lv_areaMM2") + err)),
                border_density=pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_perimeterMM") + err),
                size_age_ratio=pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + err),
                area_diameter_ratio=pl.col("tbp_lv_areaMM2") / (pl.col("clin_size_long_diam_mm") ** 2 + err),
                position_norm_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                position_angle_3d_xz=pl.arctan2(pl.col("tbp_lv_z"), pl.col("tbp_lv_x")),
                lab_chroma=(pl.col("tbp_lv_A") ** 2 + pl.col("tbp_lv_B") ** 2).sqrt(),
                lab_hue=pl.arctan2(pl.col("tbp_lv_B"), pl.col("tbp_lv_A")),
                texture_contrast=(pl.col("tbp_lv_stdL") / (pl.col("tbp_lv_L") + 1e-5)),
                texture_uniformity=(1 / (1 + pl.col("tbp_lv_color_std_mean"))),
                color_difference_AB=(pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2).sqrt(),
                color_difference_total=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over("patient_id"))
                    / (pl.col(col).std().over("patient_id") + err)
                ).alias(f"{col}_patient_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
            .to_pandas()
        )
        # .set_index(
        #     id_col
        # )

        # for col in df.select_dtypes(include=[np.float64]).columns:
        #     median_value = df[col].median()
        #     df[col] = df[col].fillna(median_value)

        df, ud_num_cols = ugly_duckling_processing(
            df.copy(), num_cols + new_num_cols + additional_num_cols, include_patient_wide_ud=True
        )

        feature_cols = (
            num_cols + new_num_cols + additional_num_cols + cat_cols + norm_cols + special_cols + ud_num_cols
        )
    elif version == 5:
        # v3にpatient_loc_normを追加
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
            # add from v1
            "asymmetry_ratio",
            "asymmetry_area_ratio",
            "color_variation_intensity",
            "color_contrast_ratio",
            "shape_irregularity",
            "border_density",
            "size_age_ratio",
            "area_diameter_ratio",
            "position_norm_3d",
            "position_angle_3d_xz",
            "lab_chroma",
            "lab_hue",
            "texture_contrast",
            "texture_uniformity",
            "color_difference_AB",
            "color_difference_total",
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "attribution",
        ]

        id_col = "isic_id"
        norm_cols = [f"{col}_patient_norm" for col in num_cols + new_num_cols + additional_num_cols] + [
            f"{col}_patient_loc_norm" for col in num_cols + new_num_cols + additional_num_cols
        ]
        special_cols = ["count_per_patient"]

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            #  add from v1
            .with_columns(
                asymmetry_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_perimeterMM") + err),
                asymmetry_area_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_areaMM2") + err),
                color_variation_intensity=pl.col("tbp_lv_norm_color") * pl.col("tbp_lv_deltaLBnorm"),
                color_contrast_ratio=pl.col("tbp_lv_deltaLBnorm") / (pl.col("tbp_lv_L") + err),
                shape_irregularity=pl.col("tbp_lv_perimeterMM")
                / (2 * np.sqrt(np.pi * pl.col("tbp_lv_areaMM2") + err)),
                border_density=pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_perimeterMM") + err),
                size_age_ratio=pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + err),
                area_diameter_ratio=pl.col("tbp_lv_areaMM2") / (pl.col("clin_size_long_diam_mm") ** 2 + err),
                position_norm_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                position_angle_3d_xz=pl.arctan2(pl.col("tbp_lv_z"), pl.col("tbp_lv_x")),
                lab_chroma=(pl.col("tbp_lv_A") ** 2 + pl.col("tbp_lv_B") ** 2).sqrt(),
                lab_hue=pl.arctan2(pl.col("tbp_lv_B"), pl.col("tbp_lv_A")),
                texture_contrast=(pl.col("tbp_lv_stdL") / (pl.col("tbp_lv_L") + 1e-5)),
                texture_uniformity=(1 / (1 + pl.col("tbp_lv_color_std_mean"))),
                color_difference_AB=(pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2).sqrt(),
                color_difference_total=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over("patient_id"))
                    / (pl.col(col).std().over("patient_id") + err)
                ).alias(f"{col}_patient_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "tbp_lv_location"]))
                    / (pl.col(col).std().over(["patient_id", "tbp_lv_location"]) + err)
                ).alias(f"{col}_patient_loc_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
            .to_pandas()
        )
        # .set_index(
        #     id_col
        # )

        # for col in df.select_dtypes(include=[np.float64]).columns:
        #     median_value = df[col].median()
        #     df[col] = df[col].fillna(median_value)

        df, ud_num_cols = ugly_duckling_processing(df.copy(), num_cols + new_num_cols + additional_num_cols)

        feature_cols = (
            num_cols + new_num_cols + additional_num_cols + cat_cols + norm_cols + special_cols + ud_num_cols
        )

    elif version == 6:
        # v3にpatient_normをpatient_loc_normに変更
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
            # add from v1
            "asymmetry_ratio",
            "asymmetry_area_ratio",
            "color_variation_intensity",
            "color_contrast_ratio",
            "shape_irregularity",
            "border_density",
            "size_age_ratio",
            "area_diameter_ratio",
            "position_norm_3d",
            "position_angle_3d_xz",
            "lab_chroma",
            "lab_hue",
            "texture_contrast",
            "texture_uniformity",
            "color_difference_AB",
            "color_difference_total",
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "attribution",
        ]

        id_col = "isic_id"
        norm_cols = [f"{col}_patient_loc_norm" for col in num_cols + new_num_cols + additional_num_cols]
        special_cols = ["count_per_patient"]

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            #  add from v1
            .with_columns(
                asymmetry_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_perimeterMM") + err),
                asymmetry_area_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_areaMM2") + err),
                color_variation_intensity=pl.col("tbp_lv_norm_color") * pl.col("tbp_lv_deltaLBnorm"),
                color_contrast_ratio=pl.col("tbp_lv_deltaLBnorm") / (pl.col("tbp_lv_L") + err),
                shape_irregularity=pl.col("tbp_lv_perimeterMM")
                / (2 * np.sqrt(np.pi * pl.col("tbp_lv_areaMM2") + err)),
                border_density=pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_perimeterMM") + err),
                size_age_ratio=pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + err),
                area_diameter_ratio=pl.col("tbp_lv_areaMM2") / (pl.col("clin_size_long_diam_mm") ** 2 + err),
                position_norm_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                position_angle_3d_xz=pl.arctan2(pl.col("tbp_lv_z"), pl.col("tbp_lv_x")),
                lab_chroma=(pl.col("tbp_lv_A") ** 2 + pl.col("tbp_lv_B") ** 2).sqrt(),
                lab_hue=pl.arctan2(pl.col("tbp_lv_B"), pl.col("tbp_lv_A")),
                texture_contrast=(pl.col("tbp_lv_stdL") / (pl.col("tbp_lv_L") + 1e-5)),
                texture_uniformity=(1 / (1 + pl.col("tbp_lv_color_std_mean"))),
                color_difference_AB=(pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2).sqrt(),
                color_difference_total=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "tbp_lv_location"]))
                    / (pl.col(col).std().over(["patient_id", "tbp_lv_location"]) + err)
                ).alias(f"{col}_patient_loc_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
            .to_pandas()
        )
        # .set_index(
        #     id_col
        # )

        # for col in df.select_dtypes(include=[np.float64]).columns:
        #     median_value = df[col].median()
        #     df[col] = df[col].fillna(median_value)

        df, ud_num_cols = ugly_duckling_processing(df.copy(), num_cols + new_num_cols + additional_num_cols)

        feature_cols = (
            num_cols + new_num_cols + additional_num_cols + cat_cols + norm_cols + special_cols + ud_num_cols
        )

    elif version == 7:
        # v3にpatient_loc_norm, patient_loc_sim_norm, patient_site_general_normを追加
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
            # add from v1
            "asymmetry_ratio",
            "asymmetry_area_ratio",
            "color_variation_intensity",
            "color_contrast_ratio",
            "shape_irregularity",
            "border_density",
            "size_age_ratio",
            "area_diameter_ratio",
            "position_norm_3d",
            "position_angle_3d_xz",
            "lab_chroma",
            "lab_hue",
            "texture_contrast",
            "texture_uniformity",
            "color_difference_AB",
            "color_difference_total",
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "attribution",
        ]

        id_col = "isic_id"
        norm_cols = (
            [f"{col}_patient_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_loc_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_loc_sim_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_site_general_norm" for col in num_cols + new_num_cols + additional_num_cols]
        )
        special_cols = ["count_per_patient"]

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            #  add from v1
            .with_columns(
                asymmetry_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_perimeterMM") + err),
                asymmetry_area_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_areaMM2") + err),
                color_variation_intensity=pl.col("tbp_lv_norm_color") * pl.col("tbp_lv_deltaLBnorm"),
                color_contrast_ratio=pl.col("tbp_lv_deltaLBnorm") / (pl.col("tbp_lv_L") + err),
                shape_irregularity=pl.col("tbp_lv_perimeterMM")
                / (2 * np.sqrt(np.pi * pl.col("tbp_lv_areaMM2") + err)),
                border_density=pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_perimeterMM") + err),
                size_age_ratio=pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + err),
                area_diameter_ratio=pl.col("tbp_lv_areaMM2") / (pl.col("clin_size_long_diam_mm") ** 2 + err),
                position_norm_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                position_angle_3d_xz=pl.arctan2(pl.col("tbp_lv_z"), pl.col("tbp_lv_x")),
                lab_chroma=(pl.col("tbp_lv_A") ** 2 + pl.col("tbp_lv_B") ** 2).sqrt(),
                lab_hue=pl.arctan2(pl.col("tbp_lv_B"), pl.col("tbp_lv_A")),
                texture_contrast=(pl.col("tbp_lv_stdL") / (pl.col("tbp_lv_L") + 1e-5)),
                texture_uniformity=(1 / (1 + pl.col("tbp_lv_color_std_mean"))),
                color_difference_AB=(pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2).sqrt(),
                color_difference_total=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over("patient_id"))
                    / (pl.col(col).std().over("patient_id") + err)
                ).alias(f"{col}_patient_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "tbp_lv_location"]))
                    / (pl.col(col).std().over(["patient_id", "tbp_lv_location"]) + err)
                ).alias(f"{col}_patient_loc_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "tbp_lv_location_simple"]))
                    / (pl.col(col).std().over(["patient_id", "tbp_lv_location_simple"]) + err)
                ).alias(f"{col}_patient_loc_sim_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "anatom_site_general"]))
                    / (pl.col(col).std().over(["patient_id", "anatom_site_general"]) + err)
                ).alias(f"{col}_patient_site_general_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
            .to_pandas()
        )
        # .set_index(
        #     id_col
        # )

        # for col in df.select_dtypes(include=[np.float64]).columns:
        #     median_value = df[col].median()
        #     df[col] = df[col].fillna(median_value)

        df, ud_num_cols = ugly_duckling_processing(df.copy(), num_cols + new_num_cols + additional_num_cols)

        feature_cols = (
            num_cols + new_num_cols + additional_num_cols + cat_cols + norm_cols + special_cols + ud_num_cols
        )

    elif version == 8:
        # v7に特徴量エンジニアリングを追加、attributionをcat_colsから削除
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
            # add from v1
            "asymmetry_ratio",
            "asymmetry_area_ratio",
            "color_variation_intensity",
            "color_contrast_ratio",
            "shape_irregularity",
            "border_density",
            "size_age_ratio",
            "area_diameter_ratio",
            "position_norm_3d",
            "position_angle_3d_xz",
            "lab_chroma",
            "lab_hue",
            "texture_contrast",
            "texture_uniformity",
            "color_difference_AB",
            "color_difference_total",
            # add from v3~7
            "position_angle_3d_yz",
            "color_change_ratio",
            "lesion_complexity_index",
            "age_adjusted_lesion_size",
            "color_uniformity_index",
            "border_sharpness_index",
            "relative_depth_3d",
            "lesion_aspect_ratio",
            "color_contrast_variation",
            "symmetry_color_index",
            "age_adjusted_complexity",
            "relative_lesion_size",
            "color_gradient_intensity",
            "lesion_density_index",
            "color_diversity_index",
            "border_irregularity_size_ratio",
            "estimated_growth_rate",
            "color_asymmetry_ratio",
            "sphericity_3d",
            "relative_position_x",
            "relative_position_y",
            "malignancy_composite_index",
            "lesion_complexity_index_patient_variation",
            "color_diversity_index_patient_variation",
            "border_irregularity_size_ratio_patient_variation",
            "tbp_lv_areaMM2_anatomical_zscore",
            "tbp_lv_norm_border_anatomical_zscore",
            "tbp_lv_norm_color_anatomical_zscore",
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            # "attribution",
        ]

        id_col = "isic_id"
        norm_cols = (
            [f"{col}_patient_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_loc_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_loc_sim_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_site_general_norm" for col in num_cols + new_num_cols + additional_num_cols]
        )
        special_cols = ["count_per_patient"]

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            #  add from v1
            .with_columns(
                asymmetry_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_perimeterMM") + err),
                asymmetry_area_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_areaMM2") + err),
                color_variation_intensity=pl.col("tbp_lv_norm_color") * pl.col("tbp_lv_deltaLBnorm"),
                color_contrast_ratio=pl.col("tbp_lv_deltaLBnorm") / (pl.col("tbp_lv_L") + err),
                shape_irregularity=pl.col("tbp_lv_perimeterMM")
                / (2 * np.sqrt(np.pi * pl.col("tbp_lv_areaMM2") + err)),
                border_density=pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_perimeterMM") + err),
                size_age_ratio=pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + err),
                area_diameter_ratio=pl.col("tbp_lv_areaMM2") / (pl.col("clin_size_long_diam_mm") ** 2 + err),
                position_norm_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                position_angle_3d_xz=pl.arctan2(pl.col("tbp_lv_z"), pl.col("tbp_lv_x")),
                lab_chroma=(pl.col("tbp_lv_A") ** 2 + pl.col("tbp_lv_B") ** 2).sqrt(),
                lab_hue=pl.arctan2(pl.col("tbp_lv_B"), pl.col("tbp_lv_A")),
                texture_contrast=(pl.col("tbp_lv_stdL") / (pl.col("tbp_lv_L") + 1e-5)),
                texture_uniformity=(1 / (1 + pl.col("tbp_lv_color_std_mean"))),
                color_difference_AB=(pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2).sqrt(),
                color_difference_total=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
            )
            #  add from v3~7
            .with_columns(
                position_angle_3d_yz=pl.arctan2(pl.col("tbp_lv_z"), pl.col("tbp_lv_y")),
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
        )
        # .set_index(
        #     id_col
        # )

        # by claude
        df = (
            df.with_columns(
                [
                    # 1. 複合的な色彩変化指標
                    (
                        (
                            pl.col("tbp_lv_deltaA") ** 2
                            + pl.col("tbp_lv_deltaB") ** 2
                            + pl.col("tbp_lv_deltaL") ** 2
                        )
                        / (pl.col("tbp_lv_A") ** 2 + pl.col("tbp_lv_B") ** 2 + pl.col("tbp_lv_L") ** 2)
                    ).alias("color_change_ratio"),
                    # 2. 病変の複雑さ指標
                    (
                        pl.col("tbp_lv_norm_border")
                        * pl.col("tbp_lv_norm_color")
                        * pl.col("tbp_lv_eccentricity")
                        * pl.col("tbp_lv_symm_2axis")
                    ).alias("lesion_complexity_index"),
                    # 3. 年齢調整済み病変サイズ
                    (pl.col("tbp_lv_areaMM2") / (pl.col("age_approx") + 1)).alias("age_adjusted_lesion_size"),
                    # 4. 色彩の均一性指標
                    (
                        1
                        - (
                            pl.col("tbp_lv_color_std_mean")
                            / (pl.col("tbp_lv_L") + pl.col("tbp_lv_A") + pl.col("tbp_lv_B"))
                        )
                    ).alias("color_uniformity_index"),
                    # 5. 境界の鋭さ指標
                    (pl.col("tbp_lv_norm_border") / pl.col("tbp_lv_perimeterMM")).alias(
                        "border_sharpness_index"
                    ),
                    # 6. 3D位置の相対的な深さ
                    (pl.col("tbp_lv_z") / (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2).sqrt()).alias(
                        "relative_depth_3d"
                    ),
                    # 7. 病変の aspect ratio
                    (pl.col("tbp_lv_perimeterMM") / pl.col("clin_size_long_diam_mm")).alias(
                        "lesion_aspect_ratio"
                    ),
                    # 8. 色彩コントラストの変動係数
                    (pl.col("tbp_lv_deltaLBnorm") / pl.col("tbp_lv_L")).alias("color_contrast_variation"),
                    # 9. 対称性と色彩の複合指標
                    (pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_color")).alias("symmetry_color_index"),
                ]
            )
            .with_columns(
                [
                    # 10. 年齢調整済み病変の複雑さ
                    (pl.col("lesion_complexity_index") / pl.col("age_approx")).alias(
                        "age_adjusted_complexity"
                    ),
                    # 11. 病変の相対的な大きさ（体の部位ごと）
                    (
                        pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_areaMM2").mean().over("anatom_site_general")
                    ).alias("relative_lesion_size"),
                    # 12. 色彩の gradient 強度
                    (
                        (
                            pl.col("tbp_lv_deltaA") ** 2
                            + pl.col("tbp_lv_deltaB") ** 2
                            + pl.col("tbp_lv_deltaL") ** 2
                        ).sqrt()
                        / pl.col("clin_size_long_diam_mm")
                    ).alias("color_gradient_intensity"),
                    # 13. 病変の密度指標
                    (
                        pl.col("tbp_lv_areaMM2")
                        / (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2)
                    ).alias("lesion_density_index"),
                    # 14. 色彩の多様性指標
                    (pl.col("tbp_lv_color_std_mean") * pl.col("tbp_lv_radial_color_std_max")).alias(
                        "color_diversity_index"
                    ),
                    # 15. 境界の不規則性と大きさの比
                    (pl.col("tbp_lv_norm_border") / pl.col("tbp_lv_areaMM2")).alias(
                        "border_irregularity_size_ratio"
                    ),
                    # 16. 病変の成長速度推定（年齢で正規化）
                    (pl.col("tbp_lv_areaMM2") / (pl.col("age_approx") + 1) ** 2).alias(
                        "estimated_growth_rate"
                    ),
                ]
            )
            .with_columns(
                [
                    # 17. 色彩の非対称性指標
                    (pl.col("tbp_lv_radial_color_std_max") / pl.col("tbp_lv_color_std_mean")).alias(
                        "color_asymmetry_ratio"
                    ),
                    # 18. 3D形状の球形度
                    (
                        (pl.col("tbp_lv_perimeterMM") ** 3 / (6 * np.pi * pl.col("tbp_lv_areaMM2") ** 2))
                        ** (1 / 3)
                    ).alias("sphericity_3d"),
                    # 19. 病変の相対的な位置（体の部位ごと）
                    (pl.col("tbp_lv_x") / pl.col("tbp_lv_x").max().over("anatom_site_general")).alias(
                        "relative_position_x"
                    ),
                    (pl.col("tbp_lv_y") / pl.col("tbp_lv_y").max().over("anatom_site_general")).alias(
                        "relative_position_y"
                    ),
                    # 20. 複合的な悪性度指標
                    (
                        pl.col("lesion_complexity_index")
                        * pl.col("color_diversity_index")
                        * pl.col("border_irregularity_size_ratio")
                        * pl.col("estimated_growth_rate")
                    ).alias("malignancy_composite_index"),
                ]
            )
        )

        # 21. 患者ごとの特徴量の変動
        for col in ["lesion_complexity_index", "color_diversity_index", "border_irregularity_size_ratio"]:
            df = df.with_columns(
                [
                    (
                        pl.col(col).std().over("patient_id") / (pl.col(col).mean().over("patient_id") + 1e-5)
                    ).alias(f"{col}_patient_variation")
                ]
            )

        # 23. 病変の特徴の z-score（体の部位ごと）
        for col in ["tbp_lv_areaMM2", "tbp_lv_norm_border", "tbp_lv_norm_color"]:
            df = df.with_columns(
                [
                    (
                        (pl.col(col) - pl.col(col).mean().over("anatom_site_general"))
                        / (pl.col(col).std().over("anatom_site_general") + 1e-5)
                    ).alias(f"{col}_anatomical_zscore")
                ]
            )

        df = (
            df.with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over("patient_id"))
                    / (pl.col(col).std().over("patient_id") + err)
                ).alias(f"{col}_patient_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "tbp_lv_location"]))
                    / (pl.col(col).std().over(["patient_id", "tbp_lv_location"]) + err)
                ).alias(f"{col}_patient_loc_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "tbp_lv_location_simple"]))
                    / (pl.col(col).std().over(["patient_id", "tbp_lv_location_simple"]) + err)
                ).alias(f"{col}_patient_loc_sim_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "anatom_site_general"]))
                    / (pl.col(col).std().over(["patient_id", "anatom_site_general"]) + err)
                ).alias(f"{col}_patient_site_general_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
        )

        df = df.to_pandas()

        df, ud_num_cols = ugly_duckling_processing(df.copy(), num_cols + new_num_cols + additional_num_cols)

        feature_cols = (
            num_cols + new_num_cols + additional_num_cols + cat_cols + norm_cols + special_cols + ud_num_cols
        )


    elif version == 9:
        # v7にattribution_normを追加
        num_cols = [
            "age_approx",  # Approximate age of patient at time of imaging.
            "clin_size_long_diam_mm",  # Maximum diameter of the lesion (mm).+
            "tbp_lv_A",  # A inside  lesion.+
            "tbp_lv_Aext",  # A outside lesion.+
            "tbp_lv_B",  # B inside  lesion.+
            "tbp_lv_Bext",  # B outside lesion.+
            "tbp_lv_C",  # Chroma inside  lesion.+
            "tbp_lv_Cext",  # Chroma outside lesion.+
            "tbp_lv_H",  # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
            "tbp_lv_Hext",  # Hue outside lesion.+
            "tbp_lv_L",  # L inside lesion.+
            "tbp_lv_Lext",  # L outside lesion.+
            "tbp_lv_areaMM2",  # Area of lesion (mm^2).+
            "tbp_lv_area_perim_ratio",  # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
            "tbp_lv_color_std_mean",  # Color irregularity, calculated as the variance of colors within the lesion's boundary.
            "tbp_lv_deltaA",  # Average A contrast (inside vs. outside lesion).+
            "tbp_lv_deltaB",  # Average B contrast (inside vs. outside lesion).+
            "tbp_lv_deltaL",  # Average L contrast (inside vs. outside lesion).+
            "tbp_lv_deltaLB",  #
            "tbp_lv_deltaLBnorm",  # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
            "tbp_lv_eccentricity",  # Eccentricity.+
            "tbp_lv_minorAxisMM",  # Smallest lesion diameter (mm).+
            "tbp_lv_nevi_confidence",  # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
            "tbp_lv_norm_border",  # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
            "tbp_lv_norm_color",  # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
            "tbp_lv_perimeterMM",  # Perimeter of lesion (mm).+
            "tbp_lv_radial_color_std_max",  # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
            "tbp_lv_stdL",  # Standard deviation of L inside  lesion.+
            "tbp_lv_stdLExt",  # Standard deviation of L outside lesion.+
            "tbp_lv_symm_2axis",  # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
            "tbp_lv_symm_2axis_angle",  # Lesion border asymmetry angle.+
            "tbp_lv_x",  # X-coordinate of the lesion on 3D TBP.+
            "tbp_lv_y",  # Y-coordinate of the lesion on 3D TBP.+
            "tbp_lv_z",  # Z-coordinate of the lesion on 3D TBP.+
        ]

        new_num_cols = [
            "lesion_size_ratio",  # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
            "lesion_shape_index",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
            "hue_contrast",  # tbp_lv_H                - tbp_lv_Hext              abs
            "luminance_contrast",  # tbp_lv_L                - tbp_lv_Lext              abs
            "lesion_color_difference",  # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
            "border_complexity",  # tbp_lv_norm_border      + tbp_lv_symm_2axis
            "color_uniformity",  # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max
            "position_distance_3d",  # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
            "perimeter_to_area_ratio",  # tbp_lv_perimeterMM      / tbp_lv_areaMM2
            "area_to_perimeter_ratio",  # tbp_lv_areaMM2          / tbp_lv_perimeterMM
            "lesion_visibility_score",  # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
            "symmetry_border_consistency",  # tbp_lv_symm_2axis       * tbp_lv_norm_border
            "consistency_symmetry_border",  # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)
            "color_consistency",  # tbp_lv_stdL             / tbp_lv_Lext
            "consistency_color",  # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
            "size_age_interaction",  # clin_size_long_diam_mm  * age_approx
            "hue_color_std_interaction",  # tbp_lv_H                * tbp_lv_color_std_mean
            "lesion_severity_index",  # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
            "shape_complexity_index",  # border_complexity       + lesion_shape_index
            "color_contrast_index",  # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm
            "log_lesion_area",  # tbp_lv_areaMM2          + 1  np.log
            "normalized_lesion_size",  # clin_size_long_diam_mm  / age_approx
            "mean_hue_difference",  # tbp_lv_H                + tbp_lv_Hext    / 2
            "std_dev_contrast",  # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
            "color_shape_composite_index",  # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
            "lesion_orientation_3d",  # tbp_lv_y                , tbp_lv_x  np.arctan2
            "overall_color_difference",  # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3
            "symmetry_perimeter_interaction",  # tbp_lv_symm_2axis       * tbp_lv_perimeterMM
            "comprehensive_lesion_index",  # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
            "color_variance_ratio",  # tbp_lv_color_std_mean   / tbp_lv_stdLExt
            "border_color_interaction",  # tbp_lv_norm_border      * tbp_lv_norm_color
            "border_color_interaction_2",
            "size_color_contrast_ratio",  # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
            "age_normalized_nevi_confidence",  # tbp_lv_nevi_confidence  / age_approx
            "age_normalized_nevi_confidence_2",
            "color_asymmetry_index",  # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max
            "volume_approximation_3d",  # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
            "color_range",  # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
            "shape_color_consistency",  # tbp_lv_eccentricity     * tbp_lv_color_std_mean
            "border_length_ratio",  # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
            "age_size_symmetry_index",  # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
            "index_age_size_symmetry",  # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
            # add from v1
            "asymmetry_ratio",
            "asymmetry_area_ratio",
            "color_variation_intensity",
            "color_contrast_ratio",
            "shape_irregularity",
            "border_density",
            "size_age_ratio",
            "area_diameter_ratio",
            "position_norm_3d",
            "position_angle_3d_xz",
            "lab_chroma",
            "lab_hue",
            "texture_contrast",
            "texture_uniformity",
            "color_difference_AB",
            "color_difference_total",
        ]

        cat_cols = [
            "sex",
            "anatom_site_general",
            "tbp_tile_type",
            "tbp_lv_location",
            "tbp_lv_location_simple",
            "attribution",
        ]

        id_col = "isic_id"
        norm_cols = (
            [f"{col}_patient_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_loc_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_loc_sim_norm" for col in num_cols + new_num_cols + additional_num_cols]
            + [f"{col}_patient_site_general_norm" for col in num_cols + new_num_cols + additional_num_cols]
            +[f"{col}_attribution_norm" for col in num_cols + new_num_cols + additional_num_cols]
        )
        special_cols = ["count_per_patient"]

        err = 1e-5

        df = (
            pl.from_dataframe(df)
            .with_columns(
                pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
            )
            .with_columns(
                pl.col(pl.Float64).fill_nan(
                    pl.col(pl.Float64).median()
                ),  # You may want to impute test data with train
            )
            .with_columns(
                lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
                lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
                hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
                luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
                lesion_color_difference=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
                border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
                color_uniformity=pl.col("tbp_lv_color_std_mean")
                / (pl.col("tbp_lv_radial_color_std_max") + err),
            )
            .with_columns(
                position_distance_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
                area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
                lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
                combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
                symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
                consistency_symmetry_border=pl.col("tbp_lv_symm_2axis")
                * pl.col("tbp_lv_norm_border")
                / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
            )
            .with_columns(
                color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
                consistency_color=pl.col("tbp_lv_stdL")
                * pl.col("tbp_lv_Lext")
                / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
                size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
                hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
                lesion_severity_index=(
                    pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")
                )
                / 3,
                shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
                color_contrast_index=pl.col("tbp_lv_deltaA")
                + pl.col("tbp_lv_deltaB")
                + pl.col("tbp_lv_deltaL")
                + pl.col("tbp_lv_deltaLBnorm"),
            )
            .with_columns(
                log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
                normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
                mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
                std_dev_contrast=(
                    (
                        pl.col("tbp_lv_deltaA") ** 2
                        + pl.col("tbp_lv_deltaB") ** 2
                        + pl.col("tbp_lv_deltaL") ** 2
                    )
                    / 3
                ).sqrt(),
                color_shape_composite_index=(
                    pl.col("tbp_lv_color_std_mean")
                    + pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 3,
                lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
                overall_color_difference=(
                    pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")
                )
                / 3,
            )
            .with_columns(
                symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
                comprehensive_lesion_index=(
                    pl.col("tbp_lv_area_perim_ratio")
                    + pl.col("tbp_lv_eccentricity")
                    + pl.col("tbp_lv_norm_color")
                    + pl.col("tbp_lv_symm_2axis")
                )
                / 4,
                color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
                border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
                border_color_interaction_2=pl.col("tbp_lv_norm_border")
                * pl.col("tbp_lv_norm_color")
                / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
                size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
                age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
                age_normalized_nevi_confidence_2=(
                    pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2
                ).sqrt(),
                color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
            )
            .with_columns(
                volume_approximation_3d=pl.col("tbp_lv_areaMM2")
                * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
                color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs()
                + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs()
                + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
                shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
                border_length_ratio=pl.col("tbp_lv_perimeterMM")
                / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
                age_size_symmetry_index=pl.col("age_approx")
                * pl.col("clin_size_long_diam_mm")
                * pl.col("tbp_lv_symm_2axis"),
                index_age_size_symmetry=pl.col("age_approx")
                * pl.col("tbp_lv_areaMM2")
                * pl.col("tbp_lv_symm_2axis"),
            )
            #  add from v1
            .with_columns(
                asymmetry_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_perimeterMM") + err),
                asymmetry_area_ratio=pl.col("tbp_lv_symm_2axis") / (pl.col("tbp_lv_areaMM2") + err),
                color_variation_intensity=pl.col("tbp_lv_norm_color") * pl.col("tbp_lv_deltaLBnorm"),
                color_contrast_ratio=pl.col("tbp_lv_deltaLBnorm") / (pl.col("tbp_lv_L") + err),
                shape_irregularity=pl.col("tbp_lv_perimeterMM")
                / (2 * np.sqrt(np.pi * pl.col("tbp_lv_areaMM2") + err)),
                border_density=pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_perimeterMM") + err),
                size_age_ratio=pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + err),
                area_diameter_ratio=pl.col("tbp_lv_areaMM2") / (pl.col("clin_size_long_diam_mm") ** 2 + err),
                position_norm_3d=(
                    pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2
                ).sqrt(),
                position_angle_3d_xz=pl.arctan2(pl.col("tbp_lv_z"), pl.col("tbp_lv_x")),
                lab_chroma=(pl.col("tbp_lv_A") ** 2 + pl.col("tbp_lv_B") ** 2).sqrt(),
                lab_hue=pl.arctan2(pl.col("tbp_lv_B"), pl.col("tbp_lv_A")),
                texture_contrast=(pl.col("tbp_lv_stdL") / (pl.col("tbp_lv_L") + 1e-5)),
                texture_uniformity=(1 / (1 + pl.col("tbp_lv_color_std_mean"))),
                color_difference_AB=(pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2).sqrt(),
                color_difference_total=(
                    pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2
                ).sqrt(),
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over("patient_id"))
                    / (pl.col(col).std().over("patient_id") + err)
                ).alias(f"{col}_patient_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "tbp_lv_location"]))
                    / (pl.col(col).std().over(["patient_id", "tbp_lv_location"]) + err)
                ).alias(f"{col}_patient_loc_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "tbp_lv_location_simple"]))
                    / (pl.col(col).std().over(["patient_id", "tbp_lv_location_simple"]) + err)
                ).alias(f"{col}_patient_loc_sim_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over(["patient_id", "anatom_site_general"]))
                    / (pl.col(col).std().over(["patient_id", "anatom_site_general"]) + err)
                ).alias(f"{col}_patient_site_general_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                (
                    (pl.col(col) - pl.col(col).mean().over("attribution"))
                    / (pl.col(col).std().over("attribution") + err)
                ).alias(f"{col}_attribution_norm")
                for col in (num_cols + new_num_cols + additional_num_cols)
            )
            .with_columns(
                count_per_patient=pl.col("isic_id").count().over("patient_id"),
            )
            .with_columns(
                pl.col(cat_cols).cast(pl.Categorical),
            )
            .to_pandas()
        )
        # .set_index(
        #     id_col
        # )

        # for col in df.select_dtypes(include=[np.float64]).columns:
        #     median_value = df[col].median()
        #     df[col] = df[col].fillna(median_value)

        df, ud_num_cols = ugly_duckling_processing(df.copy(), num_cols + new_num_cols + additional_num_cols)

        feature_cols = (
            num_cols + new_num_cols + additional_num_cols + cat_cols + norm_cols + special_cols + ud_num_cols
        )

    return df, feature_cols, cat_cols


def feature_engineering_with_dnn(df, dnn_run_name_list, version=1):
    if version == 1:
        if len(dnn_run_name_list) > 1:
            df["predictions_min"] = df[dnn_run_name_list].min(1)
            df["predictions_max"] = df[dnn_run_name_list].max(1)
            df["predictions_mean"] = df[dnn_run_name_list].mean(1)
            df["predictions_std"] = df[dnn_run_name_list].std(1)
            new_cols = ["predictions_min", "predictions_max", "predictions_mean", "predictions_std"]
        else:
            new_cols = []
    elif version == 2:
        if len(dnn_run_name_list) > 1:
            df["predictions_min"] = df[dnn_run_name_list].min(1)
            df["predictions_max"] = df[dnn_run_name_list].max(1)
            df["predictions_mean"] = df[dnn_run_name_list].mean(1)
            df["predictions_std"] = df[dnn_run_name_list].std(1)
            new_cols = dnn_run_name_list + [
                "predictions_min",
                "predictions_max",
                "predictions_mean",
                "predictions_std",
            ]
        else:
            new_cols = dnn_run_name_list
    else:
        new_cols = []

    return df, new_cols


# https://www.kaggle.com/code/richolson/isic-2024-lgbm-imagenet-public#Initial-new-features
def ugly_duckling_processing(df, num_cols, include_patient_wide_ud=False):
    ud_columns = num_cols.copy()
    new_num_cols = []

    # if false - only do location-based ugly ducklings
    # include_patient_wide_ud = False

    # counter = 0

    def calc_ugly_duckling_scores(group, grouping):
        # nonlocal counter
        # counter += 1
        # if counter % 10 == 0:
        #     print(".", end="", flush=True)
        z_scores = group[ud_columns].apply(lambda x: zscore(x, nan_policy="omit"))
        ud_scores = np.abs(z_scores)
        prefix = "ud_" if grouping == "patient" else "ud_loc_"
        ud_scores.columns = [f"{prefix}{col}" for col in ud_columns]
        return ud_scores

    print("Analyzing ducklings", end="", flush=True)
    ud_location_col = "tbp_lv_location"
    ud_scores_loc = (
        df.groupby(["patient_id", ud_location_col])[ud_columns + ["patient_id", ud_location_col]]
        .parallel_apply(lambda x: calc_ugly_duckling_scores(x, "location"))
        .reset_index(level=[0, 1], drop=True)
    )

    print("\nConcat ducklings")
    df = pd.concat([df, ud_scores_loc], axis=1)

    if include_patient_wide_ud:
        print("Analyzing ducklings (part 2)", end="", flush=True)
        ud_scores_patient = (
            df.groupby("patient_id")[ud_columns + ["patient_id"]]
            .parallel_apply(lambda x: calc_ugly_duckling_scores(x, "patient"))
            .reset_index(level=0, drop=True)
        )
        df = pd.concat([df, ud_scores_patient], axis=1)
        print()  # New line after progress indicator

    print("Extending ducklings")
    new_num_cols.extend([f"ud_loc_{col}" for col in ud_columns])
    if include_patient_wide_ud:
        new_num_cols.extend([f"ud_{col}" for col in ud_columns])

    print("Enhancing ugly duckling features", end="", flush=True)

    # 1. Percentile-based ugly duckling scores
    def calc_percentile_ud_scores(group):
        nonlocal counter
        counter += 1
        if counter % 10 == 0:
            print(".", end="", flush=True)
        percentiles = group[ud_columns].rank(pct=True)
        return percentiles.add_prefix("ud_percentile_")

    counter = 0  # Reset counter for percentile calculation
    ud_percentiles = (
        df.groupby("patient_id")[ud_columns].apply(calc_percentile_ud_scores).reset_index(level=0, drop=True)
    )
    df = pd.concat([df, ud_percentiles], axis=1)
    new_num_cols.extend([f"ud_percentile_{col}" for col in ud_columns])
    print()  # New line after progress indicator

    # 2. Ugly duckling count features
    threshold = 2.0  # You can adjust this threshold
    if include_patient_wide_ud:
        ud_count = (df[[f"ud_{col}" for col in ud_columns]].abs() > threshold).sum(axis=1)
        df["ud_count_patient"] = ud_count
        new_num_cols.append("ud_count_patient")

    ud_count_loc = (df[[f"ud_loc_{col}" for col in ud_columns]].abs() > threshold).sum(axis=1)
    df["ud_count_location"] = ud_count_loc
    new_num_cols.append("ud_count_location")

    # 3. Ugly duckling severity features
    if include_patient_wide_ud:
        df["ud_max_severity_patient"] = df[[f"ud_{col}" for col in ud_columns]].abs().max(axis=1)
        new_num_cols.append("ud_max_severity_patient")
    df["ud_max_severity_location"] = df[[f"ud_loc_{col}" for col in ud_columns]].abs().max(axis=1)
    new_num_cols.append("ud_max_severity_location")

    # 4. Ugly duckling consistency features
    if include_patient_wide_ud:
        df["ud_consistency_patient"] = df[[f"ud_{col}" for col in ud_columns]].abs().std(axis=1)
        new_num_cols.append("ud_consistency_patient")
    df["ud_consistency_location"] = df[[f"ud_loc_{col}" for col in ud_columns]].abs().std(axis=1)
    new_num_cols.append("ud_consistency_location")

    return df, new_num_cols
