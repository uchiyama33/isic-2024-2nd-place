import pandas as pd
import numpy as np
import rootutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.isic_utils.utils import SelectColumns, DNNFeatureEngineering, AddNoiseDnnPreds


class GBDTModels:
    def __init__(self, cfg, feature_cols_without_dnn, cat_cols):
        self.cfg_models = cfg.models
        self.cfg_stacking_models = cfg.get("stacking_models")
        self.over_sampling_ratio = cfg.over_sampling_ratio
        self.under_sampling_ratio = cfg.under_sampling_ratio
        self.rank_avg = cfg.rank_avg
        self.feature_cols_without_dnn = feature_cols_without_dnn
        self.cat_cols = cat_cols
        self.ensemble_weights = None
        self.num_models = len(self.cfg_models)
        self.version_dnn_fe = cfg.version_dnn_fe
        self.dnn_noise_std = cfg.dnn_noise_std

    def _preprocessing(self, X, y, cfg_model, seed, X_pl=None, y_pl=None):
        X, y = RandomOverSampler(sampling_strategy=self.over_sampling_ratio, random_state=seed).fit_resample(
            X, y
        )
        X, y = RandomUnderSampler(
            sampling_strategy=self.under_sampling_ratio, random_state=seed
        ).fit_resample(X, y)
        X = SelectColumns(
            self.feature_cols_without_dnn,
            cfg_model.use_dnn,
            cfg_model.get("not_use_col_specific_names", []),
            cfg_model.get("not_use_col_keywards", []),
            cfg_model.get("null_imp_setting", None),
        ).transform(X)
        X = AddNoiseDnnPreds(cfg_model.use_dnn, self.dnn_noise_std).transform(X)
        X = DNNFeatureEngineering(
            cfg_model.use_dnn, self.feature_cols_without_dnn, self.version_dnn_fe
        ).transform(X)
        if X_pl is not None and y_pl is not None:
            X_pl = SelectColumns(
                self.feature_cols_without_dnn,
                cfg_model.use_dnn,
                cfg_model.get("not_use_col_specific_names", []),
                cfg_model.get("not_use_col_keywards", []),
                cfg_model.get("null_imp_setting", None),
            ).transform(X_pl)
            X_pl = AddNoiseDnnPreds(cfg_model.use_dnn, self.dnn_noise_std).transform(X_pl)
            X_pl = DNNFeatureEngineering(
                cfg_model.use_dnn, self.feature_cols_without_dnn, self.version_dnn_fe
            ).transform(X_pl)
            X = pd.concat([X, X_pl])
            y = np.concatenate([y, y_pl])

        return X, y

    def _preprocessing_predict(self, X, cfg_model):
        X = SelectColumns(
            self.feature_cols_without_dnn,
            cfg_model.use_dnn,
            cfg_model.get("not_use_col_specific_names", []),
            cfg_model.get("not_use_col_keywards", []),
            cfg_model.get("null_imp_setting", None),
        ).transform(X)
        X = DNNFeatureEngineering(
            cfg_model.use_dnn, self.feature_cols_without_dnn, self.version_dnn_fe
        ).transform(X)
        return X

    def _train_lgb(self, cfg_model, X, y, seed, X_pl=None, y_pl=None, stacking=False):
        params = dict(cfg_model.params)
        params["random_state"] = seed

        if not stacking:
            X, y = self._preprocessing(X, y, cfg_model, seed, X_pl, y_pl)

        data_train = lgb.Dataset(X, y)
        model = lgb.train(params=params, train_set=data_train, num_boost_round=cfg_model.num_boost_round)

        return model

    def _train_cb(self, cfg_model, X, y, seed, X_pl=None, y_pl=None, stacking=False):
        params = dict(cfg_model.params)
        params["random_state"] = seed

        if y_pl is not None:
            y_pl = np.ones_like(y_pl).astype(y.dtype)

        if not stacking:
            X, y = self._preprocessing(X, y, cfg_model, seed, X_pl, y_pl)
            cat_cols = [item for item in self.cat_cols if item in X.columns]
            data_train = cb.Pool(X, label=y, cat_features=cat_cols)
        else:
            data_train = cb.Pool(X, label=y)

        model = cb.train(pool=data_train, params=params, num_boost_round=cfg_model.num_boost_round)

        return model

    def _train_xgb(self, cfg_model, X, y, seed, X_pl=None, y_pl=None, stacking=False):
        params = dict(cfg_model.params)
        params["random_state"] = seed

        if not stacking:
            X, y = self._preprocessing(X, y, cfg_model, seed, X_pl, y_pl)

        data_train = xgb.DMatrix(X, label=y, enable_categorical=True)
        model = xgb.train(dtrain=data_train, params=params, num_boost_round=cfg_model.num_boost_round)

        return model

    def _predict_lgb(self, models, cfg_model, X, stacking=False):
        if not stacking:
            X = self._preprocessing_predict(X, cfg_model)
        preds = [model.predict(X) for model in models]
        return np.mean(preds, axis=0)

    def _predict_cb(self, models, cfg_model, X, stacking=False):
        if not stacking:
            X = self._preprocessing_predict(X, cfg_model)
        preds = [model.predict(X, prediction_type="Probability")[:, 1] for model in models]
        return np.mean(preds, axis=0)

    def _predict_xgb(self, models, cfg_model, X, stacking=False):
        if not stacking:
            X = self._preprocessing_predict(X, cfg_model)
        X = xgb.DMatrix(X, enable_categorical=True)
        preds = [model.predict(X) for model in models]
        return np.mean(preds, axis=0)

    def fit(self, X, y, X_pl=None, y_pl=None, seed=None):
        self.models = []
        seed_count = 0
        if seed:
            seed_count += seed

        for cfg_model in tqdm(self.cfg_models, desc="training models"):
            seed_models = []
            for seed in range(cfg_model.n_seed_averaging):
                if cfg_model.type == "lgb":
                    model = self._train_lgb(cfg_model, X, y, seed + seed_count, X_pl, y_pl)

                elif cfg_model.type == "cb":
                    model = self._train_cb(cfg_model, X, y, seed + seed_count, X_pl, y_pl)

                elif cfg_model.type == "xgb":
                    model = self._train_xgb(cfg_model, X, y, seed + seed_count, X_pl, y_pl)

                seed_models.append(model)

            self.models.append(seed_models)
            seed_count += 1

    def fit_with_trained_other_model(self, X, y, trained_other_model, X_pl=None, y_pl=None, seed=None):
        self.models = []
        seed_count = 0
        if seed:
            seed_count += seed

        for cfg_model in tqdm(self.cfg_models, desc="training models"):
            seed_models = []
            for seed in range(cfg_model.n_seed_averaging):
                model = None
                for i, cfg_trained_other_model in enumerate(trained_other_model.cfg_models):
                    if cfg_trained_other_model == cfg_model:
                        model = trained_other_model.models[i][seed]
                        break

                if model is None:
                    if cfg_model.type == "lgb":
                        model = self._train_lgb(cfg_model, X, y, seed + seed_count, X_pl, y_pl)

                    elif cfg_model.type == "cb":
                        model = self._train_cb(cfg_model, X, y, seed + seed_count, X_pl, y_pl)

                    elif cfg_model.type == "xgb":
                        model = self._train_xgb(cfg_model, X, y, seed + seed_count, X_pl, y_pl)

                seed_models.append(model)

            self.models.append(seed_models)
            seed_count += 1

    def fit_stacking(self, X, y):
        self.stacking_models = []
        for cfg_model in tqdm(self.cfg_stacking_models, desc="training stacking models"):
            seed_models = []
            for seed in range(cfg_model.n_seed_averaging):
                if cfg_model.type == "lgb":
                    model = self._train_lgb(cfg_model, X, y, seed, stacking=True)

                elif cfg_model.type == "cb":
                    model = self._train_cb(cfg_model, X, y, seed, stacking=True)

                elif cfg_model.type == "xgb":
                    model = self._train_xgb(cfg_model, X, y, seed, stacking=True)

                seed_models.append(model)

            self.stacking_models.append(seed_models)

    def set_ensemble_weights(self, ensemble_weights):
        assert len(ensemble_weights) == self.num_models
        self.ensemble_weights = ensemble_weights / sum(ensemble_weights)

    def predict(self, X):
        preds_list = []
        for models, cfg_model in zip(self.models, self.cfg_models):
            if cfg_model.type == "lgb":
                preds = self._predict_lgb(models, cfg_model, X)

            elif cfg_model.type == "cb":
                preds = self._predict_cb(models, cfg_model, X)

            elif cfg_model.type == "xgb":
                preds = self._predict_xgb(models, cfg_model, X)

            preds_list.append(preds)

        if self.rank_avg:
            assert False, "0-1の範囲にする"
            df_preds = pd.DataFrame(np.array(preds_list).T)
            preds = df_preds.rank().mean(1)
        else:
            preds = np.average(np.stack(preds_list), axis=0, weights=self.ensemble_weights)

        return preds

    def predict_(self, X):
        preds_list = []
        for models, cfg_model in zip(self.models, self.cfg_models):
            if cfg_model.type == "lgb":
                preds = self._predict_lgb(models, cfg_model, X)

            elif cfg_model.type == "cb":
                preds = self._predict_cb(models, cfg_model, X)

            elif cfg_model.type == "xgb":
                preds = self._predict_xgb(models, cfg_model, X)

            preds_list.append(preds)

        return np.array(preds_list)

    def predict_stacking(self, X):
        preds_list = []
        for models, cfg_model in zip(self.stacking_models, self.cfg_stacking_models):
            if cfg_model.type == "lgb":
                preds = self._predict_lgb(models, cfg_model, X, stacking=True)

            elif cfg_model.type == "cb":
                preds = self._predict_cb(models, cfg_model, X, stacking=True)

            elif cfg_model.type == "xgb":
                preds = self._predict_xgb(models, cfg_model, X, stacking=True)

            preds_list.append(preds)

        preds = np.average(np.stack(preds_list), axis=0)

        return preds

    def get_feature_importance(self):
        feature_importances = []
        feature_names = []
        for models, cfg_model in zip(self.models, self.cfg_models):
            _feature_importances = []
            for model in models:
                if cfg_model.type == "lgb":
                    _feature_importances.append(
                        model.feature_importance(importance_type="gain")
                    )  # LightGBMの重要度を取得
                    _feature_names = model.feature_name()

                elif cfg_model.type == "cb":
                    _feature_importances.append(model.get_feature_importance())  # CatBoostの重要度を取得
                    _feature_names = model.feature_names_

                elif cfg_model.type == "xgb":
                    _feature_importances_dict = model.get_score(importance_type="gain")
                    for f in model.feature_names:
                        if f not in _feature_importances_dict:
                            _feature_importances_dict[f] = 0
                    _feature_importances_dict = sorted(_feature_importances_dict.items())
                    _feature_importances.append([y for x, y in _feature_importances_dict])
                    _feature_names = [x for x, y in _feature_importances_dict]

            avg_importance = np.mean(_feature_importances, axis=0)
            feature_importances.append(avg_importance)
            feature_names.append(_feature_names)

        return feature_importances, feature_names
