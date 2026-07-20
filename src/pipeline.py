"""
pipeline.py
===========
Assembles the full, leak-free scikit-learn Pipeline: missing-value
imputation -> ordinal encoding -> feature engineering -> column-wise
encoding/scaling -> model.

Because every step is a proper sklearn transformer, the *entire* thing can
be handed to `GridSearchCV`/`cross_val_score` (each fold fits its own
imputation statistics independently) and the *entire* thing can be
`joblib.dump`-ed after training and reloaded later to score brand-new raw
data with a single `.predict(raw_df)` call — no separate "remember to
apply the same fills to the test set" bookkeeping required.
"""

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from . import config
from .feature_engineering import FeatureEngineer
from .preprocessing import MissingValueImputer, OrdinalMapper, SkewCorrector


def build_preprocessing_pipeline(scale_numeric: bool = False) -> Pipeline:
    """The cleaning/encoding steps shared by every model.

    Parameters
    ----------
    scale_numeric:
        Tree models (XGBoost, Random Forest, Gradient Boosting) don't need
        scaled inputs. Linear models (Ridge/Lasso/ElasticNet) do. Set to
        True when building a pipeline around a linear baseline.
    """
    numeric_steps = [("skew_fix", SkewCorrector()), ("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore"),
                config.NOMINAL_COLUMNS,
            ),
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                make_column_selector(dtype_include="number"),
            ),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[
        ("imputer", MissingValueImputer()),
        ("ordinal_mapper", OrdinalMapper()),
        ("feature_engineer", FeatureEngineer()),
        ("column_transformer", column_transformer),
    ])


def build_xgb_pipeline(**xgb_params) -> Pipeline:
    """Full pipeline: preprocessing + a tuned XGBoost regressor."""
    default_params = dict(
        random_state=config.RANDOM_STATE,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    default_params.update(xgb_params)

    preprocessing = build_preprocessing_pipeline(scale_numeric=False)
    return Pipeline(steps=[
        *preprocessing.steps,
        ("model", XGBRegressor(**default_params)),
    ])
