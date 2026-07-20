"""utils.py — small shared helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

from . import config


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


rmse_scorer = make_scorer(rmse, greater_is_better=False)


def cv_rmse(estimator, X, y, folds: int = config.CV_FOLDS) -> np.ndarray:
    """Cross-validated RMSE (in log-SalePrice space, since that's the
    space we train in) across `folds` shuffled folds."""
    kfold = KFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(estimator, X, y, scoring=rmse_scorer, cv=kfold, n_jobs=-1)
    return -scores


def get_output_feature_names(fitted_pipeline, X_sample: pd.DataFrame) -> list:
    """Recovers human-readable feature names for the columns coming out of
    the fitted pipeline's ColumnTransformer, for feature-importance
    reporting.

    Rebuilt manually (rather than relying purely on sklearn's
    `get_feature_names_out` chaining through custom transformers) so this
    stays robust across sklearn versions.
    """
    pre_column_transform = fitted_pipeline[:3]  # imputer, ordinal_mapper, feature_engineer
    transformed = pre_column_transform.transform(X_sample)

    column_transformer = fitted_pipeline.named_steps["column_transformer"]
    nominal_encoder = column_transformer.named_transformers_["nominal"]
    nominal_names = list(nominal_encoder.get_feature_names_out(config.NOMINAL_COLUMNS))

    numeric_names = list(transformed.select_dtypes(include="number").columns)

    return nominal_names + numeric_names
