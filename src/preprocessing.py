"""
preprocessing.py
=================
Leak-free preprocessing for the Ames housing data.

Design note (the main upgrade over the original script)
---------------------------------------------------------
The original notebook concatenated train + test *before* computing fill
values (neighborhood-median LotFrontage, categorical modes, etc.), and
before turning categoricals into integer codes with `pd.Categorical(...).codes`.
That has two problems:

1. Statistics such as medians/modes end up computed over train+test
   combined, so the "training" pipeline has implicitly seen the test set.
   This isn't target leakage (SalePrice itself is never touched), but it
   does mean the pipeline can't be validated or reproduced the way a real
   production model would be, and cross-validation folds are no longer
   independent of each other.
2. `pd.Categorical(col).codes` assigns integer codes based on whatever
   categories happen to appear in the combined data, in alphabetical order.
   Those codes carry no ordinal meaning for a tree model, and a category
   appearing in a new fold shifts every other code around.

Here, every statistic (medians, modes) is *fit* on training data only and
stored on the transformer instance, then applied identically to validation
and test data via `.transform()`. This is what makes it safe to drop these
transformers into an sklearn `Pipeline` / `GridSearchCV` — each CV fold
fits its own statistics on its own training split automatically.
"""

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Fills missing values using domain knowledge about *why* each column
    is missing, fitting any data-dependent statistic on the training split
    only.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.lotfrontage_by_neighborhood_ = (
            X.groupby("Neighborhood")["LotFrontage"].median()
        )
        self.lotfrontage_global_median_ = X["LotFrontage"].median()
        self.mode_fills_ = {
            col: X[col].mode(dropna=True)[0]
            for col in config.MODE_FILL_CATEGORICAL
            if col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # MSSubClass is a categorical building-class *code*, not a magnitude.
        if "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype("Int64").astype(str)

        # "No such feature" categoricals.
        for col in config.NONE_FILL_CATEGORICAL:
            if col in X.columns:
                X[col] = X[col].fillna("None")

        # "No such feature" numerics -> 0.
        for col in config.ZERO_FILL_NUMERIC:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # GarageYrBlt: no garage -> same year as the house was built, so
        # a downstream "garage age" feature comes out as 0 rather than
        # a huge nonsense number. Also repairs the well-known GarageYrBlt
        # typo in the Ames test set (a value of 2207, one garage recorded
        # as built after the sale date).
        if "GarageYrBlt" in X.columns and "YearBuilt" in X.columns:
            X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
            if "YrSold" in X.columns:
                bad_year = X["GarageYrBlt"] > X["YrSold"]
                X.loc[bad_year, "GarageYrBlt"] = X.loc[bad_year, "YearBuilt"]

        # LotFrontage: median within the same neighborhood tends to be a
        # much better estimate than a global median, since lot sizes vary
        # a lot by area of town.
        if "LotFrontage" in X.columns:
            neighborhood_medians = X["Neighborhood"].map(self.lotfrontage_by_neighborhood_)
            X["LotFrontage"] = X["LotFrontage"].fillna(neighborhood_medians)
            X["LotFrontage"] = X["LotFrontage"].fillna(self.lotfrontage_global_median_)

        # Low-cardinality "probably just a data entry gap" categoricals.
        for col, fill_value in self.mode_fills_.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)

        if "Functional" in X.columns:
            # Per the data documentation: assume typical unless noted.
            X["Functional"] = X["Functional"].fillna("Typ")

        for col in config.DROP_COLUMNS:
            if col in X.columns:
                X = X.drop(columns=col)

        return X


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """Maps ordinal categorical columns to integers using fixed,
    domain-knowledge orderings (e.g. Po < Fa < TA < Gd < Ex).

    Stateless by design: the ordering is a property of the *data
    documentation*, not something that should vary by CV fold.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless (the mapping is fixed domain knowledge, not learned from
        # data) but we still record a trailing-underscore attribute so
        # sklearn's `check_is_fitted` — used e.g. when slicing a fitted
        # Pipeline — correctly recognizes this step as fitted.
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in config.ORDINAL_MAPS.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
                # Any category not in our mapping (shouldn't happen with
                # this dataset, but guards against surprises) falls back to
                # the middle of the scale rather than becoming NaN.
                if X[col].isna().any():
                    fallback = int(np.median(list(mapping.values())))
                    X[col] = X[col].fillna(fallback)
                X[col] = X[col].astype(int)
        return X


class SkewCorrector(BaseEstimator, TransformerMixin):
    """Applies a Box-Cox (1+x) transform to numeric columns whose skewness
    (measured on the training split only) exceeds a threshold.

    Tree models like XGBoost don't strictly need this, but it noticeably
    helps the linear baselines in the comparison notebook, and reduces the
    influence of extreme-value outliers on split points either way.
    """

    def __init__(self, skew_threshold: float = config.SKEW_THRESHOLD):
        self.skew_threshold = skew_threshold

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        skewed = X.apply(lambda s: s.astype(float).skew()).abs()
        self.skewed_columns_ = skewed[skewed > self.skew_threshold].index.tolist()
        self.lambdas_ = {}
        for col in self.skewed_columns_:
            values = X[col].astype(float)
            values = values.clip(lower=0)  # boxcox1p requires x > -1
            self.lambdas_[col] = boxcox_normmax(values + 1, method="mle")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for col in self.skewed_columns_:
            if col in X.columns:
                values = X[col].astype(float).clip(lower=0)
                X[col] = boxcox1p(values, self.lambdas_[col])
        return X


def remove_training_outliers(train_df: pd.DataFrame) -> pd.DataFrame:
    """Drops the two well-documented GrLivArea outliers (very large homes
    sold for anomalously low prices — recording errors per De Cock's
    original paper on the Ames dataset). Only ever applied to training
    data, and only using columns that require the (train-only) target.
    """
    mask = (
        (train_df["GrLivArea"] > config.GRLIVAREA_OUTLIER_THRESHOLD)
        & (train_df[config.TARGET] < config.GRLIVAREA_OUTLIER_SALEPRICE_THRESHOLD)
    )
    removed = int(mask.sum())
    if removed:
        print(f"Removing {removed} known outlier row(s) (large GrLivArea, low SalePrice).")
    return train_df.loc[~mask].reset_index(drop=True)
