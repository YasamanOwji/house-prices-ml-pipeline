"""
feature_engineering.py
=======================
Adds engineered features on top of the cleaned, ordinal-mapped dataframe.

All features here are computed row-wise from columns already present, so
this transformer is stateless (fit is a no-op) and identical at train and
inference time — no leakage risk.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        # Stateless transformer, but see OrdinalMapper.fit for why we still
        # set a trailing-underscore attribute here.
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # --- Size aggregates -------------------------------------------------
        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        X["TotalBathrooms"] = (
            X["FullBath"] + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
        )
        X["TotalPorchSF"] = (
            X["OpenPorchSF"] + X["3SsnPorch"] + X["EnclosedPorch"]
            + X["ScreenPorch"] + X["WoodDeckSF"]
        )

        # --- Ages --------------------------------------------------------------
        X["HouseAge"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
        X["RemodAge"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
        X["GarageAge"] = (X["YrSold"] - X["GarageYrBlt"]).clip(lower=0)
        X["IsRemodeled"] = (X["YearBuilt"] != X["YearRemodAdd"]).astype(int)
        X["IsNewHouse"] = (X["YrSold"] == X["YearBuilt"]).astype(int)

        # --- Presence flags ------------------------------------------------------
        X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
        X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
        X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
        X["Has2ndFloor"] = (X["2ndFlrSF"] > 0).astype(int)
        X["HasWoodDeck"] = (X["WoodDeckSF"] > 0).astype(int)
        X["HasOpenPorch"] = (X["OpenPorchSF"] > 0).astype(int)
        X["HasEnclosedPorch"] = (X["EnclosedPorch"] > 0).astype(int)

        # --- Interactions --------------------------------------------------------
        X["OverallGrade"] = X["OverallQual"] * X["OverallCond"]
        X["GarageGrade"] = X["GarageQual"] * X["GarageCond"]
        X["ExterGrade"] = X["ExterQual"] * X["ExterCond"]
        X["TotalQualScore"] = (
            X["ExterQual"] + X["BsmtQual"] + X["KitchenQual"] + X["GarageQual"]
        )

        return X
