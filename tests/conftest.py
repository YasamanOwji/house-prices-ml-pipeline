"""
conftest.py
===========
Generates a small, *synthetic* dataframe that matches the real Ames
Housing / Kaggle "House Prices" schema (column names + plausible value
ranges), purely so the pipeline's plumbing can be exercised in tests
without needing the actual competition data on disk.

The values themselves are random and carry no real-world meaning — they
exist only to prove the code runs correctly end-to-end, not to produce a
meaningful trained model.
"""

import numpy as np
import pandas as pd
import pytest

CATEGORY_VALUES = {
    "MSZoning": ["RL", "RM", "FV", "C (all)"],
    "Street": ["Pave", "Grvl"],
    "Alley": ["Grvl", "Pave", np.nan],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandContour": ["Lvl", "Bnk", "HLS", "Low"],
    "Utilities": ["AllPub"],
    "LotConfig": ["Inside", "Corner", "CulDSac"],
    "LandSlope": ["Gtl", "Mod", "Sev"],
    "Neighborhood": ["CollgCr", "Veenker", "NAmes", "Somerst"],
    "Condition1": ["Norm", "Feedr", "Artery"],
    "Condition2": ["Norm"],
    "BldgType": ["1Fam", "2fmCon", "TwnhsE"],
    "HouseStyle": ["2Story", "1Story", "SLvl"],
    "RoofStyle": ["Gable", "Hip"],
    "RoofMatl": ["CompShg"],
    "Exterior1st": ["VinylSd", "MetalSd", "Wd Sdng"],
    "Exterior2nd": ["VinylSd", "MetalSd", "Wd Sdng"],
    "MasVnrType": ["BrkFace", "None", np.nan],
    "ExterQual": ["Gd", "TA", "Ex"],
    "ExterCond": ["TA", "Gd", "Fa"],
    "Foundation": ["PConc", "CBlock", "BrkTil"],
    "BsmtQual": ["Gd", "TA", np.nan, "Ex"],
    "BsmtCond": ["TA", "Gd", np.nan],
    "BsmtExposure": ["No", "Gd", "Mn", np.nan],
    "BsmtFinType1": ["GLQ", "Unf", "ALQ", np.nan],
    "BsmtFinType2": ["Unf", "Rec", np.nan],
    "Heating": ["GasA"],
    "HeatingQC": ["Ex", "Gd", "TA"],
    "CentralAir": ["Y", "N"],
    "Electrical": ["SBrkr", "FuseA", np.nan],
    "KitchenQual": ["Gd", "TA", "Ex"],
    "Functional": ["Typ", "Min1", np.nan],
    "FireplaceQu": ["Gd", "TA", np.nan],
    "GarageType": ["Attchd", "Detchd", np.nan],
    "GarageFinish": ["RFn", "Unf", "Fin", np.nan],
    "GarageQual": ["TA", "Gd", np.nan],
    "GarageCond": ["TA", np.nan],
    "PavedDrive": ["Y", "N", "P"],
    "PoolQC": [np.nan, "Gd", "Ex"],
    "Fence": [np.nan, "MnPrv", "GdWo"],
    "MiscFeature": [np.nan, "Shed"],
    "SaleType": ["WD", "New", "COD"],
    "SaleCondition": ["Normal", "Abnorml", "Partial"],
}


def _make_rows(n: int, rng: np.random.Generator, with_target: bool) -> pd.DataFrame:
    data = {"Id": np.arange(1, n + 1)}
    data["MSSubClass"] = rng.choice([20, 60, 50, 120], size=n)
    for col, values in CATEGORY_VALUES.items():
        data[col] = rng.choice(values, size=n)

    data["LotFrontage"] = rng.choice([np.nan, 60, 70, 80, 21], size=n)
    data["LotArea"] = rng.integers(3000, 20000, size=n)
    data["OverallQual"] = rng.integers(1, 11, size=n)
    data["OverallCond"] = rng.integers(1, 11, size=n)
    data["YearBuilt"] = rng.integers(1900, 2010, size=n)
    data["YearRemodAdd"] = data["YearBuilt"] + rng.integers(0, 20, size=n)
    data["MasVnrArea"] = rng.choice([0, 100, 200, np.nan], size=n)
    data["BsmtFinSF1"] = rng.integers(0, 1500, size=n)
    data["BsmtFinSF2"] = np.zeros(n)
    data["BsmtUnfSF"] = rng.integers(0, 800, size=n)
    data["TotalBsmtSF"] = data["BsmtFinSF1"] + data["BsmtFinSF2"] + data["BsmtUnfSF"]
    data["1stFlrSF"] = rng.integers(500, 2000, size=n)
    data["2ndFlrSF"] = rng.choice([0, 400, 800], size=n)
    data["LowQualFinSF"] = np.zeros(n)
    data["GrLivArea"] = data["1stFlrSF"] + data["2ndFlrSF"]
    data["BsmtFullBath"] = rng.choice([0, 1, np.nan], size=n)
    data["BsmtHalfBath"] = rng.choice([0, 1, np.nan], size=n)
    data["FullBath"] = rng.integers(1, 4, size=n)
    data["HalfBath"] = rng.integers(0, 2, size=n)
    data["BedroomAbvGr"] = rng.integers(1, 6, size=n)
    data["KitchenAbvGr"] = np.ones(n)
    data["TotRmsAbvGrd"] = rng.integers(4, 12, size=n)
    data["Fireplaces"] = rng.integers(0, 3, size=n)
    data["GarageYrBlt"] = np.where(
        rng.random(n) > 0.1, data["YearBuilt"] + rng.integers(0, 5, size=n), np.nan
    )
    data["GarageCars"] = rng.choice([0, 1, 2, 3, np.nan], size=n)
    data["GarageArea"] = rng.choice([0, 250, 400, 600, np.nan], size=n)
    data["WoodDeckSF"] = rng.choice([0, 100, 200], size=n)
    data["OpenPorchSF"] = rng.choice([0, 40, 80], size=n)
    data["EnclosedPorch"] = np.zeros(n)
    data["3SsnPorch"] = np.zeros(n)
    data["ScreenPorch"] = np.zeros(n)
    data["PoolArea"] = np.zeros(n)
    data["MiscVal"] = np.zeros(n)
    data["MoSold"] = rng.integers(1, 13, size=n)
    data["YrSold"] = rng.integers(2006, 2011, size=n)

    df = pd.DataFrame(data)
    if with_target:
        base = 50000 + df["OverallQual"] * 15000 + df["GrLivArea"] * 40
        noise = rng.normal(0, 10000, size=n)
        df["SalePrice"] = (base + noise).clip(lower=30000)
    return df


@pytest.fixture
def synthetic_train_df():
    rng = np.random.default_rng(42)
    return _make_rows(60, rng, with_target=True)


@pytest.fixture
def synthetic_test_df():
    rng = np.random.default_rng(7)
    return _make_rows(20, rng, with_target=False)
