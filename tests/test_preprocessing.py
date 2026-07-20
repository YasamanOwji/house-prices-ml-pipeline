import numpy as np
import pandas as pd

from src.preprocessing import MissingValueImputer, OrdinalMapper, remove_training_outliers
from src import config


def test_missing_value_imputer_fills_none_categoricals():
    df = pd.DataFrame({
        "Neighborhood": ["CollgCr", "CollgCr", "Somerst"],
        "LotFrontage": [60.0, np.nan, 80.0],
        "PoolQC": [np.nan, np.nan, "Gd"],
        "GarageType": [np.nan, "Attchd", np.nan],
        "GarageArea": [0, 400, np.nan],
        "GarageCars": [0, 2, np.nan],
        "MasVnrArea": [np.nan, 100, 0],
        "MSZoning": ["RL", np.nan, "RL"],
        "Electrical": ["SBrkr", "SBrkr", np.nan],
        "KitchenQual": ["TA", "TA", np.nan],
        "Exterior1st": ["VinylSd", "VinylSd", np.nan],
        "Exterior2nd": ["VinylSd", "VinylSd", np.nan],
        "SaleType": ["WD", "WD", np.nan],
        "Functional": ["Typ", np.nan, "Typ"],
        "MSSubClass": [20, 60, 20],
        "Utilities": ["AllPub", "AllPub", "AllPub"],
        "YearBuilt": [2000, 1995, 1980],
        "YrSold": [2008, 2008, 2008],
        "GarageYrBlt": [np.nan, 1996, np.nan],
    })

    imputer = MissingValueImputer().fit(df)
    out = imputer.transform(df)

    assert out["PoolQC"].tolist() == ["None", "None", "Gd"]
    assert out["GarageType"].tolist() == ["None", "Attchd", "None"]
    assert out["GarageArea"].fillna(-1).tolist()[2] == 0  # zero-filled, not NaN
    assert out["MasVnrArea"].iloc[0] == 0
    assert out["MSZoning"].iloc[1] == "RL"  # mode-filled
    assert out["Functional"].iloc[1] == "Typ"  # explicit Typ fill
    assert "Utilities" not in out.columns  # dropped
    # Cast to a categorical string type (object dtype pre-pandas-3, or the
    # newer pandas StringDtype from pandas>=3.0 - either is fine here).
    assert pd.api.types.is_string_dtype(out["MSSubClass"])
    assert out["MSSubClass"].tolist() == ["20", "60", "20"]
    # LotFrontage filled with the CollgCr neighborhood median (60.0, since
    # that's the only other CollgCr row available at fit time).
    assert out["LotFrontage"].iloc[1] == 60.0
    assert not out["GarageYrBlt"].isna().any()


def test_missing_value_imputer_is_leak_free_across_splits():
    """Statistics must come from the split passed to `.fit`, not from data
    seen later at `.transform` time — this is the core leakage guard.
    """
    train = pd.DataFrame({
        "Neighborhood": ["CollgCr", "CollgCr"],
        "LotFrontage": [60.0, 80.0],  # median = 70
        "MSZoning": ["RL", "RL"],
        "Electrical": ["SBrkr", "SBrkr"],
        "KitchenQual": ["TA", "TA"],
        "Exterior1st": ["VinylSd", "VinylSd"],
        "Exterior2nd": ["VinylSd", "VinylSd"],
        "SaleType": ["WD", "WD"],
    })
    imputer = MissingValueImputer().fit(train)

    unseen = pd.DataFrame({
        "Neighborhood": ["CollgCr"],
        "LotFrontage": [np.nan],
        "MSZoning": ["RL"],
        "Electrical": ["SBrkr"],
        "KitchenQual": ["TA"],
        "Exterior1st": ["VinylSd"],
        "Exterior2nd": ["VinylSd"],
        "SaleType": ["WD"],
    })
    out = imputer.transform(unseen)
    assert out["LotFrontage"].iloc[0] == 70.0  # from train stats, not unseen row


def test_ordinal_mapper_known_values():
    df = pd.DataFrame({
        "ExterQual": ["Ex", "TA", "None"],
        "CentralAir": ["Y", "N", "Y"],
    })
    out = OrdinalMapper().fit_transform(df)
    assert out["ExterQual"].tolist() == [5, 3, 0]
    assert out["CentralAir"].tolist() == [1, 0, 1]
    assert out["ExterQual"].dtype == int


def test_remove_training_outliers_drops_documented_cases():
    df = pd.DataFrame({
        "GrLivArea": [1500, 4500, 5000, 2000],
        config.TARGET: [200000, 200000, 400000, 250000],
    })
    out = remove_training_outliers(df)
    # Row 1 (large area, low price) removed; row 2 (large area, high price) kept.
    assert len(out) == 3
    assert 4500 not in out["GrLivArea"].values
    assert 5000 in out["GrLivArea"].values
