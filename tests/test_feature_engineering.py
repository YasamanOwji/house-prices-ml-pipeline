import pandas as pd

from src.feature_engineering import FeatureEngineer


def _base_row(**overrides):
    row = dict(
        TotalBsmtSF=500, **{"1stFlrSF": 800, "2ndFlrSF": 400},
        FullBath=2, HalfBath=1, BsmtFullBath=1, BsmtHalfBath=0,
        OpenPorchSF=50, **{"3SsnPorch": 0}, EnclosedPorch=0, ScreenPorch=0, WoodDeckSF=100,
        YrSold=2010, YearBuilt=2000, YearRemodAdd=2005, GarageYrBlt=2000,
        PoolArea=0, GarageArea=300, Fireplaces=1,
        OverallQual=7, OverallCond=5,
        ExterQual=4, ExterCond=3, BsmtQual=4, KitchenQual=3,
        GarageQual=3, GarageCond=3,
    )
    row.update(overrides)
    return row


def test_feature_engineer_computes_expected_values():
    df = pd.DataFrame([_base_row()])
    out = FeatureEngineer().fit_transform(df)

    assert out["TotalSF"].iloc[0] == 500 + 800 + 400
    assert out["TotalBathrooms"].iloc[0] == 2 + 0.5 * 1 + 1 + 0.5 * 0
    assert out["TotalPorchSF"].iloc[0] == 50 + 0 + 0 + 0 + 100
    assert out["HouseAge"].iloc[0] == 10
    assert out["RemodAge"].iloc[0] == 5
    assert out["GarageAge"].iloc[0] == 10
    assert out["IsRemodeled"].iloc[0] == 1
    assert out["IsNewHouse"].iloc[0] == 0
    assert out["HasGarage"].iloc[0] == 1
    assert out["Has2ndFloor"].iloc[0] == 1
    assert out["OverallGrade"].iloc[0] == 7 * 5


def test_feature_engineer_handles_zero_feature_rows():
    df = pd.DataFrame([_base_row(**{"2ndFlrSF": 0}, GarageArea=0, PoolArea=0, Fireplaces=0)])
    out = FeatureEngineer().fit_transform(df)

    assert out["Has2ndFloor"].iloc[0] == 0
    assert out["HasGarage"].iloc[0] == 0
    assert out["HasPool"].iloc[0] == 0
    assert out["HasFireplace"].iloc[0] == 0


def test_ages_never_go_negative():
    # A YearRemodAdd earlier than YearBuilt shouldn't happen in the real
    # data, but the clip guards against it producing a negative age anyway.
    df = pd.DataFrame([_base_row(YearRemodAdd=1990, YearBuilt=2000, YrSold=1995)])
    out = FeatureEngineer().fit_transform(df)
    assert (out[["HouseAge", "RemodAge", "GarageAge"]] >= 0).all().all()
