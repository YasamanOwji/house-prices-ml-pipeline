"""
config.py
=========
Single source of truth for column groups, domain-knowledge ordinal mappings,
file paths, and model hyperparameters used across the project.

Centralizing this here (instead of scattering magic strings through the
pipeline, as the original notebook did) means every module — preprocessing,
feature engineering, training, prediction, and tests — stays in sync.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
PIPELINE_FILE = MODEL_DIR / "house_price_pipeline.joblib"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

TARGET = "SalePrice"
ID_COL = "Id"
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Columns where NaN means "feature does not exist" -> fill with the string
# "None" (categorical) rather than treating it as a genuinely missing value.
# --------------------------------------------------------------------------
NONE_FILL_CATEGORICAL = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# Columns where NaN means "0 of this thing" (numeric, no such feature).
ZERO_FILL_NUMERIC = [
    "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "MasVnrArea",
]

# Low-cardinality columns best filled with the training-set mode (a handful
# of rows only, almost certainly a data entry omission rather than a
# structural "no feature" case).
MODE_FILL_CATEGORICAL = [
    "MSZoning", "Electrical", "KitchenQual", "Exterior1st",
    "Exterior2nd", "SaleType",
]

# Dropped outright: >99% constant value, essentially zero predictive signal,
# and known to cause issues with unseen categories on the hidden test fold.
DROP_COLUMNS = ["Utilities"]

# --------------------------------------------------------------------------
# Ordinal columns: order carries real meaning, so we map to integers rather
# than one-hot encoding (which would throw away that ordering information).
# --------------------------------------------------------------------------
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLUMNS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

ORDINAL_MAPS = {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4,
                    "Min2": 5, "Min1": 6, "Typ": 7},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
    "Street": {"Grvl": 0, "Pave": 1},
}
for _col in QUALITY_COLUMNS:
    ORDINAL_MAPS[_col] = QUALITY_MAP

ORDINAL_COLUMNS = list(ORDINAL_MAPS.keys())

# --------------------------------------------------------------------------
# Nominal (unordered) categorical columns -> one-hot encoded.
# MSSubClass is numerically coded in the raw file but is actually a
# categorical building-class code, not a magnitude - cast to string.
# --------------------------------------------------------------------------
NOMINAL_COLUMNS = [
    "MSSubClass", "MSZoning", "Alley", "LandContour", "LotConfig",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
    "Foundation", "Heating", "Electrical", "GarageType", "Fence",
    "MiscFeature", "SaleType", "SaleCondition",
]

# --------------------------------------------------------------------------
# Known Ames data-quality issues, documented in Dean De Cock's original
# paper on the dataset and reproduced in most careful EDAs of it.
# --------------------------------------------------------------------------
GRLIVAREA_OUTLIER_THRESHOLD = 4000
GRLIVAREA_OUTLIER_SALEPRICE_THRESHOLD = 300000

# --------------------------------------------------------------------------
# Skew correction
# --------------------------------------------------------------------------
SKEW_THRESHOLD = 0.75

# --------------------------------------------------------------------------
# Model hyperparameter search space
# --------------------------------------------------------------------------
XGB_PARAM_GRID = {
    "model__n_estimators": [500, 1000],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

CV_FOLDS = 5
N_ITER_RANDOM_SEARCH = 25
