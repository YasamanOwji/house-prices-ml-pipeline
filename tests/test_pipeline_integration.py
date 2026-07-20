import numpy as np

from src.pipeline import build_xgb_pipeline
from src import config


def test_pipeline_fits_and_predicts_on_synthetic_data(synthetic_train_df, synthetic_test_df):
    """Not a model-quality check (the data is random) — this only proves the
    full preprocessing + model plumbing runs end-to-end without errors on
    both a training fit and a prediction on differently-shaped test data
    (including neighborhoods/categories not necessarily lined up 1:1).
    """
    X = synthetic_train_df.drop(columns=[config.TARGET, config.ID_COL])
    y = np.log1p(synthetic_train_df[config.TARGET])

    pipeline = build_xgb_pipeline(n_estimators=20, max_depth=3)
    pipeline.fit(X, y)

    X_test = synthetic_test_df.drop(columns=[config.ID_COL])
    preds = pipeline.predict(X_test)

    assert len(preds) == len(synthetic_test_df)
    assert np.isfinite(preds).all()


def test_pipeline_handles_missing_categories_in_test_only(synthetic_train_df, synthetic_test_df):
    """OneHotEncoder(handle_unknown='ignore') should stop an unseen category
    at prediction time from raising, instead of crashing the whole pipeline.
    """
    X = synthetic_train_df.drop(columns=[config.TARGET, config.ID_COL])
    y = np.log1p(synthetic_train_df[config.TARGET])
    pipeline = build_xgb_pipeline(n_estimators=10, max_depth=2)
    pipeline.fit(X, y)

    X_test = synthetic_test_df.drop(columns=[config.ID_COL]).copy()
    X_test.loc[X_test.index[0], "Neighborhood"] = "TotallyUnseenNeighborhood"
    preds = pipeline.predict(X_test)
    assert np.isfinite(preds).all()
