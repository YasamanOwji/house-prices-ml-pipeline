"""
predict.py
==========
Loads the trained pipeline and produces submission.csv from test.csv.

    python -m src.predict

Because every preprocessing step lives inside the saved pipeline (fitted
statistics included), this file does *no* preprocessing of its own beyond
dropping the Id column — there's no way for it to drift out of sync with
how the model was trained.
"""

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loader import load_raw_data


def main():
    if not config.PIPELINE_FILE.exists():
        raise FileNotFoundError(
            f"No trained pipeline found at {config.PIPELINE_FILE}. "
            "Run `python -m src.train` first."
        )

    print(f"Loading pipeline from {config.PIPELINE_FILE}...")
    pipeline = joblib.load(config.PIPELINE_FILE)

    print("Loading test data...")
    _, test_df = load_raw_data()
    X_test = test_df.drop(columns=[config.ID_COL])

    print("Predicting...")
    log_predictions = pipeline.predict(X_test)
    predictions = np.expm1(log_predictions)

    submission = pd.DataFrame({
        config.ID_COL: test_df[config.ID_COL],
        config.TARGET: predictions,
    })

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Saved {len(submission)} predictions to {config.SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
