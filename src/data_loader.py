"""
data_loader.py
===============
Thin I/O layer. Kept separate from preprocessing so tests can mock it
easily and so the raw-file layout can change without touching modeling code.
"""

import pandas as pd

from . import config


def load_raw_data(train_path=config.TRAIN_FILE, test_path=config.TEST_FILE):
    """Loads the competition train/test CSVs.

    Raises a clear error (rather than a confusing pandas traceback) if the
    files haven't been downloaded yet — see README for instructions.
    """
    missing = [p for p in (train_path, test_path) if not p.exists()]
    if missing:
        missing_str = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Could not find the competition data:\n"
            f"{missing_str}\n\n"
            "Download train.csv and test.csv from the Kaggle competition "
            "'House Prices - Advanced Regression Techniques' and place them "
            f"in {config.RAW_DATA_DIR}/ (see README.md for the exact steps)."
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df
