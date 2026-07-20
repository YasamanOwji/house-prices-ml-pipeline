"""
train.py
========
End-to-end training entry point.

    python -m src.train
    python -m src.train --n-iter 40 --cv 5

Steps
-----
1. Load raw train/test CSVs.
2. Drop the two documented GrLivArea outliers from the training rows only.
3. Hold out 10% of the training data as a final, untouched sanity check.
4. Randomized hyperparameter search (5-fold CV) for the XGBoost pipeline
   on the remaining 90%.
5. Evaluate the tuned pipeline on the untouched holdout split.
6. Refit the tuned pipeline on *all* available training data (standard
   practice once hyperparameters are fixed: more data -> better final
   model) and persist it to disk.
7. Save a feature-importance chart.
"""

import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from . import config, utils
from .data_loader import load_raw_data
from .pipeline import build_xgb_pipeline
from .preprocessing import remove_training_outliers


def parse_args():
    parser = argparse.ArgumentParser(description="Train the house price model.")
    parser.add_argument("--n-iter", type=int, default=config.N_ITER_RANDOM_SEARCH,
                         help="Number of hyperparameter combinations to sample.")
    parser.add_argument("--cv", type=int, default=config.CV_FOLDS,
                         help="Number of cross-validation folds.")
    parser.add_argument("--holdout-size", type=float, default=0.1,
                         help="Fraction of training rows held out for a final sanity check.")
    return parser.parse_args()


def main():
    args = parse_args()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df, _ = load_raw_data()
    train_df = remove_training_outliers(train_df)

    X = train_df.drop(columns=[config.TARGET, config.ID_COL])
    y = np.log1p(train_df[config.TARGET])

    X_search, X_holdout, y_search, y_holdout = train_test_split(
        X, y, test_size=args.holdout_size, random_state=config.RANDOM_STATE
    )

    print(f"Searching {args.n_iter} hyperparameter combinations "
          f"with {args.cv}-fold CV on {len(X_search)} rows...")
    pipeline = build_xgb_pipeline()
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=config.XGB_PARAM_GRID,
        n_iter=args.n_iter,
        cv=args.cv,
        scoring=utils.rmse_scorer,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        refit=True,
        verbose=1,
    )
    start = time.time()
    search.fit(X_search, y_search)
    elapsed = time.time() - start

    cv_rmse_log = -search.best_score_
    print(f"\nSearch finished in {elapsed / 60:.1f} min.")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE (log SalePrice): {cv_rmse_log:.5f}")

    # --- Fair, untouched holdout check ------------------------------------
    holdout_pred_log = search.best_estimator_.predict(X_holdout)
    holdout_rmse_log = utils.rmse(y_holdout, holdout_pred_log)
    holdout_rmse_dollars = utils.rmse(np.expm1(y_holdout), np.expm1(holdout_pred_log))
    print(f"Holdout RMSE (log SalePrice): {holdout_rmse_log:.5f}")
    print(f"Holdout RMSE (SalePrice, $):  {holdout_rmse_dollars:,.0f}")

    # --- Refit on all available training data for the deployed model ------
    print("\nRefitting best pipeline on 100% of the training data...")
    xgb_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    final_pipeline = build_xgb_pipeline(**xgb_params)
    final_pipeline.fit(X, y)

    joblib.dump(final_pipeline, config.PIPELINE_FILE)
    print(f"Saved trained pipeline to {config.PIPELINE_FILE}")

    # --- Feature importance -------------------------------------------------
    feature_names = utils.get_output_feature_names(final_pipeline, X.sample(
        min(200, len(X)), random_state=config.RANDOM_STATE))
    importances = final_pipeline.named_steps["model"].feature_importances_
    n = min(len(feature_names), len(importances))
    importance_df = (
        pd.DataFrame({"feature": feature_names[:n], "importance": importances[:n]})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x="importance", y="feature", color="#4C72B0")
    plt.title("Top 20 Most Important Features")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "feature_importance.png", dpi=150)
    print(f"Saved feature importance chart to {config.OUTPUT_DIR / 'feature_importance.png'}")

    # --- Metrics summary -----------------------------------------------------
    metrics = {
        "cv_rmse_log": cv_rmse_log,
        "holdout_rmse_log": holdout_rmse_log,
        "holdout_rmse_dollars": holdout_rmse_dollars,
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                         for k, v in search.best_params_.items()},
    }
    with open(config.OUTPUT_DIR / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved run metrics to {config.OUTPUT_DIR / 'train_metrics.json'}")


if __name__ == "__main__":
    main()
