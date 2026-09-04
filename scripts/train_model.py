"""Trains the win-probability ensemble and margin regressor on game_features,
evaluates on a held-out season, and saves the model bundle.

Split: train 2016-2024, hold out all of 2025 as the test set. A full-season
holdout (not a row-count split) avoids cutting a season in half and keeps the
evaluation honest -- the model never sees any 2025 result during training,
mirroring how it would actually be used (predict a season it hasn't seen yet).
2026 rows (if any are in game_features yet) are excluded from both train and
test -- the current season is tracked separately via the live
predictions/reconcile pipeline, not folded into this holdout-eval cycle.

Usage: .venv/bin/python scripts/train_model.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_stats_connection
from src.model import (FEATURE_COLUMNS, StackingEnsemble, evaluate_classifier,
                        evaluate_regressor, get_calibration_curve, prepare_matrix,
                        train_margin_regressor)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
HOLDOUT_SEASON = 2025


def main():
    conn = get_stats_connection()
    df = pd.read_sql("SELECT * FROM game_features ORDER BY season, week", conn)
    conn.close()

    train_df = df[df["season"] < HOLDOUT_SEASON].reset_index(drop=True)
    test_df = df[df["season"] == HOLDOUT_SEASON].reset_index(drop=True)
    print(f"Train: {len(train_df)} games ({train_df['season'].min()}-{HOLDOUT_SEASON - 1}). "
          f"Test: {len(test_df)} games ({HOLDOUT_SEASON}).")

    X_train, medians = prepare_matrix(train_df)
    X_test = test_df[FEATURE_COLUMNS].copy()
    for col in X_test.columns:
        if X_test[col].dtype == bool:
            X_test[col] = X_test[col].astype(float)
    X_test = X_test.astype(float).fillna(medians)

    y_train_class, y_test_class = train_df["home_win"], test_df["home_win"]
    y_train_margin, y_test_margin = train_df["home_margin"], test_df["home_margin"]

    print("\nTraining 5-model stacking ensemble (win probability)...")
    ensemble = StackingEnsemble()
    ensemble.fit(X_train, y_train_class)
    ensemble.feature_medians = medians

    print("Training margin regressor...")
    regressor = train_margin_regressor(X_train, y_train_margin)

    print(f"\n=== Classifier evaluation ({HOLDOUT_SEASON} holdout) ===")
    class_metrics = evaluate_classifier(ensemble, X_test, y_test_class)
    for k, v in class_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n=== Regressor evaluation ({HOLDOUT_SEASON} holdout) ===")
    reg_metrics = evaluate_regressor(regressor, X_test, y_test_margin)
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Calibration curve (predicted vs actual win %, 8 bins) ===")
    prob_true, prob_pred = get_calibration_curve(ensemble, X_test, y_test_class)
    for pt, pp in zip(prob_true, prob_pred):
        print(f"  predicted {pp:.3f} -> actual {pt:.3f}")

    print("\n=== Baseline comparisons ===")
    # market_spread convention: positive => home favored (nflverse), so it IS
    # the market's implied home margin directly -- no negation. See src/model.py's
    # docstring for the real sign-convention bug this project caught (differs
    # from CFB's CFBD-sourced spread, which uses the opposite "-7 = favored" quote).
    naive_acc = (y_test_class == 1).mean()
    print(f"  home-team-always-wins accuracy: {naive_acc:.4f}  (our model: {class_metrics['accuracy']:.4f})")
    elo_only_acc = ((test_df["elo_expected_home"] > 0.5).astype(int) == y_test_class).mean()
    print(f"  ELO-only accuracy:              {elo_only_acc:.4f}")
    spread_favorite_acc = ((test_df["market_spread"] > 0).astype(int) == y_test_class).mean()
    print(f"  spread-favorite accuracy:       {spread_favorite_acc:.4f}")
    market_mae = (test_df["market_spread"] - y_test_margin).abs().mean()
    print(f"  market MAE (vs actual margin):   {market_mae:.4f}  (our regressor: {reg_metrics['mae']:.4f})")

    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "ensemble": ensemble, "regressor": regressor, "feature_columns": FEATURE_COLUMNS,
        "feature_medians": medians, "train_seasons": (int(train_df["season"].min()), HOLDOUT_SEASON - 1),
        "holdout_season": HOLDOUT_SEASON, "classifier_metrics": class_metrics, "regressor_metrics": reg_metrics,
    }
    joblib.dump(bundle, MODEL_DIR / "nfl_model.pkl", compress=3)
    print(f"\nSaved model bundle to {MODEL_DIR / 'nfl_model.pkl'}")


if __name__ == "__main__":
    main()
