"""Refit src/bet_sizing.py's calibration constants on the model's own holdout season.

Run this after every retrain -- CALIBRATION_INTERCEPT/SLOPE describe THIS model's edges,
not edges in general, so a retrained regressor invalidates them. Prints the constants to
paste into src/bet_sizing.py plus the bucket table that says whether the fit means
anything at all: if cover rate is flat across |edge| buckets, the slope will be ~0 and
nothing will clear break-even (which is exactly the 2026-09-25 situation -- see that
module's docstring).

Usage: .venv/bin/python scripts/fit_bet_sizing.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bet_sizing import break_even
from src.db import get_stats_connection
from src.model import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "nfl_model.pkl"
HOLDOUT_SEASON = 2025
ASSUMED_SPREAD_ODDS_AMERICAN = -110
BUCKETS = [(0, 3), (3, 6), (6, 10), (10, 99)]


def scored_games(conn, bundle) -> pd.DataFrame:
    """Every completed game with a market line, scored by the CURRENT model bundle."""
    feat = pd.read_sql("SELECT * FROM game_features", conn)
    feat = feat[[c for c in feat.columns if c == "game_id" or c in FEATURE_COLUMNS]]
    games = pd.read_sql("""SELECT game_id, season, home_score, away_score, spread_line FROM games
                           WHERE home_score IS NOT NULL AND spread_line IS NOT NULL""", conn)
    df = games.merge(feat, on="game_id")
    X = df[FEATURE_COLUMNS].astype(float).fillna(bundle["feature_medians"])
    df["predicted_margin"] = bundle["regressor"].predict(X)
    df["actual_margin"] = df["home_score"] - df["away_score"]
    # nflverse convention: positive spread_line = home favored, so edge is a direct difference.
    df["edge"] = df["predicted_margin"] - df["spread_line"]
    decided = df[(df["actual_margin"] - df["spread_line"]) != 0].copy()
    decided["covered"] = np.where(decided["edge"] > 0,
                                  decided["actual_margin"] - decided["spread_line"] > 0,
                                  decided["actual_margin"] - decided["spread_line"] < 0)
    return decided


def bucket_table(df: pd.DataFrame, label: str) -> None:
    print(f"\n{label} (n={len(df)}, cover {df['covered'].mean():.1%})")
    for lo, hi in BUCKETS:
        m = df[(df["edge"].abs() >= lo) & (df["edge"].abs() < hi)]
        if len(m):
            print(f"  |edge| {lo:>2}-{hi:<3} n={len(m):>5}  cover {m['covered'].mean():>6.1%}")


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    df = scored_games(get_stats_connection(), bundle)
    holdout = df[df["season"] == HOLDOUT_SEASON]
    if holdout.empty:
        sys.exit(f"No decided {HOLDOUT_SEASON} games with a market line -- nothing to fit.")

    bucket_table(df[df["season"] < HOLDOUT_SEASON], "TRAIN seasons (in-sample, flattered by memorization)")
    bucket_table(holdout, f"HOLDOUT {HOLDOUT_SEASON} (the honest one -- fit uses this)")
    live = df[df["season"] > HOLDOUT_SEASON]
    if len(live):
        bucket_table(live, "LIVE seasons since the holdout")

    fit = LogisticRegression().fit(holdout[["edge"]].abs(), holdout["covered"].astype(int))
    intercept, slope = float(fit.intercept_[0]), float(fit.coef_[0][0])
    corr = np.corrcoef(holdout["edge"].abs(), holdout["covered"].astype(int))[0, 1]
    print(f"\ncorr(|edge|, covered) on the holdout: {corr:+.3f}")
    print(f"\nCALIBRATION_INTERCEPT = {intercept:.6f}\nCALIBRATION_SLOPE = {slope:.6f}")

    be = break_even(ASSUMED_SPREAD_ODDS_AMERICAN)
    edge_needed = (np.log(be / (1 - be)) - intercept) / slope if slope > 0 else None
    if edge_needed is None:
        print(f"\nSlope <= 0: no edge clears the {be:.1%} break-even, so every pick stakes zero.")
    else:
        print(f"\nBreak-even at {ASSUMED_SPREAD_ODDS_AMERICAN} is {be:.1%}: "
              f"needs |edge| >= {edge_needed:.1f} pts before Kelly stakes anything. "
              f"{(holdout['edge'].abs() >= edge_needed).sum()} of {len(holdout)} holdout picks clear it.")


if __name__ == "__main__":
    main()
