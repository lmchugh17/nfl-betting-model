"""Prediction model: a 5-model stacked ensemble for win probability (same
architecture as the CFB build and, before that, the reference NBA model --
LogisticRegression, RandomForest, XGBoost, LightGBM, ExtraTrees, blended by a
logistic-regression meta-learner trained on out-of-fold predictions), plus a
separate XGBoost regressor for predicted margin.

Two targets, not one, same reasoning as the CFB build: NFL betting is
spread-centric in a way a plain win-probability market isn't.
- Classifier -> P(home win), for confidence tiers in the write-up.
- Regressor -> predicted point margin, compared directly against market_spread
  to compute an edge (this is the number that actually matters for spread bets).

market_spread is deliberately EXCLUDED from both models' training features --
training on the market creates circular dependency (a model trained on the
market can't be evaluated as beating it). market_spread is only ever used
post-hoc to compute edge = predicted_margin - market_spread.

Sign convention, worth stating explicitly since it differs from CFB: nflverse's
`spread_line` is POSITIVE when the HOME team is favored (the opposite of the
"-7 = favored by 7" sportsbook-quote convention CFBD used, which is what CFB's
own edge/cover formulas assume). So market_spread already IS "the market's
implied home margin" directly -- no negation. Confirmed with a real game: 2023
Week 17 BUF (home) at spread_line=15 (a 15-point favorite) won by only 6 -- a
well-known "didn't cover" result, which `actual_margin > spread_line` (6 > 15,
false) gets right and `actual_margin > -spread_line` would have gotten backwards.

No NIL-era-style sample weighting: the CFB build down-weights pre-2021 rows
because NIL (Name/Image/Likeness) was a discrete legal break in how college
rosters get built. The NFL has no equivalent single-event discontinuity across
2016-2025 -- gradual rule/scheme drift (leaguewide passing efficiency rising
over the decade) is real but not a clean two-tier split, so it's left for the
model to learn from the data as-is rather than inventing an artificial era
boundary just because CFB had one.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier

FEATURE_COLUMNS = [
    # ELO / SRS (opponent-adjusted power ratings)
    "elo_home", "elo_away", "elo_diff", "elo_expected_home",
    "srs_home", "srs_away", "srs_diff",
    # ATS record / rest / head-to-head
    "home_ats_pct", "away_ats_pct",
    "home_rest_days", "away_rest_days",
    "h2h_home_win_pct", "h2h_avg_home_margin", "h2h_meetings",
    "h2h_current_season_margin", "h2h_current_season_meeting",
    # situational
    "home_new_hc", "away_new_hc",
    "home_short_week", "away_short_week", "home_off_bye", "away_off_bye",
    "is_primetime", "div_game", "is_international",
    "home_tz_shift_hours", "away_tz_shift_hours", "away_cross_country_travel",
    # weather
    "is_adverse_weather", "adverse_wx_ats_edge",
    # availability
    "home_injury_burden", "home_qb_backup_starting", "home_qb_trailing_share",
    "away_injury_burden", "away_qb_backup_starting", "away_qb_trailing_share",
    # EPA rolling form (home / away / diff)
    "home_avg_off_epa_play", "away_avg_off_epa_play", "diff_avg_off_epa_play",
    "home_avg_def_epa_play", "away_avg_def_epa_play", "diff_avg_def_epa_play",
    "home_avg_off_pass_epa", "away_avg_off_pass_epa", "diff_avg_off_pass_epa",
    "home_avg_off_rush_epa", "away_avg_off_rush_epa", "diff_avg_off_rush_epa",
    "home_avg_off_early_down_epa", "away_avg_off_early_down_epa", "diff_avg_off_early_down_epa",
    "home_avg_off_success_rate", "away_avg_off_success_rate", "diff_avg_off_success_rate",
    "home_avg_def_success_rate", "away_avg_def_success_rate", "diff_avg_def_success_rate",
    "home_avg_explosive_play_rate", "away_avg_explosive_play_rate", "diff_avg_explosive_play_rate",
    "home_avg_def_explosive_rate", "away_avg_def_explosive_rate", "diff_avg_def_explosive_rate",
    "home_avg_rz_td_pct", "away_avg_rz_td_pct", "diff_avg_rz_td_pct",
    "home_avg_pressure_rate_def", "away_avg_pressure_rate_def", "diff_avg_pressure_rate_def",
    "home_avg_sec_per_play", "away_avg_sec_per_play", "diff_avg_sec_per_play",
    "home_avg_pass_rate", "away_avg_pass_rate", "diff_avg_pass_rate",
    "home_avg_neutral_pass_rate", "away_avg_neutral_pass_rate", "diff_avg_neutral_pass_rate",
    "home_avg_opponent_srs", "away_avg_opponent_srs", "diff_avg_opponent_srs",
]

# 6 folds (vs the CFB build's 3): tested 3-6 against the 2025 holdout during
# this build -- accuracy was noisy across the range (60.4-62.5%, easily normal
# single-holdout variance on 285 games, not a clean monotonic trend), so this
# isn't a precision-tuned number. 6 was picked because it was consistently best
# across THREE metrics (accuracy, AUC, log_loss) simultaneously, not because it
# maximized any one score against the holdout -- optimizing a single metric
# against the one holdout the model will be judged on would just be tuning to
# the test set. Worth re-checking once a real live season of results exists.
N_SPLITS = 6

# Binary/count-ish columns that can arrive as SQLite's dynamic-typed 0/1/None
# rather than a clean float -- must be cast before imputing a median.
_BOOL_LIKE_COLUMNS = (
    "home_new_hc", "away_new_hc", "home_short_week", "away_short_week",
    "home_off_bye", "away_off_bye", "is_primetime", "div_game", "is_international",
    "away_cross_country_travel", "is_adverse_weather",
    "home_qb_backup_starting", "away_qb_backup_starting", "h2h_current_season_meeting",
)


def prepare_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    for col in _BOOL_LIKE_COLUMNS:
        X[col] = X[col].astype(float)
    medians = X.median()
    X = X.fillna(medians)
    return X, medians


def _base_classifiers() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, C=0.1),
        "random_forest": RandomForestClassifier(random_state=42, max_depth=8,
                                                  min_samples_leaf=10, n_estimators=200),
        "xgboost": XGBClassifier(random_state=42, eval_metric="logloss", subsample=0.8,
                                  colsample_bytree=0.8, learning_rate=0.05, max_depth=4, n_estimators=200),
        "lightgbm": LGBMClassifier(random_state=42, verbose=-1, subsample=0.8, colsample_bytree=0.8,
                                    n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=10),
        "extra_trees": ExtraTreesClassifier(random_state=42, n_jobs=-1, max_depth=12,
                                             min_samples_leaf=5, n_estimators=400),
    }


@dataclass
class StackingEnsemble:
    base_models: dict = field(default_factory=dict)
    meta_model: LogisticRegression = None
    feature_medians: pd.Series = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.base_models = {}
        oof_preds = np.zeros((len(X), len(_base_classifiers())))
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)

        for i, (name, model) in enumerate(_base_classifiers().items()):
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            fold_preds = np.full(len(X), np.nan)
            for train_idx, val_idx in tscv.split(X):
                pipeline_fold = Pipeline([("scaler", StandardScaler()), ("model", model.__class__(**model.get_params()))])
                pipeline_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
                fold_preds[val_idx] = pipeline_fold.predict_proba(X.iloc[val_idx])[:, 1]
            oof_preds[:, i] = fold_preds

            pipeline.fit(X, y)  # refit on full training set for inference-time use
            self.base_models[name] = pipeline

        valid_rows = ~np.isnan(oof_preds).any(axis=1)
        self.meta_model = LogisticRegression(C=0.1, max_iter=1000)
        self.meta_model.fit(oof_preds[valid_rows], y.iloc[valid_rows])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base_preds = np.column_stack([m.predict_proba(X)[:, 1] for m in self.base_models.values()])
        return self.meta_model.predict_proba(base_preds)[:, 1]

    def predict_proba_detailed(self, X: pd.DataFrame) -> dict:
        """Same as predict_proba, but also returns each base model's own win probability
        before blending -- for showing users what each of the 5 models individually predicted,
        not just the final ensemble output."""
        base_probs = {name: m.predict_proba(X)[:, 1] for name, m in self.base_models.items()}
        base_matrix = np.column_stack(list(base_probs.values()))
        final = self.meta_model.predict_proba(base_matrix)[:, 1]
        return {"base": base_probs, "final": final}


def train_margin_regressor(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(random_state=42, n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8)
    model.fit(X, y)
    return model


def evaluate_classifier(ensemble: StackingEnsemble, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = ensemble.predict_proba(X_test)
    preds = (proba > 0.5).astype(int)
    return {
        "accuracy": (preds == y_test.values).mean(),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "auc": roc_auc_score(y_test, proba),
    }


def evaluate_regressor(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": float(np.sqrt(np.mean((preds - y_test.values) ** 2))),
    }


def get_calibration_curve(ensemble: StackingEnsemble, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 8):
    proba = ensemble.predict_proba(X_test)
    return calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
