"""Predicts specific games using current team state (ELO, SRS, EPA rolling
form, ATS record, availability) computed as of right now, not "entering game
X" like the training pipeline. Loads the trained model bundle, prints a
prediction + grounded explanation facts for each requested game, and persists
the pick to nfl_predictions.db's `predictions` table.

Backtesting is implicit, not a separate flag: target games are always excluded
from "completed" state regardless of whether they've actually been played, so
passing an already-completed game_id predicts it using only what was knowable
beforehand -- an honest backtest -- while a genuinely upcoming game_id just
predicts normally. Same pattern as the CFB build.

Usage: .venv/bin/python scripts/predict_games.py <game_id> [<game_id> ...]
"""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ats_and_situational import compute_ats_results, compute_h2h_features, compute_situational_features
from src.availability import injury_burden, qb_situation
from src.db import get_pred_connection, get_stats_connection, init_all
from src.elo import HOME_ADVANTAGE_ELO, NFLElo
from src.explain import build_feature_highlights, get_shap_contributions
from src.live_state import (ATS_WINDOW, compute_current_ats_pct, compute_current_elo,
                             compute_current_epa_form, compute_current_opponent_srs,
                             compute_current_srs)
from src.model import FEATURE_COLUMNS
from src.spread_pricing import get_spread_price, load_latest_spread_prices
from src.stadiums import lookup as stadium_lookup
from src.team_names import franchise_id
from src.weather_features import compute_current_adverse_wx_ats_pct, was_game_adverse
from src.epa_features import ROLLING_WINDOW as EPA_WINDOW

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "nfl_model.pkl"

FULL_SEASON_WINDOW = max(EPA_WINDOW, ATS_WINDOW)  # both currently 8
KELLY_FRACTION_CAP = 0.25  # 25% fractional Kelly, matches the CFB/NBA-reference convention
# nflverse's games.{away,home}_spread_odds gives real closing juice for completed games, but
# for an upcoming game the only price source is a live Odds API pull (src.spread_pricing) --
# this is the fallback when that hasn't happened yet or missed this game/book.
ASSUMED_SPREAD_ODDS_AMERICAN = -110


# Placeholder thresholds on |edge| in points -- not statistically calibrated, just a
# first-pass tiering. Worth revisiting once enough real picks have accumulated (task 16).
def confidence_tier(edge: float | None) -> str | None:
    if edge is None:
        return None
    abs_edge = abs(edge)
    if abs_edge >= 7:
        return "high"
    if abs_edge >= 3:
        return "medium"
    return "low"


def moneyline_confidence_tier(win_prob_picked_side: float | None) -> str | None:
    if win_prob_picked_side is None:
        return None
    if win_prob_picked_side >= 0.75:
        return "high"
    if win_prob_picked_side >= 0.60:
        return "medium"
    return "low"


def _american_odds_to_net_decimal(odds: int) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cover_probability_and_kelly(edge: float | None, is_home_pick: bool, regressor_rmse: float,
                                 spread_odds_american: int = ASSUMED_SPREAD_ODDS_AMERICAN
                                 ) -> tuple[float | None, float | None]:
    """cover_probability is for the PICKED side specifically -- edge is defined as
    home's edge over the market (predicted_margin - market_spread, nflverse
    convention: + market_spread = home favored), so picking the away side needs
    1 minus the home cover probability. Treats the regressor's residuals as
    approximately Normal(0, rmse) around its point estimate -- rmse is the
    model's own measured error on the 2025 holdout (scripts/train_model.py),
    not an assumed number."""
    if edge is None:
        return None, None
    p_home_covers = _normal_cdf(edge / regressor_rmse)
    p_cover = p_home_covers if is_home_pick else (1 - p_home_covers)
    b = _american_odds_to_net_decimal(spread_odds_american)
    kelly_full = p_cover - (1 - p_cover) / b
    kelly = max(0.0, kelly_full) * KELLY_FRACTION_CAP
    return p_cover, kelly


def count_current_season_games(completed: list[dict], season: int) -> dict:
    counts: dict[str, int] = {}
    for g in completed:
        if g["season"] != season:
            continue
        counts[g["home_team"]] = counts.get(g["home_team"], 0) + 1
        counts[g["away_team"]] = counts.get(g["away_team"], 0) + 1
    return counts


def compute_new_hc(completed: list[dict], team: str, season: int) -> int:
    """Same definition as build_features.py's compute_new_hc_flags: this
    team's coach in its season-opening game vs. its coach in the prior
    season's final game. For a genuinely future game, "this season's opening
    coach" is just the coach on today's target-game row itself (passed by the
    caller), so this only needs the PRIOR season's final coach."""
    prior = [g for g in completed if g["season"] == season - 1
             and (g["home_team"] == team or g["away_team"] == team)]
    if not prior:
        return 0
    last = max(prior, key=lambda g: g["gameday"])
    return 1 if (last["home_team"] == team and last.get("home_coach")) or \
                (last["away_team"] == team and last.get("away_coach")) else 0


def _prior_coach(completed: list[dict], team: str, season: int) -> str | None:
    prior = [g for g in completed if g["season"] == season - 1
             and (g["home_team"] == team or g["away_team"] == team)]
    if not prior:
        return None
    last = max(prior, key=lambda g: g["gameday"])
    return last["home_coach"] if last["home_team"] == team else last["away_coach"]


def load_all_games(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT game_id, season, week, season_type, gameday, weekday, gametime,
               home_team, away_team, home_score, away_score, div_game,
               home_rest, away_rest, spread_line, roof, is_dome, temp, wind,
               stadium, home_coach, away_coach, home_qb_id, away_qb_id,
               home_qb_name, away_qb_name, neutral_site
        FROM games
    """).fetchall()
    cols = ["game_id", "season", "week", "season_type", "gameday", "weekday", "gametime",
            "home_team", "away_team", "home_score", "away_score", "div_game",
            "home_rest", "away_rest", "spread_line", "roof", "is_dome", "temp", "wind",
            "stadium", "home_coach", "away_coach", "home_qb_id", "away_qb_id",
            "home_qb_name", "away_qb_name", "neutral_site"]
    games = [dict(zip(cols, r)) for r in rows]
    for g in games:
        g["home_points"], g["away_points"] = g["home_score"], g["away_score"]  # for compute_h2h_features etc
    return games


def main():
    target_ids = sys.argv[1:]
    if not target_ids:
        sys.exit("Usage: predict_games.py <game_id> [<game_id> ...]")

    init_all()
    bundle = joblib.load(MODEL_PATH)
    stats_conn = get_stats_connection()

    all_games = load_all_games(stats_conn)
    by_id = {g["game_id"]: g for g in all_games}
    missing = [gid for gid in target_ids if gid not in by_id]
    if missing:
        sys.exit(f"Game id(s) not found: {missing}")
    targets = [by_id[gid] for gid in target_ids]

    completed = [g for g in all_games if g["home_score"] is not None and g["game_id"] not in target_ids]
    current_season = max(g["season"] for g in targets)
    games_played_this_season = count_current_season_games(completed, current_season)

    print("Computing current ELO...")
    current_elo = compute_current_elo(completed, franchise_id)

    print("Computing current SRS...")
    current_srs = compute_current_srs(completed, current_season, franchise_id)

    print("Computing current EPA rolling form...")
    epa_df = pd.read_sql("SELECT * FROM team_game_epa", stats_conn)
    epa_df = epa_df[epa_df["game_id"].isin({g["game_id"] for g in completed})]
    current_form = compute_current_epa_form(epa_df, franchise_id)
    current_opp_srs = compute_current_opponent_srs(epa_df, current_srs, franchise_id)

    print("Computing current ATS record...")
    ats_rows = compute_ats_results(completed)
    current_ats = compute_current_ats_pct(ats_rows)

    print("Computing head-to-head...")
    h2h = compute_h2h_features(all_games)  # safe: only reads scores from COMPLETED games in the list

    print("Computing adverse-weather ATS splits...")
    current_adverse_wx = compute_current_adverse_wx_ats_pct(ats_rows, {g["game_id"]: g for g in completed})

    print("Loading latest per-book spread pricing...")
    spread_prices = load_latest_spread_prices(stats_conn)

    elo_model = NFLElo()  # only used for its expected_score() static method

    records = []
    for g in targets:
        home, away, neutral = g["home_team"], g["away_team"], bool(g["neutral_site"])
        hf, af = franchise_id(home), franchise_id(away)
        elo_home, elo_away = current_elo.get(hf, 1500.0), current_elo.get(af, 1500.0)
        home_bonus = 0 if neutral else HOME_ADVANTAGE_ELO
        elo_expected_home = elo_model.expected_score(elo_home + home_bonus, elo_away)

        srs_home, srs_away = current_srs.get(hf, 0.0), current_srs.get(af, 0.0)
        h2h_f = h2h.get(g["game_id"], {})

        venue = stadium_lookup(g["stadium"])
        sit_f = compute_situational_features(g, venue["timezone"] if venue else None)

        home_ats_pct, home_ats_n = current_ats.get(home, (None, 0))
        away_ats_pct, away_ats_n = current_ats.get(away, (None, 0))

        ib_home = injury_burden(stats_conn, g["season"], g["week"], home)
        ib_away = injury_burden(stats_conn, g["season"], g["week"], away)
        qb_home = qb_situation(stats_conn, g["season"], g["week"], home, g["home_qb_id"])
        qb_away = qb_situation(stats_conn, g["season"], g["week"], away, g["away_qb_id"])

        record = {
            "game_id": g["game_id"], "home_team": home, "away_team": away,
            "home_qb_name": g["home_qb_name"], "away_qb_name": g["away_qb_name"],
            "temp": g["temp"], "wind": g["wind"], "stadium": g["stadium"],
            "min_current_season_games": min(games_played_this_season.get(home, 0),
                                             games_played_this_season.get(away, 0)),
            "elo_home": elo_home, "elo_away": elo_away, "elo_diff": elo_home - elo_away,
            "elo_expected_home": elo_expected_home,
            "srs_home": srs_home, "srs_away": srs_away, "srs_diff": srs_home - srs_away,
            "home_ats_pct": home_ats_pct, "away_ats_pct": away_ats_pct,
            "home_ats_count": home_ats_n, "away_ats_count": away_ats_n,
            "home_rest_days": g["home_rest"], "away_rest_days": g["away_rest"],
            "h2h_home_win_pct": h2h_f.get("h2h_home_win_pct"),
            "h2h_avg_home_margin": h2h_f.get("h2h_avg_home_margin"),
            "h2h_meetings": h2h_f.get("h2h_meetings"),
            "h2h_current_season_margin": h2h_f.get("h2h_current_season_margin"),
            "h2h_current_season_meeting": h2h_f.get("h2h_current_season_meeting"),
            "market_spread": g["spread_line"],
            "home_new_hc": compute_new_hc(completed, home, g["season"]),
            "away_new_hc": compute_new_hc(completed, away, g["season"]),
            "home_injury_burden": ib_home["burden"], "away_injury_burden": ib_away["burden"],
            "home_injury_out": ib_home["out_starters"], "away_injury_out": ib_away["out_starters"],
            "home_qb_backup_starting": int(qb_home["backup_starting"]),
            "away_qb_backup_starting": int(qb_away["backup_starting"]),
            "home_qb_trailing_share": qb_home["projected_qb_trailing_share"],
            "away_qb_trailing_share": qb_away["projected_qb_trailing_share"],
            **sit_f,
        }
        home_epa = current_form.get(hf, {})
        away_epa = current_form.get(af, {})
        for stat in home_epa.keys() | away_epa.keys():
            hv, av = home_epa.get(stat), away_epa.get(stat)
            record[f"home_avg_{stat}"] = hv
            record[f"away_avg_{stat}"] = av
            record[f"diff_avg_{stat}"] = (hv - av) if (hv is not None and av is not None) else None
        record["home_avg_opponent_srs"] = current_opp_srs.get(hf)
        record["away_avg_opponent_srs"] = current_opp_srs.get(af)
        if record["home_avg_opponent_srs"] is not None and record["away_avg_opponent_srs"] is not None:
            record["diff_avg_opponent_srs"] = record["home_avg_opponent_srs"] - record["away_avg_opponent_srs"]
        else:
            record["diff_avg_opponent_srs"] = None

        record["is_adverse_weather"] = int(was_game_adverse(g))
        home_wx_pct, home_wx_n = current_adverse_wx.get(home, (None, 0))
        away_wx_pct, away_wx_n = current_adverse_wx.get(away, (None, 0))
        record["home_adverse_wx_ats_pct"], record["home_adverse_wx_ats_count"] = home_wx_pct, home_wx_n
        record["away_adverse_wx_ats_pct"], record["away_adverse_wx_ats_count"] = away_wx_pct, away_wx_n
        record["adverse_wx_ats_edge"] = (
            (home_wx_pct - away_wx_pct)
            if record["is_adverse_weather"] and home_wx_pct is not None and away_wx_pct is not None
            else 0.0
        )
        records.append(record)

    df = pd.DataFrame(records)
    X = df[FEATURE_COLUMNS].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(float)
    X = X.astype(float).fillna(bundle["feature_medians"])

    detailed = bundle["ensemble"].predict_proba_detailed(X)
    win_probs = detailed["final"]
    margins = bundle["regressor"].predict(X)
    regressor_rmse = bundle["regressor_metrics"]["rmse"]
    predicted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    pred_conn = get_pred_connection()

    for i, row in df.iterrows():
        g = by_id[row["game_id"]]
        print(f"\n{'=' * 70}\n{row['away_team']} @ {row['home_team']}\n{'=' * 70}")
        print(f"Predicted: {row['home_team']} by {margins[i]:+.1f} (home win prob {win_probs[i]:.0%})")
        if row["min_current_season_games"] < FULL_SEASON_WINDOW:
            print(f"NOTE: recent-form/ATS%/opponent-SRS stats include prior-season games -- "
                  f"the less-experienced team has only played {row['min_current_season_games']} "
                  f"game(s) so far this season (need {FULL_SEASON_WINDOW} for a fully current window).")
        print("Per-model win probability (home team):")
        for name, probs in detailed["base"].items():
            print(f"  {name}: {probs[i]:.0%}")

        moneyline_pick = row["home_team"] if win_probs[i] > 0.5 else row["away_team"]
        moneyline_win_prob = float(win_probs[i] if moneyline_pick == row["home_team"] else 1 - win_probs[i])
        ml_tier = moneyline_confidence_tier(moneyline_win_prob)
        print(f"Moneyline pick: {moneyline_pick} ({moneyline_win_prob:.0%} win prob, {ml_tier} confidence)")

        edge, pick_team = None, None
        cover_prob, kelly = None, None
        spread_price, spread_price_source, spread_price_book_count = None, None, 0
        if pd.notna(row["market_spread"]):
            # nflverse convention: spread_line POSITIVE => home favored (opposite of the
            # "-7=favored" sportsbook-quote convention CFBD uses) -- see src/model.py.
            fav = row["home_team"] if row["market_spread"] > 0 else row["away_team"]
            print(f"Market: {fav} favored by {abs(row['market_spread']):.1f}")
            edge = float(margins[i] - row["market_spread"])
            pick_team = row["home_team"] if edge > 0 else row["away_team"]
            print(f"Edge: model favors {row['home_team']} by {edge:+.1f} vs. the market line -> pick {pick_team}")

            measured_price, spread_price_book_count = get_spread_price(spread_prices, pick_team)
            if measured_price is not None:
                spread_price, spread_price_source = measured_price, "measured"
                print(f"Spread price: {spread_price} (median across {spread_price_book_count} book(s))")
            else:
                spread_price, spread_price_source = ASSUMED_SPREAD_ODDS_AMERICAN, "assumed"
                print(f"Spread price: {spread_price} (assumed -- no per-book pricing available for this game)")

            cover_prob, kelly = cover_probability_and_kelly(
                edge, pick_team == row["home_team"], regressor_rmse, spread_price)
            print(f"Cover probability: {cover_prob:.0%} -> {kelly:.1%} of bankroll recommended (25% Kelly)")
        else:
            print("Market: no line available")

        contribs = get_shap_contributions(bundle["regressor"], X.iloc[[i]], FEATURE_COLUMNS)
        highlights = build_feature_highlights(row.to_dict(), contribs, row["home_team"], row["away_team"])

        print("\nTop factors:")
        for h in highlights:
            print(f"  - {h}")

        model_breakdown = {name: float(probs[i]) for name, probs in detailed["base"].items()}

        # ON CONFLICT DO UPDATE (not plain REPLACE) so tldr/bullets_json -- written
        # separately by the routine that explains the pick -- don't get wiped back to
        # NULL every time this game gets re-predicted later in the week with fresher odds.
        pred_conn.execute(
            """INSERT INTO predictions
               (game_id, predicted_at, season, week, season_type, gameday, gametime,
                home_team, away_team, predicted_margin, win_prob_home, market_spread,
                pick_team, edge, confidence_tier, cover_probability, kelly_fraction,
                spread_price, spread_price_source, spread_price_book_count,
                moneyline_pick, moneyline_win_prob, moneyline_confidence_tier,
                highlights_json, model_breakdown_json, min_current_season_games)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(game_id) DO UPDATE SET
                   predicted_at=excluded.predicted_at, season=excluded.season, week=excluded.week,
                   season_type=excluded.season_type, gameday=excluded.gameday, gametime=excluded.gametime,
                   home_team=excluded.home_team, away_team=excluded.away_team,
                   predicted_margin=excluded.predicted_margin, win_prob_home=excluded.win_prob_home,
                   market_spread=excluded.market_spread, pick_team=excluded.pick_team,
                   edge=excluded.edge, confidence_tier=excluded.confidence_tier,
                   cover_probability=excluded.cover_probability, kelly_fraction=excluded.kelly_fraction,
                   spread_price=excluded.spread_price, spread_price_source=excluded.spread_price_source,
                   spread_price_book_count=excluded.spread_price_book_count,
                   moneyline_pick=excluded.moneyline_pick, moneyline_win_prob=excluded.moneyline_win_prob,
                   moneyline_confidence_tier=excluded.moneyline_confidence_tier,
                   highlights_json=excluded.highlights_json, model_breakdown_json=excluded.model_breakdown_json,
                   min_current_season_games=excluded.min_current_season_games""",
            (
                row["game_id"], predicted_at, g["season"], g["week"], g["season_type"],
                g["gameday"], g["gametime"], row["home_team"], row["away_team"],
                float(margins[i]), float(win_probs[i]),
                float(row["market_spread"]) if pd.notna(row["market_spread"]) else None,
                pick_team, edge, confidence_tier(edge), cover_prob, kelly,
                spread_price, spread_price_source, spread_price_book_count,
                moneyline_pick, moneyline_win_prob, ml_tier,
                json.dumps(highlights), json.dumps(model_breakdown),
                int(row["min_current_season_games"]),
            ),
        )
    pred_conn.commit()
    print(f"\nSaved {len(df)} prediction(s) to nfl_predictions.db.")


if __name__ == "__main__":
    main()
