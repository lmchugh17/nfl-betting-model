"""Assembles the final per-game feature table from everything backfilled so far:
ELO, opponent-adjusted SRS, EPA-based rolling form, ATS/situational/H2H, QB
availability, injury burden, and adverse-weather ATS splits. Writes to the
`game_features` table (one row per completed 2016+ REG/POST game) for model
training (task 9). Mirrors the CFB build's assembler.

Only completed games (both scores present) are included -- this is a training
table, not a prediction input. Live inference reuses these same feature
functions directly (src/availability.py, src/ats_and_situational.py, etc. all
take an explicit (season, week, team) and are point-in-time correct already --
no separate CFB-style live_state.py reimplementation needed here, see task 11).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ats_and_situational import (compute_ats_results, compute_h2h_features,
                                      compute_rolling_ats_pct, compute_situational_features)
from src.availability import injury_burden, qb_situation
from src.db import get_stats_connection, init_stats_db
from src.elo import NFLElo
from src.epa_features import assemble_epa_game_features, compute_rolling_epa_form
from src.opponent_adjustment import compute_weekly_srs
from src.stadiums import lookup as stadium_lookup
from src.team_names import franchise_id
from src.weather_features import compute_adverse_wx_ats_pct, was_game_adverse

GAMES_COLS = """
    game_id, season, week, season_type, gameday, weekday, gametime,
    home_team, away_team, home_score, away_score, home_margin, home_win,
    div_game, home_rest, away_rest, spread_line, total_line, roof, is_dome,
    temp, wind, stadium, home_coach, away_coach, home_qb_id, away_qb_id,
    home_qb_name, away_qb_name, neutral_site
"""


def load_games(conn) -> list[dict]:
    rows = conn.execute(f"""
        SELECT {GAMES_COLS} FROM games
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
          AND season >= 2016 AND season_type IN ('REG', 'POST')
    """).fetchall()
    cols = [c.strip() for c in GAMES_COLS.split(",")]
    games = [dict(zip(cols, r)) for r in rows]
    for g in games:
        g["year"] = g["season"]                    # opponent_adjustment.py's ported key name
        g["home_points"] = g["home_score"]          # ditto -- home_points/away_points
        g["away_points"] = g["away_score"]
        g["home_franchise"] = franchise_id(g["home_team"])
        g["away_franchise"] = franchise_id(g["away_team"])
    return games


def compute_elo_features(games: list[dict]) -> dict:
    games_sorted = sorted(games, key=lambda g: (g["season"], g["gameday"]))
    elo = NFLElo()
    result = {}
    for g in games_sorted:
        elo.maybe_regress_for_new_season(g["season"])
        result[g["game_id"]] = elo.pre_game_features(
            g["home_franchise"], g["away_franchise"], g["season"], bool(g["neutral_site"])
        )
        elo.update(g["home_franchise"], g["away_franchise"], g["home_points"], g["away_points"],
                   g["season"], bool(g["neutral_site"]))
    return result


def compute_new_hc_flags(games: list[dict]) -> dict:
    """{(season, team): is_new_hc} -- team's coach for its season-opening game
    differs from that team's coach in its final game of the PRIOR season."""
    games_sorted = sorted(games, key=lambda g: (g["season"], g["gameday"]))
    last_coach_seen = {}   # team -> (season, coach_name) as of the most recent processed game
    season_open_coach = {}  # (season, team) -> coach at that team's first game of the season
    coach_prior_season_end = {}  # team -> {season: coach at team's LAST game of that season}
    by_team_season_games = {}

    for g in games_sorted:
        for side, team in (("home", g["home_team"]), ("away", g["away_team"])):
            coach = g.get(f"{side}_coach")
            key = (g["season"], team)
            if key not in season_open_coach:
                season_open_coach[key] = coach
            coach_prior_season_end.setdefault(team, {})[g["season"]] = coach

    result = {}
    for (season, team), opening_coach in season_open_coach.items():
        prior = coach_prior_season_end.get(team, {}).get(season - 1)
        result[(season, team)] = int(prior is not None and opening_coach is not None and opening_coach != prior)
    return result


def main():
    init_stats_db()
    conn = get_stats_connection()

    try:
        games = load_games(conn)
        print(f"Building features for {len(games)} completed games (2016+ REG/POST)...")
        games_by_id = {g["game_id"]: g for g in games}

        print("Computing ELO...")
        elo_features = compute_elo_features(games)

        print("Computing opponent-adjusted SRS (franchise-keyed)...")
        srs_games = [{**g, "home_team": g["home_franchise"], "away_team": g["away_franchise"]} for g in games]
        srs_lookup = compute_weekly_srs(srs_games)

        print("Computing EPA rolling form...")
        epa_df = pd.read_sql("SELECT * FROM team_game_epa", conn)
        rolling_epa = compute_rolling_epa_form(epa_df, srs_lookup, franchise_id)
        epa_features_df = assemble_epa_game_features(rolling_epa)

        print("Computing ATS record + H2H...")
        ats_rows = compute_ats_results(games)
        rolling_ats = compute_rolling_ats_pct(ats_rows)
        h2h = compute_h2h_features(games)

        print("Computing adverse-weather ATS splits...")
        adverse_wx_ats = compute_adverse_wx_ats_pct(ats_rows, games_by_id)

        print("Computing situational flags (rest/bye/primetime/div/international/travel)...")
        situational = {}
        for g in games:
            venue = stadium_lookup(g["stadium"])
            venue_tz = venue["timezone"] if venue else None
            situational[g["game_id"]] = compute_situational_features(g, venue_tz)

        print("Computing new-HC flags...")
        new_hc = compute_new_hc_flags(games)

        print("Computing QB availability + injury burden (this is the slow step)...")
        avail = {}
        for i, g in enumerate(games):
            for side, team, qb_gsis in (("home", g["home_team"], g["home_qb_id"]),
                                         ("away", g["away_team"], g["away_qb_id"])):
                ib = injury_burden(conn, g["season"], g["week"], team)
                qb = qb_situation(conn, g["season"], g["week"], team, qb_gsis)
                avail[(g["game_id"], side)] = {
                    f"{side}_injury_burden": ib["burden"],
                    f"{side}_qb_backup_starting": int(qb["backup_starting"]),
                    f"{side}_qb_trailing_share": qb["projected_qb_trailing_share"],
                }
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(games)} games...")

        print("Assembling final table...")
        records = []
        for g in games:
            gid = g["game_id"]
            elo_f = elo_features.get(gid, {})
            h2h_f = h2h.get(gid, {})
            sit_f = situational.get(gid, {})
            home_avail = avail.get((gid, "home"), {})
            away_avail = avail.get((gid, "away"), {})

            srs_home = srs_lookup.get((g["season"], g["week"], g["home_franchise"]))
            srs_away = srs_lookup.get((g["season"], g["week"], g["away_franchise"]))

            home_ats_pct, home_ats_n = rolling_ats.get((gid, g["home_team"]), (None, 0))
            away_ats_pct, away_ats_n = rolling_ats.get((gid, g["away_team"]), (None, 0))

            record = {
                "game_id": gid, "season": g["season"], "week": g["week"], "season_type": g["season_type"],
                "home_team": g["home_team"], "away_team": g["away_team"],
                "home_qb_name": g.get("home_qb_name"), "away_qb_name": g.get("away_qb_name"),
                "home_score": g["home_score"], "away_score": g["away_score"],
                "home_margin": g["home_margin"], "home_win": g["home_win"],
                "market_spread": g["spread_line"], "temp": g["temp"], "wind": g["wind"],
                "elo_home": elo_f.get("elo_home"), "elo_away": elo_f.get("elo_away"),
                "elo_diff": elo_f.get("elo_diff"), "elo_expected_home": elo_f.get("elo_expected_home"),
                "srs_home": srs_home, "srs_away": srs_away,
                "srs_diff": (srs_home - srs_away) if (srs_home is not None and srs_away is not None) else None,
                "home_ats_pct": home_ats_pct, "away_ats_pct": away_ats_pct,
                "home_ats_count": home_ats_n, "away_ats_count": away_ats_n,  # explanation-only, not a model feature
                "home_rest_days": g["home_rest"], "away_rest_days": g["away_rest"],
                "h2h_home_win_pct": h2h_f.get("h2h_home_win_pct"),
                "h2h_avg_home_margin": h2h_f.get("h2h_avg_home_margin"),
                "h2h_meetings": h2h_f.get("h2h_meetings"),
                "h2h_current_season_margin": h2h_f.get("h2h_current_season_margin"),
                "h2h_current_season_meeting": h2h_f.get("h2h_current_season_meeting"),
                "is_adverse_weather": int(was_game_adverse(g)),
                "home_new_hc": new_hc.get((g["season"], g["home_team"]), 0),
                "away_new_hc": new_hc.get((g["season"], g["away_team"]), 0),
                **sit_f, **home_avail, **away_avail,
            }
            home_wx_pct, home_wx_n = adverse_wx_ats.get((gid, g["home_team"]), (None, 0))
            away_wx_pct, away_wx_n = adverse_wx_ats.get((gid, g["away_team"]), (None, 0))
            # explanation-only, not model features
            record["home_adverse_wx_ats_pct"], record["home_adverse_wx_ats_count"] = home_wx_pct, home_wx_n
            record["away_adverse_wx_ats_pct"], record["away_adverse_wx_ats_count"] = away_wx_pct, away_wx_n
            record["adverse_wx_ats_edge"] = (
                (home_wx_pct - away_wx_pct)
                if record["is_adverse_weather"] and home_wx_pct is not None and away_wx_pct is not None
                else 0.0
            )
            records.append(record)

        final_df = pd.DataFrame(records).merge(epa_features_df, on="game_id", how="left")

        final_df.to_sql("game_features", conn, if_exists="replace", index=False)
        conn.commit()

        print(f"\nWrote {len(final_df)} rows to game_features ({len(final_df.columns)} columns).")
        non_null_pct = final_df.notna().mean().sort_values()
        print("\nColumns with the most missing data (worth knowing before modeling):")
        print(non_null_pct.head(12))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
