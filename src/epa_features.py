"""Aggregate play-by-play into one row per (game, team) of EPA / success / pace
metrics -- the `team_game_epa` table.

Raw pbp (~50k plays/season x 372 cols) is never stored. scripts/backfill_epa.py
loads a season, calls aggregate_team_game_epa(), writes the ~570 rows, and drops
the frame.

This is the CFB build's biggest single upgrade: it had only box-score yards;
the NFL gets genuine efficiency (EPA) from the same free nflverse feed.

Thresholds (explosive-play yardage, neutral-script win-probability band) are
constants here, set from the real 2021-2023 distribution -- see the comments --
not copied from another sport. Retune against a wider window if the model asks
for it (weather-threshold lesson from the CFB build).
"""
import numpy as np
import pandas as pd

# Rolling-form window: trailing games of EPA history to average, shifted by one
# so a game's features never include itself. 8 games (vs the CFB build's 4 for a
# 13-game season) -- the 17-game NFL season affords a longer, more stable window;
# min_periods=3 so a team isn't flagged with a feature until it has a real sample.
ROLLING_WINDOW = 8
MIN_PERIODS = 3

ROLL_COLS = [
    "off_epa_play", "def_epa_play", "off_pass_epa", "off_rush_epa", "off_early_down_epa",
    "off_success_rate", "def_success_rate", "explosive_play_rate", "def_explosive_rate",
    "rz_td_pct", "pressure_rate_def", "sec_per_play", "pass_rate", "neutral_pass_rate",
]

# Explosive play: pass gains >= 15, rush gains >= 10. In 2021-2023 pbp that flags
# ~13.5% of pass plays and ~10% of rush plays -- the widely-used Sharp/Baldwin
# definition, and a big enough slice to be a stable per-game rate.
EXPLOSIVE_PASS_YDS = 15
EXPLOSIVE_RUSH_YDS = 10

# Neutral game script: win probability between these bounds and > 2 min left in
# the half -- i.e. play-calling not yet distorted by score/clock.
NEUTRAL_WP_LO, NEUTRAL_WP_HI = 0.20, 0.80
NEUTRAL_MIN_HALF_SECONDS = 120


def _offensive_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """The standard EPA/play denominator: every play that was fundamentally a
    pass or a rush (including ones wiped out by penalty), minus QB kneels/spikes."""
    m = ((pbp["pass"] == 1) | (pbp["rush"] == 1)) & (pbp["qb_kneel"] == 0) & (pbp["qb_spike"] == 0)
    return pbp.loc[m].copy()


def _parse_top(top: str) -> float:
    if not isinstance(top, str) or ":" not in top:
        return np.nan
    m, s = top.split(":")
    return int(m) * 60 + int(s)


def _offense_by_team(off: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, posteam) offensive aggregates."""
    off = off.copy()
    off["is_pass"] = (off["pass"] == 1).astype(int)
    off["is_rush"] = (off["rush"] == 1).astype(int)
    off["is_early_down"] = off["down"].isin([1, 2])
    off["explosive"] = (
        ((off["pass"] == 1) & (off["yards_gained"] >= EXPLOSIVE_PASS_YDS))
        | ((off["rush"] == 1) & (off["yards_gained"] >= EXPLOSIVE_RUSH_YDS))
    ).astype(int)
    wp = off["vegas_wp"].fillna(off["wp"])
    off["neutral"] = (
        wp.between(NEUTRAL_WP_LO, NEUTRAL_WP_HI)
        & (off["half_seconds_remaining"] > NEUTRAL_MIN_HALF_SECONDS)
    )

    rows = []
    for (game_id, team), g in off.groupby(["game_id", "posteam"], sort=False):
        pass_g, rush_g = g[g["is_pass"] == 1], g[g["is_rush"] == 1]
        early = g[g["is_early_down"]]
        neut = g[g["neutral"]]
        top_secs = (
            g.drop_duplicates("fixed_drive")["drive_time_of_possession"].map(_parse_top).sum()
        )
        rows.append({
            "game_id": game_id,
            "team": team,
            "season": g["season"].iloc[0],
            "week": g["week"].iloc[0],
            "off_epa_play": g["epa"].mean(),
            "off_success_rate": g["success"].mean(),
            "off_pass_epa": pass_g["epa"].mean() if len(pass_g) else np.nan,
            "off_rush_epa": rush_g["epa"].mean() if len(rush_g) else np.nan,
            "off_early_down_epa": early["epa"].mean() if len(early) else np.nan,
            "explosive_play_rate": g["explosive"].mean(),
            "plays": len(g),
            "sec_per_play": (top_secs / len(g)) if len(g) else np.nan,
            "pass_rate": g["is_pass"].sum() / (g["is_pass"].sum() + g["is_rush"].sum()),
            "neutral_pass_rate": (
                neut["is_pass"].sum() / (neut["is_pass"].sum() + neut["is_rush"].sum())
                if len(neut) and (neut["is_pass"].sum() + neut["is_rush"].sum()) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _pressure_by_defense(pbp: pd.DataFrame) -> pd.Series:
    """Per (game_id, defteam) pressure rate: sacks + QB hits per opponent dropback."""
    db = pbp[pbp["qb_dropback"] == 1].copy()
    db["pressured"] = ((db["sack"] == 1) | (db["qb_hit"] == 1)).astype(int)
    return db.groupby(["game_id", "defteam"])["pressured"].mean()


def _redzone_td_pct(pbp: pd.DataFrame) -> pd.Series:
    """Per (game_id, posteam): of drives that reached the opponent 20, share that
    ended in a TD."""
    drives = pbp.drop_duplicates(["game_id", "fixed_drive"])
    rz = drives[drives["drive_inside20"] == 1]
    return rz.groupby(["game_id", "posteam"]).apply(
        lambda d: (d["fixed_drive_result"] == "Touchdown").mean(), include_groups=False
    )


def aggregate_team_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """pbp for one or more seasons -> one row per (game_id, team). REG + POST only."""
    pbp = pbp[pbp["season_type"].isin(["REG", "POST"])].copy()

    off = _offensive_plays(pbp)
    off_agg = _offense_by_team(off)
    pressure = _pressure_by_defense(pbp)
    rz = _redzone_td_pct(pbp)

    # team <-> opponent within each game, plus home/away
    meta = pbp.drop_duplicates("game_id").set_index("game_id")[["home_team", "away_team"]]

    out = []
    for game_id, g in off_agg.groupby("game_id", sort=False):
        if len(g) != 2:
            continue  # a game with only one team's offense charted -- skip rather than guess
        teams = g["team"].tolist()
        for i, team in enumerate(teams):
            opp = teams[1 - i]
            me = g[g["team"] == team].iloc[0]
            them = g[g["team"] == opp].iloc[0]
            home = meta.loc[game_id, "home_team"] if game_id in meta.index else None
            out.append({
                "game_id": game_id,
                "team": team,
                "opponent": opp,
                "season": int(me["season"]),
                "week": int(me["week"]),
                "is_home": int(team == home) if home is not None else None,
                "off_epa_play": me["off_epa_play"],
                "def_epa_play": them["off_epa_play"],
                "off_pass_epa": me["off_pass_epa"],
                "off_rush_epa": me["off_rush_epa"],
                "off_early_down_epa": me["off_early_down_epa"],
                "off_success_rate": me["off_success_rate"],
                "def_success_rate": them["off_success_rate"],
                "explosive_play_rate": me["explosive_play_rate"],
                "def_explosive_rate": them["explosive_play_rate"],
                "rz_td_pct": rz.get((game_id, team), np.nan),
                "pressure_rate_def": pressure.get((game_id, team), np.nan),
                "plays": int(me["plays"]),
                "sec_per_play": me["sec_per_play"],
                "pass_rate": me["pass_rate"],
                "neutral_pass_rate": me["neutral_pass_rate"],
            })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Rolling form (training-time feature engineering, task 8) -- mirrors the CFB
# build's box_score_features.py pattern, but team_game_epa is already one row
# per (game, team) with an `opponent` column, so no build_long_format() step is
# needed here -- that CFB module existed only to reshape wide home/away rows
# into this same long shape, which team_game_epa already has.
# --------------------------------------------------------------------------- #
def compute_rolling_epa_form(epa_df: pd.DataFrame, srs_lookup: dict, franchise_of) -> pd.DataFrame:
    """epa_df: team_game_epa rows (game_id, team, opponent, season, week, is_home,
    + ROLL_COLS). srs_lookup: {(season, week, franchise): srs} from
    opponent_adjustment.compute_weekly_srs, keyed on FRANCHISE ids.
    franchise_of: src.team_names.franchise_id -- rolling groups on franchise so a
    mid-window relocation (SD->LAC 2017, OAK->LV 2020) doesn't reset the trailing
    window; the roster carried over with the team.
    """
    df = epa_df.copy()
    df["franchise"] = df["team"].map(franchise_of)
    df["opp_franchise"] = df["opponent"].map(franchise_of)
    df = df.sort_values(["franchise", "season", "week"])

    for col in ROLL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"avg_{col}"] = (
            df.groupby("franchise")[col]
            .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean())
        )

    df["opponent_srs"] = df.apply(
        lambda r: srs_lookup.get((r["season"], r["week"], r["opp_franchise"])), axis=1
    )
    df["avg_opponent_srs"] = (
        df.groupby("franchise")["opponent_srs"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean())
    )
    return df


def assemble_epa_game_features(rolling_df: pd.DataFrame) -> pd.DataFrame:
    """Pivots the per-team rolling form back to one row per game with
    home_/away_/diff_ columns, matching the CFB build's assembly pattern."""
    feature_cols = [c for c in rolling_df.columns if c.startswith("avg_")]

    home = rolling_df[rolling_df["is_home"] == 1][["game_id"] + feature_cols].set_index("game_id")
    away = rolling_df[rolling_df["is_home"] == 0][["game_id"] + feature_cols].set_index("game_id")

    home = home.add_prefix("home_")
    away = away.add_prefix("away_")
    joined = home.join(away, how="inner")

    for col in feature_cols:
        joined[f"diff_{col}"] = joined[f"home_{col}"] - joined[f"away_{col}"]

    return joined.reset_index()
