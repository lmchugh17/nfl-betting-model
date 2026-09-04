"""Turns a game's model prediction + feature values into the grounded facts
that drive a pick explanation.

Only the deterministic half lives here: extract the actual top-contributing
features for THIS specific game via XGBoost's native SHAP (pred_contribs) on
the margin regressor, then render each into a factual sentence using the
game's real numbers. No LLM call in this module -- these facts are true by
construction. `get_shap_contributions` and `build_feature_highlights` are
ported from the CFB build verbatim (sport-agnostic); only `_describe_feature`
is NFL-specific.

The prose-writing half (turning these facts into a TL;DR + bullets) is done
natively by the Claude Code session that runs the weekly pipeline, not by a
separate Anthropic API call -- same as the CFB build. See scripts/predict_games.py
(task 11) for how these highlights get used.

Row-dict contract: `row` is a game_features row (dict) merged with a few
lookups scripts/predict_games.py already has on hand:
  - home_qb_name / away_qb_name        (games.home_qb_name/away_qb_name)
  - temp / wind                        (games.temp/wind)
  - home_ats_count / away_ats_count, home_adverse_wx_ats_pct /
    home_adverse_wx_ats_count / away_adverse_wx_ats_pct / away_adverse_wx_ats_count
    (all explanation-only columns build_features.py already writes to
    game_features alongside the model features)
Optional, richer-if-present:
  - home_injury_out / away_injury_out  (list[str] of player names -- from
    src.availability.injury_burden()['out_starters']; falls back to a generic
    sentence using just the burden score if omitted)
"""
import pandas as pd
import xgboost as xgb

from src.weather_features import ADVERSE_TEMP_F, ADVERSE_WIND_MPH

TOP_N_FACTORS = 5

# Rolling EPA/situational stats where a HIGHER value is better for the team
# that has it. Defensive "allowed" stats are the opposite (lower = better
# defense) -- see LOWER_IS_BETTER below. Stats not in either set (pace,
# pass-rate, opponent strength) don't have a universal "better" direction and
# get their own dedicated phrasing instead of the generic leader/trailer frame.
HIGHER_IS_BETTER = {
    "off_epa_play", "off_pass_epa", "off_rush_epa", "off_early_down_epa",
    "off_success_rate", "explosive_play_rate", "rz_td_pct", "pressure_rate_def",
}
LOWER_IS_BETTER = {"def_epa_play", "def_success_rate", "def_explosive_rate"}

READABLE_STAT = {
    "off_epa_play": "offensive EPA per play", "def_epa_play": "defensive EPA per play allowed",
    "off_pass_epa": "passing EPA per play", "off_rush_epa": "rushing EPA per play",
    "off_early_down_epa": "early-down EPA per play", "off_success_rate": "offensive success rate",
    "def_success_rate": "defensive success rate allowed", "explosive_play_rate": "explosive-play rate",
    "def_explosive_rate": "explosive-play rate allowed", "rz_td_pct": "red-zone touchdown rate",
    "pressure_rate_def": "pass-rush pressure rate",
}


def get_shap_contributions(regressor, X_row: pd.DataFrame, feature_columns: list) -> dict:
    """Per-game SHAP contributions toward the predicted margin, via XGBoost's
    built-in TreeSHAP (no separate `shap` package needed)."""
    dmat = xgb.DMatrix(X_row[feature_columns])
    contribs = regressor.get_booster().predict(dmat, pred_contribs=True)[0]
    return dict(zip(feature_columns, contribs[:-1]))  # last column is the bias term


def _describe_weather_condition(row: dict) -> str:
    """Names whichever adverse-weather threshold(s) this game's forecast actually
    crossed -- thresholds match src.weather_features so the description never
    drifts from what the feature itself was gated on. No precipitation branch
    (unlike the CFB build): nflverse carries no historical precip field at all,
    see src/weather_features.py."""
    temp, wind = row.get("temp"), row.get("wind")
    parts = []
    if temp is not None and temp <= ADVERSE_TEMP_F:
        parts.append(f"{temp:.0f}°F temperatures")
    if wind is not None and wind >= ADVERSE_WIND_MPH:
        parts.append(f"{wind:.0f} mph wind")
    return " and ".join(parts) if parts else "adverse weather"


def _describe_epa_diff(feature: str, row: dict, home_team: str, away_team: str) -> str | None:
    stat = feature.replace("diff_avg_", "")
    home_val, away_val, diff = row.get(f"home_avg_{stat}"), row.get(f"away_avg_{stat}"), row.get(feature)
    if home_val is None or away_val is None or not pd.notna(diff):
        return None

    if stat == "opponent_srs":
        tougher, easier = (home_team, away_team) if diff > 0 else (away_team, home_team)
        tougher_val, easier_val = max(home_val, away_val), min(home_val, away_val)
        return (f"{tougher} has faced tougher recent competition than {easier} "
                f"(average opponent rating {tougher_val:+.1f} vs {easier_val:+.1f}).")
    if stat == "sec_per_play":
        faster, slower = (home_team, away_team) if home_val < away_val else (away_team, home_team)
        return f"{faster} has been playing at a faster pace recently ({min(home_val, away_val):.1f}s vs {max(home_val, away_val):.1f}s per play)."
    if stat in ("pass_rate", "neutral_pass_rate"):
        more, less = (home_team, away_team) if diff > 0 else (away_team, home_team)
        label = "in neutral game script" if stat == "neutral_pass_rate" else "overall"
        return f"{more} has leaned more pass-heavy {label} recently ({max(home_val, away_val):.0%} vs {min(home_val, away_val):.0%})."

    if stat in LOWER_IS_BETTER:
        leader, trailer, l_val, t_val = (
            (home_team, away_team, home_val, away_val) if home_val < away_val
            else (away_team, home_team, away_val, home_val)
        )
    elif stat in HIGHER_IS_BETTER:
        leader, trailer, l_val, t_val = (
            (home_team, away_team, home_val, away_val) if diff > 0
            else (away_team, home_team, away_val, home_val)
        )
    else:
        return None
    readable = READABLE_STAT.get(stat, stat)
    fmt = "{:.0%}" if "rate" in stat or "pct" in stat else "{:+.3f}" if "epa" in stat else "{:.2f}"
    return f"{leader} has the edge in recent {readable} ({fmt.format(l_val)} vs {fmt.format(t_val)}, last 8 games)."


def _describe_feature(feature: str, row: dict, home_team: str, away_team: str) -> str | None:
    """Renders one feature into a grounded, factual sentence using this game's real values."""
    g = lambda k, default=None: row.get(k, default)

    if feature == "elo_diff":
        diff = g("elo_diff")
        is_home_leader = diff > 0
        leader, trailer = (home_team, away_team) if is_home_leader else (away_team, home_team)
        leader_elo, trailer_elo = (
            (g("elo_home"), g("elo_away")) if is_home_leader else (g("elo_away"), g("elo_home"))
        )
        return (f"{leader} carries a {abs(diff):.0f}-point ELO&Dagger; advantage over {trailer} "
                f"({leader_elo:.0f} vs {trailer_elo:.0f}).")
    if feature == "elo_expected_home":
        return f"Pre-game ELO&Dagger; gives {home_team} a {g('elo_expected_home'):.0%} win probability."
    if feature == "srs_diff":
        diff = g("srs_diff")
        is_home_leader = diff > 0
        leader, trailer = (home_team, away_team) if is_home_leader else (away_team, home_team)
        leader_srs, trailer_srs = (
            (g("srs_home"), g("srs_away")) if is_home_leader else (g("srs_away"), g("srs_home"))
        )
        return (f"{leader} has been the better team once schedule strength is accounted for, "
                f"outrating {trailer} by {abs(diff):.1f} points of opponent-adjusted margin "
                f"({leader_srs:+.1f} vs {trailer_srs:+.1f}).")
    if feature in ("home_ats_pct", "away_ats_pct"):
        team = home_team if feature == "home_ats_pct" else away_team
        pct = g(feature)
        n = g("home_ats_count" if feature == "home_ats_pct" else "away_ats_count")
        if not pd.notna(pct) or not n:
            return None
        return f"{team} has covered the spread in {pct:.0%} of its last {int(n)} game{'s' if n != 1 else ''}."
    if feature in ("home_rest_days", "away_rest_days"):
        team = home_team if feature == "home_rest_days" else away_team
        days = g(feature)
        if days is None:
            return None
        bye = g("home_off_bye" if feature == "home_rest_days" else "away_off_bye")
        short = g("home_short_week" if feature == "home_rest_days" else "away_short_week")
        if bye:
            return f"{team} enters off a bye week ({days:.0f} days rest)."
        if short:
            return f"{team} is on a short week ({days:.0f} days rest)."
        return f"{team} has {days:.0f} days of rest."
    if feature in ("home_off_bye", "away_off_bye", "home_short_week", "away_short_week"):
        return None  # already folded into the rest_days sentence above
    if feature == "h2h_avg_home_margin" and g("h2h_meetings", 0):
        margin = g("h2h_avg_home_margin")
        favored, other = (home_team, away_team) if margin > 0 else (away_team, home_team)
        return (f"Over their last {int(g('h2h_meetings'))} meetings, {favored} has outscored {other} "
                f"by an average of {abs(margin):.1f} points.")
    if feature in ("h2h_current_season_margin", "h2h_current_season_meeting"):
        margin = g("h2h_current_season_margin")
        if not g("h2h_current_season_meeting") or margin is None:
            return None
        winner, loser = (home_team, away_team) if margin > 0 else (away_team, home_team)
        return f"These teams already met this season -- {winner} beat {loser} by {abs(margin):.0f}."
    if feature == "is_adverse_weather":
        if not g("is_adverse_weather"):
            return None
        return f"This game is forecast for {_describe_weather_condition(row)}."
    if feature == "adverse_wx_ats_edge":
        edge = g("adverse_wx_ats_edge")
        if not g("is_adverse_weather") or not edge:
            return None
        home_pct, away_pct = g("home_adverse_wx_ats_pct"), g("away_adverse_wx_ats_pct")
        home_n, away_n = g("home_adverse_wx_ats_count"), g("away_adverse_wx_ats_count")
        if home_pct is None or away_pct is None:
            return None
        leader, trailer = (home_team, away_team) if edge > 0 else (away_team, home_team)
        leader_pct, trailer_pct = (home_pct, away_pct) if edge > 0 else (away_pct, home_pct)
        leader_n, trailer_n = (home_n, away_n) if edge > 0 else (away_n, home_n)
        return (f"In its last {int(leader_n)} game{'s' if leader_n != 1 else ''} with "
                f"{_describe_weather_condition(row)}, {leader} has covered the spread {leader_pct:.0%} "
                f"of the time, versus {trailer}'s {trailer_pct:.0%} over its last {int(trailer_n)} "
                f"such game{'s' if trailer_n != 1 else ''}.")
    if feature in ("home_new_hc", "away_new_hc"):
        team = home_team if feature == "home_new_hc" else away_team
        if not g(feature):
            return None
        return f"{team} is playing under a new head coach this season."
    if feature == "div_game":
        return "This is a divisional matchup." if g("div_game") else None
    if feature == "is_primetime":
        return "This is a prime-time game." if g("is_primetime") else None
    if feature == "is_international":
        return f"This game is played internationally, at {g('stadium', 'a neutral site')}." if g("is_international") else None
    if feature in ("home_tz_shift_hours", "away_tz_shift_hours", "away_cross_country_travel"):
        shift = g("away_tz_shift_hours")
        if shift is None or abs(shift) < 2:
            return None
        direction = "east" if shift > 0 else "west"
        return f"{away_team} is traveling {direction} across {abs(shift):.0f}+ time zones for this game."
    if feature in ("home_injury_burden", "away_injury_burden"):
        team = home_team if feature == "home_injury_burden" else away_team
        burden = g(feature)
        if not burden:
            return None
        out = g("home_injury_out" if feature == "home_injury_burden" else "away_injury_out")
        if out:
            names = ", ".join(out[:3]) + (f" and {len(out) - 3} other{'s' if len(out) - 3 != 1 else ''}" if len(out) > 3 else "")
            return f"{team} is missing key contributors: {names}."
        return f"{team} is dealing with more starter unavailability than usual this week."
    if feature in ("home_qb_backup_starting", "away_qb_backup_starting"):
        is_home = feature == "home_qb_backup_starting"
        if not g(feature):
            return None
        team = home_team if is_home else away_team
        qb_name = g("home_qb_name" if is_home else "away_qb_name")
        share = g("home_qb_trailing_share" if is_home else "away_qb_trailing_share")
        if qb_name is None:
            return f"{team} is not starting its usual quarterback this week."
        if share is not None:
            return f"{team} starts {qb_name} at quarterback, not its usual starter (only {share:.0%} of recent offensive snaps)."
        return f"{team} starts {qb_name} at quarterback, not its usual starter."
    if feature in ("home_qb_trailing_share", "away_qb_trailing_share"):
        return None  # folded into the backup-starting sentence above when that's what's driving it
    if feature.startswith("diff_avg_"):
        return _describe_epa_diff(feature, row, home_team, away_team)
    return None


def build_feature_highlights(row: dict, contributions: dict, home_team: str, away_team: str,
                              top_n: int = TOP_N_FACTORS) -> list[str]:
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    highlights = []
    for feature, _ in ranked:
        sentence = _describe_feature(feature, row, home_team, away_team)
        if sentence:
            highlights.append(sentence)
        if len(highlights) >= top_n:
            break
    return highlights
