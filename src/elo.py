"""ELO ratings for NFL franchises, ported from the CFB build's FiveThirtyEight-
style formula (margin-of-victory multiplier + logistic expected score), with
constants recalibrated for the NFL rather than blindly carried over:

- K=20 (vs CFB's 40): NFL rosters and coaching staffs are far more stable
  year-over-year than CFB's transfer-portal/graduation churn, and single-game
  variance shouldn't move a rating as fast. Still just a starting point --
  worth backtesting once the model exists (task 9).
- Season regression factor=0.80 (vs CFB's 0.6): pulls teams only 20% of the way
  to the mean at each season boundary (vs CFB's 40%), reflecting NFL's much
  lower roster turnover.
- Home advantage=40 ELO points (vs CFB's 65, an unvalidated placeholder there
  too): modern NFL home-field advantage is smaller than CFB's and has been
  trending down. **2020 downweight**: that season was played with no/minimal
  fans across the league -- HOME_ADVANTAGE_2020 defaults to 0 for regular-season
  games that year specifically (not a neutral-site game -- an empty-stadium
  game), rather than assuming the normal crowd-driven bonus still applied.
- Ratings are keyed on whatever string the caller passes in -- build_features.py
  passes `franchise_id(team)` (src/team_names.py) so a rating carries across a
  relocation (STL->LA, SD->LAC, OAK->LV) instead of resetting, since the roster
  and coaching staff are what these ratings are really tracking.
"""
from collections import defaultdict

K_FACTOR = 20
HOME_ADVANTAGE_ELO = 40       # placeholder pending backtesting, same caveat as the CFB build
HOME_ADVANTAGE_2020 = 0       # no/minimal fans leaguewide
SEASON_REGRESSION_FACTOR = 0.80
INITIAL_RATING = 1500.0
NO_FAN_SEASON = 2020


class NFLElo:
    def __init__(self, k: float = K_FACTOR, home_advantage: float = HOME_ADVANTAGE_ELO):
        self.k = k
        self.home_advantage = home_advantage
        self.ratings = defaultdict(lambda: INITIAL_RATING)
        self._current_season = None

    def get_rating(self, team) -> float:
        return self.ratings[team]

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def margin_multiplier(point_diff: float, elo_diff: float) -> float:
        mov = abs(point_diff)
        return ((mov + 3) ** 0.8) / (7.5 + 0.006 * abs(elo_diff))

    def _home_bonus(self, season: int, neutral_site: bool) -> float:
        if neutral_site:
            return 0.0
        return HOME_ADVANTAGE_2020 if season == NO_FAN_SEASON else self.home_advantage

    def maybe_regress_for_new_season(self, season):
        if self._current_season is not None and season != self._current_season:
            mean_elo = sum(self.ratings.values()) / len(self.ratings) if self.ratings else INITIAL_RATING
            for team in list(self.ratings.keys()):
                self.ratings[team] = (
                    SEASON_REGRESSION_FACTOR * self.ratings[team]
                    + (1 - SEASON_REGRESSION_FACTOR) * mean_elo
                )
        self._current_season = season

    def pre_game_features(self, home_team, away_team, season: int, neutral_site: bool) -> dict:
        """Call BEFORE update() for a game -- these are the pre-game (no-leakage) values."""
        home_bonus = self._home_bonus(season, neutral_site)
        r_home = self.get_rating(home_team) + home_bonus
        r_away = self.get_rating(away_team)
        return {
            "elo_home": self.get_rating(home_team),
            "elo_away": self.get_rating(away_team),
            "elo_diff": self.get_rating(home_team) - self.get_rating(away_team),
            "elo_expected_home": self.expected_score(r_home, r_away),
        }

    def update(self, home_team, away_team, home_points: int, away_points: int,
               season: int, neutral_site: bool):
        home_bonus = self._home_bonus(season, neutral_site)
        r_home = self.get_rating(home_team) + home_bonus
        r_away = self.get_rating(away_team)
        e_home = self.expected_score(r_home, r_away)
        s_home = 1.0 if home_points > away_points else (0.5 if home_points == away_points else 0.0)
        elo_diff = r_home - r_away
        m = self.margin_multiplier(home_points - away_points, elo_diff)
        self.ratings[home_team] = self.get_rating(home_team) + self.k * m * (s_home - e_home)
        self.ratings[away_team] = self.get_rating(away_team) + self.k * m * ((1 - s_home) - (1 - e_home))
