"""Bet sizing for the spread pick -- replaced the original sizing on 2026-09-25.

Ported from the CFB project's own src/bet_sizing.py (`c6abd9c`), but with the
calibration REFIT on NFL data. CFB's constants describe CFB's edges; copying them
would have been the exact mistake the cross-project rule warns about.

The original cover probability, Phi(edge / regressor RMSE), treated the model's
disagreement with the market as if the market carried no information of its own, so
it badly overstated the chance of covering. Measured on the 35 graded 2026 picks it
had produced by 2026-09-25: it predicted 61.3% cover against 37.1% actual, and its
ordering was inverted -- the picks it rated 60-65% covered 20% (n=5) while those it
rated under 55% covered 45% (n=11). The $500 paper bankroll sized off it fell to
$369.54 over 36 picks.

cover_probability() is a logistic fit of "did the pick cover" on |edge| over the 2025
holdout's 284 decided spread picks (scripts/train_model.py's HOLDOUT_SEASON), the same
shape CFB uses. Refit with scripts/fit_bet_sizing.py whenever the model is retrained.

**What the fit actually found, and why nothing is staked right now (2026-09-25).**
On the 2025 holdout, |edge| carries no information about covering: cover rate by edge
bucket is 51.4% / 44.7% / 51.2% / 55.6% (0-3 / 3-6 / 6-10 / 10+ points, n=140/94/41/9),
and corr(|edge|, covered) = +0.006 (p=0.93). So the fitted slope is near zero
(0.0040) and the calibrated probability is ~49-50% for every pick regardless of edge
size -- below the 52.4% break-even at -110, so kelly_fraction() returns 0 and every
pick is staked at nothing. That is the honest state of this model, not a bug in the
port: on the training seasons the same edges "cover" 83.8% rising to 100% at 10+
points, which is the margin regressor having memorized training margins, and it does
not survive out of sample (2025: 49.3%, 2026 live: 40.6%).

Picks and their explanations still publish as analysis -- they are just not staked.
Sizing turns back on by itself, with no code change, if a future refit finds a real
relationship between edge and covering (the logistic will then clear break-even at
some edge and Kelly will size it).

**Deliberately NOT ported from CFB**: its blowout no-bet rule. That rule exists
because CFB's regressor takes the underdog in ~2/3 of games with |market_spread| >= 21
and the NIL/portal-era market shifted under it. NFL spreads that large are rare, and
NFL's problem is not confined to a bucket -- the edge signal is flat everywhere -- so
importing a CFB-shaped patch would add a rule this data does not support.
"""
import math

# Fit 2026-09-25 by scripts/fit_bet_sizing.py on the 2025 holdout (284 decided picks).
CALIBRATION_INTERCEPT = -0.042321
CALIBRATION_SLOPE = 0.003951
KELLY_MULTIPLIER = 0.25
MAX_BET_FRACTION = 0.02  # ceiling on any single stake, unreachable while the fit is flat
# The paper bankroll restarted at $500 with this sizing: only picks whose game kicks off
# at or after this date count toward it (user decision 2026-09-25). Compared against
# games.gameday, which is a plain 'YYYY-MM-DD' US-Eastern date string.
BANKROLL_RESTART_GAMEDAY = "2026-09-27"
NO_BET_REASON = (
    "not staked: on the 2025 holdout this model's edge size had no relationship to "
    "covering (corr +0.006, p=0.93), so the calibrated cover probability sits below the "
    "break-even a -110 price needs. The pick and its reasoning still stand as analysis"
)


def american_odds_to_net_decimal(odds: int) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100


def break_even(odds: int) -> float:
    return 1 / (1 + american_odds_to_net_decimal(odds))


def cover_probability(edge: float | None) -> float | None:
    """Calibrated probability that the spread pick covers, from the size of its edge.
    The pick is always the side the edge points to, so only |edge| matters."""
    if edge is None:
        return None
    z = CALIBRATION_INTERCEPT + CALIBRATION_SLOPE * abs(edge)
    return 1 / (1 + math.exp(-z))


def kelly_fraction(p_cover: float | None, odds: int) -> float | None:
    """Fraction of bankroll to stake: 25% Kelly, capped, 0 when p_cover is below break-even."""
    if p_cover is None:
        return None
    b = american_odds_to_net_decimal(odds)
    full = p_cover - (1 - p_cover) / b
    return min(max(0.0, full) * KELLY_MULTIPLIER, MAX_BET_FRACTION)


def size_bet(edge: float | None, odds: int) -> tuple[float | None, float | None]:
    p = cover_probability(edge)
    return p, kelly_fraction(p, odds)
