"""Projected-starter identification and injury-burden / QB-availability signals.

Design notes (from what the data actually looks like):

- **Snap share is the starter signal.** A player's trailing snap share at their
  position IS the working definition of "starter". (nflverse depth charts were
  evaluated and dropped -- 40+ MB, no 2025 data, little added value.) Week 1 has
  no current-season snaps, so it falls back to the prior season's snap leaders.

- **Season-ending injuries vanish from the injury report.** A player moved to IR
  (e.g. Aaron Rodgers after Week 1, 2023) is not listed Out -- he's just gone.
  So "unavailable" = listed Out/Doubtful *OR* a former heavy-snap starter who has
  suddenly played ~0 snaps. `projected_starters` + a trailing-snap check catches
  both; the injury report alone does not.

Thresholds and position weights are module constants, tuned properly against
holdout performance in Task 8 -- treat the values here as starting points.
"""
from collections import defaultdict

STARTER_MIN_SHARE = 0.55      # trailing snap share to count as a starter at a position
STARTER_LOOKBACK_GAMES = 4
DROPPED_OUT_SHARE = 0.10      # a former starter now below this -> treat as unavailable (IR/benched)

REPORT_WEIGHT = {"Out": 1.0, "Doubtful": 0.6, "Questionable": 0.15}

# Rough positional importance for injury-burden weighting. QB dominates; then the
# trench / coverage spots Vegas actually moves lines on; skill players next.
POSITION_WEIGHT = defaultdict(lambda: 0.5, {
    "QB": 4.0,
    "T": 1.3, "G": 1.1, "C": 1.1, "OL": 1.2, "OT": 1.3, "OG": 1.1,
    "EDGE": 1.3, "DE": 1.2, "DT": 1.1, "OLB": 1.1, "DL": 1.1,
    "CB": 1.2, "S": 1.0, "SS": 1.0, "FS": 1.0, "DB": 1.1,
    "WR": 1.1, "RB": 0.9, "TE": 0.8, "LB": 0.9, "ILB": 0.9, "MLB": 0.9,
})


# --------------------------------------------------------------------------- #
# crosswalk + trailing snap share
# --------------------------------------------------------------------------- #
def gsis_pfr_maps(conn) -> tuple[dict, dict]:
    rows = conn.execute("SELECT gsis_id, pfr_id FROM players WHERE pfr_id IS NOT NULL").fetchall()
    g2p = {g: p for g, p in rows}
    p2g = {p: g for g, p in rows}
    return g2p, p2g


def _player_snap_rows(conn, pfr_id, season, before_week, limit):
    return conn.execute(
        """SELECT week, offense_pct, defense_pct FROM snap_counts
           WHERE pfr_player_id = ? AND season = ? AND week < ?
           ORDER BY week DESC LIMIT ?""",
        (pfr_id, season, before_week, limit),
    ).fetchall()


def trailing_snap_share(conn, pfr_id, season, before_week,
                        n_games=STARTER_LOOKBACK_GAMES) -> float | None:
    rows = _player_snap_rows(conn, pfr_id, season, before_week, n_games)
    if not rows:
        return None
    return sum(max(o or 0.0, d or 0.0) for _, o, d in rows) / len(rows)


# --------------------------------------------------------------------------- #
# projected starters
# --------------------------------------------------------------------------- #
def projected_starters(conn, season, week, team,
                       min_share=STARTER_MIN_SHARE,
                       n_games=STARTER_LOOKBACK_GAMES) -> dict:
    """{gsis_id: {'pfr_id','player','position','trailing_share'}} for players whose
    trailing snap share this season >= min_share. Falls back to the prior
    season's final games when this season has no snaps yet (week 1)."""
    _, p2g = gsis_pfr_maps(conn)

    # candidate players: anyone who took a snap for this team recently
    if week > 1:
        cand = conn.execute(
            """SELECT DISTINCT pfr_player_id, player, position FROM snap_counts
               WHERE team = ? AND season = ? AND week < ?""",
            (team, season, week),
        ).fetchall()
        lookup_season, lookup_before = season, week
    else:
        cand = conn.execute(
            """SELECT DISTINCT pfr_player_id, player, position FROM snap_counts
               WHERE team = ? AND season = ?""",
            (team, season - 1),
        ).fetchall()
        lookup_season, lookup_before = season - 1, 99

    starters = {}
    for pfr_id, player, position in cand:
        share = trailing_snap_share(conn, pfr_id, lookup_season, lookup_before, n_games)
        if share is not None and share >= min_share:
            gsis = p2g.get(pfr_id)
            if gsis:
                starters[gsis] = {
                    "pfr_id": pfr_id, "player": player,
                    "position": position, "trailing_share": round(share, 3),
                }
    return starters


def team_recent_weeks(conn, season, team, before_week, k=2) -> list[int]:
    """The k most recent weeks this team has snap data for, before `before_week`."""
    rows = conn.execute(
        """SELECT DISTINCT week FROM snap_counts
           WHERE team = ? AND season = ? AND week < ?
           ORDER BY week DESC LIMIT ?""",
        (team, season, before_week, k),
    ).fetchall()
    return [r[0] for r in rows]


def recent_availability(conn, pfr_id, season, team, before_week) -> str:
    """'active' / 'reduced' / 'absent' over the team's last 1-2 games.

    Absence of a snap_counts row means the player did not dress (IR, healthy
    scratch, suspension) -- the injury report hides these once a player is moved
    to IR, so this is how a season-ending injury like Burrow 2023 or Rodgers 2023
    actually shows up.
    """
    weeks = team_recent_weeks(conn, season, team, before_week, k=2)
    if not weeks:
        return "active"  # no basis to say otherwise (week 1)
    shares = []
    for wk in weeks:
        row = conn.execute(
            "SELECT offense_pct, defense_pct FROM snap_counts WHERE pfr_player_id=? AND season=? AND week=?",
            (pfr_id, season, wk),
        ).fetchone()
        shares.append(0.0 if row is None else max(row[0] or 0.0, row[1] or 0.0))
    if max(shares) <= DROPPED_OUT_SHARE:
        return "absent"
    if shares[0] <= DROPPED_OUT_SHARE:      # most recent game barely played
        return "reduced"
    return "active"


# --------------------------------------------------------------------------- #
# injury burden
# --------------------------------------------------------------------------- #
def _report_status(conn, season, week, team, gsis_id):
    row = conn.execute(
        "SELECT report_status FROM injuries WHERE season=? AND week=? AND team=? AND gsis_id=?",
        (season, week, team, gsis_id),
    ).fetchone()
    return row[0] if row else None


def injury_burden(conn, season, week, team) -> dict:
    """Position-weighted count of unavailable non-QB projected starters.

    Unavailable = listed Out/Doubtful on this week's report, OR a former starter
    who did not dress in the team's last 1-2 games (catches IR/scratch the report
    hides once a player moves to IR).

    QBs are deliberately excluded -- qb_situation() is the dedicated QB signal, and
    counting a hurt backup QB here while the real starter is healthy would
    double-model the position. Returns {'burden', 'out_starters', 'detail'}.
    """
    starters = projected_starters(conn, season, week, team)
    burden = 0.0
    out_list, detail = [], []
    for gsis, info in starters.items():
        if (info["position"] or "") == "QB":
            continue
        status = _report_status(conn, season, week, team, gsis)
        weight = REPORT_WEIGHT.get(status, 0.0)
        reason = status
        if weight < 1.0 and week > 1:
            avail = recent_availability(conn, info["pfr_id"], season, team, week)
            if avail == "absent":
                weight, reason = 1.0, "not_dressed"          # IR / scratch the report hides
            elif avail == "reduced" and weight < 0.6:
                weight, reason = 0.6, "reduced_role"
        if weight == 0.0:
            continue
        pos_w = POSITION_WEIGHT[info["position"] or ""]
        contrib = weight * pos_w * info["trailing_share"]
        burden += contrib
        entry = {"player": info["player"], "position": info["position"],
                 "reason": reason, "share": info["trailing_share"], "contribution": round(contrib, 3)}
        detail.append(entry)
        if reason in ("Out", "not_dressed"):
            out_list.append(info["player"])
    detail.sort(key=lambda d: -d["contribution"])
    return {"burden": round(burden, 3), "out_starters": out_list, "detail": detail}


# --------------------------------------------------------------------------- #
# QB availability
# --------------------------------------------------------------------------- #
def primary_qb(conn, season, through_week, team) -> str | None:
    """gsis_id of the team's main QB this season = most QB offensive snaps
    through `through_week` (exclusive)."""
    _, p2g = gsis_pfr_maps(conn)
    row = conn.execute(
        """SELECT pfr_player_id, SUM(offense_snaps) s FROM snap_counts
           WHERE team=? AND season=? AND week < ? AND position='QB'
           GROUP BY pfr_player_id ORDER BY s DESC LIMIT 1""",
        (team, season, through_week),
    ).fetchone()
    if not row:
        return None
    return p2g.get(row[0])


def qb_situation(conn, season, week, team, projected_qb_gsis: str | None) -> dict:
    """projected_qb_gsis is games.{home,away}_qb_id for this matchup. Flags a
    non-primary starter (backup / just-traded-for / rookie taking over)."""
    primary = primary_qb(conn, season, week, team)
    backup_starting = (
        projected_qb_gsis is not None and primary is not None
        and projected_qb_gsis != primary
    )
    # how established is the projected guy? (trailing snap share this season)
    proj_share = None
    if projected_qb_gsis:
        g2p, _ = gsis_pfr_maps(conn)
        pfr = g2p.get(projected_qb_gsis)
        if pfr:
            proj_share = trailing_snap_share(conn, pfr, season, week, n_games=4)
    return {
        "primary_qb_gsis": primary,
        "projected_qb_gsis": projected_qb_gsis,
        "backup_starting": bool(backup_starting),
        "projected_qb_trailing_share": proj_share,
        "week1_or_unknown": primary is None,
    }
