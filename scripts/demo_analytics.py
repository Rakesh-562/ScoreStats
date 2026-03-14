"""
scripts/demo_analytics.py
==========================

Standalone demonstration of the ScoreStat analytics engine.

No database or Flask application context is required — all computation
uses synthetic data so the script can be run immediately after cloning
the repository::

    python scripts/demo_analytics.py

Output format mirrors what the API would return, making it easy to cross-
reference with the live endpoints once the server is running.
"""

from __future__ import annotations

import math
import os
import sys

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.analytics_service import (
    BattingStats,
    BowlingStats,
    SquadComposition,
    TeamSelector,
    compute_batting_profile,
    compute_bowling_profile,
    gini_coefficient,
    shannon_entropy,
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_W = 62  # total line width
_RULE = "═" * _W
_THIN = "─" * _W


def _header(title: str) -> None:
    print(f"\n{_RULE}")
    print(f"  {title}")
    print(_RULE)


def _section(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {_THIN}")


def _row(label: str, value: str, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{label:<28}{value}")


def _print_batting(s: BattingStats) -> None:
    _row("Player",               s.player_name)
    _row("Innings / Total runs", f"{s.innings_played} inns  ·  {s.total_runs} runs")
    _row("Average",              f"{s.average:.2f}")
    _row("Strike rate",          f"{s.strike_rate:.1f}")
    _row("Boundary %",           f"{s.boundary_pct * 100:.1f}%")
    _row("Gini coefficient",     f"{s.gini:.4f}   (0 = consistent, 1 = erratic)")
    _row("Coeff. of variation",  f"{s.cv:.4f}")
    _row("Batting Index",        f"{s.batting_index:.4f}")
    label = (
        "consistent" if s.gini < 0.30
        else "moderate" if s.gini < 0.50
        else "erratic"
    )
    _row("Consistency label",    label)


def _print_bowling(s: BowlingStats) -> None:
    avg_str = f"{s.bowling_average:.2f}" if math.isfinite(s.bowling_average) else "—"
    _row("Player",               s.player_name)
    _row("Balls / Wickets",      f"{s.balls_bowled} balls  ·  {s.wickets} wkts")
    _row("Economy rate",         f"{s.economy_rate:.2f} rpo")
    _row("Bowling average",      avg_str)
    _row("Dot ball %",           f"{s.dot_pct * 100:.1f}%")
    _row("Shannon entropy (H)",  f"{s.entropy:.4f} bits")
    _row("Bowling Index",        f"{s.bowling_index:.4f}")


def _print_selection(selection: dict) -> None:
    _header("TEAM SELECTION  —  TOPSIS closeness coefficient")
    for group, players in selection.items():
        _section(group.upper())
        print(
            f"    {'Rank':<5} {'Player':<22} {'Closeness':>10} "
            f"{'Index':>8} {'XI':>4}"
        )
        print(f"    {_THIN}")
        for ps in players:
            tick = "✓" if ps.selected else " "
            print(
                f"    {ps.rank:<5} {ps.player_name:<22} {ps.closeness:>10.4f} "
                f"{ps.raw_index:>8.4f} {tick:>4}"
            )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

#
# Two contrasting batting profiles:
#   Kohli  — narrow score distribution (low Gini)  → consistent
#   Rohit  — wide distribution with outlier knocks  → explosive but erratic
#

kohli = BattingStats(
    player_id=1,
    player_name="Virat Kohli",
    innings_played=10,
    total_runs=480,
    balls_faced=360,
    dismissals=8,
    score_per_innings=[55, 48, 62, 70, 45, 52, 60, 38, 50, 0],
    boundary_balls=48,
    dot_balls_faced=120,
)

rohit = BattingStats(
    player_id=2,
    player_name="Rohit Sharma",
    innings_played=10,
    total_runs=423,
    balls_faced=310,
    dismissals=9,
    score_per_innings=[62, 5, 78, 3, 90, 0, 44, 55, 82, 4],
    boundary_balls=52,
    dot_balls_faced=100,
)

dhoni = BattingStats(
    player_id=3,
    player_name="MS Dhoni",
    innings_played=10,
    total_runs=310,
    balls_faced=200,
    dismissals=6,
    score_per_innings=[45, 38, 22, 60, 55, 30, 28, 10, 15, 7],
    boundary_balls=30,
    dot_balls_faced=60,
)

#
# Two contrasting bowling profiles:
#   Bumrah — dot-heavy, low entropy  → difficult to score off
#   Jadeja — more boundaries conceded, higher entropy
#

bumrah_outcomes = (
    ["dot"] * 40 + ["1"] * 12 + ["2"] * 5 + ["4"] * 6 + ["6"] * 2 + ["W"] * 7
)
jadeja_outcomes = (
    ["dot"] * 25 + ["1"] * 18 + ["2"] * 8 + ["4"] * 12 + ["6"] * 4 + ["W"] * 5
)

bumrah = BowlingStats(
    player_id=4,
    player_name="Jasprit Bumrah",
    balls_bowled=72,
    runs_conceded=148,
    wickets=7,
    ball_outcomes=bumrah_outcomes,
    wicket_overs=[2, 5, 8, 11, 14, 17, 19],
)

jadeja = BowlingStats(
    player_id=5,
    player_name="Ravindra Jadeja",
    balls_bowled=72,
    runs_conceded=192,
    wickets=5,
    ball_outcomes=jadeja_outcomes,
    wicket_overs=[3, 7, 12, 16, 18],
)

# Compute all profiles.
for b in (kohli, rohit, dhoni):
    compute_batting_profile(b)
for b in (bumrah, jadeja):
    compute_bowling_profile(b)

player_roles: dict[int, str] = {
    kohli.player_id:  "batsman",
    rohit.player_id:  "batsman",
    dhoni.player_id:  "wicket-keeper",
    bumrah.player_id: "bowler",
    jadeja.player_id: "all-rounder",
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_header("ScoreStat  —  Analytics Engine Demo")

_header("BATTING PROFILES")
for batter in (kohli, rohit, dhoni):
    _print_batting(batter)
    print(f"  {_THIN}")

print()
print("  Gini interpretation")
print(f"  {_THIN}")
for b in (kohli, rohit, dhoni):
    label = (
        "consistent" if b.gini < 0.30
        else "moderate" if b.gini < 0.50
        else "erratic"
    )
    print(f"    {b.player_name:<22}  Gini={b.gini:.4f}  CV={b.cv:.4f}  →  {label}")

_header("BOWLING PROFILES")
for bowler in (bumrah, jadeja):
    _print_bowling(bowler)
    print(f"  {_THIN}")

print()
print("  Entropy interpretation")
print(f"  {_THIN}")
for b in (bumrah, jadeja):
    print(
        f"    {b.player_name:<22}  H={b.entropy:.4f} bits  "
        f"dot%={b.dot_pct * 100:.0f}%"
    )

# TOPSIS selection with custom composition (same as default, shown explicitly).
comp = SquadComposition(batsmen=4, wicket_keepers=1, all_rounders=0, bowlers=2)
selection = TeamSelector.select_xi(
    batting_stats=[kohli, rohit, dhoni],
    bowling_stats=[bumrah, jadeja],
    player_roles=player_roles,
    composition=comp,
)
_print_selection(selection)

print()
print("  Method notes")
print(f"  {_THIN}")
print("    TOPSIS ranks each player by Euclidean distance from the ideal-best")
print("    and ideal-worst reference vectors across all normalised criteria.")
print("    Closeness → 1.0 means the player dominates on every criterion.")
print()
