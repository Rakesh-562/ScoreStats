"""
app/services/analytics_service.py
==================================

Analytics engine for ScoreStat.

Responsibility
--------------
This module owns three distinct layers:

1. **Pure mathematics** — stateless functions with no I/O side-effects.
   These can be unit-tested without a database or Flask context.

2. **Domain data-classes** — immutable value objects that carry raw inputs
   and their computed derivatives.  ``field(init=False)`` marks every derived
   attribute so the caller cannot accidentally supply a pre-computed value.

3. **AnalyticsService** — a thin, DB-aware façade that translates SQLAlchemy
   rows into domain objects, delegates computation to the pure layer, and
   returns results to the route layer.  It never touches ``flask.request``.

Statistical methods
-------------------
Batting
    Gini coefficient
        Measures inequality in a batter's score distribution.
        Borrowed from welfare economics; here G ≈ 0 means the batter scores
        similarly every innings (consistent), G ≈ 1 means one outlier knock
        surrounded by ducks (erratic).

    Coefficient of Variation  (CV = σ / μ)
        Dimensionless dispersion.  Complements Gini by capturing spread
        relative to the mean rather than cumulative inequality.

    Batting Index
        Weighted composite normalised to [0, 1]:
        BI = 0.35·(avg/100) + 0.25·(SR/200) + 0.25·(1−G) + 0.15·boundary%

Bowling
    Shannon entropy  (H = −Σ pᵢ log₂ pᵢ)
        Measures unpredictability of ball outcomes.  Low H with a high dot
        fraction is ideal; high H driven by boundaries is bad.

    Dot-entropy score
        Combined signal: sqrt(dot_fraction) × (1 − H / log₂8).

    Bowling Index
        BI = 0.35·(1−eco_norm) + 0.40·(1−avg_norm) + 0.25·dot_entropy_score

Team selection
    TOPSIS  (Hwang & Yoon, 1981)
        Multi-criteria decision method from operations research.
        Each player's Euclidean distance to the ideal-best and ideal-worst
        reference vectors determines a closeness coefficient C ∈ [0, 1].
        C → 1 means the player dominates on every criterion.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Final, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Roles that contribute to batting pool in team selection.
BATTING_ROLES: Final[frozenset[str]] = frozenset(
    {"batsman", "wicket-keeper", "all-rounder"}
)

#: Roles that contribute to bowling pool in team selection.
BOWLING_ROLES: Final[frozenset[str]] = frozenset({"bowler", "all-rounder"})

#: Wicket types that credit the bowler (run-outs do not).
BOWLER_WICKET_TYPES: Final[frozenset[str]] = frozenset(
    {"bowled", "caught", "lbw", "stumped", "hit_wicket", "hit-wicket", "caught_and_bowled"}
)

#: Normalisation ceiling for batting average.
_AVG_CAP: Final[float] = 100.0

#: Normalisation ceiling for strike rate (T20 upper bound).
_SR_CAP: Final[float] = 200.0

#: Normalisation ceiling for economy rate (worst-case T20).
_ECO_CAP: Final[float] = 12.0

#: Normalisation ceiling for bowling average.
_BOWL_AVG_CAP: Final[float] = 50.0

#: Entropy normalisation base (8 distinct outcome categories).
_ENTROPY_BASE: Final[float] = math.log2(8)

# Batting Index weights — must sum to 1.
_BI_W_AVG: Final[float] = 0.35
_BI_W_SR: Final[float] = 0.25
_BI_W_CONSISTENCY: Final[float] = 0.25
_BI_W_BOUNDARY: Final[float] = 0.15

# Bowling Index weights — must sum to 1.
_BWI_W_ECO: Final[float] = 0.35
_BWI_W_AVG: Final[float] = 0.40
_BWI_W_ENTROPY: Final[float] = 0.25

# TOPSIS criteria weights (batting and bowling share the same structure).
_TOPSIS_WEIGHTS: Final[list[float]] = [0.40, 0.25, 0.20, 0.15]
_TOPSIS_HIGHER_IS_BETTER: Final[list[bool]] = [True, True, True, True]

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class AnalyticsError(Exception):
    """Base class for all analytics-layer errors."""


class InsufficientDataError(AnalyticsError):
    """Raised when a player or innings has no ball-level data to analyse."""


class PlayerNotFoundError(AnalyticsError):
    """Raised when a requested player_id does not exist in the database."""


class InningsNotFoundError(AnalyticsError):
    """Raised when a requested innings_id does not exist in the database."""


class MatchNotFoundError(AnalyticsError):
    """Raised when a requested match_id does not exist in the database."""


class TeamNotFoundError(AnalyticsError):
    """Raised when a requested team_id does not exist in the database."""


# ---------------------------------------------------------------------------
# Domain data-classes
# ---------------------------------------------------------------------------


@dataclass
class BattingStats:
    """
    Carries raw batting inputs and their computed derivatives.

    Attributes marked ``field(init=False)`` are populated by
    :func:`compute_batting_profile`.  Callers must not set them directly.
    """

    player_id: int
    player_name: str
    innings_played: int
    total_runs: int
    balls_faced: int
    dismissals: int
    score_per_innings: list[int]  #: runs in each individual innings
    boundary_balls: int           #: balls that produced a boundary (4 or 6)
    dot_balls_faced: int

    # --- derived (populated by compute_batting_profile) ---
    average: float = field(init=False, default=0.0)
    strike_rate: float = field(init=False, default=0.0)
    gini: float = field(init=False, default=0.0)
    cv: float = field(init=False, default=0.0)
    boundary_pct: float = field(init=False, default=0.0)
    batting_index: float = field(init=False, default=0.0)


@dataclass
class BowlingStats:
    """
    Carries raw bowling inputs and their computed derivatives.

    Attributes marked ``field(init=False)`` are populated by
    :func:`compute_bowling_profile`.
    """

    player_id: int
    player_name: str
    balls_bowled: int
    runs_conceded: int
    wickets: int
    ball_outcomes: list[str]  #: per-ball label: "dot"|"1"|"2"|"4"|"6"|"W"|"extra"
    wicket_overs: list[int]   #: over number of each wicket taken

    # --- derived ---
    economy_rate: float = field(init=False, default=0.0)
    bowling_average: float = field(init=False, default=0.0)
    dot_pct: float = field(init=False, default=0.0)
    entropy: float = field(init=False, default=0.0)
    bowling_index: float = field(init=False, default=0.0)


@dataclass
class PlayerScore:
    """
    TOPSIS output for one player.

    ``closeness`` is the relative closeness coefficient C ∈ [0, 1].
    C → 1 means the player is closest to the ideal reference vector.
    """

    player_id: int
    player_name: str
    role: str
    closeness: float   #: relative closeness coefficient
    raw_index: float   #: batting_index or bowling_index, whichever applies
    rank: int = field(default=0)
    selected: bool = field(default=False)


# ---------------------------------------------------------------------------
# Pure mathematical functions
# ---------------------------------------------------------------------------


def gini_coefficient(values: Sequence[float]) -> float:
    """
    Gini coefficient of a non-negative distribution.

    .. math::
        G = \\frac{2 \\sum_{i=1}^{n} i \\cdot x_i}{n \\sum_{i=1}^{n} x_i} - \\frac{n+1}{n}

    where :math:`x_i` are the values sorted in ascending order and
    :math:`i` is the 1-based rank.

    Returns
    -------
    float
        Value in [0, 1].  Returns 0.0 for empty or all-zero sequences.

    Notes
    -----
    Applied to a batter's innings scores:

    * G ≈ 0 → scores are uniformly distributed (consistent batter)
    * G ≈ 1 → one large outlier score, remainder near zero (erratic batter)
    """
    xs = sorted(max(0.0, v) for v in values)
    n = len(xs)
    total = sum(xs)

    if n == 0 or total == 0.0:
        return 0.0

    weighted_sum = sum(rank * x for rank, x in enumerate(xs, start=1))
    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


def shannon_entropy(outcomes: Sequence[str]) -> float:
    """
    Shannon entropy of a categorical outcome distribution.

    .. math::
        H = -\\sum_{k} p_k \\log_2 p_k

    Returns
    -------
    float
        Entropy in bits.  Returns 0.0 for an empty sequence.

    Notes
    -----
    Applied to a bowler's per-ball outcomes, low entropy combined with a
    high dot-ball fraction indicates tight, difficult-to-score-off bowling.
    """
    if not outcomes:
        return 0.0

    n = len(outcomes)
    counts = Counter(outcomes)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def dot_entropy_score(outcomes: Sequence[str]) -> float:
    """
    Composite score in [0, 1] that rewards dot-heavy, low-boundary bowling.

    .. math::
        S = \\sqrt{f_{\\text{dot}}} \\times \\left(1 - \\frac{H}{\\log_2 8}\\right)

    The square-root term dampens extreme dot fractions; the entropy term
    penalises the unpredictability that comes from conceding boundaries.

    Returns
    -------
    float
        0.0 for an empty sequence.
    """
    if not outcomes:
        return 0.0

    dot_fraction = outcomes.count("dot") / len(outcomes)
    normalised_h = shannon_entropy(outcomes) / _ENTROPY_BASE
    return math.sqrt(dot_fraction) * (1.0 - normalised_h)


def coefficient_of_variation(values: Sequence[float]) -> float:
    """
    Population coefficient of variation (CV = σ / μ).

    Dimensionless — enables comparison across players with different means.

    Returns
    -------
    float
        0.0 when the sequence is empty or the mean is zero.
    """
    n = len(values)
    if n == 0:
        return 0.0

    mu = sum(values) / n
    if mu == 0.0:
        return 0.0

    variance = sum((x - mu) ** 2 for x in values) / n
    return math.sqrt(variance) / mu


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------


def compute_batting_profile(stats: BattingStats) -> BattingStats:
    """
    Compute and store all derived batting metrics on *stats* in-place.

    **Batting Index formula**::

        BI = 0.35 · (avg / 100)
           + 0.25 · (SR  / 200)
           + 0.25 · (1 − Gini)
           + 0.15 · boundary_pct

    Each component is normalised to [0, 1] before weighting so the
    contributions are comparable regardless of scale.

    Parameters
    ----------
    stats:
        A fully initialised :class:`BattingStats` instance.

    Returns
    -------
    BattingStats
        The same object, with derived fields populated.
    """
    stats.average = (
        stats.total_runs / stats.dismissals
        if stats.dismissals > 0
        else float(stats.total_runs)
    )
    stats.strike_rate = (
        (stats.total_runs / stats.balls_faced) * 100.0
        if stats.balls_faced > 0
        else 0.0
    )
    stats.gini = gini_coefficient(stats.score_per_innings)
    stats.cv = coefficient_of_variation([float(s) for s in stats.score_per_innings])
    stats.boundary_pct = (
        stats.boundary_balls / stats.balls_faced if stats.balls_faced > 0 else 0.0
    )

    avg_norm = min(stats.average, _AVG_CAP) / _AVG_CAP
    sr_norm = min(stats.strike_rate, _SR_CAP) / _SR_CAP
    consistency = 1.0 - stats.gini

    stats.batting_index = (
        _BI_W_AVG * avg_norm
        + _BI_W_SR * sr_norm
        + _BI_W_CONSISTENCY * consistency
        + _BI_W_BOUNDARY * stats.boundary_pct
    )
    return stats


def compute_bowling_profile(stats: BowlingStats) -> BowlingStats:
    """
    Compute and store all derived bowling metrics on *stats* in-place.

    **Bowling Index formula**::

        BI = 0.35 · (1 − eco_norm)
           + 0.40 · (1 − avg_norm)
           + 0.25 · dot_entropy_score

    where::

        eco_norm  = economy_rate  / 12   (12 rpo ≈ worst-case T20)
        avg_norm  = bowling_avg   / 50   (50 runs/wkt ≈ very expensive)

    A higher Bowling Index always indicates a better bowler.

    Parameters
    ----------
    stats:
        A fully initialised :class:`BowlingStats` instance.

    Returns
    -------
    BowlingStats
        The same object, with derived fields populated.
    """
    overs = stats.balls_bowled / 6.0
    stats.economy_rate = stats.runs_conceded / overs if overs > 0.0 else 0.0
    stats.bowling_average = (
        stats.runs_conceded / stats.wickets
        if stats.wickets > 0
        else math.inf
    )
    stats.dot_pct = (
        stats.ball_outcomes.count("dot") / len(stats.ball_outcomes)
        if stats.ball_outcomes
        else 0.0
    )
    stats.entropy = shannon_entropy(stats.ball_outcomes)

    eco_norm = min(stats.economy_rate, _ECO_CAP) / _ECO_CAP
    finite_avg = stats.bowling_average if math.isfinite(stats.bowling_average) else _BOWL_AVG_CAP
    avg_norm = min(finite_avg, _BOWL_AVG_CAP) / _BOWL_AVG_CAP
    de_score = dot_entropy_score(stats.ball_outcomes)

    stats.bowling_index = (
        _BWI_W_ECO * (1.0 - eco_norm)
        + _BWI_W_AVG * (1.0 - avg_norm)
        + _BWI_W_ENTROPY * de_score
    )
    return stats


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------


def topsis(
    players: list[tuple[int, str, str, list[float]]],
    weights: list[float],
    higher_is_better: list[bool],
) -> list[PlayerScore]:
    """
    Rank players using TOPSIS (Hwang & Yoon, 1981).

    Algorithm
    ---------
    1. Build the decision matrix **X** where ``X[i][j]`` is player *i*'s
       value on criterion *j*.
    2. **Vector normalise**: ``r[i][j] = X[i][j] / ‖X[:,j]‖₂``
    3. **Weight**: ``v[i][j] = w[j] · r[i][j]``
    4. **Ideal solutions**:

       * ``A⁺[j]`` = best value in column *j* (max if ``higher_is_better[j]``)
       * ``A⁻[j]`` = worst value in column *j*

    5. **Euclidean distances**:

       * ``d⁺[i] = ‖v[i] − A⁺‖₂``
       * ``d⁻[i] = ‖v[i] − A⁻‖₂``

    6. **Closeness coefficient**: ``C[i] = d⁻[i] / (d⁺[i] + d⁻[i])``
       ``C ∈ [0, 1]``; ``C → 1`` indicates proximity to the ideal solution.

    Parameters
    ----------
    players:
        ``[(player_id, name, role, [c₁, c₂, …, cₙ]), …]``
    weights:
        Criterion weights; must sum to 1.
    higher_is_better:
        ``True`` if a larger value of that criterion is preferable.

    Returns
    -------
    list[PlayerScore]
        Sorted descending by closeness coefficient; ``rank`` is 1-based.
    """
    if not players:
        return []

    n_players = len(players)
    n_criteria = len(weights)
    matrix: list[list[float]] = [p[3] for p in players]

    # Step 2 — vector normalisation
    col_norms: list[float] = [
        math.sqrt(sum(matrix[i][j] ** 2 for i in range(n_players))) or 1.0
        for j in range(n_criteria)
    ]
    normalised: list[list[float]] = [
        [matrix[i][j] / col_norms[j] for j in range(n_criteria)]
        for i in range(n_players)
    ]

    # Step 3 — weighted normalised matrix
    weighted: list[list[float]] = [
        [weights[j] * normalised[i][j] for j in range(n_criteria)]
        for i in range(n_players)
    ]

    # Step 4 — ideal solutions
    ideal_best: list[float] = []
    ideal_worst: list[float] = []
    for j in range(n_criteria):
        col = [weighted[i][j] for i in range(n_players)]
        if higher_is_better[j]:
            ideal_best.append(max(col))
            ideal_worst.append(min(col))
        else:
            ideal_best.append(min(col))
            ideal_worst.append(max(col))

    # Steps 5–6 — distances and closeness coefficient
    scores: list[PlayerScore] = []
    for i, (pid, name, role, _) in enumerate(players):
        d_plus = math.sqrt(
            sum((weighted[i][j] - ideal_best[j]) ** 2 for j in range(n_criteria))
        )
        d_minus = math.sqrt(
            sum((weighted[i][j] - ideal_worst[j]) ** 2 for j in range(n_criteria))
        )
        denom = d_plus + d_minus
        closeness = d_minus / denom if denom > 0.0 else 0.0
        scores.append(
            PlayerScore(
                player_id=pid,
                player_name=name,
                role=role,
                closeness=closeness,
                raw_index=0.0,  # filled by TeamSelector
            )
        )

    scores.sort(key=lambda ps: ps.closeness, reverse=True)
    for rank, ps in enumerate(scores, start=1):
        ps.rank = rank

    return scores


# ---------------------------------------------------------------------------
# TeamSelector
# ---------------------------------------------------------------------------


@dataclass
class SquadComposition:
    """Defines how many players of each role form the final XI."""

    batsmen: int = 4
    wicket_keepers: int = 1
    all_rounders: int = 2
    bowlers: int = 4

    def as_quota_map(self) -> dict[str, int]:
        return {
            "batsman": self.batsmen,
            "wicket-keeper": self.wicket_keepers,
            "all-rounder": self.all_rounders,
            "bowler": self.bowlers,
        }

    @property
    def total(self) -> int:
        return self.batsmen + self.wicket_keepers + self.all_rounders + self.bowlers


class TeamSelector:
    """
    Selects an optimal XI using TOPSIS applied separately to batting
    and bowling candidate pools.

    The default :attr:`DEFAULT_COMPOSITION` picks
    4 batsmen · 1 wicket-keeper · 2 all-rounders · 4 bowlers = 11.

    All-rounders are evaluated in *both* pools and their selection quota
    is satisfied from the batting pool (since batting consistency is
    typically the primary criterion for an all-rounder pick).
    """

    DEFAULT_COMPOSITION: Final[SquadComposition] = SquadComposition()

    # ------------------------------------------------------------------
    # Criteria vectors
    # ------------------------------------------------------------------

    @staticmethod
    def _batting_criteria(s: BattingStats) -> list[float]:
        """[batting_index, avg_norm, sr_norm, consistency]"""
        return [
            s.batting_index,
            min(s.average, _AVG_CAP) / _AVG_CAP,
            min(s.strike_rate, _SR_CAP) / _SR_CAP,
            1.0 - s.gini,
        ]

    @staticmethod
    def _bowling_criteria(s: BowlingStats) -> list[float]:
        """[bowling_index, eco_inverse_norm, dot_pct, dot_entropy_score]"""
        eco_inv = (1.0 / max(s.economy_rate, 0.1)) / (1.0 / 0.1)  # normalise to [0,1]
        return [
            s.bowling_index,
            eco_inv,
            s.dot_pct,
            dot_entropy_score(s.ball_outcomes),
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @classmethod
    def select_xi(
        cls,
        batting_stats: list[BattingStats],
        bowling_stats: list[BowlingStats],
        player_roles: dict[int, str],
        composition: Optional[SquadComposition] = None,
    ) -> dict[str, list[PlayerScore]]:
        """
        Apply TOPSIS to batting and bowling pools and mark selected players.

        Parameters
        ----------
        batting_stats:
            Computed :class:`BattingStats` for all candidates.
        bowling_stats:
            Computed :class:`BowlingStats` for all candidates.
        player_roles:
            ``{player_id: role_string}`` — used to allocate players to pools.
        composition:
            Optional override for squad size per role.
            Defaults to :attr:`DEFAULT_COMPOSITION`.

        Returns
        -------
        dict
            ``{"batsmen": [PlayerScore, …], "bowlers": [PlayerScore, …]}``
            with ``.selected`` flags set on the chosen XI members.
        """
        comp = composition or cls.DEFAULT_COMPOSITION
        bat_index = {s.player_id: s for s in batting_stats}
        bowl_index = {s.player_id: s for s in bowling_stats}

        batting_pool = [
            (s.player_id, s.player_name, player_roles.get(s.player_id, "batsman"),
             cls._batting_criteria(s))
            for s in batting_stats
            if player_roles.get(s.player_id) in BATTING_ROLES
        ]
        bowling_pool = [
            (s.player_id, s.player_name, player_roles.get(s.player_id, "bowler"),
             cls._bowling_criteria(s))
            for s in bowling_stats
            if player_roles.get(s.player_id) in BOWLING_ROLES
        ]

        bat_ranked = topsis(batting_pool, _TOPSIS_WEIGHTS, _TOPSIS_HIGHER_IS_BETTER)
        bowl_ranked = topsis(bowling_pool, _TOPSIS_WEIGHTS, _TOPSIS_HIGHER_IS_BETTER)

        for ps in bat_ranked:
            ps.raw_index = bat_index[ps.player_id].batting_index
        for ps in bowl_ranked:
            ps.raw_index = bowl_index[ps.player_id].bowling_index

        cls._mark_selected(bat_ranked, bowl_ranked, player_roles, comp)

        return {"batsmen": bat_ranked, "bowlers": bowl_ranked}

    @staticmethod
    def _mark_selected(
        bat_ranked: list[PlayerScore],
        bowl_ranked: list[PlayerScore],
        player_roles: dict[int, str],
        comp: SquadComposition,
    ) -> None:
        """Set ``PlayerScore.selected = True`` for the top quota per role."""
        quota_map = comp.as_quota_map()
        filled: dict[str, int] = {role: 0 for role in quota_map}

        for pool in (bat_ranked, bowl_ranked):
            for ps in pool:
                role = player_roles.get(ps.player_id, "")
                if role in quota_map and filled[role] < quota_map[role] and not ps.selected:
                    ps.selected = True
                    filled[role] += 1


# ---------------------------------------------------------------------------
# AnalyticsService — DB-aware façade
# ---------------------------------------------------------------------------


class AnalyticsService:
    """
    Translates SQLAlchemy rows into domain objects and delegates to the
    pure analytics layer.

    All class methods must be called within a Flask application context
    so that ``db.session`` is available.

    Design notes
    ------------
    * Imports of ``app.extensions`` and ``app.models`` are deferred to the
      method body.  This keeps the module importable without a Flask app
      (e.g. in unit tests that mock the DB layer).
    * Queries use explicit column filtering where possible to avoid loading
      unnecessary data.
    * All ``ValueError`` / ``LookupError`` conditions are converted to the
      typed exceptions defined at the top of this module so that the route
      layer can handle them uniformly.
    """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _outcome_label(ball) -> str:
        """Convert a Ball ORM row to a categorical outcome string."""
        if ball.wicket_type and ball.wicket_type in BOWLER_WICKET_TYPES:
            return "W"
        runs = ball.runs_scored or 0
        if ball.extras:
            return "extra"
        if runs == 0:
            return "dot"
        if runs in (4, 6):
            return str(runs)
        return str(runs)

    @classmethod
    def _build_batting_stats(cls, player, balls: list) -> BattingStats:
        """Aggregate a list of Ball rows for one batsman into a BattingStats."""
        innings_scores: dict[int, int] = defaultdict(int)
        for b in balls:
            innings_scores[b.inning_id] += b.runs_scored or 0

        return BattingStats(
            player_id=player.id,
            player_name=player.name,
            innings_played=len(innings_scores),
            total_runs=sum(b.runs_scored or 0 for b in balls),
            balls_faced=len(balls),
            dismissals=sum(1 for b in balls if b.dismissed_player_id == player.id),
            score_per_innings=list(innings_scores.values()),
            boundary_balls=sum(1 for b in balls if (b.runs_scored or 0) in (4, 6)),
            dot_balls_faced=sum(
                1 for b in balls if not b.runs_scored and not b.extras
            ),
        )

    @classmethod
    def _build_bowling_stats(cls, player, balls: list) -> BowlingStats:
        """Aggregate a list of Ball rows for one bowler into a BowlingStats."""
        return BowlingStats(
            player_id=player.id,
            player_name=player.name,
            balls_bowled=len(balls),
            runs_conceded=sum(
                (b.runs_scored or 0) + (b.extras or 0) for b in balls
            ),
            wickets=sum(
                1 for b in balls
                if b.wicket_type and b.wicket_type in BOWLER_WICKET_TYPES
            ),
            ball_outcomes=[cls._outcome_label(b) for b in balls],
            wicket_overs=[b.over_number for b in balls if b.wicket_type],
        )

    @staticmethod
    def _empty_batting_stats(player) -> BattingStats:
        return compute_batting_profile(
            BattingStats(
                player_id=player.id,
                player_name=player.name,
                innings_played=0,
                total_runs=0,
                balls_faced=0,
                dismissals=0,
                score_per_innings=[],
                boundary_balls=0,
                dot_balls_faced=0,
            )
        )

    @staticmethod
    def _empty_bowling_stats(player) -> BowlingStats:
        return compute_bowling_profile(
            BowlingStats(
                player_id=player.id,
                player_name=player.name,
                balls_bowled=0,
                runs_conceded=0,
                wickets=0,
                ball_outcomes=[],
                wicket_overs=[],
            )
        )

    @classmethod
    def _career_batting_or_empty(cls, player) -> BattingStats:
        try:
            return cls.player_career_batting(player.id)
        except InsufficientDataError:
            return cls._empty_batting_stats(player)

    @classmethod
    def _career_bowling_or_empty(cls, player) -> BowlingStats:
        try:
            return cls.player_career_bowling(player.id)
        except InsufficientDataError:
            return cls._empty_bowling_stats(player)

    @staticmethod
    def _selection_to_lookup(selection: dict[str, list[PlayerScore]]) -> dict[int, PlayerScore]:
        lookup: dict[int, PlayerScore] = {}
        for pool in selection.values():
            for score in pool:
                if score.selected:
                    lookup[score.player_id] = score
        return lookup

    @staticmethod
    def _selected_team_summary(
        team,
        selection: dict[str, list[PlayerScore]],
        batting_index: dict[int, BattingStats],
        bowling_index: dict[int, BowlingStats],
    ) -> dict:
        selected_lookup = AnalyticsService._selection_to_lookup(selection)
        squad = []
        for player in team.players.order_by("Player.name").all():
            if player.id not in selected_lookup:
                continue
            batting = batting_index[player.id]
            bowling = bowling_index[player.id]
            choice = selected_lookup[player.id]
            squad.append({
                "player_id": player.id,
                "player_name": player.name,
                "role": player.role,
                "selection_score": round(choice.closeness, 4),
                "batting_index": round(batting.batting_index, 4),
                "gini_coefficient": round(batting.gini, 4),
                "consistency_label": (
                    "consistent" if batting.gini < 0.30
                    else "moderate" if batting.gini < 0.50
                    else "erratic"
                ),
                "bowling_index": round(bowling.bowling_index, 4),
                "economy_rate": round(bowling.economy_rate, 2),
                "entropy": round(bowling.entropy, 4),
                "career_runs": batting.total_runs,
                "career_wickets": bowling.wickets,
            })

        batting_values = [batting_index[p["player_id"]].batting_index for p in squad]
        bowling_values = [bowling_index[p["player_id"]].bowling_index for p in squad]
        experience_values = [
            batting_index[p["player_id"]].innings_played + bowling_index[p["player_id"]].balls_bowled / 6.0
            for p in squad
        ]
        role_counts = Counter(p["role"] for p in squad)

        batting_strength = sum(batting_values) / len(batting_values) if batting_values else 0.0
        bowling_strength = sum(bowling_values) / len(bowling_values) if bowling_values else 0.0
        experience_strength = sum(experience_values) / len(experience_values) if experience_values else 0.0
        balance_strength = min(len(squad) / 11.0, 1.0) * 0.5 + min(len(role_counts) / 4.0, 1.0) * 0.5
        overall_rating = (
            batting_strength * 0.40
            + bowling_strength * 0.35
            + min(experience_strength / 20.0, 1.0) * 0.15
            + balance_strength * 0.10
        )

        return {
            "team_id": team.id,
            "team_name": team.name,
            "team_short_name": team.short_name,
            "selected_xi": squad,
            "role_counts": dict(role_counts),
            "strengths": {
                "batting": round(batting_strength, 4),
                "bowling": round(bowling_strength, 4),
                "experience": round(experience_strength, 2),
                "balance": round(balance_strength, 4),
                "overall": round(overall_rating, 4),
            },
            "rankings": {
                "batsmen": [
                    {
                        "player_id": ps.player_id,
                        "player_name": ps.player_name,
                        "role": ps.role,
                        "rank": ps.rank,
                        "closeness": round(ps.closeness, 4),
                        "index": round(ps.raw_index, 4),
                        "selected": ps.selected,
                    }
                    for ps in selection["batsmen"]
                ],
                "bowlers": [
                    {
                        "player_id": ps.player_id,
                        "player_name": ps.player_name,
                        "role": ps.role,
                        "rank": ps.rank,
                        "closeness": round(ps.closeness, 4),
                        "index": round(ps.raw_index, 4),
                        "selected": ps.selected,
                    }
                    for ps in selection["bowlers"]
                ],
            },
        }

    @staticmethod
    def _win_probability(team_a_rating: float, team_b_rating: float) -> tuple[float, float]:
        exponent_a = math.exp(team_a_rating * 5.0)
        exponent_b = math.exp(team_b_rating * 5.0)
        total = exponent_a + exponent_b
        if total == 0:
            return 50.0, 50.0
        return (round(exponent_a / total * 100, 1), round(exponent_b / total * 100, 1))

    @classmethod
    def _build_team_preview(cls, team) -> tuple[dict, dict[int, BattingStats], dict[int, BowlingStats]]:
        players = team.players.order_by("Player.name").all()
        if not players:
            raise InsufficientDataError(f"Team team_id={team.id} has no players.")

        player_roles = {player.id: player.role for player in players}
        batting_stats = [cls._career_batting_or_empty(player) for player in players]
        bowling_stats = [cls._career_bowling_or_empty(player) for player in players]
        batting_lookup = {stats.player_id: stats for stats in batting_stats}
        bowling_lookup = {stats.player_id: stats for stats in bowling_stats}
        selection = TeamSelector.select_xi(batting_stats, bowling_stats, player_roles)
        summary = cls._selected_team_summary(team, selection, batting_lookup, bowling_lookup)
        return summary, batting_lookup, bowling_lookup

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def batting_profiles_for_innings(cls, innings_id: int) -> list[BattingStats]:
        """
        Return computed batting profiles for every batsman in one innings.

        Parameters
        ----------
        innings_id:
            Primary key of the :class:`~app.models.Inning` row.

        Raises
        ------
        InningsNotFoundError
            If no Ball rows exist for the given innings_id.
        """
        from app.extensions import db
        from app.models import Ball, Player

        balls = (
            db.session.query(Ball)
            .filter(Ball.innings_id == innings_id)
            .order_by(Ball.over_number, Ball.ball_number)
            .all()
        )
        if not balls:
            raise InningsNotFoundError(
                f"No ball data found for innings_id={innings_id}."
            )

        striker_balls: dict[int, list] = defaultdict(list)
        for b in balls:
            striker_balls[b.striker_id].append(b)

        profiles: list[BattingStats] = []
        for pid, pballs in striker_balls.items():
            player = db.session.get(Player, pid)
            if player is None:
                logger.warning("Striker player_id=%d not found in Player table; skipping.", pid)
                continue
            stats = cls._build_batting_stats(player, pballs)
            profiles.append(compute_batting_profile(stats))

        return profiles

    @classmethod
    def bowling_profiles_for_innings(cls, innings_id: int) -> list[BowlingStats]:
        """
        Return computed bowling profiles for every bowler in one innings.

        Parameters
        ----------
        innings_id:
            Primary key of the :class:`~app.models.Inning` row.

        Raises
        ------
        InningsNotFoundError
            If no Ball rows exist for the given innings_id.
        """
        from app.extensions import db
        from app.models import Ball, Player

        balls = (
            db.session.query(Ball)
            .filter(Ball.innings_id == innings_id)
            .order_by(Ball.over_number, Ball.ball_number)
            .all()
        )
        if not balls:
            raise InningsNotFoundError(
                f"No ball data found for innings_id={innings_id}."
            )

        bowler_balls: dict[int, list] = defaultdict(list)
        for b in balls:
            bowler_balls[b.bowler_id].append(b)

        profiles: list[BowlingStats] = []
        for pid, pballs in bowler_balls.items():
            player = db.session.get(Player, pid)
            if player is None:
                logger.warning("Bowler player_id=%d not found in Player table; skipping.", pid)
                continue
            stats = cls._build_bowling_stats(player, pballs)
            profiles.append(compute_bowling_profile(stats))

        return profiles

    @classmethod
    def select_xi_for_match(
        cls,
        match_id: int,
        player_roles: Optional[dict[int, str]] = None,
        composition: Optional[SquadComposition] = None,
    ) -> dict[str, list[PlayerScore]]:
        """
        Run TOPSIS-based XI selection over all innings in a match.

        Parameters
        ----------
        match_id:
            Primary key of the :class:`~app.models.Match` row.
        player_roles:
            Optional ``{player_id: role}`` override.
            When omitted, roles are read from :class:`~app.models.Player`.
        composition:
            Optional squad composition override.

        Raises
        ------
        MatchNotFoundError
            If no innings rows exist for the given match_id.
        InsufficientDataError
            If no ball data can be found across all innings.
        """
        from app.extensions import db
        from app.models import Inning, Player

        innings_list = (
            db.session.query(Inning)
            .filter(Inning.match_id == match_id)
            .all()
        )
        if not innings_list:
            raise MatchNotFoundError(
                f"No innings found for match_id={match_id}."
            )

        all_bat: list[BattingStats] = []
        all_bowl: list[BowlingStats] = []

        for inning in innings_list:
            try:
                all_bat.extend(cls.batting_profiles_for_innings(inning.id))
                all_bowl.extend(cls.bowling_profiles_for_innings(inning.id))
            except InningsNotFoundError:
                logger.debug("innings_id=%d has no ball data; skipping.", inning.id)

        if not all_bat and not all_bowl:
            raise InsufficientDataError(
                f"No ball data found for any innings in match_id={match_id}."
            )

        if player_roles is None:
            all_ids = {s.player_id for s in all_bat} | {s.player_id for s in all_bowl}
            players = (
                db.session.query(Player)
                .filter(Player.id.in_(all_ids))
                .all()
            )
            player_roles = {p.id: p.role for p in players}

        return TeamSelector.select_xi(all_bat, all_bowl, player_roles, composition)

    @classmethod
    def pre_match_team_preview(cls, match_id: int) -> dict:
        from app.extensions import db
        from app.models import Match

        match = db.session.get(Match, match_id)
        if match is None:
            raise MatchNotFoundError(f"Match match_id={match_id} not found.")

        if match.team1 is None or match.team2 is None:
            raise MatchNotFoundError(f"Match match_id={match_id} is missing team assignments.")

        team1_summary, _, _ = cls._build_team_preview(match.team1)
        team2_summary, _, _ = cls._build_team_preview(match.team2)

        team1_rating = team1_summary["strengths"]["overall"]
        team2_rating = team2_summary["strengths"]["overall"]
        team1_win, team2_win = cls._win_probability(team1_rating, team2_rating)

        aspect_labels = {
            "batting": "Batting",
            "bowling": "Bowling",
            "experience": "Experience",
            "balance": "Squad Balance",
            "overall": "Overall",
        }
        comparisons = []
        for key, label in aspect_labels.items():
            team1_value = team1_summary["strengths"][key]
            team2_value = team2_summary["strengths"][key]
            if abs(team1_value - team2_value) < 0.02:
                edge = "Even"
            elif team1_value > team2_value:
                edge = team1_summary["team_name"]
            else:
                edge = team2_summary["team_name"]
            comparisons.append({
                "aspect": key,
                "label": label,
                "team_1_value": team1_value,
                "team_2_value": team2_value,
                "edge": edge,
            })

        favored = team1_summary["team_name"] if team1_win >= team2_win else team2_summary["team_name"]

        return {
            "match_id": match.id,
            "match_status": match.status,
            "match_type": match.match_type,
            "team_1": team1_summary,
            "team_2": team2_summary,
            "comparison": comparisons,
            "win_probability": {
                "team_1": {
                    "team_id": team1_summary["team_id"],
                    "team_name": team1_summary["team_name"],
                    "chance": team1_win,
                },
                "team_2": {
                    "team_id": team2_summary["team_id"],
                    "team_name": team2_summary["team_name"],
                    "chance": team2_win,
                },
                "favored_team": favored,
            },
        }

    @classmethod
    def player_career_batting(cls, player_id: int) -> BattingStats:
        """
        Aggregate every innings ever faced by one batsman into a career profile.

        Parameters
        ----------
        player_id:
            Primary key of the :class:`~app.models.Player` row.

        Raises
        ------
        PlayerNotFoundError
            If the player_id does not exist.
        InsufficientDataError
            If the player has never faced a ball.
        """
        from app.extensions import db
        from app.models import Ball, Player

        player = db.session.get(Player, player_id)
        if player is None:
            raise PlayerNotFoundError(f"Player player_id={player_id} not found.")

        balls = (
            db.session.query(Ball)
            .filter(Ball.striker_id == player_id)
            .order_by(Ball.innings_id, Ball.over_number, Ball.ball_number)
            .all()
        )
        if not balls:
            raise InsufficientDataError(
                f"Player player_id={player_id} has no batting data."
            )

        stats = cls._build_batting_stats(player, balls)
        return compute_batting_profile(stats)

    @classmethod
    def player_career_bowling(cls, player_id: int) -> BowlingStats:
        """
        Aggregate every delivery ever bowled by one bowler into a career profile.

        Parameters
        ----------
        player_id:
            Primary key of the :class:`~app.models.Player` row.

        Raises
        ------
        PlayerNotFoundError
            If the player_id does not exist.
        InsufficientDataError
            If the player has never bowled a ball.
        """
        from app.extensions import db
        from app.models import Ball, Player

        player = db.session.get(Player, player_id)
        if player is None:
            raise PlayerNotFoundError(f"Player player_id={player_id} not found.")

        balls = (
            db.session.query(Ball)
            .filter(Ball.bowler_id == player_id)
            .order_by(Ball.innings_id, Ball.over_number, Ball.ball_number)
            .all()
        )
        if not balls:
            raise InsufficientDataError(
                f"Player player_id={player_id} has no bowling data."
            )

        stats = cls._build_bowling_stats(player, balls)
        return compute_bowling_profile(stats)
