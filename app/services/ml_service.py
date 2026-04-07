"""
app/services/ml_service.py
===========================

Machine-learning layer for ScoreStat.

Architecture
------------
1.  **Feature Engineering** — converts raw SQLAlchemy rows into numeric
    feature vectors with zero external I/O.

2.  **ScorePredictor** — a RandomForestRegressor trained on (team_batting_features,
    opponent_bowling_features) → predicted innings total.  One model is trained
    per innings slot (innings 1 / innings 2) so that the second-innings model
    can implicitly learn chase dynamics.

3.  **WinnerClassifier** — a RandomForestClassifier that takes both predicted
    innings totals and the feature vectors of both teams and outputs a win
    probability for team_1.

All models are trained lazily (on first prediction call) and cached in module-
level variables so they survive the lifetime of a Gunicorn worker.

Design principles
-----------------
*   No Flask request context is required in the service layer.
*   Every public method is a @classmethod so the service is used without
    instantiation.
*   The module is importable without a running Flask app (useful for tests).
*   sklearn objects are not persisted to disk in this iteration — they are
    re-trained from the DB each time the worker restarts or when explicitly
    invalidated via ``MLService.invalidate_cache()``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from app.services.team_profiles import TeamProfileService
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy sklearn imports (keep module importable without sklearn installed)
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — ML predictions unavailable.")

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_score_model_inn1: Optional["RandomForestRegressor"] = None
_score_model_inn2: Optional["RandomForestRegressor"] = None
_winner_model: Optional["RandomForestClassifier"] = None
_scaler_inn1: Optional["StandardScaler"] = None
_scaler_inn2: Optional["StandardScaler"] = None
_scaler_winner: Optional["StandardScaler"] = None
_models_trained: bool = False


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

@dataclass
class TeamMatchFeatures:
    """
    Numeric feature vector for one team in one match context.

    All values are floats in their natural scale — scaling is done inside
    the service before passing to sklearn.
    """
    team_id: int
    team_name: str

    # Batting aggregates (last N matches, default N=10)
    avg_runs_scored: float = 0.0
    avg_wickets_lost: float = 0.0
    avg_run_rate: float = 0.0
    top_order_avg_sr: float = 0.0        # avg SR of top-3 batsmen
    boundary_rate: float = 0.0           # boundaries per over
    dot_ball_pct_batting: float = 0.0
    avg_partnerships_runs: float = 0.0
    innings_consistency: float = 0.0     # 1 - Gini of recent totals

    # Bowling aggregates
    avg_runs_conceded: float = 0.0
    avg_wickets_taken: float = 0.0
    avg_economy: float = 0.0
    dot_ball_pct_bowling: float = 0.0
    avg_extras_conceded: float = 0.0
    bowling_consistency: float = 0.0     # 1 - Gini of wickets-per-match

    # Head-to-head
    h2h_wins: int = 0
    h2h_losses: int = 0
    h2h_avg_margin_runs: float = 0.0

    # Best XI aggregates (from TOPSIS / analytics layer)
    xi_avg_batting_index: float = 0.0
    xi_avg_bowling_index: float = 0.0
    xi_batting_depth: float = 0.0        # proportion of all-rounders + batsmen
    xi_bowling_depth: float = 0.0        # proportion of all-rounders + bowlers

    def to_batting_vector(self) -> list[float]:
        """Feature vector when this team is batting."""
        return [
            self.avg_runs_scored,
            self.avg_wickets_lost,
            self.avg_run_rate,
            self.top_order_avg_sr,
            self.boundary_rate,
            self.dot_ball_pct_batting,
            self.avg_partnerships_runs,
            self.innings_consistency,
            self.xi_avg_batting_index,
            self.xi_batting_depth,
        ]

    def to_bowling_vector(self) -> list[float]:
        """Feature vector when this team is bowling."""
        return [
            self.avg_runs_conceded,
            self.avg_wickets_taken,
            self.avg_economy,
            self.dot_ball_pct_bowling,
            self.avg_extras_conceded,
            self.bowling_consistency,
            self.xi_avg_bowling_index,
            self.xi_bowling_depth,
        ]

    def to_full_vector(self) -> list[float]:
        """Combined vector for winner classification."""
        return self.to_batting_vector() + self.to_bowling_vector() + [
            float(self.h2h_wins),
            float(self.h2h_losses),
            self.h2h_avg_margin_runs,
        ]


# ---------------------------------------------------------------------------
# Gini helper (duplicated from analytics_service to avoid circular import)
# ---------------------------------------------------------------------------

def _gini(values: list[float]) -> float:
    xs = sorted(max(0.0, v) for v in values)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total == 0.0:
        return 0.0
    weighted = sum(r * x for r, x in enumerate(xs, 1))
    return (2.0 * weighted) / (n * total) - (n + 1) / n


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Builds :class:`TeamMatchFeatures` from the database."""

    LOOKBACK = 15  # max past matches to consider

    @classmethod
    def build(cls, team_id: int, opponent_id: int, over_limit: int = 20):

        team_profile = TeamProfileService.get(team_id)
        opponent_profile = TeamProfileService.get(opponent_id)

        feat = TeamMatchFeatures(
            team_id=team_id,
            team_name=team_profile.team_name,
        )

        # ── Batting (from team profile)
        feat.avg_runs_scored = team_profile.avg_runs
        feat.avg_wickets_lost = team_profile.avg_wickets
        feat.avg_run_rate = team_profile.run_rate
        feat.boundary_rate = team_profile.boundary_rate
        feat.innings_consistency = team_profile.batting_consistency
        feat.top_order_avg_sr = team_profile.run_rate * 12  # scaled proxy

        # ── Bowling (from opponent profile)
        feat.avg_runs_conceded = opponent_profile.avg_conceded
        feat.avg_wickets_taken = opponent_profile.avg_wickets_taken
        feat.avg_economy = opponent_profile.economy
        feat.dot_ball_pct_bowling = opponent_profile.dot_ball_pct or 0.0
        feat.bowling_consistency = opponent_profile.bowling_consistency

        # ── XI strength (TOPSIS already inside profile)
        feat.xi_avg_batting_index = team_profile.batting_index
        feat.xi_avg_bowling_index = team_profile.bowling_index

        # ── Depth (approximation)
        feat.xi_batting_depth =  (team_profile.batting_index / 200)
        feat.xi_bowling_depth = 0.6 + (team_profile.bowling_index / 200)

        # ── Head-to-head (keep your existing logic OR simplify)
        feat.h2h_wins = team_profile.h2h_wins_vs.get(opponent_id, 0)
        feat.h2h_losses = team_profile.h2h_losses_vs.get(opponent_id, 0)
        feat.h2h_avg_margin_runs = team_profile.h2h_avg_margin_runs_vs.get(opponent_id, 0.0)
        feat.dot_ball_pct_batting = team_profile.dot_ball_pct or 0.0
        feat.avg_partnerships_runs = team_profile.avg_partnership or 0.0

        return feat


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------

class TrainingDataBuilder:
    """Builds (X, y) pairs from completed matches in the DB."""

    @classmethod
    def build_score_samples(cls) -> tuple[list, list, list, list]:
        """
        Returns (X_inn1, y_inn1, X_inn2, y_inn2).

        Each X row = batting_vector(batting_team) + bowling_vector(bowling_team).
        y = actual innings total.
        """
        from app.extensions import db
        from app.models import Match, Inning

        matches = (
            Match.query
            .filter_by(status="completed")
            .order_by(Match.match_date.desc())
            .limit(200)
            .all()
        )

        X1, y1, X2, y2 = [], [], [], []

        for match in matches:
            try:
                innings_list = (
                    Inning.query
                    .filter_by(match_id=match.id)
                    .order_by(Inning.innings_number)
                    .all()
                )
                if len(innings_list) < 2:
                    continue

                inn1, inn2 = innings_list[0], innings_list[1]
                bat1_id = inn1.batting_team_id
                bow1_id = inn1.bowling_team_id

                feat_bat = FeatureBuilder.build(bat1_id, bow1_id, match.over_limit or 20)
                feat_bowl = FeatureBuilder.build(bow1_id, bat1_id, match.over_limit or 20)

                row1 = feat_bat.to_batting_vector() + feat_bowl.to_bowling_vector()
                X1.append(row1)
                y1.append(float(inn1.total_runs))

                row2 = feat_bowl.to_batting_vector() + feat_bat.to_bowling_vector() + [float(inn1.total_runs)]
                X2.append(row2)
                y2.append(float(inn2.total_runs))

            except Exception as exc:
                logger.debug("Skipping match %d for training: %s", match.id, exc)
                continue

        return X1, y1, X2, y2

    @classmethod
    def build_winner_samples(cls) -> tuple[list, list]:
        """
        Returns (X_winner, y_winner).

        X row = full_vector(team1) + full_vector(team2) + [inn1_runs, inn2_runs].
        y = 1 if team1 won, 0 if team2 won.
        """
        from app.extensions import db
        from app.models import Match, Inning

        matches = (
            Match.query
            .filter(Match.status == "completed", Match.winner_id.isnot(None))
            .order_by(Match.match_date.desc())
            .limit(200)
            .all()
        )

        X, y = [], []

        for match in matches:
            try:
                innings_list = (
                    Inning.query.filter_by(match_id=match.id)
                    .order_by(Inning.innings_number)
                    .all()
                )
                if len(innings_list) < 2:
                    continue

                inn1, inn2 = innings_list[0], innings_list[1]
                t1_id = match.team_1_id
                t2_id = match.team_2_id

                feat1 = FeatureBuilder.build(t1_id, t2_id, match.over_limit or 20)
                feat2 = FeatureBuilder.build(t2_id, t1_id, match.over_limit or 20)

                row = feat1.to_full_vector() + feat2.to_full_vector() + [
                    float(inn1.total_runs), float(inn2.total_runs)
                ]
                X.append(row)
                y.append(1 if match.winner_id == t1_id else 0)

            except Exception as exc:
                logger.debug("Skipping winner sample for match %d: %s", match.id, exc)
                continue

        return X, y


# ---------------------------------------------------------------------------
# Model trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Trains RandomForest models and stores them in module-level cache."""

    SCORE_PARAMS = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": 1,
    }
    WINNER_PARAMS = {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": 1,
        "class_weight": "balanced",
    }

    @classmethod
    def train(cls) -> dict:
        """
        Train all three models.  Returns a status dict.
        Sets module-level cache variables.
        """
        global _score_model_inn1, _score_model_inn2, _winner_model
        global _scaler_inn1, _scaler_inn2, _scaler_winner
        global _models_trained

        if not _SKLEARN_AVAILABLE:
            return {"status": "error", "detail": "scikit-learn not installed"}

        X1, y1, X2, y2 = TrainingDataBuilder.build_score_samples()
        Xw, yw = TrainingDataBuilder.build_winner_samples()

        status = {}

        # ── Innings 1 score model ─────────────────────────────────────────
        if len(X1) >= 5:
            _scaler_inn1 = StandardScaler()
            X1_s = _scaler_inn1.fit_transform(X1)
            _score_model_inn1 = RandomForestRegressor(**cls.SCORE_PARAMS)
            _score_model_inn1.fit(X1_s, y1)
            status["inn1_samples"] = len(X1)
        else:
            status["inn1_samples"] = len(X1)
            status["inn1_warning"] = "Insufficient data — using heuristic fallback"

        # ── Innings 2 score model ─────────────────────────────────────────
        if len(X2) >= 5:
            _scaler_inn2 = StandardScaler()
            X2_s = _scaler_inn2.fit_transform(X2)
            _score_model_inn2 = RandomForestRegressor(**cls.SCORE_PARAMS)
            _score_model_inn2.fit(X2_s, y2)
            status["inn2_samples"] = len(X2)
        else:
            status["inn2_samples"] = len(X2)
            status["inn2_warning"] = "Insufficient data — using heuristic fallback"

        # ── Winner classifier ─────────────────────────────────────────────
        if len(Xw) >= 5 and len(set(yw)) > 1:
            _scaler_winner = StandardScaler()
            Xw_s = _scaler_winner.fit_transform(Xw)
            _winner_model = RandomForestClassifier(**cls.WINNER_PARAMS)
            _winner_model.fit(Xw_s, yw)
            status["winner_samples"] = len(Xw)
        else:
            status["winner_samples"] = len(Xw)
            status["winner_warning"] = "Insufficient data — using feature-based heuristic"

        _models_trained = True
        status["status"] = "trained"
        return status


# ---------------------------------------------------------------------------
# Heuristic fallbacks (used when training data is sparse)
# ---------------------------------------------------------------------------

def _heuristic_score(feat_bat: "TeamMatchFeatures", feat_bowl: "TeamMatchFeatures",
                     over_limit: int = 20) -> float:
    """Simple heuristic when ML model is unavailable."""
    base = feat_bat.avg_runs_scored if feat_bat.avg_runs_scored > 0 else (over_limit * 7.5)
    bowl_adj = (feat_bowl.avg_economy / 8.0) if feat_bowl.avg_economy > 0 else 1.0
    return round(base * (1 / max(bowl_adj, 0.5)), 1)


def _heuristic_win_prob(feat1: "TeamMatchFeatures", feat2: "TeamMatchFeatures") -> float:
    """Simple heuristic win probability for team_1."""
    s1 = (feat1.xi_avg_batting_index * 0.5 + feat1.xi_avg_bowling_index * 0.5 + 0.001)
    s2 = (feat2.xi_avg_batting_index * 0.5 + feat2.xi_avg_bowling_index * 0.5 + 0.001)
    return round(s1 / (s1 + s2), 3)


# ---------------------------------------------------------------------------
# Public MLService API
# ---------------------------------------------------------------------------

@dataclass
class MatchPrediction:
    """Complete prediction output for a prospective match."""
    team1_id: int
    team2_id: int
    team1_name: str
    team2_name: str

    # Batting-first team (from toss or user selection)
    batting_first_id: int
    batting_first_name: str

    # Innings score predictions
    innings1_predicted: float = 0.0
    innings2_predicted: float = 0.0
    innings1_range: tuple[float, float] = field(default=(0.0, 0.0))
    innings2_range: tuple[float, float] = field(default=(0.0, 0.0))

    # Winner
    win_probability_team1: float = 0.0
    win_probability_team2: float = 0.0
    predicted_winner_id: int = 0
    predicted_winner_name: str = ""

    # Feature snapshots
    team1_features: Optional[TeamMatchFeatures] = None
    team2_features: Optional[TeamMatchFeatures] = None

    # Model provenance
    model_used: str = "ml"          # "ml" | "heuristic" | "mixed"
    training_status: dict = field(default_factory=dict)

    # Best XI (from TOPSIS)
    team1_xi: list = field(default_factory=list)
    team2_xi: list = field(default_factory=list)

    # Narrative analysis for the matchup UI
    analysis_summary: str = ""
    analysis_points: list[str] = field(default_factory=list)


class MLService:
    """
    Public facade.  All methods are class-level and require a Flask app context.
    """

    @classmethod
    def invalidate_cache(cls) -> None:
        """Force re-training on next prediction."""
        global _models_trained
        _models_trained = False

    @classmethod
    def ensure_trained(cls) -> dict:
        """Train models if not already trained."""
        global _models_trained
        if not _models_trained:
            return ModelTrainer.train()
        return {"status": "already_trained"}

    @classmethod
    def predict_match(
        cls,
        match_id: int,
        batting_first_id: Optional[int] = None,
    ) -> "MatchPrediction":
        """
        Full prediction pipeline for a match that is scheduled or live.

        Parameters
        ----------
        match_id:
            PK of the Match row.
        batting_first_id:
            Which team bats first.  If None, inferred from toss or defaults to team_1.

        Returns
        -------
        MatchPrediction
            Complete prediction with scores, win probability, and best XI.
        """
        from app.extensions import db
        from app.models import Match

        match = db.session.get(Match, match_id)
        if match is None:
            raise ValueError(f"Match {match_id} not found")

        t1_id = match.team_1_id
        t2_id = match.team_2_id
        over_limit = match.over_limit or 20

        # Determine batting order
        if batting_first_id is None:
            if match.toss_winner and match.toss_decision:
                if match.toss_decision == "bat":
                    batting_first_id = match.toss_winner
                else:
                    batting_first_id = t2_id if match.toss_winner == t1_id else t1_id
            else:
                batting_first_id = t1_id

        bowling_first_id = t2_id if batting_first_id == t1_id else t1_id

        # Build features
        feat_bat = FeatureBuilder.build(batting_first_id, bowling_first_id, over_limit)
        feat_bowl = FeatureBuilder.build(bowling_first_id, batting_first_id, over_limit)
        feat_t1 = feat_bat if batting_first_id == t1_id else feat_bowl
        feat_t2 = feat_bowl if batting_first_id == t1_id else feat_bat

        # Ensure models
        train_status = cls.ensure_trained()

        # ── Innings 1 prediction ──────────────────────────────────────────
        inn1_pred, inn1_lo, inn1_hi, inn1_model_used = cls._predict_score_inn1(feat_bat, feat_bowl)

        # ── Innings 2 prediction ──────────────────────────────────────────
        inn2_pred, inn2_lo, inn2_hi, inn2_model_used = cls._predict_score_inn2(
            feat_bowl, feat_bat, inn1_pred
        )

        # ── Winner probability ────────────────────────────────────────────
        win_prob_bat, winner_model_used = cls._predict_winner(
            feat_bat, feat_bowl, inn1_pred, inn2_pred
        )

        if batting_first_id == t1_id:
            win_prob_t1 = win_prob_bat
        else:
            win_prob_t1 = 1.0 - win_prob_bat

        win_prob_t1 = round(win_prob_t1, 3)
        win_prob_t2 = round(1.0 - win_prob_t1, 3)

        winner_id = t1_id if win_prob_t1 >= 0.5 else t2_id

        # ── Best XI ───────────────────────────────────────────────────────
        xi1, xi2 = cls._get_best_xi(match_id, t1_id, t2_id)

        model_modes = {inn1_model_used, inn2_model_used, winner_model_used}
        if model_modes == {"ml"}:
            model_used = "ml"
        elif model_modes == {"heuristic"}:
            model_used = "heuristic"
        else:
            model_used = "mixed"

        pred = MatchPrediction(
            team1_id=t1_id,
            team2_id=t2_id,
            team1_name=feat_t1.team_name,
            team2_name=feat_t2.team_name,
            batting_first_id=batting_first_id,
            batting_first_name=feat_bat.team_name,
            innings1_predicted=round(inn1_pred, 1),
            innings2_predicted=round(inn2_pred, 1),
            innings1_range=(round(inn1_lo, 1), round(inn1_hi, 1)),
            innings2_range=(round(inn2_lo, 1), round(inn2_hi, 1)),
            win_probability_team1=win_prob_t1,
            win_probability_team2=win_prob_t2,
            predicted_winner_id=winner_id,
            predicted_winner_name=feat_t1.team_name if winner_id == t1_id else feat_t2.team_name,
            team1_features=feat_t1,
            team2_features=feat_t2,
            model_used=model_used,
            training_status=train_status,
            team1_xi=xi1,
            team2_xi=xi2,
        )
        return pred

    @classmethod
    def predict_teams(
        cls,
        team1_id: int,
        team2_id: int,
        batting_first_id: Optional[int] = None,
        over_limit: int = 20,
    ) -> "MatchPrediction":
        """Predict a matchup directly from two selected teams."""
        if team1_id == team2_id:
            raise ValueError("Please select two different teams.")
        if batting_first_id is not None and batting_first_id not in {team1_id, team2_id}:
            raise ValueError("batting_first_id must belong to one of the selected teams.")

        return cls._predict_from_team_ids(
            t1_id=team1_id,
            t2_id=team2_id,
            batting_first_id=batting_first_id or team1_id,
            over_limit=over_limit or 20,
            match_id=None,
        )

    @classmethod
    def _predict_from_team_ids(
        cls,
        t1_id: int,
        t2_id: int,
        batting_first_id: int,
        over_limit: int,
        match_id: Optional[int],
    ) -> "MatchPrediction":
        from app.extensions import db
        from app.models import Team

        team1 = db.session.get(Team, t1_id)
        team2 = db.session.get(Team, t2_id)
        if team1 is None or team2 is None:
            raise ValueError("One or both selected teams were not found.")

        bowling_first_id = t2_id if batting_first_id == t1_id else t1_id

        feat_bat = FeatureBuilder.build(batting_first_id, bowling_first_id, over_limit)
        feat_bowl = FeatureBuilder.build(bowling_first_id, batting_first_id, over_limit)
        feat_t1 = feat_bat if batting_first_id == t1_id else feat_bowl
        feat_t2 = feat_bowl if batting_first_id == t1_id else feat_bat

        train_status = cls.ensure_trained()

        inn1_pred, inn1_lo, inn1_hi, inn1_model_used = cls._predict_score_inn1(feat_bat, feat_bowl)
        inn2_pred, inn2_lo, inn2_hi, inn2_model_used = cls._predict_score_inn2(
            feat_bowl, feat_bat, inn1_pred
        )
        win_prob_bat, winner_model_used = cls._predict_winner(
            feat_bat, feat_bowl, inn1_pred, inn2_pred
        )

        if batting_first_id == t1_id:
            win_prob_t1 = win_prob_bat
        else:
            win_prob_t1 = 1.0 - win_prob_bat

        win_prob_t1 = round(win_prob_t1, 3)
        win_prob_t2 = round(1.0 - win_prob_t1, 3)
        winner_id = t1_id if win_prob_t1 >= 0.5 else t2_id

        xi1, xi2 = cls._get_best_xi(match_id, team1, team2)
        analysis_summary, analysis_points = cls._build_match_analysis(
            feat_t1=feat_t1,
            feat_t2=feat_t2,
            win_prob_t1=win_prob_t1,
            batting_first_id=batting_first_id,
            team1_id=t1_id,
            team2_id=t2_id,
            xi1=xi1,
            xi2=xi2,
        )

        model_modes = {inn1_model_used, inn2_model_used, winner_model_used}
        if model_modes == {"ml"}:
            model_used = "ml"
        elif model_modes == {"heuristic"}:
            model_used = "heuristic"
        else:
            model_used = "mixed"

        return MatchPrediction(
            team1_id=t1_id,
            team2_id=t2_id,
            team1_name=feat_t1.team_name,
            team2_name=feat_t2.team_name,
            batting_first_id=batting_first_id,
            batting_first_name=feat_bat.team_name,
            innings1_predicted=round(inn1_pred, 1),
            innings2_predicted=round(inn2_pred, 1),
            innings1_range=(round(inn1_lo, 1), round(inn1_hi, 1)),
            innings2_range=(round(inn2_lo, 1), round(inn2_hi, 1)),
            win_probability_team1=win_prob_t1,
            win_probability_team2=win_prob_t2,
            predicted_winner_id=winner_id,
            predicted_winner_name=feat_t1.team_name if winner_id == t1_id else feat_t2.team_name,
            team1_features=feat_t1,
            team2_features=feat_t2,
            model_used=model_used,
            training_status=train_status,
            team1_xi=xi1,
            team2_xi=xi2,
            analysis_summary=analysis_summary,
            analysis_points=analysis_points,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _predict_score_inn1(cls, feat_bat, feat_bowl) -> tuple[float, float, float, str]:
        row = feat_bat.to_batting_vector() + feat_bowl.to_bowling_vector()
        if _score_model_inn1 is not None and _scaler_inn1 is not None:
            try:
                X = _scaler_inn1.transform([row])
                # Per-tree predictions for confidence interval
                tree_preds = np.array([t.predict(X)[0] for t in _score_model_inn1.estimators_])
                pred = float(np.mean(tree_preds))
                lo = float(np.percentile(tree_preds, 10))
                hi = float(np.percentile(tree_preds, 90))
                return pred, lo, hi, "ml"
            except Exception as exc:
                logger.warning("Inn1 ML prediction failed: %s", exc)

        pred = _heuristic_score(feat_bat, feat_bowl)
        return pred, pred * 0.85, pred * 1.15, "heuristic"

    @classmethod
    def _predict_score_inn2(cls, feat_bat, feat_bowl, inn1_total) -> tuple[float, float, float, str]:
        row = feat_bat.to_batting_vector() + feat_bowl.to_bowling_vector() + [inn1_total]
        if _score_model_inn2 is not None and _scaler_inn2 is not None:
            try:
                X = _scaler_inn2.transform([row])
                tree_preds = np.array([t.predict(X)[0] for t in _score_model_inn2.estimators_])
                pred = float(np.mean(tree_preds))
                lo = float(np.percentile(tree_preds, 10))
                hi = float(np.percentile(tree_preds, 90))
                return pred, lo, hi, "ml"
            except Exception as exc:
                logger.warning("Inn2 ML prediction failed: %s", exc)

        pred = _heuristic_score(feat_bat, feat_bowl)
        return pred, pred * 0.85, pred * 1.15, "heuristic"

    @classmethod
    def _predict_winner(cls, feat_bat, feat_bowl, inn1, inn2) -> tuple[float, str]:
        row = feat_bat.to_full_vector() + feat_bowl.to_full_vector() + [inn1, inn2]
        if _winner_model is not None and _scaler_winner is not None:
            try:
                X = _scaler_winner.transform([row])
                prob = float(_winner_model.predict_proba(X)[0][1])
                return prob, "ml"
            except Exception as exc:
                logger.warning("Winner ML prediction failed: %s", exc)

        prob = _heuristic_win_prob(feat_bat, feat_bowl)
        return prob, "heuristic"

    @classmethod
    def _get_best_xi(cls, match_id: Optional[int], team1_or_id, team2_or_id):
        """Fetch XI from analytics preview, with a team-based fallback."""
        try:
            from app.services.analytics_service import AnalyticsService
            if match_id is not None:
                preview = AnalyticsService.pre_match_team_preview(match_id)
                xi1 = preview["team_1"]["selected_xi"]
                xi2 = preview["team_2"]["selected_xi"]
                return xi1, xi2

            team1_summary, _, _ = AnalyticsService._build_team_preview(team1_or_id)
            team2_summary, _, _ = AnalyticsService._build_team_preview(team2_or_id)
            return team1_summary["selected_xi"], team2_summary["selected_xi"]
        except Exception as exc:
            logger.debug("Best XI fetch failed: %s", exc)
            return [], []

    @classmethod
    def _build_match_analysis(
        cls,
        feat_t1: TeamMatchFeatures,
        feat_t2: TeamMatchFeatures,
        win_prob_t1: float,
        batting_first_id: int,
        team1_id: int,
        team2_id: int,
        xi1: list,
        xi2: list,
    ) -> tuple[str, list[str]]:
        team1_name = feat_t1.team_name
        team2_name = feat_t2.team_name
        favored_name = team1_name if win_prob_t1 >= 0.5 else team2_name
        favored_prob = win_prob_t1 if win_prob_t1 >= 0.5 else 1.0 - win_prob_t1

        points: list[str] = []

        if feat_t1.avg_run_rate > feat_t2.avg_run_rate + 0.35:
            points.append(
                f"{team1_name} bring the stronger batting tempo with a recent run rate of "
                f"{feat_t1.avg_run_rate:.2f} versus {feat_t2.avg_run_rate:.2f}."
            )
        elif feat_t2.avg_run_rate > feat_t1.avg_run_rate + 0.35:
            points.append(
                f"{team2_name} have the sharper scoring profile, running at "
                f"{feat_t2.avg_run_rate:.2f} compared with {feat_t1.avg_run_rate:.2f}."
            )
        else:
            points.append("Both batting groups arrive with very similar scoring momentum.")

        if feat_t1.avg_economy + 0.3 < feat_t2.avg_economy:
            points.append(
                f"{team1_name} look tighter with the ball, conceding {feat_t1.avg_economy:.2f} per over "
                f"against {feat_t2.avg_economy:.2f}."
            )
        elif feat_t2.avg_economy + 0.3 < feat_t1.avg_economy:
            points.append(
                f"{team2_name} have the cleaner bowling numbers, with an economy of "
                f"{feat_t2.avg_economy:.2f} against {feat_t1.avg_economy:.2f}."
            )

        if feat_t1.h2h_wins or feat_t2.h2h_wins:
            if feat_t1.h2h_wins > feat_t2.h2h_wins:
                points.append(f"Head-to-head history leans toward {team1_name} ({feat_t1.h2h_wins}-{feat_t2.h2h_wins}).")
            elif feat_t2.h2h_wins > feat_t1.h2h_wins:
                points.append(f"Head-to-head history favors {team2_name} ({feat_t2.h2h_wins}-{feat_t1.h2h_wins}).")
            else:
                points.append(f"The head-to-head record is level at {feat_t1.h2h_wins}-{feat_t2.h2h_wins}.")

        if feat_t1.xi_batting_depth > feat_t2.xi_batting_depth + 0.1:
            points.append(f"{team1_name} appear deeper with the bat, which improves their recovery potential.")
        elif feat_t2.xi_batting_depth > feat_t1.xi_batting_depth + 0.1:
            points.append(f"{team2_name} look deeper with the bat and should handle collapses better.")

        if xi1 and xi2:
            points.append("The projection is based on each side's strongest available XI profile.")

        batting_first_name = team1_name if batting_first_id == team1_id else team2_name
        chase_name = team2_name if batting_first_id == team1_id else team1_name
        points.append(f"The projection assumes {batting_first_name} bat first and {chase_name} chase.")

        summary = (
            f"{favored_name} start as slight favorites with a projected win chance of about "
            f"{favored_prob * 100:.1f}%, based on recent scoring, bowling control, squad depth, and head-to-head context."
        )
        return summary, points[:5]

    @classmethod
    def get_feature_importances(cls) -> dict:
        """Return feature importance dicts for both score models."""
        batting_labels = [
            "avg_runs_scored", "avg_wickets_lost", "avg_run_rate",
            "top_order_sr", "boundary_rate", "dot_pct_bat",
            "avg_partnership", "innings_consistency",
            "xi_batting_index", "xi_batting_depth",
        ]
        bowling_labels = [
            "avg_conceded", "avg_wkts_taken", "avg_economy",
            "dot_pct_bowl", "avg_extras",
            "bowl_consistency", "xi_bowling_index", "xi_bowling_depth",
        ]
        result = {}
        if _score_model_inn1:
            imp = _score_model_inn1.feature_importances_
            labels = batting_labels + bowling_labels
            result["inn1"] = dict(zip(labels, [round(float(v), 4) for v in imp]))
        if _score_model_inn2:
            imp = _score_model_inn2.feature_importances_
            labels = batting_labels + bowling_labels + ["inn1_total"]
            result["inn2"] = dict(zip(labels, [round(float(v), 4) for v in imp]))
        return result
