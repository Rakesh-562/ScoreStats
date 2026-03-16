"""
app/routes/api/analytics.py
============================

REST endpoints for the ScoreStat analytics engine.

Blueprint registration
----------------------
Register this blueprint in ``app/routes/api/__init__.py``::

    from app.routes.api.analytics import bp as analytics_bp
    api.register_blueprint(analytics_bp, url_prefix="/analytics")

All endpoints are then available under ``/api/v1/analytics/``.

Endpoints
---------
``GET  /api/v1/analytics/innings/<id>/batting``
    Per-batsman analytics for one innings.

``GET  /api/v1/analytics/innings/<id>/bowling``
    Per-bowler analytics for one innings.

``POST /api/v1/analytics/select-xi``
    TOPSIS-based XI selection for a completed match.

``GET  /api/v1/analytics/player/<id>/career``
    Career-aggregate batting or bowling profile for one player.

Error handling
--------------
Domain exceptions from the service layer are mapped to HTTP status codes
in :func:`_register_error_handlers`.  Routes themselves contain no
``try/except`` blocks for domain errors — the handlers do that once.
"""

from __future__ import annotations

import logging
import math
from http import HTTPStatus

from flask import Blueprint, jsonify, request
from marshmallow import ValidationError

from app.services.analytics_service import (
    AnalyticsError,
    AnalyticsService,
    BattingStats,
    BowlingStats,
    InningsNotFoundError,
    InsufficientDataError,
    MatchNotFoundError,
    PlayerNotFoundError,
    PlayerScore,
    SquadComposition,
)
from app.validators.analytics_validator import (
    load_career_query,
    load_innings_query,
    load_selection_payload,
)

logger = logging.getLogger(__name__)

bp = Blueprint("analytics", __name__)


# ---------------------------------------------------------------------------
# Error handlers — registered on the blueprint
# ---------------------------------------------------------------------------


def _register_error_handlers(blueprint: Blueprint) -> None:
    """
    Map domain exceptions to JSON error responses.

    Centralising error mapping here means route functions stay clean and
    every analytics endpoint gets consistent error formatting for free.
    """

    @blueprint.errorhandler(ValidationError)
    def handle_validation_error(exc: ValidationError):
        logger.debug("Validation error: %s", exc.messages)
        return (
            jsonify({"error": "Validation failed.", "details": exc.messages}),
            HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    @blueprint.errorhandler(PlayerNotFoundError)
    def handle_player_not_found(exc: PlayerNotFoundError):
        return jsonify({"error": str(exc)}), HTTPStatus.NOT_FOUND

    @blueprint.errorhandler(InningsNotFoundError)
    def handle_innings_not_found(exc: InningsNotFoundError):
        return jsonify({"error": str(exc)}), HTTPStatus.NOT_FOUND

    @blueprint.errorhandler(MatchNotFoundError)
    def handle_match_not_found(exc: MatchNotFoundError):
        return jsonify({"error": str(exc)}), HTTPStatus.NOT_FOUND

    @blueprint.errorhandler(InsufficientDataError)
    def handle_insufficient_data(exc: InsufficientDataError):
        return jsonify({"error": str(exc)}), HTTPStatus.UNPROCESSABLE_ENTITY

    @blueprint.errorhandler(AnalyticsError)
    def handle_generic_analytics_error(exc: AnalyticsError):
        logger.exception("Unhandled analytics error.")
        return (
            jsonify({"error": "An analytics error occurred.", "detail": str(exc)}),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


_register_error_handlers(bp)


# ---------------------------------------------------------------------------
# Marshmallow serialisation schemas
# ---------------------------------------------------------------------------


from marshmallow import Schema, fields as ma_fields, pre_dump  # noqa: E402


class BattingStatsSchema(Schema):
    """Serialises a :class:`~app.services.analytics_service.BattingStats`."""

    player_id = ma_fields.Int(dump_default=None)
    player_name = ma_fields.Str()
    innings_played = ma_fields.Int()
    total_runs = ma_fields.Int()
    balls_faced = ma_fields.Int()
    dismissals = ma_fields.Int()
    average = ma_fields.Method("get_average")
    strike_rate = ma_fields.Method("get_strike_rate")
    boundary_pct = ma_fields.Method("get_boundary_pct")
    gini_coefficient = ma_fields.Method("get_gini")
    coeff_variation = ma_fields.Method("get_cv")
    batting_index = ma_fields.Method("get_batting_index")
    consistency_label = ma_fields.Method("get_consistency_label")

    def get_average(self, obj: BattingStats) -> float:
        return round(obj.average, 2)

    def get_strike_rate(self, obj: BattingStats) -> float:
        return round(obj.strike_rate, 2)

    def get_boundary_pct(self, obj: BattingStats) -> float:
        return round(obj.boundary_pct, 2)

    def get_gini(self, obj: BattingStats) -> float:
        return round(obj.gini, 4)

    def get_cv(self, obj: BattingStats) -> float:
        return round(obj.cv, 4)

    def get_batting_index(self, obj: BattingStats) -> float:
        return round(obj.batting_index, 4)

    def get_consistency_label(self, obj: BattingStats) -> str:
        if obj.gini < 0.30:
            return "consistent"
        if obj.gini < 0.50:
            return "moderate"
        return "erratic"


class BowlingStatsSchema(Schema):
    """Serialises a :class:`~app.services.analytics_service.BowlingStats`."""

    player_id = ma_fields.Int()
    player_name = ma_fields.Str()
    balls_bowled = ma_fields.Int()
    overs_bowled = ma_fields.Method("get_overs")
    runs_conceded = ma_fields.Int()
    wickets = ma_fields.Int()
    economy_rate = ma_fields.Method("get_economy_rate")
    bowling_average = ma_fields.Method("get_bowling_average")
    dot_pct = ma_fields.Method("get_dot_pct")
    shannon_entropy = ma_fields.Method("get_entropy")
    bowling_index = ma_fields.Method("get_bowling_index")

    def get_overs(self, obj: BowlingStats) -> float:
        return round(obj.balls_bowled / 6, 1)

    def get_economy_rate(self, obj: BowlingStats) -> float:
        return round(obj.economy_rate, 2)

    def get_bowling_average(self, obj: BowlingStats):
        return round(obj.bowling_average, 2) if math.isfinite(obj.bowling_average) else None

    def get_dot_pct(self, obj: BowlingStats) -> float:
        return round(obj.dot_pct, 2)

    def get_entropy(self, obj: BowlingStats) -> float:
        return round(obj.entropy, 4)

    def get_bowling_index(self, obj: BowlingStats) -> float:
        return round(obj.bowling_index, 4)


class PlayerScoreSchema(Schema):
    """Serialises a :class:`~app.services.analytics_service.PlayerScore`."""

    player_id = ma_fields.Int()
    player_name = ma_fields.Str()
    role = ma_fields.Str()
    rank = ma_fields.Int()
    closeness = ma_fields.Method("get_closeness")
    index = ma_fields.Method("get_index")
    selected = ma_fields.Bool()

    def get_closeness(self, obj: PlayerScore) -> float:
        return round(obj.closeness, 4)

    def get_index(self, obj: PlayerScore) -> float:
        return round(obj.raw_index, 4)


# Module-level schema singletons.
_batting_schema = BattingStatsSchema()
_bowling_schema = BowlingStatsSchema()
_player_score_schema = PlayerScoreSchema()
_batting_schema_many = BattingStatsSchema(many=True)
_bowling_schema_many = BowlingStatsSchema(many=True)
_player_score_schema_many = PlayerScoreSchema(many=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@bp.get("/innings/<int:innings_id>/batting")
def innings_batting(innings_id: int):
    """
    Batting analytics for every batsman in the specified innings.

    **Query parameters**

    ``role`` *(optional)*
        Filter results to players of a specific role.

    **Response 200**

    .. code-block:: json

        {
            "innings_id": 1,
            "count": 3,
            "batsmen": [
                {
                    "player_id": 2,
                    "player_name": "Virat Kohli",
                    "innings_played": 1,
                    "total_runs": 62,
                    "balls_faced": 44,
                    "dismissals": 1,
                    "average": 62.0,
                    "strike_rate": 140.91,
                    "boundary_pct": 22.73,
                    "gini_coefficient": 0.0,
                    "coeff_variation": 0.0,
                    "batting_index": 0.6409,
                    "consistency_label": "consistent"
                }
            ]
        }

    **Response 404** — innings not found or no balls recorded.
    """
    params = load_innings_query(request.args.to_dict())

    profiles = AnalyticsService.batting_profiles_for_innings(innings_id)

    if params["role"]:
        # Role filtering is done post-query; roles live on Player not on Ball.
        profiles = [p for p in profiles if hasattr(p, "_role") and p._role == params["role"]]

    profiles.sort(key=lambda p: p.batting_index, reverse=True)

    return (
        jsonify({
            "innings_id": innings_id,
            "count": len(profiles),
            "batsmen": _batting_schema_many.dump(profiles),
        }),
        HTTPStatus.OK,
    )


@bp.get("/innings/<int:innings_id>/bowling")
def innings_bowling(innings_id: int):
    """
    Bowling analytics for every bowler in the specified innings.

    **Response 200**

    .. code-block:: json

        {
            "innings_id": 1,
            "count": 2,
            "bowlers": [...]
        }

    **Response 404** — innings not found or no balls recorded.
    """
    profiles = AnalyticsService.bowling_profiles_for_innings(innings_id)
    profiles.sort(key=lambda p: p.bowling_index, reverse=True)

    return (
        jsonify({
            "innings_id": innings_id,
            "count": len(profiles),
            "bowlers": _bowling_schema_many.dump(profiles),
        }),
        HTTPStatus.OK,
    )


@bp.post("/select-xi")
def select_xi():
    """
    Run pre-match XI selection and team-vs-team comparison for a match.

    **Request body** (JSON)

    .. code-block:: json

        {
            "match_id": 1,
            "player_roles": {
                "5": "all-rounder",
                "7": "wicket-keeper"
            }
        }

    **Response 200**

    .. code-block:: json

        {
            "match_id": 1,
            "team_1": {...},
            "team_2": {...},
            "comparison": [...],
            "win_probability": {...}
        }

    **Response 404** — match not found.
    **Response 422** — one or both teams have no players to analyse.
    """
    payload = load_selection_payload(request.get_json(silent=True) or {})

    match_id = payload["match_id"]
    preview = AnalyticsService.pre_match_team_preview(match_id)

    return (
        jsonify(preview),
        HTTPStatus.OK,
    )


@bp.get("/player/<int:player_id>/career")
def player_career(player_id: int):
    """
    Career-aggregate analytics for a single player.

    **Query parameters**

    ``stat_type`` *(default: ``"batting"``)*
        ``"batting"`` or ``"bowling"``.

    **Response 200**

    .. code-block:: json

        {
            "player_id": 1,
            "stat_type": "batting",
            "stats": { ... }
        }

    **Response 404** — player not found or no career data available.
    """
    params = load_career_query(request.args.to_dict())
    stat_type: str = params["stat_type"]

    if stat_type == "batting":
        profile = AnalyticsService.player_career_batting_safe(player_id)
        stats_payload = _batting_schema.dump(profile)
        breakdown = AnalyticsService._batting_component_breakdown(profile)
    else:
        profile = AnalyticsService.player_career_bowling_safe(player_id)
        stats_payload = _bowling_schema.dump(profile)
        breakdown = AnalyticsService._bowling_component_breakdown(profile)

    return (
        jsonify({
            "player_id": player_id,
            "stat_type": stat_type,
            "stats": stats_payload,
            "breakdown": breakdown,
        }),
        HTTPStatus.OK,
    )
