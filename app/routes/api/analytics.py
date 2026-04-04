"""
app/routes/api/analytics.py  — fixed + extended (matplotlib charts)
=====================================================================
Bug fix: player_career no longer calls missing *_safe methods.

New endpoints:
  GET /api/v1/analytics/player/<id>/innings-history   raw per-innings data
  GET /api/v1/analytics/player/<id>/charts            base64 matplotlib PNGs
  GET /api/v1/analytics/match/<id>/team-overview      raw match data
  GET /api/v1/analytics/match/<id>/charts             base64 matplotlib PNGs
"""

from __future__ import annotations

import logging
import math
from http import HTTPStatus

from flask import Blueprint, jsonify, request
from marshmallow import ValidationError
from app.services.prediction_service import predict_innings

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
    compute_batting_profile,
    compute_bowling_profile,
)
from app.validators.analytics_validator import (
    load_career_query,
    load_innings_query,
    load_selection_payload,
)

logger = logging.getLogger(__name__)
analytics_bp = Blueprint("analytics", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_player_or_404(player_id: int):
    from app.extensions import db
    from app.models import Player
    player = db.session.get(Player, player_id)
    if not player:
        raise PlayerNotFoundError(f"Player player_id={player_id} not found.")
    return player


def _build_innings_data(innings_list):
    """Shared logic: query balls + scorecards for a list of Inning objects."""
    from app.extensions import db
    from app.models import Ball, BattingScorecard, BowlingScorecard, Player

    result = []
    for innings in innings_list:
        balls = (Ball.query.filter_by(inning_id=innings.id)
                 .order_by(Ball.over_number, Ball.ball_number).all())

        over_data: dict = {}
        for ball in balls:
            ov = ball.over_number
            if ov not in over_data:
                over_data[ov] = {"runs": 0, "wickets": 0}
            over_data[ov]["runs"] += (ball.runs_scored or 0) + (ball.extra_runs or 0)
            if ball.is_wicket:
                over_data[ov]["wickets"] += 1

        cumulative = 0
        progression = []
        for ov_num in sorted(over_data.keys()):
            cumulative += over_data[ov_num]["runs"]
            progression.append({
                "over": ov_num + 1,
                "runs_this_over": over_data[ov_num]["runs"],
                "cumulative_runs": cumulative,
                "wickets_this_over": over_data[ov_num]["wickets"],
            })

        batting_scs = (BattingScorecard.query.filter_by(innings_id=innings.id)
                       .order_by(BattingScorecard.runs.desc()).all())
        batsmen = []
        for sc in batting_scs:
            p = db.session.get(Player, sc.player_id)
            batsmen.append({
                "player_id": sc.player_id,
                "player_name": p.name if p else "Unknown",
                "runs": sc.runs or 0, "balls_faced": sc.balls_faced or 0,
                "fours": sc.fours or 0, "sixes": sc.sixes or 0,
                "strike_rate": round(sc.strike_rate or 0, 2),
                "is_out": sc.is_out, "dismissal_type": sc.dismissal_type,
            })

        bowling_scs = (BowlingScorecard.query.filter_by(innings_id=innings.id)
                       .order_by(BowlingScorecard.wickets_taken.desc(),
                                 BowlingScorecard.economy_rate.asc()).all())
        bowlers = []
        for sc in bowling_scs:
            p = db.session.get(Player, sc.player_id)
            bowlers.append({
                "player_id": sc.player_id,
                "player_name": p.name if p else "Unknown",
                "wickets": sc.wickets_taken or 0,
                "runs_conceded": sc.runs_conceded or 0,
                "overs": round(sc.overs_bowled or 0, 1),
                "economy": round(sc.economy_rate or 0, 2),
                "maidens": sc.maidens or 0,
            })

        boundaries = {
            "fours":   sum(1 for b in balls if b.runs_scored == 4),
            "sixes":   sum(1 for b in balls if b.runs_scored == 6),
            "dots":    sum(1 for b in balls if not b.runs_scored and not b.extra_runs and b.is_legal_delivery),
            "ones":    sum(1 for b in balls if b.runs_scored == 1 and b.is_legal_delivery),
            "twos":    sum(1 for b in balls if b.runs_scored == 2 and b.is_legal_delivery),
            "extras":  sum(b.extra_runs or 0 for b in balls),
            "wides":   sum(1 for b in balls if b.extra_type == "wide"),
            "no_balls":sum(1 for b in balls if b.extra_type == "no-ball"),
        }

        result.append({
            "innings_id": innings.id,
            "innings_number": innings.innings_number,
            "batting_team_id": innings.batting_team_id,
            "total_runs": innings.total_runs,
            "total_wickets": innings.total_wickets,
            "total_overs": innings.total_overs,
            "target": innings.target,
            "run_progression": progression,
            "batsmen": batsmen,
            "bowlers": bowlers,
            "boundaries": boundaries,
        })
    return result


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

def _register_error_handlers(blueprint: Blueprint) -> None:
    @blueprint.errorhandler(ValidationError)
    def handle_validation_error(exc):
        return jsonify({"error": "Validation failed.", "details": exc.messages}), HTTPStatus.UNPROCESSABLE_ENTITY

    @blueprint.errorhandler(PlayerNotFoundError)
    def handle_player_not_found(exc):
        return jsonify({"error": str(exc)}), HTTPStatus.NOT_FOUND

    @blueprint.errorhandler(InningsNotFoundError)
    def handle_innings_not_found(exc):
        return jsonify({"error": str(exc)}), HTTPStatus.NOT_FOUND

    @blueprint.errorhandler(MatchNotFoundError)
    def handle_match_not_found(exc):
        return jsonify({"error": str(exc)}), HTTPStatus.NOT_FOUND

    @blueprint.errorhandler(InsufficientDataError)
    def handle_insufficient_data(exc):
        return jsonify({"error": str(exc)}), HTTPStatus.UNPROCESSABLE_ENTITY

    @blueprint.errorhandler(AnalyticsError)
    def handle_generic_analytics_error(exc):
        logger.exception("Unhandled analytics error.")
        return jsonify({"error": "An analytics error occurred.", "detail": str(exc)}), HTTPStatus.INTERNAL_SERVER_ERROR


_register_error_handlers(analytics_bp)


# ---------------------------------------------------------------------------
# Marshmallow schemas
# ---------------------------------------------------------------------------

from marshmallow import Schema, fields as ma_fields  # noqa: E402


class BattingStatsSchema(Schema):
    player_id         = ma_fields.Int(dump_default=None)
    player_name       = ma_fields.Str()
    innings_played    = ma_fields.Int()
    total_runs        = ma_fields.Int()
    balls_faced       = ma_fields.Int()
    dismissals        = ma_fields.Int()
    average           = ma_fields.Method("get_average")
    strike_rate       = ma_fields.Method("get_strike_rate")
    boundary_pct      = ma_fields.Method("get_boundary_pct")
    gini_coefficient  = ma_fields.Method("get_gini")
    coeff_variation   = ma_fields.Method("get_cv")
    batting_index     = ma_fields.Method("get_batting_index")
    consistency_label = ma_fields.Method("get_consistency_label")

    def get_average(self, obj):  return round(obj.average, 2)
    def get_strike_rate(self, obj): return round(obj.strike_rate, 2)
    def get_boundary_pct(self, obj): return round(obj.boundary_pct, 2)
    def get_gini(self, obj):     return round(obj.gini, 4)
    def get_cv(self, obj):       return round(obj.cv, 4)
    def get_batting_index(self, obj): return round(obj.batting_index, 4)
    def get_consistency_label(self, obj):
        if obj.gini < 0.30: return "consistent"
        if obj.gini < 0.50: return "moderate"
        return "erratic"


class BowlingStatsSchema(Schema):
    player_id       = ma_fields.Int()
    player_name     = ma_fields.Str()
    balls_bowled    = ma_fields.Int()
    overs_bowled    = ma_fields.Method("get_overs")
    runs_conceded   = ma_fields.Int()
    wickets         = ma_fields.Int()
    economy_rate    = ma_fields.Method("get_economy_rate")
    bowling_average = ma_fields.Method("get_bowling_average")
    dot_pct         = ma_fields.Method("get_dot_pct")
    shannon_entropy = ma_fields.Method("get_entropy")
    bowling_index   = ma_fields.Method("get_bowling_index")

    def get_overs(self, obj): return round(obj.balls_bowled / 6, 1)
    def get_economy_rate(self, obj): return round(obj.economy_rate, 2)
    def get_bowling_average(self, obj):
        return round(obj.bowling_average, 2) if math.isfinite(obj.bowling_average) else None
    def get_dot_pct(self, obj): return round(obj.dot_pct, 2)
    def get_entropy(self, obj): return round(obj.entropy, 4)
    def get_bowling_index(self, obj): return round(obj.bowling_index, 4)


_batting_schema      = BattingStatsSchema()
_bowling_schema      = BowlingStatsSchema()
_batting_schema_many = BattingStatsSchema(many=True)
_bowling_schema_many = BowlingStatsSchema(many=True)


# ---------------------------------------------------------------------------
# Existing routes
# ---------------------------------------------------------------------------

@analytics_bp.get("/innings/<int:innings_id>/batting")
def innings_batting(innings_id: int):
    params   = load_innings_query(request.args.to_dict())
    profiles = AnalyticsService.batting_profiles_for_innings(innings_id)
    profiles.sort(key=lambda p: p.batting_index, reverse=True)
    return jsonify({"innings_id": innings_id, "count": len(profiles),
                    "batsmen": _batting_schema_many.dump(profiles)}), HTTPStatus.OK


@analytics_bp.get("/innings/<int:innings_id>/bowling")
def innings_bowling(innings_id: int):
    profiles = AnalyticsService.bowling_profiles_for_innings(innings_id)
    profiles.sort(key=lambda p: p.bowling_index, reverse=True)
    return jsonify({"innings_id": innings_id, "count": len(profiles),
                    "bowlers": _bowling_schema_many.dump(profiles)}), HTTPStatus.OK


@analytics_bp.post("/select-xi")
def select_xi():
    payload = load_selection_payload(request.get_json(silent=True) or {})
    preview = AnalyticsService.pre_match_team_preview(payload["match_id"])
    return jsonify(preview), HTTPStatus.OK


@analytics_bp.get("/player/<int:player_id>/career")
def player_career(player_id: int):
    """Fixed: no longer calls missing *_safe methods."""
    params    = load_career_query(request.args.to_dict())
    stat_type = params["stat_type"]
    if stat_type == "batting":
        try:
            profile = AnalyticsService.player_career_batting(player_id)
        except InsufficientDataError:
            profile = AnalyticsService._empty_batting_stats(_get_player_or_404(player_id))
            compute_batting_profile(profile)
        stats_payload = _batting_schema.dump(profile)
        breakdown     = AnalyticsService._batting_component_breakdown(profile)
    else:
        try:
            profile = AnalyticsService.player_career_bowling(player_id)
        except InsufficientDataError:
            profile = AnalyticsService._empty_bowling_stats(_get_player_or_404(player_id))
            compute_bowling_profile(profile)
        stats_payload = _bowling_schema.dump(profile)
        breakdown     = AnalyticsService._bowling_component_breakdown(profile)
    return jsonify({"player_id": player_id, "stat_type": stat_type,
                    "stats": stats_payload, "breakdown": breakdown}), HTTPStatus.OK
@analytics_bp.get("/innings/<int:innings_id>/predict")
def predict_innings_score(innings_id: int):
    """
    Predict projected final score and win probability for an active innings.
    Called by the match frontend after every ball update.
    """
    result = predict_innings(innings_id)
    return jsonify(result), HTTPStatus.OK

# ---------------------------------------------------------------------------
# New: raw innings-history
# ---------------------------------------------------------------------------

@analytics_bp.get("/player/<int:player_id>/innings-history")
def player_innings_history(player_id: int):
    from app.extensions import db
    from app.models import BattingScorecard, BowlingScorecard, Inning, Match, Player

    player = db.session.get(Player, player_id)
    if not player:
        return jsonify({"error": "Player not found"}), HTTPStatus.NOT_FOUND

    batting_scs = (BattingScorecard.query.filter_by(player_id=player_id)
                   .join(Inning, BattingScorecard.innings_id == Inning.id)
                   .join(Match, Inning.match_id == Match.id)
                   .order_by(Match.match_date.asc()).all())
    batting_history = []
    for sc in batting_scs:
        innings = db.session.get(Inning, sc.innings_id)
        match   = db.session.get(Match, innings.match_id) if innings else None
        batting_history.append({
            "innings_id": sc.innings_id, "match_id": innings.match_id if innings else None,
            "match_date": match.match_date.isoformat() if match and match.match_date else None,
            "runs": sc.runs or 0, "balls_faced": sc.balls_faced or 0,
            "strike_rate": round(sc.strike_rate or 0, 2),
            "fours": sc.fours or 0, "sixes": sc.sixes or 0,
            "is_out": sc.is_out, "dismissal_type": sc.dismissal_type,
        })

    bowling_scs = (BowlingScorecard.query.filter_by(player_id=player_id)
                   .join(Inning, BowlingScorecard.innings_id == Inning.id)
                   .join(Match, Inning.match_id == Match.id)
                   .order_by(Match.match_date.asc()).all())
    bowling_history = []
    for sc in bowling_scs:
        innings = db.session.get(Inning, sc.innings_id)
        match   = db.session.get(Match, innings.match_id) if innings else None
        bowling_history.append({
            "innings_id": sc.innings_id,
            "match_date": match.match_date.isoformat() if match and match.match_date else None,
            "wickets": sc.wickets_taken or 0, "runs_conceded": sc.runs_conceded or 0,
            "overs": round(sc.overs_bowled or 0, 1), "economy": round(sc.economy_rate or 0, 2),
            "maidens": sc.maidens or 0,
        })

    return jsonify({"player_id": player_id, "player_name": player.name,
                    "role": player.role, "batting_history": batting_history,
                    "bowling_history": bowling_history}), HTTPStatus.OK


# ---------------------------------------------------------------------------
# New: player charts (matplotlib → base64)
# ---------------------------------------------------------------------------

@analytics_bp.get("/player/<int:player_id>/charts")
def player_charts(player_id: int):
    from app.extensions import db
    from app.models import BattingScorecard, BowlingScorecard, Inning, Match, Player
    from app.services.chart_service import (
        player_runs_trend, player_bowling_trend,
        player_scoring_mix, player_radar,
    )

    player = db.session.get(Player, player_id)
    if not player:
        return jsonify({"error": "Player not found"}), HTTPStatus.NOT_FOUND

    batting_scs = (BattingScorecard.query.filter_by(player_id=player_id)
                   .join(Inning, BattingScorecard.innings_id == Inning.id)
                   .join(Match, Inning.match_id == Match.id)
                   .order_by(Match.match_date.asc()).all())
    batting_history = [{"runs": sc.runs or 0, "balls_faced": sc.balls_faced or 0,
                        "strike_rate": round(sc.strike_rate or 0, 2),
                        "fours": sc.fours or 0, "sixes": sc.sixes or 0}
                       for sc in batting_scs]

    bowling_scs = (BowlingScorecard.query.filter_by(player_id=player_id)
                   .join(Inning, BowlingScorecard.innings_id == Inning.id)
                   .join(Match, Inning.match_id == Match.id)
                   .order_by(Match.match_date.asc()).all())
    bowling_history = [{"wickets": sc.wickets_taken or 0,
                        "runs_conceded": sc.runs_conceded or 0,
                        "economy": round(sc.economy_rate or 0, 2)}
                       for sc in bowling_scs]

    preferred_stat_type = "bowling" if (player.role or "").lower() == "bowler" else "batting"

    if preferred_stat_type == "bowling":
        try:
            career = AnalyticsService.player_career_bowling(player_id)
            compute_bowling_profile(career)
        except InsufficientDataError:
            career = AnalyticsService._empty_bowling_stats(player)
            compute_bowling_profile(career)
        career_stats = _bowling_schema.dump(career)
        breakdown    = AnalyticsService._bowling_component_breakdown(career)
    else:
        try:
            career = AnalyticsService.player_career_batting(player_id)
            compute_batting_profile(career)
        except InsufficientDataError:
            career = AnalyticsService._empty_batting_stats(player)
            compute_batting_profile(career)
        career_stats = _batting_schema.dump(career)
        breakdown    = AnalyticsService._batting_component_breakdown(career)

    return jsonify({
        "player_name":      player.name,
        "role":             player.role,
        "runs_trend":       player_runs_trend(batting_history),
        "bowling_trend":    player_bowling_trend(bowling_history),
        "scoring_mix":      player_scoring_mix(batting_history),
        "radar":            player_radar(career_stats, breakdown, preferred_stat_type),
        "radar_stat_type":  preferred_stat_type,
        "career_stats":     career_stats,
        "career_breakdown": breakdown,
    }), HTTPStatus.OK


# ---------------------------------------------------------------------------
# New: match team overview (raw)
# ---------------------------------------------------------------------------

@analytics_bp.get("/match/<int:match_id>/team-overview")
def match_team_overview(match_id: int):
    from app.extensions import db
    from app.models import Inning, Match
    match = db.session.get(Match, match_id)
    if not match:
        return jsonify({"error": "Match not found"}), HTTPStatus.NOT_FOUND
    innings_list = Inning.query.filter_by(match_id=match_id).order_by(Inning.innings_number).all()
    return jsonify({"match_id": match_id, "match_type": match.match_type,
                    "over_limit": match.over_limit,
                    "innings": _build_innings_data(innings_list)}), HTTPStatus.OK


# ---------------------------------------------------------------------------
# New: match charts (matplotlib → base64)
# ---------------------------------------------------------------------------

@analytics_bp.get("/match/<int:match_id>/charts")
def match_charts(match_id: int):
    from app.extensions import db
    from app.models import Inning, Match
    from app.services.chart_service import (
        match_run_progression, match_over_run_rate,
        match_batting_contributions, match_bowling_figures,
        match_boundary_breakdown,
    )

    match = db.session.get(Match, match_id)
    if not match:
        return jsonify({"error": "Match not found"}), HTTPStatus.NOT_FOUND

    innings_list = Inning.query.filter_by(match_id=match_id).order_by(Inning.innings_number).all()
    innings_data = _build_innings_data(innings_list)

    innings_charts = []
    for inn in innings_data:
        innings_charts.append({
            "innings_number": inn["innings_number"],
            "total_runs":     inn["total_runs"],
            "total_wickets":  inn["total_wickets"],
            "total_overs":    inn["total_overs"],
            "target":         inn.get("target"),
            "batting":        match_batting_contributions(inn["batsmen"],  inn["innings_number"]),
            "bowling":        match_bowling_figures(inn["bowlers"],        inn["innings_number"]),
            "boundary_donut": match_boundary_breakdown(inn["boundaries"],  inn["innings_number"]),
        })

    return jsonify({
        "match_id":        match_id,
        "run_progression": match_run_progression(innings_data),
        "over_run_rate":   match_over_run_rate(innings_data),
        "innings":         innings_charts,
    }), HTTPStatus.OK
