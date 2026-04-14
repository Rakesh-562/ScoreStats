"""
app/routes/api/prediction.py
=============================

REST endpoints for the ML-based match prediction layer.

All endpoints are registered under ``/api/v1/prediction/`` via the
parent api_bp blueprint in ``app/routes/api/__init__.py``.

Endpoints
---------
``POST /api/v1/prediction/train``
    Trigger (re-)training of all ML models from completed matches.

``POST /api/v1/prediction/teams``
    Full pre-match prediction for two selected teams:
    - Best XI for both teams
    - Predicted innings score for both teams
    - Win probability
    - Narrative matchup analysis

``POST /api/v1/prediction/match/<id>``
    Legacy match-based prediction for scheduled/live matches.

``GET  /api/v1/prediction/feature-importances``
    Returns feature importance vectors from trained models.
"""

from __future__ import annotations

from http import HTTPStatus

from flask import Blueprint, jsonify, request

prediction_bp = Blueprint("prediction", __name__)


def _prediction_to_payload(pred):
    def _feat_dict(f):
        if f is None:
            return {}
        return {
            "team_id": f.team_id,
            "team_name": f.team_name,
            "avg_runs_scored": round(f.avg_runs_scored, 1),
            "avg_wickets_lost": round(f.avg_wickets_lost, 1),
            "avg_run_rate": round(f.avg_run_rate, 2),
            "top_order_avg_sr": round(f.top_order_avg_sr, 1),
            "boundary_rate": round(f.boundary_rate, 2),
            "dot_ball_pct_batting": round(f.dot_ball_pct_batting, 3),
            "avg_partnerships_runs": round(f.avg_partnerships_runs, 1),
            "innings_consistency": round(f.innings_consistency, 3),
            "avg_runs_conceded": round(f.avg_runs_conceded, 1),
            "avg_wickets_taken": round(f.avg_wickets_taken, 1),
            "avg_economy": round(f.avg_economy, 2),
            "dot_ball_pct_bowling": round(f.dot_ball_pct_bowling, 3),
            "avg_extras_conceded": round(f.avg_extras_conceded, 1),
            "bowling_consistency": round(f.bowling_consistency, 3),
            "h2h_wins": f.h2h_wins,
            "h2h_losses": f.h2h_losses,
            "h2h_avg_margin_runs": round(f.h2h_avg_margin_runs, 1),
            "xi_avg_batting_index": round(f.xi_avg_batting_index, 4),
            "xi_avg_bowling_index": round(f.xi_avg_bowling_index, 4),
            "xi_batting_depth": round(f.xi_batting_depth, 3),
            "xi_bowling_depth": round(f.xi_bowling_depth, 3),
        }

    return {
        "team1_id": pred.team1_id,
        "team2_id": pred.team2_id,
        "team1_name": pred.team1_name,
        "team2_name": pred.team2_name,
        "batting_first_id": pred.batting_first_id,
        "batting_first_name": pred.batting_first_name,
        "innings1_predicted": pred.innings1_predicted,
        "innings1_range": list(pred.innings1_range),
        "innings2_predicted": pred.innings2_predicted,
        "innings2_range": list(pred.innings2_range),
        "win_probability_team1": pred.win_probability_team1,
        "win_probability_team2": pred.win_probability_team2,
        "predicted_winner_id": pred.predicted_winner_id,
        "predicted_winner_name": pred.predicted_winner_name,
        "win_probability_note": pred.win_probability_note,
        "model_used": pred.model_used,
        "training_status": pred.training_status,
        "team1_xi": pred.team1_xi,
        "team2_xi": pred.team2_xi,
        "team1_features": _feat_dict(pred.team1_features),
        "team2_features": _feat_dict(pred.team2_features),
        "analysis_summary": pred.analysis_summary,
        "analysis_points": pred.analysis_points,
    }


@prediction_bp.post("/train")
def train_models():
    """
    Trigger model training from all completed matches in the database.

    Returns
    -------
    JSON with training status and sample counts.
    """
    try:
        from app.services.ml_service import MLService
        MLService.invalidate_cache()
        status = MLService.ensure_trained()
        return jsonify({"success": True, "training": status}), HTTPStatus.OK
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), HTTPStatus.INTERNAL_SERVER_ERROR


@prediction_bp.post("/teams")
def predict_teams():
    """Predict a matchup from two user-selected teams."""
    try:
        from app.services.ml_service import MLService

        data = request.get_json(silent=True) or {}
        team1_id = data.get("team1_id")
        team2_id = data.get("team2_id")
        batting_first_id = data.get("batting_first_id")
        over_limit = data.get("over_limit")

        if team1_id is None or team2_id is None:
            return jsonify({
                "success": False,
                "error": "team1_id and team2_id are required.",
            }), HTTPStatus.BAD_REQUEST

        pred = MLService.predict_teams(
            int(team1_id),
            int(team2_id),
            batting_first_id=int(batting_first_id) if batting_first_id is not None else None,
            over_limit=int(over_limit) if over_limit is not None else 20,
        )
        return jsonify({"success": True, "prediction": _prediction_to_payload(pred)}), HTTPStatus.OK

    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), HTTPStatus.BAD_REQUEST
    except Exception as exc:
        import traceback
        return jsonify({
            "success": False,
            "error": str(exc),
            "trace": traceback.format_exc(),
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@prediction_bp.get("/match/<int:match_id>")
@prediction_bp.post("/match/<int:match_id>")
def predict_match(match_id: int):
    """
    Full pre-match prediction.

    Optional JSON body
    ------------------
    ``batting_first_id`` : int
        Override which team bats first. If omitted, inferred from toss.

    Response 200
    ------------
    .. code-block:: json

        {
            "success": true,
            "prediction": {
                "team1_id": 1,
                "team2_id": 2,
                "team1_name": "Mumbai Indians",
                "team2_name": "Chennai Super Kings",
                "batting_first_id": 1,
                "batting_first_name": "Mumbai Indians",
                "innings1_predicted": 167.4,
                "innings1_range": [148.2, 186.6],
                "innings2_predicted": 154.8,
                "innings2_range": [137.1, 172.5],
                "win_probability_team1": 0.613,
                "win_probability_team2": 0.387,
                "predicted_winner_id": 1,
                "predicted_winner_name": "Mumbai Indians",
                "model_used": "ml",
                "team1_xi": [...],
                "team2_xi": [...],
                "team1_features": {...},
                "team2_features": {...}
            }
        }
    """
    try:
        from app.services.ml_service import MLService

        data = request.get_json(silent=True) or {}
        batting_first_id = data.get("batting_first_id") or request.args.get("batting_first_id", type=int)

        pred = MLService.predict_match(match_id, batting_first_id=batting_first_id)
        return jsonify({"success": True, "prediction": _prediction_to_payload(pred)}), HTTPStatus.OK

    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), HTTPStatus.BAD_REQUEST
    except Exception as exc:
        import traceback
        return jsonify({
            "success": False,
            "error": str(exc),
            "trace": traceback.format_exc()
        }), HTTPStatus.INTERNAL_SERVER_ERROR


@prediction_bp.get("/feature-importances")
def feature_importances():
    """Return feature importances from trained RandomForest models."""
    try:
        from app.services.ml_service import MLService
        MLService.ensure_trained()
        importances = MLService.get_feature_importances()
        return jsonify({"success": True, "importances": importances}), HTTPStatus.OK
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), HTTPStatus.INTERNAL_SERVER_ERROR
