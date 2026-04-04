""""
app/services/prediction_service.py
 
This file is the bridge between your saved ML models and your Flask routes.
 
When a route calls predict_innings(innings_id), this file:
  1. Loads the model from disk (only once — then keeps it in memory)
  2. Reads the current innings state from your DB
  3. Builds a feature vector
  4. Asks the model to predict
  5. Returns the result as a plain dict
 
You never call this file directly from templates.
The route in analytics.py calls it.
"""
 
from __future__ import annotations
 
import json
import logging
import os
 
logger = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────────────────────────────────────
# Model cache
#
# These three variables live at module level.
# The first time predict_innings() is called, _load_models() fills them.
# Every call after that reuses the same objects from memory.
# This means the .pkl file is only read from disk ONCE per server restart.
# ─────────────────────────────────────────────────────────────────────────────
 
_score_model  = None   # RandomForestRegressor
_win_model    = None   # RandomForestClassifier
_feature_meta = None   # dict with feature name lists
_loaded       = False  # flag so we don't retry on every request
 
 
def _load_models():
    """Load models from models_ml/ folder into memory."""
    global _score_model, _win_model, _feature_meta, _loaded
 
    if _loaded:
        return  # already done
 
    score_path = os.path.join('models_ml', 'score_predictor.pkl')
    win_path   = os.path.join('models_ml', 'win_predictor.pkl')
    meta_path  = os.path.join('models_ml', 'feature_meta.json')
 
    if not os.path.exists(score_path):
        logger.warning(
            "Score predictor not found at %s. "
            "Run: python scripts/build_training_data.py "
            "then: python scripts/train_prediction_model.py",
            score_path
        )
        _loaded = True
        return
 
    try:
        import joblib
        _score_model = joblib.load(score_path)
        logger.info("Loaded score predictor")
 
        if os.path.exists(win_path):
            _win_model = joblib.load(win_path)
            logger.info("Loaded win predictor")
 
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                _feature_meta = json.load(f)
 
        _loaded = True
 
    except Exception as exc:
        logger.error("Failed to load ML models: %s", exc)
        _loaded = True  # stop retrying on every request
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
#
# CRITICAL: The feature names and order here must exactly match
# what train_prediction_model.py used for BASE_FEATURES and WIN_FEATURES.
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_features(innings, last_ball):
    """
    Read the current state of the innings from the DB and
    turn it into a feature vector for the ML model.
 
    Parameters
    ----------
    innings   : Inning ORM object  (the active innings)
    last_ball : Ball ORM object    (the most recent ball)
 
    Returns
    -------
    dict or None
        None if there isn't enough data yet (less than 1 over bowled).
    """
    from app.models import BattingScorecard, BowlingScorecard, Ball
 
    over_limit       = innings.match.over_limit or 20
    total_ball_limit = over_limit * 6
 
    # Count legal balls so far in this innings
    balls_bowled = (
        Ball.query
        .filter_by(inning_id=innings.id, is_legal_delivery=True)
        .count()
    )
 
    # We need at least 1 full over before predicting
    if balls_bowled < 6:
        return None
 
    balls_remaining  = max(0, total_ball_limit - balls_bowled)
    overs_completed  = balls_bowled // 6
    crr = (innings.total_runs / (balls_bowled / 6.0)) if balls_bowled > 0 else 0.0
 
    # 2nd innings only
    rrr            = 0.0
    runs_to_target = 0
    target         = innings.target or 0
 
    if innings.innings_number == 2 and target > 0:
        runs_to_target  = max(0, target - innings.total_runs)
        overs_remaining = balls_remaining / 6.0
        rrr = (runs_to_target / overs_remaining) if overs_remaining > 0 else 99.0
 
    # Current batter (striker from last ball)
    striker_sc = BattingScorecard.query.filter_by(
        innings_id=innings.id,
        player_id=last_ball.batsman_id
    ).first()
 
    if striker_sc and striker_sc.balls_faced:
        striker_runs        = striker_sc.runs or 0
        striker_balls       = striker_sc.balls_faced or 0
        striker_strike_rate = striker_sc.strike_rate or 0.0
        striker_fours       = striker_sc.fours or 0
        striker_sixes       = striker_sc.sixes or 0
    else:
        striker_runs = striker_balls = striker_fours = striker_sixes = 0
        striker_strike_rate = 0.0
 
    # Current bowler
    bowler_sc = BowlingScorecard.query.filter_by(
        innings_id=innings.id,
        player_id=last_ball.bowler_id
    ).first()
 
    if bowler_sc and bowler_sc.balls_bowled:
        bowler_economy = bowler_sc.economy_rate or 0.0
        bowler_wickets = bowler_sc.wickets_taken or 0
        bowler_dots    = bowler_sc.dots or 0
    else:
        bowler_economy = bowler_wickets = bowler_dots = 0
 
    return {
        # Game state
        'runs_so_far':         innings.total_runs,
        'wickets_fallen':      innings.total_wickets,
        'balls_bowled':        balls_bowled,
        'balls_remaining':     balls_remaining,
        'overs_completed':     overs_completed,
        'current_run_rate':    round(crr, 2),
        # 2nd innings
        'required_run_rate':   round(rrr, 2),
        'runs_to_target':      runs_to_target,
        'target':              target,
        # Batter
        'striker_runs':        striker_runs,
        'striker_balls':       striker_balls,
        'striker_strike_rate': round(striker_strike_rate, 2),
        'striker_fours':       striker_fours,
        'striker_sixes':       striker_sixes,
        # Bowler
        'bowler_economy':      round(bowler_economy, 2),
        'bowler_wickets':      bowler_wickets,
        'bowler_dots':         bowler_dots,
        # Context
        'innings_number':      innings.innings_number,
        'over_limit':          over_limit,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main public function — called from analytics.py route
# ─────────────────────────────────────────────────────────────────────────────
 
def predict_innings(innings_id: int) -> dict:
    """
    Predict the final score and win probability for an active innings.
 
    Called by:
        app/routes/api/analytics.py  →  predict_innings_score()
 
    Parameters
    ----------
    innings_id : int
        Primary key of the innings to predict.
 
    Returns
    -------
    dict with these keys:
 
        available         bool    False if models not trained or not enough data
        message           str     Human-readable status or explanation
 
        innings_id        int
        innings_number    int     1 or 2
        current_score     str     e.g. "87/3"
        current_overs     float   e.g. 10.4
 
        predicted_score   int     e.g. 162
        score_range_low   int     e.g. 148   (15th percentile of trees)
        score_range_high  int     e.g. 176   (85th percentile of trees)
 
        win_probability   float   0.0–1.0, only for 2nd innings, else None
        confidence        str     "high" / "medium" / "low"
    """
    _load_models()
 
    # If models don't exist yet, return gracefully
    if _score_model is None:
        return {
            'available': False,
            'message': (
                'Prediction model not trained yet. '
                'Run: python scripts/build_training_data.py '
                'then: python scripts/train_prediction_model.py'
            )
        }
 
    from app.models import Inning, Ball
    from app.extensions import db
    import numpy as np
 
    innings = db.session.get(Inning, innings_id)
    if not innings:
        return {'available': False, 'message': f'Innings {innings_id} not found.'}
 
    if innings.is_completed:
        return {
            'available':   False,
            'message':     'Innings already completed.',
            'final_score': innings.total_runs
        }
 
    last_ball = (
        Ball.query
        .filter_by(inning_id=innings_id)
        .order_by(Ball.id.desc())
        .first()
    )
 
    if not last_ball:
        return {'available': False, 'message': 'No balls recorded yet.'}
 
    features = _build_features(innings, last_ball)
 
    if features is None:
        return {
            'available': False,
            'message':   'Need at least 1 complete over before predicting.'
        }
 
    # ── Score prediction ───────────────────────────────────────────────────
 
    # Get feature order from the saved meta file
    # This guarantees the same order as training
    base_features = _feature_meta['base_features'] if _feature_meta else [
        'runs_so_far', 'wickets_fallen', 'balls_bowled', 'balls_remaining',
        'overs_completed', 'current_run_rate', 'striker_runs', 'striker_balls',
        'striker_strike_rate', 'striker_fours', 'striker_sixes',
        'bowler_economy', 'bowler_wickets', 'bowler_dots', 'over_limit',
    ]
 
    feature_vector = [[features[f] for f in base_features]]
 
    predicted_score = int(_score_model.predict(feature_vector)[0])
 
    # Get predictions from each individual tree to build a confidence range
    tree_preds = np.array([
        tree.predict(feature_vector)[0]
        for tree in _score_model.estimators_
    ])
    score_low  = int(np.percentile(tree_preds, 15))
    score_high = int(np.percentile(tree_preds, 85))
    spread     = score_high - score_low
 
    # Confidence based on how much the trees disagree
    if spread <= 15:
        confidence = 'high'
    elif spread <= 30:
        confidence = 'medium'
    else:
        confidence = 'low'
 
    # ── Win probability (2nd innings only) ────────────────────────────────
 
    win_probability = None
 
    if innings.innings_number == 2 and _win_model is not None and innings.target:
        win_features = _feature_meta['win_features'] if _feature_meta else base_features
        win_vector   = [[features.get(f, 0) for f in win_features]]
 
        # predict_proba returns [[prob_lose, prob_win]]
        proba = _win_model.predict_proba(win_vector)[0]
        win_probability = round(float(proba[1]), 3)
 
    # ── Message ───────────────────────────────────────────────────────────
 
    current_score_str = f"{innings.total_runs}/{innings.total_wickets}"
    overs_done        = innings.total_overs
 
    if innings.innings_number == 1:
        message = (
            f"After {overs_done} overs ({current_score_str}), "
            f"projected total: {score_low}–{score_high} runs."
        )
    else:
        runs_needed = max(0, (innings.target or 0) - innings.total_runs)
        over_limit  = innings.match.over_limit or 20
        balls_left  = max(0, int((over_limit - overs_done) * 6))
        message = (
            f"Need {runs_needed} off {balls_left} balls. "
            f"Projected: {score_low}–{score_high}. "
            f"Win chance: {int((win_probability or 0.5) * 100)}%."
        )
 
    return {
        'available':        True,
        'innings_id':       innings_id,
        'innings_number':   innings.innings_number,
        'current_score':    current_score_str,
        'current_overs':    innings.total_overs,
        'predicted_score':  predicted_score,
        'score_range_low':  score_low,
        'score_range_high': score_high,
        'win_probability':  win_probability,
        'confidence':       confidence,
        'message':          message,
    }