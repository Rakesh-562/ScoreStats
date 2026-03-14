from flask import Blueprint, jsonify, render_template, request

from app.models import Ball, Inning, Match, Player, Team
from app.services import BallService, InningsService, MatchService, StatisticsService


main_bp = Blueprint("main", __name__)


@main_bp.route("/create_match", methods=["POST"])
def create_match():
    data = request.get_json(silent=True) or {}
    team_1_id = data.get("team_1_id", data.get("team1_id"))
    team_2_id = data.get("team_2_id", data.get("team2_id"))

    try:
        match = MatchService.create_match(
            team_1_id=team_1_id,
            team_2_id=team_2_id,
            over_limit=data.get("over_limit", 20),
            match_type=data.get("match_type", "t20"),
            match_date=data.get("match_date"),
        )
        return jsonify({"message": "Match created successfully", "match_id": match.id}), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 400


@main_bp.route("/start_inning/<int:match_id>", methods=["POST"])
def start_inning(match_id):
    data = request.get_json(silent=True) or {}
    batting_team_id = data.get("batting_team_id", data.get("batting_team"))
    bowling_team_id = data.get("bowling_team_id", data.get("bowling_team"))
    innings_number = data.get("innings_number", data.get("inning_number", 1))

    try:
        inning_obj = InningsService.start_innings(
            match_id=match_id,
            batting_team_id=batting_team_id,
            bowling_team_id=bowling_team_id,
            innings_number=innings_number,
        )
        return jsonify({"message": "Inning started successfully", "inning_id": inning_obj.id}), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 400


@main_bp.route("/api/record_ball", methods=["POST"])
def api_record_ball():
    data = request.get_json(silent=True) or {}
    try:
        new_ball = BallService.record_ball(**data)
        return jsonify({"message": "Ball recorded successfully", "ball_id": new_ball.id}), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 400


@main_bp.route("/match/<int:match_id>/summary", methods=["GET"])
def get_match_summary(match_id):
    balls = Ball.query.join(Inning).filter(Inning.match_id == match_id).all()
    total_runs = sum(ball.total_runs for ball in balls)
    total_wickets = sum(1 for ball in balls if ball.is_wicket)

    if balls:
        last_ball = max(balls, key=lambda b: (b.over_number, b.ball_number))
        overs = f"{last_ball.over_number}.{last_ball.ball_number}"
    else:
        overs = "0.0"

    return jsonify(
        {
            "match_id": match_id,
            "total_runs": total_runs,
            "total_wickets": total_wickets,
            "overs": overs,
        }
    ), 200


@main_bp.route("/")
def home():
    teams = Team.query.all()
    players = Player.query.limit(10).all()
    live_matches = Match.query.filter_by(status="live").order_by(Match.match_date.desc()).all()
    recent_matches = Match.query.filter(Match.status != "live").order_by(Match.match_date.desc()).limit(15).all()
    live_count = len(live_matches)
    # Build id→name map so template doesn't show raw IDs
    team_map = {t.id: t.name for t in teams}
    return render_template("home.html", teams=teams, live_matches=live_matches,
                           recent_matches=recent_matches, players=players,
                           live_count=live_count, team_map=team_map)


@main_bp.route("/match/<int:match_id>")
def score_match(match_id):
    from app.extensions import db as _db
    match = _db.session.get(Match, match_id)
    if not match:
        from flask import abort
        abort(404)
    innings_list = Inning.query.filter_by(match_id=match_id).order_by(Inning.innings_number).all()
    innings_1 = innings_list[0] if len(innings_list) > 0 else None
    innings_2 = innings_list[1] if len(innings_list) > 1 else None

    batting_scorecard_1, bowling_scorecard_1 = [], []
    batting_scorecard_2, bowling_scorecard_2 = [], []
    if innings_1:
        batting_scorecard_1 = StatisticsService.get_batting_scorecard(innings_1.id)
        bowling_scorecard_1 = StatisticsService.get_bowling_scorecard(innings_1.id)
    if innings_2:
        batting_scorecard_2 = StatisticsService.get_batting_scorecard(innings_2.id)
        bowling_scorecard_2 = StatisticsService.get_bowling_scorecard(innings_2.id)

    # Resolve team names
    team1 = db.session.get(Team, match.team_1_id)
    team2 = db.session.get(Team, match.team_2_id)

    return render_template(
        "match_scorecard.html",
        match=match,
        team1=team1,
        team2=team2,
        innings_1=innings_1,
        innings_2=innings_2,
        batting_scorecard_1=batting_scorecard_1,
        bowling_scorecard_1=bowling_scorecard_1,
        batting_scorecard_2=batting_scorecard_2,
        bowling_scorecard_2=bowling_scorecard_2,
    )


@main_bp.route("/stats/<int:player_id>")
def player_stats(player_id):
    player = Player.query.get_or_404(player_id)
    stats = StatisticsService.get_player_career_stats(player_id)
    return render_template("player_profile.html", player=player, stats=stats)


@main_bp.route("/predict_winner/<int:team_a>/<int:team_b>")
def predict_winner(team_a, team_b):
    return jsonify(
        {
            "success": False,
            "error": "Prediction endpoint is not implemented yet.",
            "team_a": team_a,
            "team_b": team_b,
        }
    ), 501