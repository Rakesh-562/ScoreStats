from flask import Blueprint,render_template
from app.models import Match,Team,Player,Inning
from app.services import StatisticsService
pages_bp=Blueprint('pages',__name__)

@pages_bp.route('/')
def home():
    '''Dashboard-Home Page'''
    teams = Team.query.all()
    players = Player.query.limit(10).all()
    matches = Match.query.all()
    live_matches = Match.query.filter_by(status='live').order_by(Match.match_date.desc()).all()
    recent_matches = Match.query.filter(Match.status != 'live').order_by(Match.match_date.desc()).limit(15).all()
    live_count = len(live_matches)
    # Build id→name map so template doesn't show raw IDs
    team_map = {t.id: t.name for t in teams}
    
    return render_template(
        'home.html', 
        teams=teams, 
        matches=matches,
        live_matches=live_matches, 
        recent_matches=recent_matches, 
        players=players, 
        live_count=live_count, 
        team_map=team_map
    )

@pages_bp.route('/match/<int:match_id>')
def match_details(match_id):
    match = Match.query.get_or_404(match_id)
        
    innings_list = Inning.query.filter_by(match_id=match_id).order_by(Inning.innings_number).all()
    innings_1 = innings_list[0] if len(innings_list) > 0 else None
    innings_2 = innings_list[1] if len(innings_list) > 1 else None
    current_innings = next((i for i in innings_list if not i.is_completed), None)
    team1_players = match.team1.players.all() if match.team1 else []
    team2_players = match.team2.players.all() if match.team2 else []
    
    batting_scorecard_1, bowling_scorecard_1 = [], []
    batting_scorecard_2, bowling_scorecard_2 = [], []
    
    if innings_1:
        batting_scorecard_1 = StatisticsService.get_batting_scorecard(innings_1.id)
        bowling_scorecard_1 = StatisticsService.get_bowling_scorecard(innings_1.id)
    if innings_2:
        batting_scorecard_2 = StatisticsService.get_batting_scorecard(innings_2.id)
        bowling_scorecard_2 = StatisticsService.get_bowling_scorecard(innings_2.id)
        
    return render_template(
        'match_details.html',
        match=match,
        team1=match.team1,
        team2=match.team2,
        innings_1=innings_1,
        innings_2=innings_2,
        current_innings=current_innings,
        team1_players=team1_players,
        team2_players=team2_players,
        batting_scorecard_1=batting_scorecard_1,
        bowling_scorecard_1=bowling_scorecard_1,
        batting_scorecard_2=batting_scorecard_2,
        bowling_scorecard_2=bowling_scorecard_2
    )

@pages_bp.route('/player/<int:player_id>')
def player_profile(player_id):
    player=Player.query.get_or_404(player_id)
    stats=StatisticsService.get_player_career_stats(player_id)
    return render_template('player_profile.html',player=player,stats=stats)

@pages_bp.route('/teams/<int:team_id>')
def team_detail(team_id):
    from app.services.team_profiles import TeamProfileService

    team=Team.query.get_or_404(team_id)
    players=Player.query.filter_by(team_id=team_id).all()
    
    # Group players by role
    players_by_role = {}
    for p in players:
        role = p.role or 'Other'
        if role not in players_by_role:
            players_by_role[role] = []
        players_by_role[role].append(p)
        
    matches=Match.query.filter((Match.team_1_id==team_id)|(Match.team_2_id==team_id)).order_by(Match.match_date.desc()).all()
    team_profile = TeamProfileService.get(team_id)
    h2h_summary = TeamProfileService.get_head_to_head_summary(team_id)
    return render_template(
        'team_details.html',
        team=team,
        players=players,
        players_by_role=players_by_role,
        matches=matches,
        team_profile=team_profile,
        h2h_summary=h2h_summary,
    )

@pages_bp.route('/teams')
def teams_list():
    teams=Team.query.all()
    return render_template('teams_list.html',teams=teams)

@pages_bp.route('/players')
def players_list():
    players=Player.query.all()
    return render_template('players_list.html',players=players)

@pages_bp.route('/matches')
def matches_list():
    matches=Match.query.order_by(Match.match_date.desc()).all()
    live_count=Match.query.filter_by(status='live').count()
    return render_template('matches_list.html',matches=matches,live_count=live_count)

@pages_bp.route('/analytics')
def analytics_dashboard():
    matches = Match.query.order_by(Match.match_date.desc()).all()
    players = Player.query.order_by(Player.name.asc()).all()
    innings = Inning.query.order_by(Inning.match_id.desc(), Inning.innings_number.asc()).all()
    return render_template(
        'analytics.html',
        matches=matches,
        players=players,
        innings=innings,
    )
@pages_bp.route('/prediction')
def prediction_dashboard():
    """ML-powered pre-match prediction page."""
    teams = Team.query.order_by(Team.name.asc()).all()
    return render_template('prediction.html', teams=teams)
