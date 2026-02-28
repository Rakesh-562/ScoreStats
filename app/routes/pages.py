from flask import Blueprint,render_template
from app.models import Match,Team,Player,Inning
from app.services import StatisticsService
pages_bp=Blueprint('pages',__name__)
@pages_bp.route('/')
def home():
    '''Dashboard-Home Page'''
    matches=Match.query.order_by(Match.match_date.desc()).limit(5).all()
    team=Team.query.all()
    players=Player.query.limit(10).all()
    live_count=Match.query.filter_by(status='live').count()
    return render_template('home.html',matches=matches,teams=team,players=players,live_count=live_count)
@pages_bp.route('/match/<int:match_id>')
def match_details(match_id):
    match=Match.query.get_or_404(match_id)
    innings_list=Inning.query.filter_by(match_id=match_id).order_by(Inning.innings_number).all()
    innings_1=innings_list[0] if len(innings_list)>0 else None
    innings_2=innings_list[1] if len(innings_list)>1 else None
    batting_scorecard_1=[]
    bowling_scorecard_1=[]
    if innings_1:
        batting_scorecard_1=StatisticsService.get_batting_scorecard(innings_1.id)
        bowling_scorecard_1=StatisticsService.get_bowling_scorecard(innings_1.id)
    
    return render_template('match_details.html',match=match,innings_1=innings_1,innings_2=innings_2,batting_scorecard_1=batting_scorecard_1,bowling_scorecard_1=bowling_scorecard_1)
@pages_bp.route('/player/<int:player_id>')
def player_profile(player_id):
    player=Player.query.get_or_404(player_id)
    stats=StatisticsService.get_player_career_stats(player_id)
    return render_template('player_profile.html',player=player,stats=stats)

@pages_bp.route('/teams/<int:team_id>')
def team_detail(team_id):
    team=Team.query.get_or_404(team_id)
    players=Player.query.filter_by(team_id=team_id).all()
    matches=Match.query.filter((Match.team_1_id==team_id)|(Match.team_2_id==team_id)).order_by(Match.match_date.desc()).all()
    return render_template('team_details.html',team=team,players=players,matches=matches)
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
    return render_template('matches_list.html',matches=matches)
