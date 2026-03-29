from flask import Blueprint
api_bp = Blueprint('api', __name__)
from . import matches,teams,players,innings,balls
from .teams import teams_bp
from .players import player_bp
from .matches import matches_bp
from .balls import balls_bp
from .analytics import analytics_bp
 
api_bp.register_blueprint(teams_bp,url_prefix='/teams')
api_bp.register_blueprint(player_bp,url_prefix='/players')
api_bp.register_blueprint(matches_bp,url_prefix='/matches')
api_bp.register_blueprint(balls_bp,url_prefix='/balls')
api_bp.register_blueprint(analytics_bp,url_prefix='/analytics')