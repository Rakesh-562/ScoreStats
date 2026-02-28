from flask import Flask
from app.config import config
from app.websockets import register_socket_events
from app.extensions import db,socketio,cache,login_manager,migrate
def create_app(config_name="default"):
    """ Applicaion factory function 
    Creates and config Flask application
     Args:
      config_name;'development','production','testing'or 'default'
       returns:
        Configured Flask Application. """
    app=Flask(__name__)
    app.config.from_object(config[config_name])
    db.init_app(app)
    migrate.init_app(app,db)
    socketio.init_app(app)
    register_socket_events(socketio)
    cache.init_app(app)
    # login_manager.init_app(app)
    from app.routes import register_blueprints
    register_blueprints(app)
    # from .middleware.error_handlers import register_error_handlers
    # register_error_handlers(app)
    with app.app_context():
        from app.models import (
            Team, Player, Match, Inning, Ball,
            BattingScorecard, BowlingScorecard, Partnership
        )
        if config_name=="development":
            db.create_all()
    return app


















