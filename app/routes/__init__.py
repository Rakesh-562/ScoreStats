# from .main import main_bp
from .api import api_bp
from .pages import pages_bp
def register_blueprints(app):
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(pages_bp)