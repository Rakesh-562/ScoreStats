from flask import Flask,render_template

from config import *

from extensions import db
def create_app():
    app= Flask(__name__)
    app.config.from_object("config")
    db.init_app(app)
    from routes import main
    app.register_blueprint(main)
    with app.app_context():
        db.create_all()
    return app
app= create_app()
if __name__== "__main__":
    app.run(debug= True)


















