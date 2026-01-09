from extensions import db
class Team(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    team_id = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)
class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    team1 = db.Column(db.String(100), db.ForeignKey('team.id'), nullable=False)
    team2 = db.Column(db.String(100), db.ForeignKey('team.id'), nullable=False)
    over_limit = db.Column(db.Integer, default=20)
    status = db.Column(db.String(50), default="scheduled")
class Innings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    match_id = db.Column(db.Integer, db.ForeignKey('match.id'), nullable=False)
    batting_team = db.Column(db.String(100), db.ForeignKey('team.id'), nullable=False)
    bowling_team = db.Column(db.String(100), db.ForeignKey('team.id'), nullable=False)
    innings_number = db.Column(db.Integer, nullable=False)
    is_completed = db.Column(db.Boolean, default=False)
class Ball(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    inning_id = db.Column(db.Integer, db.ForeignKey('innings.id'), nullable=False)
    over= db.Column(db.Integer, default=0)
    balls= db.Column(db.Integer, default=0)
    striker_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    non_striker_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    bowler_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    runs = db.Column(db.Integer, default=0)
    wickets = db.Column(db.Integer, default=0)
    extras = db.Column(db.Integer, default=0)
    is_wicket = db.Column(db.Boolean, default=False)
    dismissed_player_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=True)