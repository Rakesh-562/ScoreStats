from datetime import datetime
from app.extensions import db
class Match(db.Model):
    """
    Represent a Match
    A match is played between two teams and has multiple innings
    """
    __tablename__='match'
    id=db.Column(db.Integer,primary_key=True)
    # Info
    team_1_id=db.Column(db.Integer,db.ForeignKey('team.id'),nullable=False,comment="Team A ID")
    team_2_id=db.Column(db.Integer,db.ForeignKey('team.id'),nullable=False,comment="Team B ID")
    match_date=db.Column(db.DateTime,nullable=False,comment="Date of the Match")
    match_type=db.Column(db.String(50),default= 't20',nullable=False,comment="Type of Match (Test, ODI, T20)")
    over_limit=db.Column(db.Integer,comment="Over limit for limited overs matches")
    created_at=db.Column(db.DateTime,default=datetime.utcnow,nullable=False,comment="Record Creation Timestamp")
    updated_at=db.Column(db.DateTime,default=datetime.utcnow,onupdate=datetime.utcnow,nullable=False,comment="Record Update Timestamp")
    status=db.Column(db.String(50),default='scheduled',nullable=False,comment="Match Status (scheduled, ongoing, completed)")
    toss_winner=db.Column(db.Integer,db.ForeignKey('team.id'),comment="Team ID who won the toss")
    toss_team = db.relationship("Team", foreign_keys=[toss_winner], backref="toss_wins")
    toss_decision=db.Column(db.String(10),comment="Toss Decision (bat/field)")
    # Relationships
    innings=db.relationship("Inning",backref="match",lazy='dynamic',cascade='all,delete-orphan',order_by='Inning.innings_number')
    team1 = db.relationship("Team", foreign_keys=[team_1_id], backref="matches_as_team1")
    team2 = db.relationship("Team", foreign_keys=[team_2_id], backref="matches_as_team2")
    
    winner_id=db.Column(db.Integer,db.ForeignKey('team.id'),comment="Team ID who won the match")
    winner_team = db.relationship("Team", foreign_keys=[winner_id], backref="wins")
    
    win_margin=db.Column(db.String(50),comment="Margin of Victory (runs/wickets)")
    
    man_of_the_match=db.Column(db.Integer,db.ForeignKey('player.id'),comment="PlayerID awarded Man of the Match")
    mom_player = db.relationship("Player", foreign_keys=[man_of_the_match], backref="man_of_the_match_awards")
    def __init__(self,**kwargs):
        super(Match,self).__init__(**kwargs)
        if self.team_1_id and self.team_2_id and self.team_1_id==self.team_2_id:
            raise ValueError("A team cannot play against itself.")
    def __repr__(self):
        return f"<Match {self.team_1_id} vs {self.team_2_id} on {self.match_date.date()}>"
    def to_dict(self):
        """Convert Match object to dictionary"""
        return {
            'id':self.id,
            'team_1_id':self.team_1_id,
            'team_2_id':self.team_2_id,
            'match_date':self.match_date.isoformat(),
            'match_type':self.match_type,
            'over_limit':self.over_limit,
            'status':self.status,
            'toss_winner':self.toss_winner,
            'toss_decision':self.toss_decision,
            'winner_id':self.winner_id,
            'win_margin':self.win_margin,
            'created_at':self.created_at.isoformat(),
            'updated_at':self.updated_at.isoformat()
        }