from random import random, choice
from app.extensions import db
from app.models import Match,Team,Inning
from datetime import datetime
class MatchService:
    '''
    Handles match lifecycle and operations.
    
    Responsibilities:
    - Create matches
    - Update match status
    - Record toss
    - Get match details
    
    '''
    @staticmethod
    def create_match(team_1_id=None, team_2_id=None, over_limit=None, match_type=None, match_date=None, **kwargs):
        '''
        create a new match        
        '''
        # Backward-compat for older callers using team1_id/team2_id.
        if team_1_id is None:
            team_1_id = kwargs.pop("team1_id", None)
        if team_2_id is None:
            team_2_id = kwargs.pop("team2_id", None)

        team1=db.session.get(Team,team_1_id)
        team2=db.session.get(Team,team_2_id)
        if not team1 or not team2:
            raise ValueError(f"Invalid team ids")
        if team_1_id==team_2_id:
            raise ValueError(f"Team cannot play against itself")
        match=Match(
            team_1_id=team_1_id,
            team_2_id=team_2_id,
            match_type=match_type,
            over_limit=over_limit,
            match_date=match_date or datetime.utcnow(),
            status='scheduled',
        )
        db.session.add(match)
        db.session.commit()
        return match
    @staticmethod 
    def record_toss(match_id,toss_winner_id,toss_decision):
        '''
        record toss result
        '''
        match =db.session.get(Match,match_id)
        if not match:
            raise ValueError(f"Match {match_id} not found")
        toss_winner_id=random.choice([match.team_1_id,match.team_2_id]) if toss_winner_id is None or toss_winner_id not in (match.team_1_id,match.team_2_id) else toss_winner_id
        toss_decision=choice(['bat','field']) if toss_decision not in ['bat','field'] else toss_decision
        match.toss_winner=toss_winner_id
        match.toss_decision=toss_decision
        db.session.commit()
        return match
    @staticmethod
    def get_match_summary(match_id):
        match=db.session.get(Match,match_id)
        if not match:
            raise ValueError(f"Match {match_id} not found")
        innings=Inning.query.filter_by(match_id=match_id).order_by(Inning.innings_number).all()
        return {
            'match_details':match.to_dict(),
            'innings':[i.to_dict() for i in innings],
            'status':match.status,
            "result":{
                'winner_id':match.winner_id,
                'win_margin':match.win_margin
            } if match.status=='completed' else None
        }
    @staticmethod
    def get_live_matches():
        return Match.query.filter_by(status='live').all()
    @staticmethod
    def update_match_status(match_id,status):
        valid_statuses=['scheduled','live','completed','abandoned']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status, must be one of {valid_statuses}")
        match=db.session.get(Match,match_id)
        if not match:
            raise ValueError(f"Match {match_id} not found")
        match.status=status
        db.session.commit()
        return match
    
