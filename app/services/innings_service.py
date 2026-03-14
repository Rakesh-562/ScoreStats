from app.extensions import db
from app.models import Inning,Match
from datetime import datetime
class InningsService:
    '''
    Manages innings lifecycle
     Responsibilities:
    - Start a new innings
    - Complete an innings
    - Get innings status
    - Calculate required run rate
    '''
    @staticmethod
    def start_innings(match_id,batting_team_id,bowling_team_id,innings_number=None, **kwargs):
        '''
        Start a new innings
        
        :param match_id: Which match
        :param batting_team_id: Team batting
        :param bowling_team_id: team bowling
        :param innings_number: 1 or 2 (or 3,4 for test matches)
        '''
        if innings_number is None:
            innings_number = kwargs.pop("inning_number", None)
        existing=Inning.query.filter_by(match_id=match_id,innings_number=innings_number).first()
        if existing:
            raise ValueError(f"Innings {innings_number} already exists")
        match=Match.query.get(match_id)
        if not match:
            raise ValueError(f"Match {match_id} not found")
        target=None
        if innings_number==2:
            first_innings=Inning.query.filter_by(match_id=match_id,innings_number=1).first()
            if first_innings:
                target=first_innings.total_runs+1
        innings=Inning(
            match_id=match_id,
            batting_team_id=batting_team_id,
            bowling_team_id=bowling_team_id,
            innings_number=innings_number,
            target=target,
        )
        db.session.add(innings)
        if match.status=='scheduled':
            match.status='live'
        db.session.commit()
        return innings
    @staticmethod
    def complete_innings(innings_id):
        innings=db.session.get(Inning,innings_id)
        if not innings:
            raise ValueError(f"Innings {innings_id} not found")
        innings.is_completed=True
        innings.updated_at=datetime.utcnow()
        db.session.commit()
        InningsService._check_match_completion(innings.match_id)
        return innings
    @staticmethod
    def _check_match_completion(match_id):
        match=db.session.get(Match,match_id)
        if not match:
            return None
        innings_list=Inning.query.filter_by(match_id=match_id).all()
        if len(innings_list)==2 and all(i.is_completed for i in innings_list):
            first_innings=next(i for i in innings_list if i.innings_number==1)
            second_innings=next(i for i in innings_list if i.innings_number==2)
            if second_innings.total_runs>first_innings.total_runs:
                match.winner_id=second_innings.batting_team_id
                wickets_remaining=10-second_innings.total_wickets
                match.win_margin=f"by {wickets_remaining} wickets"
            elif first_innings.total_runs>second_innings.total_runs:
                match.winner_id=first_innings.batting_team_id
                run_difference=first_innings.total_runs-second_innings.total_runs
                match.win_margin=f"by {run_difference} runs"
            else:
                match.win_margin='match tied'
            match.status='completed'
            db.session.commit()
        return match

    @staticmethod
    def get_innings_summary(inning_id):
        innings=db.session.get(Inning,inning_id)
        if not innings:
            return None
        from app.services.statistics_service import StatisticsService
        return {
            "innings_details":innings.to_dict(),
            "batting_scorecard":StatisticsService.get_batting_scorecard(inning_id),
            "bowling_scorecard":StatisticsService.get_bowling_scorecard(inning_id),
            "partnerships":StatisticsService.get_partnerships(inning_id),
            "run_rate":innings.run_rate,
            "required_run_rate":innings.required_run_rate
        }
