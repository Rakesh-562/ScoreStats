from app.extensions import db 
from app.models import Ball,Inning,BattingScorecard,BowlingScorecard,Partnership,Player
from sqlalchemy.exc import SQLAlchemyError
class BallService:
    '''
    Docstring for BallService
    handles all the ball recordings logic with cricket rules 
    Responsibilities:
    -Record a ball with validation 
    -auto-increment over/ball numbers
    -Rotate strikers on odd runs
    -Swap strikers at over completion
    -Updateall aggregation tables 
    -handle extras (wide,no-ball)
    '''
    @staticmethod
    def record_ball(innings_id,striker_id,non_striker_id,bowler_id,runs=0,extras=0,extra_type=None,is_wicket=False,wicket_type=None,dismissed_player_id=None,fielder_id=None,**kwargs):
        """      
        record a single ball 
        """
        try:
            if runs < 0 or runs > 6:
                raise ValueError("Runs must be between 0 and 6")
            if extras < 0:
                raise ValueError("Extras must be 0 or more")
            if striker_id == non_striker_id:
                raise ValueError("Striker and non-striker must be different players")
            innings=db.session.get(Inning,innings_id)
            if not innings:
                raise ValueError(f"Inning {innings_id} not found")
            if innings.is_completed:
                raise ValueError("cannot record ball in completed innings")
            BallService._validate_player_teams(innings, striker_id, non_striker_id, bowler_id)
            BallService._validate_batsmen_not_out(innings_id, striker_id, non_striker_id)
            last_ball=Ball.query.filter_by(inning_id=innings_id)\
                .order_by(Ball.id.desc())\
                .first()
            is_free_hit = BallService._is_free_hit_next(last_ball) if last_ball else False
            if dismissed_player_id is None:
                dismissed_player_id = kwargs.pop("dismissed_player_id", None) # Fix: Typo in kwarg key
            if is_wicket:
                if is_free_hit and wicket_type != "run-out":
                    raise ValueError("On a free hit, only run-out can result in wicket")
                if dismissed_player_id is None:
                    dismissed_player_id = striker_id
                if dismissed_player_id not in (striker_id, non_striker_id):
                    raise ValueError("dismissed_player_id must be striker or non-striker for this ball")
            if last_ball:
                expected_striker, expected_non_striker = BallService._expected_next_batsmen(last_ball)
                if expected_striker is not None and striker_id != expected_striker:
                    raise ValueError("Invalid striker for next ball as per previous delivery")
                if expected_non_striker is not None and non_striker_id != expected_non_striker:
                    raise ValueError("Invalid non-striker for next ball as per previous delivery")
                if last_ball.is_legal_delivery and last_ball.ball_number == 6 and bowler_id == last_ball.bowler_id:
                    raise ValueError("Bowler cannot bowl consecutive overs")

            max_bowler_overs = BallService._get_bowler_max_overs(innings.match)
            if max_bowler_overs is not None:
                legal_balls_by_bowler = Ball.query.filter_by(
                    inning_id=innings_id,
                    bowler_id=bowler_id,
                    is_legal_delivery=True
                ).count()
                if legal_balls_by_bowler >= int(max_bowler_overs * 6):
                    raise ValueError(f"Bowler has reached the over limit ({max_bowler_overs}) for this match type")
            if extra_type in ['wide','no-ball'] and extras == 0:
                extras = 1
            is_legal_delivery = extra_type not in ['wide','no-ball']
            if not last_ball:
                over_number=0
                ball_number=1
            else:
                over_number=last_ball.over_number
                ball_number=last_ball.ball_number
                if last_ball.is_legal_delivery:
                    if ball_number==6:
                        over_number+=1
                        ball_number=1
                    else:
                        ball_number+=1
            ball=Ball(
                inning_id=innings_id,
                over_number=over_number,
                ball_number=ball_number,
                batsman_id=striker_id,
                non_striker_id=non_striker_id,
                bowler_id=bowler_id,
                runs_scored=runs,
                extra_runs=extras,
                extra_type=extra_type,
                is_wicket=is_wicket,
                wicket_type=wicket_type,
                dismissed_player_id=dismissed_player_id,
                fielder_id=fielder_id,
                is_legal_delivery=is_legal_delivery,
            )
            db.session.add(ball)
            BallService._update_batting_scorecard(ball)
            BallService._update_bowling_scorecard(ball)
            BallService._update_partnership(ball)
            BallService._update_innings(ball)
            if is_wicket:
                BallService._handle_wicket(ball)
            db.session.commit()
            return ball
        except SQLAlchemyError as e:
            db.session.rollback()
            raise ValueError(f"DataBase error:{str(e)}")

    @staticmethod
    def _validate_player_teams(innings, striker_id, non_striker_id, bowler_id):
        striker = db.session.get(Player, striker_id)
        non_striker = db.session.get(Player, non_striker_id)
        bowler = db.session.get(Player, bowler_id)
        if not striker or not non_striker or not bowler:
            raise ValueError("Invalid player selected for striker/non-striker/bowler")
        if striker.team_id != innings.batting_team_id or non_striker.team_id != innings.batting_team_id:
            raise ValueError("Striker and non-striker must belong to batting team")
        if bowler.team_id != innings.bowling_team_id:
            raise ValueError("Bowler must belong to bowling team")

    @staticmethod
    def _validate_batsmen_not_out(innings_id, striker_id, non_striker_id):
        striker_sc = BattingScorecard.query.filter_by(innings_id=innings_id, player_id=striker_id).first()
        non_striker_sc = BattingScorecard.query.filter_by(innings_id=innings_id, player_id=non_striker_id).first()
        if striker_sc and striker_sc.is_out:
            raise ValueError("Striker is already out and cannot bat again in this innings")
        if non_striker_sc and non_striker_sc.is_out:
            raise ValueError("Non-striker is already out and cannot bat again in this innings")

    @staticmethod
    def _is_free_hit_next(last_ball):
        # Free hit becomes active after a no-ball, and remains active through
        # subsequent illegal deliveries (e.g., wide) until a legal ball occurs.
        cursor = last_ball
        while cursor:
            if cursor.extra_type == "no-ball":
                return True
            if cursor.is_legal_delivery:
                return False
            cursor = (
                Ball.query
                .filter(Ball.inning_id == last_ball.inning_id, Ball.id < cursor.id)
                .order_by(Ball.id.desc())
                .first()
            )
        return False

    @staticmethod
    def is_free_hit_next(innings_id):
        last_ball = Ball.query.filter_by(inning_id=innings_id).order_by(Ball.id.desc()).first()
        if not last_ball:
            return False
        return BallService._is_free_hit_next(last_ball)

    @staticmethod
    def _strike_rotates_for_ball(ball):
        # Strike changes on odd completed runs (bat runs + run byes/leg-byes).
        total_running_runs = ball.runs_scored
        if ball.extra_type in ['bye', 'leg-bye']:
            total_running_runs += ball.extra_runs
        elif ball.extra_type in ['wide', 'no-ball'] and ball.extra_runs > 1:
            total_running_runs += (ball.extra_runs - 1)
        return total_running_runs % 2 == 1

    @staticmethod
    def _expected_next_batsmen(last_ball):
        striker = last_ball.batsman_id
        non_striker = last_ball.non_striker_id
        if last_ball.is_wicket:
            if last_ball.dismissed_player_id == last_ball.batsman_id:
                striker = None
            elif last_ball.dismissed_player_id == last_ball.non_striker_id:
                non_striker = None
            if last_ball.is_legal_delivery and last_ball.ball_number == 6:
                striker, non_striker = non_striker, striker
        else:
            if BallService._strike_rotates_for_ball(last_ball):
                striker, non_striker = non_striker, striker
            if last_ball.is_legal_delivery and last_ball.ball_number == 6:
                striker, non_striker = non_striker, striker
        return striker, non_striker
    @staticmethod
    def _update_batting_scorecard(ball):
        scorecard=BattingScorecard.query.filter_by(innings_id=ball.inning_id,player_id=ball.batsman_id).first()
        if not scorecard:
            existing_batsmen=BattingScorecard.query.filter_by(innings_id=ball.inning_id).count()
            scorecard=BattingScorecard(innings_id=ball.inning_id,player_id=ball.batsman_id,batting_position=existing_batsmen+1)
            db.session.add(scorecard)
        scorecard.update_stats(ball)
    @staticmethod
    def _update_bowling_scorecard(ball):
        scorecard = BowlingScorecard.query.filter_by(innings_id=ball.inning_id,player_id=ball.bowler_id).first()
        if not scorecard:
            scorecard = BowlingScorecard(innings_id=ball.inning_id,player_id=ball.bowler_id)
            db.session.add(scorecard)
        
        scorecard.update_stats(ball)
    @staticmethod
    def _update_partnership(ball):
        partnership=Partnership.query.filter_by(inning_id=ball.inning_id,is_active=True).first()
        if not partnership:
            wickets_fallen=Partnership.query.filter_by(inning_id=ball.inning_id).count()
            partnership=Partnership(
                inning_id=ball.inning_id,
                batsman1_id=ball.batsman_id,
                batsman2_id=ball.non_striker_id,
                wickets_fallen=wickets_fallen,
                runs_scored=0,
                balls_faced=0,
                is_active=True,
            )
            db.session.add(partnership)
        # Defensive defaults for nullable legacy rows
        partnership.runs_scored = partnership.runs_scored or 0
        partnership.balls_faced = partnership.balls_faced or 0
        partnership.wickets_fallen = partnership.wickets_fallen or 0
        partnership.runs_scored+=(ball.runs_scored+ball.extra_runs)
        if ball.is_legal_delivery:
            partnership.balls_faced+=1
        if ball.is_wicket:
            partnership.is_active=False
    
    @staticmethod
    def _update_innings(ball):
        innings=db.session.get(Inning,ball.inning_id)
        innings.total_runs+=(ball.runs_scored+ball.extra_runs)
        innings.extras+=ball.extra_runs
        if ball.is_wicket:
            innings.total_wickets+=1
        if ball.is_legal_delivery:
            total_balls=Ball.query.filter_by(inning_id=ball.inning_id,is_legal_delivery=True).count()
            innings.total_overs=total_balls//6 +(total_balls%6)/10
        match=innings.match
        total_legal_balls=Ball.query.filter_by(inning_id=ball.inning_id,is_legal_delivery=True).count()
        if innings.total_wickets>=10:
            innings.is_completed=True
        elif match.over_limit is not None and total_legal_balls>=match.over_limit*6:
            innings.is_completed=True
        elif innings.target is not None and innings.total_runs>=innings.target:
            innings.is_completed=True
    @staticmethod
    def _handle_wicket(ball):
        """Handle wicket-specific logic"""
        # Partnership already ended in _update_partnership
        # Next batsman will come in - handled by frontend/next ball
        pass
    @staticmethod
    def _get_bowler_max_overs(match):
        """
        Return max overs per bowler by format.
        T20 -> 4, ODI -> 10, else derive from over_limit//5 when available.
        """
        if not match:
            return None
        match_type = (match.match_type or "").lower()
        if "t20" in match_type:
            return 4
        if "odi" in match_type:
            return 10
        if match.over_limit:
            return max(1, int(match.over_limit // 5))
        return None
    @staticmethod
    def get_current_batsmen(innings_id):
        last_ball=Ball.query.filter_by(inning_id=innings_id).order_by(Ball.id.desc()).first()
        if not last_ball:
            return None,None
        return BallService._expected_next_batsmen(last_ball)
    @staticmethod
    def get_over_summary(innings_id,over_number):
        balls=Ball.query.filter_by(inning_id=innings_id,over_number=over_number).order_by(Ball.ball_number).all()
        return {
            'over_number':over_number+1,
            'balls':[ball.to_dict() for ball in balls],
            'total_runs':sum(ball.runs_scored+ball.extra_runs for ball in balls),
            'is_complete':len(balls)==6,
            'wickets':sum(1 for ball in balls if ball.is_wicket)
        }
    
