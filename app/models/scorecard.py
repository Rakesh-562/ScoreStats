from app.extensions import db
class BattingScorecard(db.Model):
    """
    Docstring for BattingScorecard
    batting statistics from the player 
    updated automatically each time a ball is recorded  
    """
    __tablename__="batting_scorecard"
    id=db.Column(db.Integer,primary_key=True)
    #References 
    innings_id=db.Column(db.Integer,db.ForeignKey("inning.id",ondelete='CASCADE'),nullable=False,index=True)
    player_id=db.Column(db.Integer,db.ForeignKey("player.id"),nullable=False,index=True)
    player=db.relationship("Player",foreign_keys=[player_id],lazy='joined')
    # Batting stats
    runs=db.Column(db.Integer,default=0,comment="Total runs scored")
    balls_faced=db.Column(db.Integer,default=0,comment="Legal deliveries faced")
    sixes=db.Column(db.Integer,default=0,comment="Number of sixes") 
    fours=db.Column(db.Integer,default=0,comment="Number of fours") 
    dots=db.Column(db.Integer,default=0,comment="Number of dots") 
    # Calculated Stats
    strike_rate=db.Column(db.Float,default=0,comment="(runs/balls_faced)*100")
    is_out=db.Column(db.Boolean,default=False)
    dismissal_type=db.Column(db.String(30),comment="How Player got out")
    bowler_id=db.Column(db.Integer,db.ForeignKey("player.id"),comment="Bowler who took wicket")
    fielder_id=db.Column(db.Integer,db.ForeignKey("player.id"),comment="Fielder involved in dismissal")
    batting_position=db.Column(db.Integer,comment="1=opener,3=first down")
    # unique constraint to prevent duplicate entries
    __table_args__=(db.UniqueConstraint('innings_id','player_id',name='unique_inning_player'),)
    def __repr__(self):
        return f"<BattingScorecard Player:{self.player_id} Innings:{self.innings_id} Runs:{self.runs} Balls:{self.balls_faced}>"
    def to_dict(self):
        """Convert BattingScorecard object to dictionary"""
        return {
            'id':self.id,
            'innings_id':self.innings_id,
            'player_id':self.player_id,
            'runs':self.runs,
            'balls_faced':self.balls_faced,
            'sixes':self.sixes,
            'fours':self.fours,
            'dots':self.dots,
            'strike_rate':self.strike_rate,
            'is_out':self.is_out,
            'dismissal_type':self.dismissal_type,
            'bowler_id':self.bowler_id,
            'fielder_id':self.fielder_id,
            'batting_position':self.batting_position
        }
    def update_stats(self,ball):
        """Update batting statistics based on a new ball delivery"""
        if ball.batsman_id != self.player_id:
            return  # Not the batsman for this scorecard
        # Defensive defaults for nullable legacy rows
        self.balls_faced = self.balls_faced or 0
        self.runs = self.runs or 0
        self.fours = self.fours or 0
        self.sixes = self.sixes or 0
        self.dots = self.dots or 0
        self.balls_faced += 1 if ball.is_legal_delivery else 0
        self.runs += ball.runs_scored
        if ball.runs_scored == 4:
            self.fours += 1
        elif ball.runs_scored == 6:
            self.sixes += 1
        elif ball.runs_scored == 0 and ball.is_legal_delivery:
            self.dots += 1
        # Update strike rate
        if self.balls_faced > 0:
            self.strike_rate = (self.runs / self.balls_faced) * 100
        # Handle dismissal
        if ball.is_wicket and ball.dismissed_player_id == self.player_id:
            self.is_out = True
            self.dismissal_type = ball.wicket_type
            self.bowler_id = ball.bowler_id
            self.fielder_id = ball.fielder_id
class BowlingScorecard(db.Model):
    """
    Docstring for BowlingScorecard
    """
    __tablename__="bowling_scorecard"
    id=db.Column(db.Integer,primary_key=True)
    #References
    innings_id=db.Column(db.Integer,db.ForeignKey("inning.id",ondelete='CASCADE'),nullable=False,index=True)
    player_id=db.Column(db.Integer,db.ForeignKey("player.id"),nullable=False,index=True )
    player=db.relationship("Player",foreign_keys=[player_id],lazy='joined')
    # Bowling stats
    overs_bowled=db.Column(db.Float,default=0.0,comment="Total overs bowled")
    balls_bowled=db.Column(db.Integer,default=0,comment="Total balls bowled")
    maidens=db.Column(db.Integer,default=0,comment="Number of maiden overs") 
    wides=db.Column(db.Integer,default=0,comment="Total wides bowled")
    no_balls=db.Column(db.Integer,default=0,comment="Total no balls bowled")
    dots=db.Column(db.Integer,default=0,comment="Total dot balls bowled")   
    runs_conceded=db.Column(db.Integer,default=0,comment="Total runs conceded")
    wickets_taken=db.Column(db.Integer,default=0,comment="Total wickets taken") 
    extras_conceded=db.Column(db.Integer,default=0,comment="Total extras conceded")
    # Calculated Stats
    economy_rate=db.Column(db.Float,default=0.0,comment="runs_conceded/overs_bowled")
    bowling_average=db.Column(db.Float,default=0.0,comment="runs_conceded/wickets_taken")
    strike_rate=db.Column(db.Float,default=0.0,comment="balls_bowled/wickets_taken")
    # unique constraint to prevent duplicate entries
    __table_args__=(db.UniqueConstraint('innings_id','player_id',name='unique_bowling_inning_player'),)
    def __repr__(self):
        return f"<BowlingScorecard Player:{self.player_id} Innings:{self.innings_id} Wickets:{self.wickets_taken} Runs:{self.runs_conceded}>"
    def to_dict(self):
        """Convert BowlingScorecard object to dictionary"""
        return {
            'id':self.id,
            'innings_id':self.innings_id,
            'player_id':self.player_id,
            'overs_bowled':self.overs_bowled,
            'balls_bowled':self.balls_bowled,
            'maidens':self.maidens,
            'wides':self.wides,
            'no_balls':self.no_balls,
            'dots':self.dots,
            'runs_conceded':self.runs_conceded,
            'wickets_taken':self.wickets_taken,
            'extras_conceded':self.extras_conceded,
            'economy_rate':self.economy_rate,
            'bowling_average':self.bowling_average,
            'strike_rate':self.strike_rate
        }
    def update_stats(self,ball):
        """Update bowling statistics based on a new ball delivery"""
        if ball.bowler_id != self.player_id:
            return  # Not the bowler for this scorecard
        # Defensive defaults for nullable legacy rows
        self.balls_bowled = self.balls_bowled or 0
        self.overs_bowled = self.overs_bowled or 0.0
        self.maidens = self.maidens or 0
        self.wides = self.wides or 0
        self.no_balls = self.no_balls or 0
        self.dots = self.dots or 0
        self.runs_conceded = self.runs_conceded or 0
        self.wickets_taken = self.wickets_taken or 0
        self.extras_conceded = self.extras_conceded or 0
        # Update balls bowled
        if ball.is_legal_delivery:
            self.balls_bowled += 1
            # Update overs bowled
            self.overs_bowled = self.balls_bowled // 6 + (self.balls_bowled % 6) / 10.0
            if self.balls_bowled % 6 == 0:
                # Check for maiden over
                completed_over = (self.balls_bowled // 6) - 1
                last_over_balls = [b for b in ball.inning.balls if b.over_number == completed_over and b.bowler_id == self.player_id]
                if last_over_balls and sum((b.runs_scored + b.extra_runs) for b in last_over_balls) == 0:
                    self.maidens += 1
        # Update runs conceded
        self.runs_conceded += ball.runs_scored + ball.extra_runs
        # Update extras conceded
        self.extras_conceded += ball.extra_runs
        # Update wides and no balls
        if ball.extra_type == 'wide':
            self.wides += 1
        elif ball.extra_type == 'no-ball':
            self.no_balls += 1
        # Update dots
        if ball.runs_scored == 0 and ball.extra_runs == 0:
            self.dots += 1
        # Update wickets taken
        bowler_wicket_types = {'bowled', 'caught', 'lbw', 'stumped', 'hit-wicket'}
        if ball.is_wicket and ball.bowler_id == self.player_id and (ball.wicket_type in bowler_wicket_types):
            self.wickets_taken += 1
        # Update calculated stats
        if self.overs_bowled > 0:
            self.economy_rate = self.runs_conceded / self.overs_bowled
        if self.wickets_taken > 0:
            self.bowling_average = self.runs_conceded / self.wickets_taken
            self.strike_rate = self.balls_bowled / self.wickets_taken
        else:
            self.bowling_average = 0.0
            self.strike_rate = 0.0
