from marshmallow import Schema,fields,validates,ValidationError,validate
class BallRecordSchema(Schema):
    innings_id=fields.Int(required=True,error_messages={"required":"inning_id is required"})
    striker_id=fields.Int(required=True,error_messages={"required":"striker id is required"})
    non_striker_id=fields.Int(required=True,error_messages={"required":"non striker id is required"})
    bowler_id=fields.Int(required=True,error_messages={"required":"bowler id is required"})
    runs=fields.Int(required=True,validate=validate.Range(min=0,max=6),  error_messages={"required":"runs is required","validator_failed":"Runs must be between 0 to 6"})
    extras=fields.Int(required=False,load_default=0,validate=validate.Range(min=0))
    extras_type=fields.Str(required=False,validate= lambda x: x in ['wide','no-ball','bye','leg-bye','penalty'],allow_none=True)
    is_wicket=fields.Bool(required=False,load_default=False)
    wicket_type=fields.Str(required=False,validate=lambda x: x in ["bowled","caught",'lbw','run-out','stumped','hit-wicket','retired-hurt','obstructing-field'],allow_none=True)
    dismissed_player_id=fields.Int(required=False,allow_none=True)
    fielder_id=fields.Int(required=False,allow_none=True)
    @validates("is_wicket")
    def validate_wicket_details(self, value, **kwargs):
        pass
class InningsStartSchema(Schema):
    inning_number=fields.Int(required=True,validate=lambda x: 1<=x <=4)
    match_id=fields.Int(required=True)
    batting_team_id=fields.Int(required=True)
    bowling_team_id=fields.Int(required=True)
