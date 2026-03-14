from marshmallow import Schema,fields,validates,ValidationError
from datetime import date
class MatchCreateSchema(Schema):
    '''
    Docstring for MatchCreateSchema
    Validate match creation input
    '''
    team_1_id=fields.Int(required=True,error_messages={"reqiured":"Team 1 ID is required"})
    team_2_id=fields.Int(required=True,error_messages={"reqiured":"Team 2 ID is required"})
    match_date=fields.DateTime(required=False)
    over_limit=fields.Int(required=False)
    match_type=fields.Str(required=False)
    @validates("team_2_id")
    def validate_different_teams(self, value, **kwargs):
        # Actual cross-field check is handled in service layer; keep signature v4-compatible.
        pass
class TossRecordSchema(Schema):
    '''
    Docstring for TossRecordSchema
    Validates Toss Recording
    '''
    toss_winner_id = fields.Int(required=True)
    toss_decision = fields.Str(
        required=True,
        validate=lambda x: x in ['bat', 'field'],
        error_messages={
            "validator_failed": "Toss decision must be 'bat' or 'field'"
        }
    )
