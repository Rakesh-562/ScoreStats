from flask import Blueprint,request,jsonify
from app.models import Team
from app.extensions import db
from app.services.deletion_service import DeletionService
from app.validators import TeamCreateSchema,TeamUpdateSchema
from marshmallow import ValidationError
teams_bp=Blueprint('team',__name__)

@teams_bp.route('', methods=['GET'])
def get_all_teams():
    team=Team.query.all()
    return jsonify( 
        {"success":True,"count":len(team),"teams":[t.to_dict() for t in team]}),200
@teams_bp.route('/<int:team_id>', methods=['GET'])
def get_team(team_id):
    team=Team.query.get(team_id)
    if not team:
        return jsonify({
            'success':False,
            'message':'Team not found'
        }),404
    return jsonify({
        'success':True,
        'team':team.to_dict()
    }),200
@teams_bp.route('', methods=['POST'])
def create_team():
    try:
        data=TeamCreateSchema().load(request.json)
        team=Team(**data)
        db.session.add(team)
        db.session.commit()
        return jsonify({
        'success':True,
        'team':team.to_dict()
    }),201
    except ValidationError as err:
        return jsonify({
            'success':False,
            'errors':err.messages
        }),400
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success':False,
            'message':str(e)
        }),500
@teams_bp.route('/<int:team_id>', methods=['PUT'])
def update_team(team_id):
    team=Team.query.get(team_id)
    if not team:
        return jsonify({
            'success':False,
            'message':'Team not found'
        }),404
    try:
        data=TeamUpdateSchema().load(request.json)
        for key,value in data.items():
            setattr(team,key,value)
        db.session.commit()
        return jsonify({
            'success':True,
            'team':team.to_dict()
        }),200
    except ValidationError as err:
        return jsonify({
            'success':False,
            'errors':err.messages
        }),400
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success':False,
            'message':str(e)
        }),500
@teams_bp.route('/<int:team_id>', methods=['DELETE'])
def delete_team(team_id):
    try:
        result = DeletionService.delete_team(team_id)
        return jsonify({
            'success':True,
            'message':'Team deleted successfully',
            **result,
        }),200
    except ValueError as exc:
        return jsonify({
            'success':False,
            'message':str(exc)
        }),404
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success':False,
            'message':str(e)
        }),500
   
  
