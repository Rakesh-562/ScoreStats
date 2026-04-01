from flask import Blueprint,request,jsonify
from app.models import Player 
from app.extensions import db
from app.validators import PlayerCreateSchema,PlayerUpdateSchema
from marshmallow import ValidationError
player_bp=Blueprint('palyer',__name__)
@player_bp.route('',methods=['GET'])
def get_all_players():
    query=Player.query
    team_id=request.args.get('team_id',type=int)
    if team_id:
        query=query.filter_by(team_id=team_id)
    players=query.all()
    return jsonify({
        'success':True,'count':len(players),'players':[player.to_dict() for player in players]
    }),200
@player_bp.route('/<int:player_id>',methods=['GET'])
def get_player(player_id):
    player=Player.query.get(player_id)
    if not player:
        return jsonify({
            'success':False,'error':'Player not found'
        }),404
    return jsonify(
        {'success':True,'player':player.to_dict()}
    ),200
@player_bp.route('',methods=['POST'])
def create_player():
    try:
        schema=PlayerCreateSchema()
        data=schema.load(request.get_json())
        player=Player(**data)
        db.session.add(player)
        db.session.commit()
        return jsonify({
            'success':True,'message':'Player created successfully','player':player.to_dict()
        }),201
    except ValidationError as e:
        return jsonify({
            'success':False,
            'errors':e.messages
        }),400
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success':False,'error':str(e)
        }),500
@player_bp.route('/<int:player_id>',methods=['PUT'])
def update_player(player_id):
    player=db.session.get(Player, player_id)
    if not player:
        return jsonify({
            'success':False,'error':'player not found'
        }),404
    try:
        schema=PlayerUpdateSchema()
        data=schema.load(request.get_json())
        for key,val in data.items():
            setattr(player,key,val)
        db.session.commit()
        return jsonify({
            'success':True,
            'message':'Palyer updated successfully',
            'player':player.to_dict()
        }),200
    except ValidationError as e:
        return jsonify({'success': False, 'errors': e.messages}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
@player_bp.route('/<int:player_id>',methods=['DELETE'])
def delete_player(player_id):
    player=db.session.get(Player, player_id)
    if not player:
        return jsonify({'success':False,'error':'Player not found'}),404
    try:
        db.session.delete(player)
        db.session.commit()
        return jsonify({
            'success':True,
            'message':'Player deleted successfully'
        }) ,200
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

    
