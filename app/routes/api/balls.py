from flask import Blueprint,request,jsonify
from app.services import BallService
from app.validators import BallRecordSchema,InningsStartSchema
from marshmallow import ValidationError
from app.websockets.match_socket import (emit_ball_update,emit_score_update,emit_innings_complete,emit_match_status_change)
balls_bp=Blueprint("balls",__name__)
@balls_bp.route('/record', methods=['POST'])
def record_ball():
    try :
        schema=BallRecordSchema()
        data=schema.load(request.get_json())
        if data.get('is_wicket') and not data.get('wicket_type'):
            return jsonify({
                'success':False,
                'error':'wicket_type is required when is_wicket is true'
            }),400
        ball=BallService.record_ball(**data)
        from app.models import Inning
        innings=Inning.query.get(data['innings_id'])
        return jsonify({
            'success':True,
            'message':'Ball recorded successfully',
            'ball':ball.to_dict(),
            'inning_summary':{
                  'total_runs': innings.total_runs,
                'total_wickets': innings.total_wickets,
                'total_overs': innings.total_overs,
                'run_rate': innings.run_rate
            }
        }),201
    except ValidationError as e:
        return jsonify({
            'success':False,
            'errors':e.messages
        }),400
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500
@balls_bp.route('/innings/start',methods=['POST'])
def start_innings():
    try:
        from app.services import InningsService
        schema=InningsStartSchema()
        data=schema.load(request.get_json())
        innings=InningsService.start_innings(**data)
        return jsonify({
            'success':True,
            'message':'Innings started successfully',
            'innings':innings.to_dict()
        }),201
    except ValidationError as e:
        return jsonify({
            'success':False,
            'errors':e.messages
        }),400
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),400
@balls_bp.route('/over/<int:innings_id>/<int:over_number>',methods=['GET'])
def get_over_summary(innings_id,over_number):
    summary=BallService.get_over_summary(innings_id,over_number)
    return jsonify({'success':True,'over_summary':summary}),200
        
