from flask import Blueprint,request,jsonify
from app.services import BallService, StatisticsService
from app.validators import BallRecordSchema,InningsStartSchema
from marshmallow import ValidationError
from app.extensions import socketio
from app.websockets.match_socket import (emit_ball_update,emit_score_update,emit_innings_complete,emit_match_status_change)
balls_bp=Blueprint("balls",__name__)
@balls_bp.route('/record', methods=['POST'])
def record_ball():
    try :
        schema=BallRecordSchema()
        data=schema.load(request.get_json())
        # Keep compatibility with schema field name.
        if 'extras_type' in data and 'extra_type' not in data:
            data['extra_type'] = data.pop('extras_type')
        if data.get('is_wicket') and not data.get('wicket_type'):
            return jsonify({
                'success':False,
                'error':'wicket_type is required when is_wicket is true'
            }),400
        ball=BallService.record_ball(**data)
        from app.models import Inning
        innings=Inning.query.get(data['innings_id'])
        match_id = innings.match_id

        # Realtime updates to all viewers in this match room.
        emit_ball_update(socketio, match_id, ball)
        emit_score_update(socketio, match_id, innings)
        if innings.is_completed:
            emit_innings_complete(
                socketio,
                match_id,
                innings.innings_number,
                {
                    'runs': innings.total_runs,
                    'wickets': innings.total_wickets,
                    'overs': innings.total_overs,
                },
            )
            emit_match_status_change(
                socketio,
                match_id,
                innings.match.status,
                innings.match.win_margin,
            )
            from app.analytics.auto_update import trigger_post_match_analytics
            trigger_post_match_analytics(match_id)
        return jsonify({
            'success':True,
            'message':'Ball recorded successfully',
            'ball':ball.to_dict(),
            'batting_scorecard': StatisticsService.get_batting_scorecard(innings.id),
            'bowling_scorecard': StatisticsService.get_bowling_scorecard(innings.id),
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

@balls_bp.route('/innings/<int:innings_id>/summary', methods=['GET'])
def get_innings_summary(innings_id):
    from app.models import Inning
    innings = Inning.query.get(innings_id)
    if not innings:
        return jsonify({'success': False, 'error': 'Innings not found'}), 404
    return jsonify({
        'success': True,
        'innings': innings.to_dict(),
        'batting_scorecard': StatisticsService.get_batting_scorecard(innings_id),
        'bowling_scorecard': StatisticsService.get_bowling_scorecard(innings_id)
    }), 200

@balls_bp.route('/innings/<int:innings_id>/state', methods=['GET'])
def get_innings_state(innings_id):
    from app.models import Inning
    innings = Inning.query.get(innings_id)
    if not innings:
        return jsonify({'success': False, 'error': 'Innings not found'}), 404
    striker_id, non_striker_id = BallService.get_current_batsmen(innings_id)
    from app.models import BattingScorecard
    out_player_ids = [
        sc.player_id for sc in BattingScorecard.query.filter_by(innings_id=innings_id, is_out=True).all()
    ]
    return jsonify({
        'success': True,
        'innings_id': innings_id,
        'striker_id': striker_id,
        'non_striker_id': non_striker_id,
        'out_player_ids': out_player_ids,
        'free_hit_next': BallService.is_free_hit_next(innings_id),
        'is_completed': innings.is_completed,
    }), 200
        
