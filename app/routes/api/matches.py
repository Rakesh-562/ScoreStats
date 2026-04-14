from flask import Blueprint,request,jsonify
from app.services import MatchService
from app.services.deletion_service import DeletionService
from app.validators import MatchCreateSchema,TossRecordSchema
from marshmallow import ValidationError
matches_bp=Blueprint('matches',__name__)
@matches_bp.route('',methods=['GET'])
def get_all_matches():
    from app.models import Match
    status=request.args.get('status')
    query=Match.query
    if status:
        query=query.filter_by(status=status)
    matches=query.order_by(Match.match_date.desc()).all()
    return jsonify({
        'success':True,
        'count':len(matches),
        'matches':[match.to_dict() for match in matches]
    }),200
@matches_bp.route('/<int:match_id>',methods=['GET'])
def get_match(match_id):
    summary=MatchService.get_match_summary(match_id)
    if not summary:
        return jsonify({'success':False,'error':'Match not found'}),404
    return jsonify({
        'success':True,'match':summary
    }),200
@matches_bp.route('',methods=['POST'])
def create_match():
    try :
        schema=MatchCreateSchema()
        data=schema.load(request.get_json())
        match=MatchService.create_match(**data)
        return jsonify({
            'success':True,
            'message':'Match created successfully',
            'match':match.to_dict()
        }),201
    except ValidationError as e:
        return jsonify({
            'success':False,
            'error':e.messages    
        }),400
    except Exception as e:
        return jsonify({
            'success':False,
            'error':str(e)
        }),400
@matches_bp.route('/<int:match_id>/toss',methods=['POST'])
def record_toss(match_id):
    try :
        schema=TossRecordSchema()
        data=schema.load(request.get_json())
        match=MatchService.record_toss(match_id,**data)
        return jsonify({
            'success':True,
            'message':'toss recorded successfully',
            'match':match.to_dict()
        }),200
    except ValidationError as e:
        return jsonify({'success':False,'errors':e.messages}),400
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),400
@matches_bp.route('/live',methods=['GET'])
def get_live_matches():
    matches=MatchService.get_live_matches()
    return jsonify({
        'success':True,
        'count':len(matches),
        'matches':[m.to_dict() for m in matches]
    }),200


@matches_bp.route('/<int:match_id>', methods=['DELETE'])
def delete_match(match_id):
    try:
        result = DeletionService.delete_match(match_id)
        return jsonify({
            'success': True,
            'message': 'Match deleted successfully',
            **result,
        }), 200
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500
