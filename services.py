from models import Ball
from extensions import db
def record_ball(data):
    print("Recording ball with data:", data)
    inning_id = data.get("inning_id")
    last_ball = Ball.query.filter_by(inning_id=inning_id).order_by(Ball.id.desc()).first()
    if not last_ball:
        over = 0
        balls = 1
        striker_id = data["striker_id"]
        non_striker_id = data["non_striker_id"]
    else:
        if last_ball.balls == 6:
            over = last_ball.over + 1
            balls = 1
            striker_id = last_ball.non_striker_id
            non_striker_id = last_ball.striker_id
        else:
            over = last_ball.over
            balls = last_ball.balls + 1
            striker_id = last_ball.striker_id
            non_striker_id = last_ball.non_striker_id
    runs = data.get("runs", 0)

    if data["runs"] % 2 == 1:
        striker_id, non_striker_id = non_striker_id, striker_id
    new_ball = Ball(
        inning_id=inning_id,
        over=over,
        balls=balls,
        striker_id=striker_id,
        non_striker_id=non_striker_id,
        bowler_id=data.get("bowler_id"),
        runs=data.get("runs", 0),
        wickets=data.get("wickets", 0),
        extras=data.get("extras", 0),
        dismissed_player_id=data.get("dismissed_player_id")
    )
    db.session.add(new_ball)
    db.session.commit()
    return new_ball