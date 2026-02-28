# app/websockets/match_socket.py
"""
WebSocket event handlers and server-side emitters for live match updates.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW IT FITS IN YOUR ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  app/__init__.py (factory)
      └── register_socket_events(socketio)
              └── register_match_events(socketio)   ← this file

  app/routes/api/balls.py  (REST scorer endpoint)
      └── emit_ball_update(socketio, match_id, ball)   ← imported from here
      └── emit_score_update(socketio, match_id, innings)
      └── emit_innings_complete(...)
      └── emit_match_status_change(...)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOCKET ROOMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each live match gets its own room: "match_<id>"
  → emit to "match_42" reaches ONLY viewers of match 42
  → 1000 users watching 100 matches = zero cross-talk

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVENTS  (Server → Client)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  connected         ACK on initial WebSocket open
  match_joined      Full current state snapshot sent after join_match
  ball_update       New delivery recorded (runs, wicket, extras)
  score_update      Innings total updated (runs/wickets/overs)
  innings_complete  An innings has ended (all-out or overs done)
  match_status      Match lifecycle change (live → completed/abandoned)
  error             Something went wrong

EVENTS  (Client → Server)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  join_match        Subscribe to a match room
  leave_match       Unsubscribe from a match room
  ping_match        Health-check / latency probe
"""

from datetime import datetime

from flask import request
from flask_socketio import join_room, leave_room, emit

from app.extensions import db
from app.models import Match, Inning, Ball


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _room(match_id: int) -> str:
    """
    Canonical room name for a match.
    Centralised here so it never drifts between files.

    >>> _room(42)
    'match_42'
    """
    return f"match_{match_id}"


def _get_live_innings(match_id: int):
    """
    Return the currently active (incomplete) innings for a match.
    Returns None if the match hasn't started yet.
    """
    return (
        Inning.query
        .filter_by(match_id=match_id, is_complete=False)
        .order_by(Inning.innings_number.desc())
        .first()
    )


def _get_recent_balls(match_id: int, limit: int = 12) -> list[dict]:
    """
    Return the last `limit` deliveries for a match in chronological order.
    Used to hydrate the ball feed when a client first joins.
    """
    balls = (
        Ball.query
        .join(Inning)
        .filter(Inning.match_id == match_id)
        .order_by(Ball.id.desc())
        .limit(limit)
        .all()
    )
    return [b.to_dict() for b in reversed(balls)]


def _build_commentary(ball: "Ball") -> str:
    """
    Auto-generate a one-line commentary string for a delivery.
    Called inside emit_ball_update before broadcasting.

    Examples
    --------
    "WICKET! Bumrah gets Kohli (bowled)"
    "SIX! Sharma pulls Shami over mid-wicket"
    "Dot ball. Bumrah to Sharma"
    """
    batsman = ball.batsman.name if ball.batsman else "Batsman"
    bowler  = ball.bowler.name  if ball.bowler  else "Bowler"

    if ball.is_wicket:
        return f"WICKET! {bowler} gets {batsman} ({ball.dismissal_type or 'out'})"
    if ball.runs_scored == 6:
        return f"SIX! {bowler} to {batsman}"
    if ball.runs_scored == 4:
        return f"FOUR! {bowler} to {batsman}"
    if ball.is_wide:
        return f"Wide. {bowler} to {batsman}"
    if ball.is_no_ball:
        return f"No ball! {bowler} to {batsman}, {ball.runs_scored} run(s)"
    if ball.runs_scored == 0:
        return f"Dot ball. {bowler} to {batsman}"
    return f"{ball.runs_scored} run(s). {bowler} to {batsman}"


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def register_match_events(socketio):
    """
    Attach all match-related Socket.IO event handlers to the
    given socketio instance.

    Called once from app/websockets/__init__.py during app startup.
    """

    # ── CONNECTION LIFECYCLE ──────────────────────────────────────────────────

    @socketio.on('connect')
    def handle_connect():
        """
        Fires automatically the moment a client opens a WebSocket connection.
        `request.sid` is Socket.IO's unique session ID for this tab/client.
        """
        print(f"[WS] Client connected  sid={request.sid}")
        emit('connected', {
            'status': 'ok',
            'sid': request.sid,
            'message': 'Connected to Cricket live server',
        })

    @socketio.on('disconnect')
    def handle_disconnect():
        """
        Fires when a client closes their tab, loses network, or times out.
        Socket.IO automatically removes them from all rooms — no manual cleanup needed.
        """
        print(f"[WS] Client disconnected  sid={request.sid}")

    # ── ROOM MANAGEMENT ───────────────────────────────────────────────────────

    @socketio.on('join_match')
    def handle_join_match(data):
        """
        Client subscribes to live updates for a specific match.

        Expected payload
        ----------------
        { "match_id": 42 }

        Server responds with
        --------------------
        'match_joined' event carrying the full current state so the UI
        is never blank while waiting for the next ball event.
        """
        match_id = data.get('match_id')

        if not match_id:
            emit('error', {'message': 'match_id is required'})
            return

        match = Match.query.get(match_id)
        if not match:
            emit('error', {'message': f'Match {match_id} not found'})
            return

        room = _room(match_id)
        join_room(room)
        print(f"[WS] {request.sid} → joined {room}")

        # Send a state snapshot immediately (avoids blank UI on first connect)
        current_innings = _get_live_innings(match_id)
        emit('match_joined', {
            'match_id':       match_id,
            'match':          match.to_dict(),
            'current_innings': current_innings.to_dict() if current_innings else None,
            'recent_balls':   _get_recent_balls(match_id, limit=12),
        })

    @socketio.on('leave_match')
    def handle_leave_match(data):
        """
        Client unsubscribes (e.g. navigates to a different page).

        Expected payload
        ----------------
        { "match_id": 42 }
        """
        match_id = data.get('match_id')
        if match_id:
            room = _room(match_id)
            leave_room(room)
            print(f"[WS] {request.sid} ← left {room}")
            emit('match_left', {'match_id': match_id})

    # ── UTILITY ───────────────────────────────────────────────────────────────

    @socketio.on('ping_match')
    def handle_ping(data):
        """
        Debug / health-check.  Client sends ping → server replies with pong.
        Useful for measuring round-trip latency and verifying the connection.

        Expected payload:  { "match_id": 42 }
        Response event:    'pong_match'
        """
        emit('pong_match', {
            'match_id':  data.get('match_id'),
            'timestamp': datetime.utcnow().isoformat(),
            'sid':       request.sid,
        })


# ─────────────────────────────────────────────────────────────────────────────
# SERVER-SIDE EMITTERS
# Called from app/routes/api/balls.py (and other routes) — NOT by clients.
# ─────────────────────────────────────────────────────────────────────────────

def emit_ball_update(socketio_instance, match_id: int, ball: "Ball") -> None:
    """
    Broadcast a new delivery to every client watching this match.

    Called by: app/routes/api/balls.py  after a Ball is committed to DB.

    Payload sent to clients
    -----------------------
    {
      "id":            301,
      "over_number":   4,
      "ball_number":   3,
      "runs_scored":   4,
      "is_wicket":     false,
      "is_wide":       false,
      "is_no_ball":    false,
      "batsman_name":  "Rohit Sharma",
      "bowler_name":   "Jasprit Bumrah",
      "commentary":    "FOUR! Bumrah to Sharma"
    }
    """
    room    = _room(match_id)
    payload = ball.to_dict()
    payload['commentary'] = _build_commentary(ball)

    socketio_instance.emit('ball_update', payload, room=room)
    print(f"[WS] ball_update → {room}  runs={payload['runs_scored']}")


def emit_score_update(socketio_instance, match_id: int, innings: "Inning") -> None:
    """
    Broadcast the updated innings total after each delivery.

    Gives clients the live score header:   MI 87/3 (10.4 ov)

    Called by: app/routes/api/balls.py  immediately after emit_ball_update.
    """
    socketio_instance.emit('score_update', innings.to_dict(), room=_room(match_id))


def emit_innings_complete(
    socketio_instance,
    match_id: int,
    innings_number: int,
    final_score: dict,
) -> None:
    """
    Broadcast that an innings has ended (all-out or overs exhausted).

    Args
    ----
    final_score : dict
        { "runs": 183, "wickets": 8, "overs": "20.0" }

    Called by: app/routes/api/balls.py  when _check_innings_complete() returns True.
    """
    room = _room(match_id)
    socketio_instance.emit('innings_complete', {
        'match_id':       match_id,
        'innings_number': innings_number,
        'final_score':    final_score,
    }, room=room)
    print(f"[WS] innings_complete → {room}  innings={innings_number}")


def emit_match_status_change(
    socketio_instance,
    match_id: int,
    new_status: str,
    result_summary: str = None,
) -> None:
    """
    Broadcast a match lifecycle change.

    new_status values
    -----------------
    'live'       — match has started
    'completed'  — result decided
    'abandoned'  — weather/other stoppage

    result_summary examples
    -----------------------
    "Mumbai Indians won by 5 wickets"
    "Chennai Super Kings won by 23 runs"

    Called by: app/routes/api/balls.py  when match.status changes.
    """
    room = _room(match_id)
    socketio_instance.emit('match_status', {
        'match_id':       match_id,
        'status':         new_status,
        'result_summary': result_summary,
    }, room=room)
    print(f"[WS] match_status → {room}  status={new_status}")