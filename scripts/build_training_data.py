"""
scripts/build_training_data.py
 
Run this from your project root:
    python scripts/build_training_data.py
 
What it does:
    - Connects to your cricket.db database
    - Reads every completed innings
    - For each innings, takes a snapshot of the game state at each ball
    - Saves everything to data/training_data.csv
 
Run this again every time you want to retrain with new match data.
"""
 
import os
import sys
 
# This line makes Python find your 'app' folder
# Because this script is inside scripts/, we go one level up (..)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
import pandas as pd
from app import create_app
from app.extensions import db
from app.models import Ball, Inning
 
 
def get_batter_profile_at_this_moment(player_id, innings_id):
    """
    Look up how the current batsman is performing SO FAR in this innings.
    We read from BattingScorecard which your BallService already keeps updated.
    """
    from app.models import BattingScorecard
 
    sc = BattingScorecard.query.filter_by(
        innings_id=innings_id,
        player_id=player_id
    ).first()
 
    if sc and sc.balls_faced and sc.balls_faced > 0:
        return {
            'striker_runs':        sc.runs or 0,
            'striker_balls':       sc.balls_faced or 0,
            'striker_strike_rate': round(sc.strike_rate or 0.0, 2),
            'striker_fours':       sc.fours or 0,
            'striker_sixes':       sc.sixes or 0,
        }
 
    # Batsman just arrived — no data yet, use zeros
    return {
        'striker_runs':        0,
        'striker_balls':       0,
        'striker_strike_rate': 0.0,
        'striker_fours':       0,
        'striker_sixes':       0,
    }
 
 
def get_bowler_profile_at_this_moment(player_id, innings_id):
    """
    Look up how the current bowler is performing SO FAR in this innings.
    We read from BowlingScorecard which your BallService already keeps updated.
    """
    from app.models import BowlingScorecard
 
    sc = BowlingScorecard.query.filter_by(
        innings_id=innings_id,
        player_id=player_id
    ).first()
 
    if sc and sc.balls_bowled and sc.balls_bowled > 0:
        return {
            'bowler_economy':  round(sc.economy_rate or 0.0, 2),
            'bowler_wickets':  sc.wickets_taken or 0,
            'bowler_dots':     sc.dots or 0,
        }
 
    return {
        'bowler_economy': 0.0,
        'bowler_wickets': 0,
        'bowler_dots':    0,
    }
 
 
def build_snapshots_for_one_innings(innings):
    """
    Takes ONE completed innings and returns a list of rows.
    Each row = game state at one ball + what the final score was.
 
    Example:
        At ball 24 (over 4):  score is 42/1, RR=7.0
        Final score was:      164/6
        So the row says:      "from this state, the innings ended at 164"
 
    The model learns this mapping for hundreds of balls across all innings.
    """
    rows = []
 
    # Get all balls in order
    balls = (
        Ball.query
        .filter_by(inning_id=innings.id)
        .order_by(Ball.id.asc())
        .all()
    )
 
    if not balls:
        return rows
 
    # This is what we want to PREDICT — the final score of this innings
    final_score = innings.total_runs
 
    # We track state manually as we walk through each ball
    running_runs        = 0
    running_wickets     = 0
    running_legal_balls = 0
 
    over_limit       = innings.match.over_limit or 20
    total_ball_limit = over_limit * 6
 
    for ball in balls:
 
        # ── State BEFORE this ball is bowled ──────────────────────────────
        # We want to predict FROM the current state, not after the ball
 
        balls_remaining  = total_ball_limit - running_legal_balls
        overs_completed  = running_legal_balls // 6
        overs_as_decimal = running_legal_balls / 6.0
 
        # Current run rate
        crr = (running_runs / overs_as_decimal) if overs_as_decimal > 0 else 0.0
 
        # Required run rate (only makes sense for 2nd innings)
        rrr            = 0.0
        runs_to_target = 0
        target         = innings.target or 0
 
        if innings.innings_number == 2 and target > 0:
            runs_to_target  = max(0, target - running_runs)
            overs_remaining = balls_remaining / 6.0
            rrr = (runs_to_target / overs_remaining) if overs_remaining > 0 else 99.0
 
        # Player profiles at this exact moment
        batter_data = get_batter_profile_at_this_moment(ball.batsman_id, innings.id)
        bowler_data = get_bowler_profile_at_this_moment(ball.bowler_id, innings.id)
 
        # Only record snapshots from over 1 onwards
        # (first 6 balls have very little signal)
        if running_legal_balls >= 6:
            row = {
                # ── Game state numbers ─────────────────────────────────
                'runs_so_far':         running_runs,
                'wickets_fallen':      running_wickets,
                'balls_bowled':        running_legal_balls,
                'balls_remaining':     balls_remaining,
                'overs_completed':     overs_completed,
                'current_run_rate':    round(crr, 2),
 
                # ── 2nd innings features ───────────────────────────────
                'required_run_rate':   round(rrr, 2),
                'runs_to_target':      runs_to_target,
                'target':              target,
 
                # ── Current batter ─────────────────────────────────────
                **batter_data,
 
                # ── Current bowler ─────────────────────────────────────
                **bowler_data,
 
                # ── Match context ──────────────────────────────────────
                'innings_number':      innings.innings_number,
                'over_limit':          over_limit,
 
                # ── LABEL: what we want the model to predict ───────────
                'final_score':         final_score,
 
                # ── Extra info for win label (used in training only) ───
                'match_id':            innings.match_id,
                'batting_team_id':     innings.batting_team_id,
            }
            rows.append(row)
 
        # ── Update state AFTER this ball ──────────────────────────────────
        running_runs += (ball.runs_scored + ball.extra_runs)
        if ball.is_wicket:
            running_wickets += 1
        if ball.is_legal_delivery:
            running_legal_balls += 1
 
    return rows
 
 
def main():
    print("\n" + "=" * 60)
    print("BUILDING TRAINING DATA")
    print("=" * 60)
 
    app = create_app()
 
    with app.app_context():
 
        # Only use COMPLETED innings — we know the final score for those
        completed_innings = (
            Inning.query
            .filter_by(is_completed=True)
            .all()
        )
 
        print(f"\nCompleted innings found: {len(completed_innings)}")
 
        if len(completed_innings) == 0:
            print("\nNo completed innings yet.")
            print("Record some matches first using your scorer.")
            print("Then run this script again.")
            return
 
        if len(completed_innings) < 10:
            print(f"\nWARNING: Only {len(completed_innings)} innings available.")
            print("Predictions will work but won't be very accurate.")
            print("Aim for 20+ innings before relying on predictions.")
 
        all_rows = []
 
        for innings in completed_innings:
            snapshots = build_snapshots_for_one_innings(innings)
            all_rows.extend(snapshots)
 
            print(
                f"  Innings ID {innings.id:3d} | "
                f"Match {innings.match_id:3d} | "
                f"Inn #{innings.innings_number} | "
                f"Final: {innings.total_runs}/{innings.total_wickets} | "
                f"Snapshots: {len(snapshots)}"
            )
 
        if not all_rows:
            print("\nNo rows generated. Are your innings really completed?")
            return
 
        df = pd.DataFrame(all_rows)
 
        # Create data/ folder if it doesn't exist
        os.makedirs('data', exist_ok=True)
        output_path = 'data/training_data.csv'
        df.to_csv(output_path, index=False)
 
        print(f"\nTotal rows saved: {len(df)}")
        print(f"Saved to: {output_path}")
        print(f"\nColumn list:")
        for col in df.columns:
            print(f"  {col}")
 
 
if __name__ == '__main__':
    main()
 