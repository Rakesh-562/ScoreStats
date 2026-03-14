import sys
import os
# Ensure the project root is on the path so `app` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.extensions import db
from app.models import Team,Player
from app.services import MatchService,InningsService,BallService
from faker import Faker
import random
from datetime import datetime, timedelta
fake=Faker()
TEAMS = [
    {'name': 'Mumbai Indians', 'short_name': 'MI'},
    {'name': 'Chennai Super Kings', 'short_name': 'CSK' },
    {'name': 'Royal Challengers Bangalore', 'short_name': 'RCB'},
    {'name': 'Kolkata Knight Riders', 'short_name': 'KKR' },
    {'name': 'Delhi Capitals', 'short_name': 'DC'}
]

# Indian cricket player names for realism
PLAYER_NAMES = [
    "Virat Kohli", "Rohit Sharma", "MS Dhoni", "Jasprit Bumrah", "Ravindra Jadeja",
    "KL Rahul", "Hardik Pandya", "Rishabh Pant", "Shubman Gill", "Mohammed Siraj",
    "Yuzvendra Chahal", "Shreyas Iyer", "Suryakumar Yadav", "Ishan Kishan", "Axar Patel",
    "Ravichandran Ashwin", "Mohammed Shami", "Kuldeep Yadav", "Shikhar Dhawan", "Prithvi Shaw",
    "Sanju Samson", "Deepak Chahar", "Shardul Thakur", "Washington Sundar", "Krunal Pandya",
    "Rahul Tripathi", "Nitish Rana", "Venkatesh Iyer", "Ruturaj Gaikwad", "Devdutt Padikkal"
]
def clear_db():
    """Clear existing data from the database."""
    print("Clearing existing data...")
    db.drop_all()
    db.create_all()
    print("Database cleared and recreated.")
def create_teams():
    print("Creating teams...")
    team_ids = []
    for team_data in TEAMS:
        team = Team(**team_data)
        db.session.add(team)
        db.session.flush()  # assigns team.id immediately
        team_ids.append(team.id)
        print(f"Created team: {team.name} (ID: {team.id})")
    db.session.commit()
    return team_ids

def create_players(team_ids):
    print("Creating players...")
    batting_styles = ['right-handed', 'left-handed']
    bowling_styles = ['right-arm fast', 'left-arm fast', 'right-arm spin', 'left-arm spin']
    player_name_pool = PLAYER_NAMES.copy()
    random.shuffle(player_name_pool)
    name_index = 0
    all_players = []
    used_jerseys = set()

    for team_id in team_ids:
        team_players = []
        # Build an XI per team so "all out" maps to 10 wickets.
        roles_distribution = (['wicket-keeper'] * 1) + (['batsman'] * 4) + (['bowler'] * 3) + (['all-rounder'] * 3)
        for role in roles_distribution:
            name = player_name_pool[name_index] if name_index < len(player_name_pool) else fake.name()
            name_index += 1

            jersey = random.randint(1, 99)
            while jersey in used_jerseys:
                jersey = random.randint(1, 99)
            used_jerseys.add(jersey)

            player = Player(
                name=name,
                jersey_number=jersey,
                team_id=team_id,
                role=role,
                batting_style=random.choice(batting_styles),
                bowling_style=random.choice(bowling_styles) if role in ['bowler', 'all-rounder'] else None
            )
            db.session.add(player)
            team_players.append(player)

        all_players.extend(team_players)
        print(f"Created {len(team_players)} players for team ID: {team_id}")

    db.session.commit()
    print(f"Total players created: {len(all_players)}")
    return all_players
def create_matches(team1,team2,match_num):
    print(f"Creating matches between Team {team1} and Team {team2}... ")
    match=MatchService.create_match(team_1_id=team1,team_2_id=team2,match_date=datetime.strptime(f"2024-07-{match_num+1:02d} 19:00:00", "%Y-%m-%d %H:%M:%S"),over_limit=20,match_type='T20')
    print(f"Created match ID: {match.id} between Team {team1} and Team {team2}")
    toss_winner=random.choice([team1,team2])
    # Model expects 'bat' or 'field' (not 'bowl')
    toss_decision=random.choice(['bat','field'])
    # Persist toss to DB
    MatchService.record_toss(match.id,toss_winner,toss_decision)
    if toss_decision=='bat':
        batting_first=toss_winner
        bowling_first=team2 if toss_winner==team1 else team1    
    else:
        bowling_first=toss_winner
        batting_first=team2 if toss_winner==team1 else team1
    print(f"Toss winner: Team {toss_winner}, Decision: {toss_decision}, Batting first: Team {batting_first}, Bowling first: Team {bowling_first}")
    print(f"Simulating innings 1 for Match ID: {match.id}...")
    innings1_runs=simulate_innings(match.id,batting_first, bowling_first,1)
    print(f"Simulating innings 2 for Match ID: {match.id}...")
    innings2_runs=simulate_innings(match.id,bowling_first, batting_first,2)
    print(f"Match ID: {match.id} completed. Innings 1 runs: {innings1_runs}, Innings 2 runs: {innings2_runs}")
    return match
def simulate_innings(match_id,batting_team,bowling_team,innings_num):
    innings=InningsService.start_innings(match_id=match_id,batting_team_id=batting_team,bowling_team_id=bowling_team,innings_number=innings_num)
    print(f"Created innings ID: {innings.id} for Match ID: {match_id}, Batting Team: {batting_team}, Innings Number: {innings_num}")
    batting_order = Player.query.filter(
        Player.team_id == batting_team
    ).order_by(Player.id.asc()).all()
    
    bowlers = Player.query.filter(
        Player.team_id == bowling_team,
        Player.role.in_(['bowler', 'all-rounder'])
    ).all()
    if not bowlers:
        bowlers = Player.query.filter(Player.team_id == bowling_team).all()
    if len(batting_order) < 2 or not bowlers:
        print(f"No valid batsmen or bowlers found for teams {batting_team} and {bowling_team}. Skipping innings simulation.")
        return 0
    striker=batting_order[0]
    non_striker=batting_order[1]
    current_bowler=random.choice(bowlers)
    next_batsman_index=2
    max_wickets=max(0, len(batting_order) - 1)
    total_runs=0
    wickets=0
    balls_bowled=0
    max_balls=120
    for ball_num in range(1,max_balls+1):
        # Check if innings should end
        if wickets >= max_wickets:
            break
        
        # Check if target reached (2nd innings)
        if innings_num == 2:
            target = innings.target
            if target and total_runs >= target:
                print(f"Target reached!")
                break
        
        # Decide ball outcome (weighted probabilities)
        runs = random.choices(
            [0, 1, 2, 3, 4, 6],
            weights=[35, 25, 15, 5, 15, 5]  # Realistic T20 distribution
        )[0]
        
        # Extra probability (8%)
        is_extra = random.random() < 0.08
        extra_runs = 0
        extra_type = None
        
        if is_extra:
            # 'bye' is a legal delivery; 'wide'/'no-ball' are not
            extra_type = random.choice(['wide', 'no-ball', 'bye', 'leg-bye'])
            extra_runs = 1
            is_legal = extra_type not in ['wide', 'no-ball']
        else:
            is_legal = True

        # Wicket probability (5% base, increases in death overs)
        # Use balls_bowled (legal balls so far) for accurate over tracking
        over_num = balls_bowled // 6
        wicket_probability = 0.05 + (0.02 if over_num >= 15 else 0)
        is_wicket = is_legal and (random.random() < wicket_probability)
        
        # Record the ball
        try:
            ball = BallService.record_ball(
                innings_id=innings.id,
                striker_id=striker.id,
                non_striker_id=non_striker.id,
                bowler_id=current_bowler.id,
                runs=runs,
                extras=extra_runs,
                extra_type=extra_type,
                is_wicket=is_wicket,
                wicket_type='bowled' if is_wicket else None,
                dismissed_player_id=striker.id if is_wicket else None
            )
            
            total_runs += (runs + extra_runs)
            
            if is_legal:
                balls_bowled += 1
            
            # Handle wicket
            if is_wicket:
                wickets += 1
                if next_batsman_index < len(batting_order):
                    striker = batting_order[next_batsman_index]
                    next_batsman_index += 1
                else:
                    break  # No more batsmen
            
            # Rotate strike on odd runs
            elif runs % 2 == 1:
                striker, non_striker = non_striker, striker
            
            # Change bowler every over (6 legal balls)
            if balls_bowled > 0 and balls_bowled % 6 == 0:
                striker, non_striker = non_striker, striker  # Over change
                current_bowler = random.choice([b for b in bowlers if b.id != current_bowler.id] or bowlers)
        
        except Exception as e:
            print(f"Error recording ball: {e}")
            break
    InningsService.complete_innings(innings_id=innings.id)
    return total_runs
def print_summary():
    print("\n" + "="*60)
    print("DATABASE SUMMARY")
    print("="*60)
    
    teams = Team.query.all()
    players = Player.query.all()
    from app.models import Match, Inning, Ball
    matches = Match.query.all()
    innings = Inning.query.all()
    balls = Ball.query.all()
    
    print(f"\n Teams: {len(teams)}")
    for team in teams:
        player_count = Player.query.filter_by(team_id=team.id).count()
        print(f"   • {team.name} ({team.short_name}) - {player_count} players")
    
    print(f"\n Total Players: {len(players)}")
    print(f"   • Batsmen: {len([p for p in players if p.role == 'batsman'])}")
    print(f"   • Bowlers: {len([p for p in players if p.role == 'bowler'])}")
    print(f"   • All-rounders: {len([p for p in players if p.role == 'all-rounder'])}")
    print(f"   • Wicket-keepers: {len([p for p in players if p.role == 'wicket-keeper'])}")
    
    print(f"\n Matches: {len(matches)}")
    completed = len([m for m in matches if m.status == 'completed'])
    print(f"   • Completed: {completed}")
    
    print(f"\n Innings: {len(innings)}")
    print(f"\n Balls Recorded: {len(balls)}")
    print(f"   • Average per match: {len(balls) // len(matches) if matches else 0}")
    
    # Calculate total runs
    total_runs = sum(i.total_runs for i in innings)
    print(f"\n Total Runs Scored: {total_runs}")
    
    print("\n" + "="*60)
    print("DATABASE SEEDED SUCCESSFULLY!")
    print("="*60)
def main():
    print("\n" + "="*60)
    print("SEEDING DATABASE WITH SAMPLE DATA")
    print("="*60)
    app=create_app()
    with app.app_context():
        clear_db()
        team_ids=create_teams()
        create_players(team_ids)
        print("\nSimulating matches and innings...")
        print("-"*60)
        num_matches=10
        for i in range(num_matches):
            team1,team2=random.sample(team_ids,2)
            create_matches(team1,team2,i+1)
        print_summary()
if __name__=="__main__":
    main()
