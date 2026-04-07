import sys
import os
import json
from datetime import datetime
from itertools import count
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.extensions import db
from sqlalchemy import func

from app.models import Team, Player, Match, Inning, Ball, BattingScorecard, BowlingScorecard, Partnership

DEFAULT_UPLOAD_DIR = Path(
    r"C:\Users\rakesh\AppData\Local\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm\LocalState\sessions\4DB5D8D8384B3366D726FA545C9866C1A6B757C3\transfers\2026-14\2024 IPL DATA"
)

# Global registries (persist across all match files)
team_map    = {}   # team_name -> Team object
player_map  = {}   # player_name -> Player object
jersey_pool = count(1)


def resolve_upload_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    env_dir = os.environ.get("IPL_UPLOAD_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return DEFAULT_UPLOAD_DIR


def discover_json_files(upload_dir: Path) -> list[Path]:
    return sorted(upload_dir.rglob("*.json"))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_short_name(name: str, existing: set) -> str:
    """Build a unique ≤6-char short name from the team name."""
    initials = ''.join(w[0] for w in name.split()).upper()
    short = initials[:6]
    base, i = short, 2
    while short in existing:
        short = base[:5] + str(i)
        i += 1
    return short


def get_or_create_team(name: str) -> Team:
    if name in team_map:
        return team_map[name]
    existing_shorts = {t.short_name for t in team_map.values()}
    team = Team(name=name, short_name=make_short_name(name, existing_shorts))
    db.session.add(team)
    db.session.flush()
    team_map[name] = team
    print(f"    + Team: {name} ({team.short_name})")
    return team


def get_or_create_player(name: str, team_name: str) -> Player:
    if name in player_map:
        return player_map[name]
    team = team_map.get(team_name) or list(team_map.values())[0]
    player = Player(
        name=name,
        jersey_number=next(jersey_pool),
        team_id=team.id,
        role='batsman',   # default; no role info in source data
        is_active=True,
    )
    db.session.add(player)
    db.session.flush()
    player_map[name] = player
    return player


def resolve_extra_type(extras_detail: dict):
    """Map Cricsheet extra keys → our schema values."""
    if 'wides'    in extras_detail: return 'wide'
    if 'noballs'  in extras_detail: return 'no-ball'
    if 'byes'     in extras_detail: return 'bye'
    if 'legbyes'  in extras_detail: return 'leg-bye'
    return None


# ──────────────────────────────────────────────
# Per-innings processing
# ──────────────────────────────────────────────

def process_innings(match: Match, inn_num: int, inn_data: dict,
                    batting_team_name: str, bowling_team_name: str) -> Inning:
    batting_team  = team_map[batting_team_name]
    bowling_team  = team_map[bowling_team_name]

    target_info   = inn_data.get('target') or {}
    target_runs   = target_info.get('runs')

    inning = Inning(
        match_id       = match.id,
        batting_team_id= batting_team.id,
        bowling_team_id= bowling_team.id,
        innings_number = inn_num,
        is_completed   = True,
        target         = target_runs,
    )
    db.session.add(inning)
    db.session.flush()

    # Per-innings state
    batting_scorecards  = {}   # player_name -> BattingScorecard
    bowling_scorecards  = {}   # player_name -> BowlingScorecard
    batting_position    = 1
    seen_batters        = set()

    total_runs    = 0
    total_wickets = 0
    total_extras  = 0
    legal_balls   = 0

    # Partnership tracking
    current_partnership      = None
    current_partnership_pair = None   # frozenset of two names

    for over_data in inn_data.get('overs', []):
        over_num   = over_data['over']
        ball_in_over = 0   # counts only legal deliveries for ball_number

        for delivery in over_data.get('deliveries', []):
            batter_name      = delivery['batter']
            bowler_name      = delivery['bowler']
            non_striker_name = delivery['non_striker']

            runs_info  = delivery.get('runs', {})
            runs_scored = runs_info.get('batter', 0)
            extra_runs  = runs_info.get('extras', 0)

            extra_type = resolve_extra_type(delivery.get('extras', {}))
            is_legal   = extra_type not in ('wide', 'no-ball')

            # Wicket info
            wickets   = delivery.get('wickets', [])
            is_wicket = len(wickets) > 0
            wicket_type = dismissed_player_name = fielder_name = None
            if is_wicket:
                w = wickets[0]
                wicket_type            = w.get('kind')
                dismissed_player_name  = w.get('player_out')
                fielders               = w.get('fielders', [])
                if fielders:
                    fielder_name = fielders[0].get('name')

            # Resolve Player objects (create on-the-fly if missing)
            batter       = player_map.get(batter_name)      or get_or_create_player(batter_name, batting_team_name)
            non_striker  = player_map.get(non_striker_name) or get_or_create_player(non_striker_name, batting_team_name)
            bowler       = player_map.get(bowler_name)      or get_or_create_player(bowler_name, bowling_team_name)
            dismissed_pl = player_map.get(dismissed_player_name) if dismissed_player_name else None
            fielder      = player_map.get(fielder_name) if fielder_name else None

            if is_legal:
                ball_in_over += 1
                legal_balls  += 1

            # ── Ball record ──────────────────────────
            ball = Ball(
                inning_id           = inning.id,
                over_number         = over_num,
                ball_number         = ball_in_over,
                batsman_id          = batter.id,
                non_striker_id      = non_striker.id,
                bowler_id           = bowler.id,
                runs_scored         = runs_scored,
                is_wicket           = is_wicket,
                wicket_type         = wicket_type,
                extra_type          = extra_type,
                extra_runs          = extra_runs,
                dismissed_player_id = dismissed_pl.id if dismissed_pl else None,
                fielder_id          = fielder.id if fielder else None,
                is_legal_delivery   = is_legal,
            )
            db.session.add(ball)
            db.session.flush()

            # ── Batting scorecard ────────────────────
            for player_name, player_obj in [(batter_name, batter), (non_striker_name, non_striker)]:
                if player_name not in seen_batters:
                    sc = BattingScorecard(
                        innings_id      = inning.id,
                        player_id       = player_obj.id,
                        batting_position= batting_position,
                    )
                    db.session.add(sc)
                    db.session.flush()
                    batting_scorecards[player_name] = sc
                    seen_batters.add(player_name)
                    batting_position += 1

            batting_scorecards[batter_name].update_stats(ball)

            # ── Bowling scorecard ────────────────────
            if bowler_name not in bowling_scorecards:
                sc = BowlingScorecard(
                    innings_id = inning.id,
                    player_id  = bowler.id,
                )
                db.session.add(sc)
                db.session.flush()
                bowling_scorecards[bowler_name] = sc
            bowling_scorecards[bowler_name].update_stats(ball)

            # ── Partnership ──────────────────────────
            pair = frozenset([batter_name, non_striker_name])
            if current_partnership_pair != pair:
                # Close old partnership
                if current_partnership:
                    current_partnership.is_active = False
                # Open new one
                current_partnership = Partnership(
                    inning_id   = inning.id,
                    batsman1_id = batter.id,
                    batsman2_id = non_striker.id,
                    is_active   = True,
                )
                db.session.add(current_partnership)
                db.session.flush()
                current_partnership_pair = pair

            current_partnership.runs_scored += runs_scored
            if is_legal:
                current_partnership.balls_faced += 1
            if is_wicket:
                current_partnership.wickets_fallen += 1
                current_partnership.is_active = False
                current_partnership      = None
                current_partnership_pair = None

            # ── Running totals ───────────────────────
            total_runs    += runs_scored + extra_runs
            total_extras  += extra_runs
            if is_wicket:
                total_wickets += 1

    # Close any open partnership at innings end
    if current_partnership:
        current_partnership.is_active = False

    # Update inning summary
    inning.total_runs    = total_runs
    inning.total_wickets = total_wickets
    inning.extras        = total_extras
    inning.total_overs   = (legal_balls // 6) + (legal_balls % 6) / 10.0
    db.session.flush()

    return inning


# ──────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────

def process_file(filepath: Path):
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    info         = data['info']
    teams_in_match = info['teams']   # [batting_first_team, other_team] — order varies

    # Create teams
    for t in teams_in_match:
        get_or_create_team(t)

    # Create players from the squad list
    for team_name, player_names in info.get('players', {}).items():
        for pname in player_names:
            get_or_create_player(pname, team_name)

    # Match metadata
    match_date       = datetime.strptime(info['dates'][0], '%Y-%m-%d')
    team1            = team_map[teams_in_match[0]]
    team2            = team_map[teams_in_match[1]]
    toss_winner_name = info['toss']['winner']
    toss_decision    = info['toss']['decision']   # 'bat' / 'field'
    toss_winner_team = team_map.get(toss_winner_name)

    outcome    = info.get('outcome', {})
    winner_name = outcome.get('winner')
    winner_team = team_map.get(winner_name) if winner_name else None

    win_by  = outcome.get('by', {})
    win_margin = (f"{win_by['runs']} runs"    if 'runs'    in win_by else
                  f"{win_by['wickets']} wickets" if 'wickets' in win_by else None)

    mom_names  = info.get('player_of_match', [])
    mom_player = player_map.get(mom_names[0]) if mom_names else None

    match = Match(
        team_1_id        = team1.id,
        team_2_id        = team2.id,
        match_date       = match_date,
        match_type       = info.get('match_type', 'T20'),
        over_limit       = info.get('overs', 20),
        status           = 'completed',
        toss_winner      = toss_winner_team.id if toss_winner_team else None,
        toss_decision    = toss_decision,
        winner_id        = winner_team.id if winner_team else None,
        win_margin       = win_margin,
        man_of_the_match = mom_player.id if mom_player else None,
    )
    db.session.add(match)
    db.session.flush()

    # Process each innings
    innings_data = data.get('innings', [])
    for inn_num, inn_data in enumerate(innings_data, 1):
        batting_team_name  = inn_data['team']
        bowling_team_name  = next(t for t in teams_in_match if t != batting_team_name)
        process_innings(match, inn_num, inn_data, batting_team_name, bowling_team_name)

    print(f"  OK  {team1.short_name} vs {team2.short_name}  |  {match_date.date()}  |  Winner: {winner_name or 'N/A'}")
    return match


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)
    print(f"  Teams              : {Team.query.count()}")
    print(f"  Players            : {Player.query.count()}")
    print(f"  Matches            : {Match.query.count()}")
    print(f"  Innings            : {Inning.query.count()}")
    print(f"  Balls              : {Ball.query.count()}")
    print(f"  Batting Scorecards : {BattingScorecard.query.count()}")
    print(f"  Bowling Scorecards : {BowlingScorecard.query.count()}")
    print(f"  Partnerships       : {Partnership.query.count()}")
    print("=" * 60)
    print("IPL DATA SEEDED SUCCESSFULLY!")
    print("=" * 60)


def assign_player_roles():
    print("\nAssigning player roles from match involvement...")

    batting_counts = dict(
        db.session.query(Ball.batsman_id, func.count(Ball.id))
        .group_by(Ball.batsman_id)
        .all()
    )
    bowling_counts = dict(
        db.session.query(Ball.bowler_id, func.count(Ball.id))
        .group_by(Ball.bowler_id)
        .all()
    )
    stumping_counts = dict(
        db.session.query(Ball.fielder_id, func.count(Ball.id))
        .filter(Ball.wicket_type == "stumped", Ball.fielder_id.isnot(None))
        .group_by(Ball.fielder_id)
        .all()
    )

    role_totals = {"batsman": 0, "bowler": 0, "all-rounder": 0, "wicket-keeper": 0}

    for player in Player.query.all():
        batting_balls = batting_counts.get(player.id, 0)
        bowling_balls = bowling_counts.get(player.id, 0)
        stumpings = stumping_counts.get(player.id, 0)

        if stumpings > 0:
            role = "wicket-keeper"
        elif bowling_balls >= 60 and batting_balls >= 60:
            role = "all-rounder"
        elif bowling_balls >= 60:
            role = "bowler"
        else:
            role = "batsman"

        player.role = role
        role_totals[role] += 1

    db.session.commit()
    print("Role distribution:", role_totals)


def main():
    print("\n" + "=" * 60)
    print("  SEEDING DATABASE WITH REAL IPL DATA")
    print("=" * 60)

    upload_dir = resolve_upload_dir()

    app = create_app()
    with app.app_context():
        if not upload_dir.exists():
            raise FileNotFoundError(f"Upload directory not found: {upload_dir}")

        json_files = discover_json_files(upload_dir)
        if not json_files:
            raise FileNotFoundError(f"No JSON match files found under: {upload_dir}")

        print("\nDropping & recreating all tables...")
        db.drop_all()
        db.create_all()
        print("Done.\n")

        print(f"Found {len(json_files)} match files in {upload_dir}\n")

        for fpath in json_files:
            fname = fpath.name
            print(f"Processing {fname} ...")
            try:
                process_file(fpath)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()

        assign_player_roles()
        print_summary()


if __name__ == '__main__':
    main()
