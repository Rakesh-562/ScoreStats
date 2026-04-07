from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TeamProfile:
    team_id: int
    team_name: str

    # Batting
    avg_runs: float = 0.0
    avg_wickets: float = 0.0
    run_rate: float = 0.0
    boundary_rate: float = 0.0
    batting_consistency: float = 0.0

    # Bowling
    avg_conceded: float = 0.0
    avg_wickets_taken: float = 0.0
    economy: float = 0.0
    dot_ball_pct: float = 0.0
    bowling_consistency: float = 0.0

    # Strength (TOPSIS)
    batting_index: float = 0.0
    bowling_index: float = 0.0

    # Trends
    recent_form: float = 0.0   # last 3 matches
    dot_ball_pct: float = 0.0
    avg_partnership: float = 0.0

    # Head to head
    h2h_wins_vs: dict[int, int] = field(default_factory=dict)
    h2h_losses_vs: dict[int, int] = field(default_factory=dict)
    h2h_avg_margin_runs_vs: dict[int, float] = field(default_factory=dict)


class TeamProfileService:
    _cache = {}

    @classmethod
    def invalidate(cls, team_id=None):
        if team_id:
            cls._cache.pop(team_id, None)
        else:
            cls._cache.clear()
    @classmethod
    def get(cls, team_id: int) -> TeamProfile:
        if team_id in cls._cache:
            return cls._cache[team_id]

        profile = cls._build(team_id)
        cls._cache[team_id] = profile
        return profile
    @classmethod
    def _build(cls, team_id: int) -> TeamProfile:
        from app.extensions import db
        from app.models import Match, Inning, Ball, Team, Partnership

        team = db.session.get(Team, team_id)

        profile = TeamProfile(
            team_id=team_id,
            team_name=team.name if team else f"Team {team_id}"
        )

        matches = (
            Match.query
            .filter(
                ((Match.team_1_id == team_id) | (Match.team_2_id == team_id)),
                Match.status == "completed"
            )
            .order_by(Match.match_date.desc())
            .limit(15)
            .all()
        )

        runs, wickets, run_rates = [], [], []
        conceded, wkts, economies = [], [], []
        dot_ball_pcts, partnerships = [], []
        h2h_margin_totals: dict[int, list[float]] = {}

        for m in matches:
            opponent_id = m.team_2_id if m.team_1_id == team_id else m.team_1_id

            # Batting innings
            inn = Inning.query.filter_by(
                match_id=m.id,
                batting_team_id=team_id
            ).first()

            if inn:
                runs.append(inn.total_runs)
                wickets.append(inn.total_wickets)
                overs = inn.total_overs or 1
                run_rates.append(inn.total_runs / overs)
                balls = Ball.query.filter_by(inning_id=inn.id).all()
                legal_balls = [b for b in balls if b.is_legal_delivery]
                if legal_balls:
                    dots = sum(1 for b in legal_balls if (b.runs_scored or 0) == 0 and not b.extra_type)
                    dot_ball_pcts.append(dots / len(legal_balls))

                pships = Partnership.query.filter_by(inning_id=inn.id).all()
                if pships:
                    partnerships.extend(float(p.runs_scored or 0) for p in pships)

            # Bowling innings
            inn_b = Inning.query.filter_by(
                match_id=m.id,
                bowling_team_id=team_id
            ).first()

            if inn_b:
                conceded.append(inn_b.total_runs)
                wkts.append(inn_b.total_wickets)
                overs = inn_b.total_overs or 1
                economies.append(inn_b.total_runs / overs)

            if opponent_id is not None and m.winner_id is not None:
                if m.winner_id == team_id:
                    profile.h2h_wins_vs[opponent_id] = profile.h2h_wins_vs.get(opponent_id, 0) + 1
                elif m.winner_id == opponent_id:
                    profile.h2h_losses_vs[opponent_id] = profile.h2h_losses_vs.get(opponent_id, 0) + 1

                margin_value = cls._extract_margin_runs(m.win_margin)
                if margin_value is not None:
                    h2h_margin_totals.setdefault(opponent_id, []).append(margin_value)

        # Helper
        def avg(lst):
            return sum(lst)/len(lst) if lst else 0.0

        profile.avg_runs = avg(runs)
        profile.avg_wickets = avg(wickets)
        profile.run_rate = avg(run_rates)

        profile.avg_conceded = avg(conceded)
        profile.avg_wickets_taken = avg(wkts)
        profile.economy = avg(economies)
        profile.dot_ball_pct = avg(dot_ball_pcts)
        profile.avg_partnership = avg(partnerships)

        # Consistency (reuse your Gini logic)
        from app.services.ml_service import _gini

        profile.batting_consistency = 1 - _gini(runs)
        profile.bowling_consistency = 1 - _gini(wkts)

        # 🔥 TOPSIS Integration
        try:
            from app.services.analytics_service import AnalyticsService

            players = team.players.all() if team else []

            batting_stats = [
                AnalyticsService._career_batting_or_empty(p)
                for p in players
            ]
            bowling_stats = [
                AnalyticsService._career_bowling_or_empty(p)
                for p in players
            ]

            if batting_stats:
                profile.batting_index = avg([s.batting_index for s in batting_stats])

            if bowling_stats:
                profile.bowling_index = avg([s.bowling_index for s in bowling_stats])

        except Exception:
            pass

        # 🔥 Recent form (last 3 matches)
        last3 = matches[:3]
        wins = sum(1 for m in last3 if m.winner_id == team_id)
        profile.recent_form = wins / 3 if last3 else 0.0

        profile.h2h_avg_margin_runs_vs = {
            opponent_id: avg(margins)
            for opponent_id, margins in h2h_margin_totals.items()
        }
        return profile

    @staticmethod
    def _extract_margin_runs(win_margin: Optional[str]) -> Optional[float]:
        if not win_margin:
            return None
        text = win_margin.strip().lower()
        if "run" not in text:
            return None
        for token in text.replace("-", " ").split():
            if token.isdigit():
                return float(token)
        return None

    @classmethod
    def get_head_to_head_summary(cls, team_id: int) -> list[dict]:
        from app.extensions import db
        from app.models import Team

        profile = cls.get(team_id)
        opponent_ids = set(profile.h2h_wins_vs) | set(profile.h2h_losses_vs) | set(profile.h2h_avg_margin_runs_vs)
        if not opponent_ids:
            return []

        opponents = {
            team.id: team
            for team in Team.query.filter(Team.id.in_(opponent_ids)).all()
        }

        summary = []
        for opponent_id in opponent_ids:
            wins = profile.h2h_wins_vs.get(opponent_id, 0)
            losses = profile.h2h_losses_vs.get(opponent_id, 0)
            summary.append({
                "opponent_id": opponent_id,
                "opponent_name": opponents.get(opponent_id).name if opponents.get(opponent_id) else f"Team {opponent_id}",
                "wins": wins,
                "losses": losses,
                "matches": wins + losses,
                "avg_margin_runs": round(profile.h2h_avg_margin_runs_vs.get(opponent_id, 0.0), 1),
            })

        return sorted(summary, key=lambda item: (-item["matches"], -item["wins"], item["opponent_name"]))
