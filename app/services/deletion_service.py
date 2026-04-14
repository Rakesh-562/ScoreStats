from __future__ import annotations

from sqlalchemy import or_

from app.extensions import db
from app.models import (
    Ball,
    BattingScorecard,
    BowlingScorecard,
    Inning,
    Match,
    Partnership,
    Player,
    Team,
)


class DeletionService:
    """Centralized cleanup for destructive deletes with related rows."""

    @staticmethod
    def _delete_match_rows(match: Match) -> None:
        innings_ids = [inning.id for inning in match.innings.order_by(Inning.innings_number).all()]

        if innings_ids:
            Ball.query.filter(Ball.inning_id.in_(innings_ids)).delete(synchronize_session=False)
            Partnership.query.filter(Partnership.inning_id.in_(innings_ids)).delete(synchronize_session=False)
            BattingScorecard.query.filter(BattingScorecard.innings_id.in_(innings_ids)).delete(synchronize_session=False)
            BowlingScorecard.query.filter(BowlingScorecard.innings_id.in_(innings_ids)).delete(synchronize_session=False)
            Inning.query.filter(Inning.id.in_(innings_ids)).delete(synchronize_session=False)

        db.session.delete(match)

    @classmethod
    def delete_match(cls, match_id: int) -> dict:
        match = db.session.get(Match, match_id)
        if match is None:
            raise ValueError("Match not found")

        cls._delete_match_rows(match)
        db.session.commit()
        return {"match_id": match_id}

    @staticmethod
    def _delete_player_rows(player: Player) -> None:
        Match.query.filter(Match.man_of_the_match == player.id).update(
            {Match.man_of_the_match: None},
            synchronize_session=False,
        )

        BattingScorecard.query.filter(
            or_(
                BattingScorecard.player_id == player.id,
                BattingScorecard.bowler_id == player.id,
                BattingScorecard.fielder_id == player.id,
            )
        ).delete(synchronize_session=False)

        BowlingScorecard.query.filter(BowlingScorecard.player_id == player.id).delete(synchronize_session=False)

        Partnership.query.filter(
            or_(
                Partnership.batsman1_id == player.id,
                Partnership.batsman2_id == player.id,
            )
        ).delete(synchronize_session=False)

        Ball.query.filter(
            or_(
                Ball.batsman_id == player.id,
                Ball.non_striker_id == player.id,
                Ball.bowler_id == player.id,
                Ball.dismissed_player_id == player.id,
                Ball.fielder_id == player.id,
            )
        ).delete(synchronize_session=False)

        db.session.delete(player)

    @classmethod
    def delete_player(cls, player_id: int) -> dict:
        player = db.session.get(Player, player_id)
        if player is None:
            raise ValueError("Player not found")

        affected_match_ids = {
            match_id
            for (match_id,) in (
                db.session.query(Inning.match_id)
                .join(Ball, Ball.inning_id == Inning.id)
                .filter(
                    or_(
                        Ball.batsman_id == player.id,
                        Ball.non_striker_id == player.id,
                        Ball.bowler_id == player.id,
                        Ball.dismissed_player_id == player.id,
                        Ball.fielder_id == player.id,
                    )
                )
                .distinct()
                .all()
            )
        }
        affected_match_ids.update(
            match_id
            for (match_id,) in (
                db.session.query(Inning.match_id)
                .join(BattingScorecard, BattingScorecard.innings_id == Inning.id)
                .filter(
                    or_(
                        BattingScorecard.player_id == player.id,
                        BattingScorecard.bowler_id == player.id,
                        BattingScorecard.fielder_id == player.id,
                    )
                )
                .distinct()
                .all()
            )
        )
        affected_match_ids.update(
            match_id
            for (match_id,) in (
                db.session.query(Inning.match_id)
                .join(BowlingScorecard, BowlingScorecard.innings_id == Inning.id)
                .filter(BowlingScorecard.player_id == player.id)
                .distinct()
                .all()
            )
        )

        if affected_match_ids:
            for match in Match.query.filter(Match.id.in_(affected_match_ids)).all():
                cls._delete_match_rows(match)

        cls._delete_player_rows(player)
        db.session.commit()
        return {
            "player_id": player_id,
            "team_id": player.team_id,
            "deleted_matches": len(affected_match_ids),
        }

    @classmethod
    def delete_team(cls, team_id: int) -> dict:
        team = db.session.get(Team, team_id)
        if team is None:
            raise ValueError("Team not found")

        match_ids = [
            m.id for m in Match.query.filter(
                or_(Match.team_1_id == team_id, Match.team_2_id == team_id)
            ).all()
        ]
        for match in Match.query.filter(Match.id.in_(match_ids)).all() if match_ids else []:
            cls._delete_match_rows(match)

        for player in team.players.all():
            cls._delete_player_rows(player)

        Inning.query.filter(
            or_(Inning.batting_team_id == team_id, Inning.bowling_team_id == team_id)
        ).delete(synchronize_session=False)

        db.session.delete(team)
        db.session.commit()
        return {"team_id": team_id, "deleted_matches": len(match_ids)}
