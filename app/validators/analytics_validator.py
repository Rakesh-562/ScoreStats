"""
app/validators/analytics_validator.py
=======================================

Marshmallow schemas for the analytics API layer.

Each schema is responsible for one endpoint's input contract:
  * type coercion   — ``fields.Int`` rejects "abc", accepts "5"
  * presence checks — ``required=True`` fields raise if absent
  * value checks    — ``validate.OneOf`` rejects unknown enum strings

Schemas are kept separate from routes so they can be imported and
reused in tests without a running Flask application.
"""

from __future__ import annotations

from marshmallow import Schema, ValidationError, fields, post_load, validate

# ---------------------------------------------------------------------------
# Allowed enum values — single source of truth
# ---------------------------------------------------------------------------

VALID_ROLES = ("batsman", "bowler", "all-rounder", "wicket-keeper")
VALID_STAT_TYPES = ("batting", "bowling")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class InningsAnalyticsQuerySchema(Schema):
    """
    Query-string schema for::

        GET /api/v1/analytics/innings/<id>/batting
        GET /api/v1/analytics/innings/<id>/bowling

    All fields are optional; omitting ``role`` returns all players.
    """

    role = fields.Str(
        load_default=None,
        validate=validate.OneOf(
            VALID_ROLES,
            error="role must be one of: {choices}.",
        ),
        metadata={"description": "Filter results to one player role."},
    )


class MatchSelectionSchema(Schema):
    """
    JSON body schema for::

        POST /api/v1/analytics/select-xi

    ``player_roles`` is an optional override map.
    When absent, roles are read directly from the Player table.
    """

    match_id = fields.Int(
        required=True,
        strict=True,
        metadata={"description": "Primary key of the Match to analyse."},
    )
    player_roles = fields.Dict(
        keys=fields.Int(strict=True),
        values=fields.Str(
            validate=validate.OneOf(
                VALID_ROLES,
                error="Each role value must be one of: {choices}.",
            )
        ),
        load_default=None,
        metadata={
            "description": (
                "Optional per-player role override. "
                "Keys are player_id integers; values are role strings."
            )
        },
    )

    @post_load
    def coerce_role_keys(self, data: dict, **kwargs) -> dict:
        """
        Ensure player_role keys are integers even when the client sends
        JSON objects with string keys (which is the JSON spec default).
        """
        if data.get("player_roles"):
            data["player_roles"] = {
                int(k): v for k, v in data["player_roles"].items()
            }
        return data


class PlayerCareerQuerySchema(Schema):
    """
    Query-string schema for::

        GET /api/v1/analytics/player/<id>/career
    """

    stat_type = fields.Str(
        load_default="batting",
        validate=validate.OneOf(
            VALID_STAT_TYPES,
            error="stat_type must be 'batting' or 'bowling'.",
        ),
        metadata={"description": "Which career aggregate to return."},
    )


# ---------------------------------------------------------------------------
# Module-level schema singletons (avoids re-instantiation on every request)
# ---------------------------------------------------------------------------

_innings_query_schema = InningsAnalyticsQuerySchema()
_match_selection_schema = MatchSelectionSchema()
_player_career_schema = PlayerCareerQuerySchema()


# ---------------------------------------------------------------------------
# Convenience loaders (used by route handlers)
# ---------------------------------------------------------------------------


def load_innings_query(args: dict) -> dict:
    """
    Deserialise and validate GET query parameters for innings endpoints.

    Raises
    ------
    marshmallow.ValidationError
        On invalid or missing required fields.
    """
    return _innings_query_schema.load(args)


def load_selection_payload(data: dict) -> dict:
    """
    Deserialise and validate the JSON body for the select-XI endpoint.

    Raises
    ------
    marshmallow.ValidationError
        On invalid or missing required fields.
    """
    return _match_selection_schema.load(data)


def load_career_query(args: dict) -> dict:
    """
    Deserialise and validate GET query parameters for career endpoints.

    Raises
    ------
    marshmallow.ValidationError
        On invalid or missing required fields.
    """
    return _player_career_schema.load(args)
