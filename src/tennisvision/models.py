"""Serializable analysis contracts shared by the API, engine, and exports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Point(Record):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class InitialScore(Record):
    points: tuple[int, int] = (0, 0)
    games: tuple[int, int] = (0, 0)
    sets: tuple[int, int] = (0, 0)
    best_of: Literal[3, 5] = 3
    tiebreak: bool = False
    tiebreak_target: Literal[7, 10] = 7

    @model_validator(mode="after")
    def legal(self):
        if min(*self.points, *self.games, *self.sets) < 0:
            raise ValueError("Score values cannot be negative.")
        if max(self.sets) >= (self.best_of + 1) // 2:
            raise ValueError("The initial match must be unfinished.")
        if max(self.games) > 6 or (
            max(self.games) >= 6 and abs(self.games[0] - self.games[1]) >= 2
        ):
            raise ValueError("Use the current unfinished set's game score.")
        if self.tiebreak != (self.games == (6, 6)):
            raise ValueError("A tiebreak requires a 6-6 game score.")
        if (
            self.tiebreak
            and max(self.points) >= self.tiebreak_target
            and abs(self.points[0] - self.points[1]) >= 2
        ):
            raise ValueError("The initial tiebreak must be unfinished.")
        if not self.tiebreak:
            if max(self.points) > 4 or (4 in self.points and sorted(self.points) != [3, 4]):
                raise ValueError("Point indices must be 0,1,2,3 or a valid 40-AD pair.")
        return self


class AnalysisConfig(Record):
    players: tuple[str, str] = ("Player A", "Player B")
    near_player: Literal[0, 1] = 0
    # Far left, far right, near right, near left: singles court baseline corners.
    court: list[Point] = Field(
        default_factory=lambda: [
            Point(x=0.32, y=0.22),
            Point(x=0.68, y=0.22),
            Point(x=0.90, y=0.90),
            Point(x=0.10, y=0.90),
        ],
        min_length=4,
        max_length=4,
    )
    detector: Literal["classical", "tracknet"] = "classical"
    inference_fps: float = Field(default=8, ge=2, le=30)
    pose_fps: float = Field(default=8, ge=2, le=30)
    lost_seconds: float = Field(default=1.8, ge=0.8, le=5)
    initial: InitialScore = Field(default_factory=InitialScore)

    @model_validator(mode="after")
    def names(self):
        if any(not p.strip() or len(p) > 30 for p in self.players):
            raise ValueError("Player names must contain 1 to 30 characters.")
        return self


class Event(Record):
    id: str
    time: float = Field(ge=0)
    winner: Literal[0, 1] | None = None
    status: Literal["pending", "confirmed", "rejected"] = "pending"
    source: Literal["vision", "manual"] = "vision"
    evidence_score: float = Field(default=0, ge=0, le=1)
    reason: str = Field(default="", max_length=2000)
    start: float = Field(default=0, ge=0)
    hits: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def confirmed_winner(self):
        if self.status == "confirmed" and self.winner is None:
            raise ValueError("A confirmed point needs a winner.")
        return self


class Review(Record):
    events: list[Event] = Field(max_length=10000)


class BallLabel(Record):
    frame: int = Field(ge=0)
    visible: bool
    x: float | None = Field(default=None, ge=0, le=1)
    y: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def coordinates(self):
        if self.visible and (self.x is None or self.y is None):
            raise ValueError("A visible ball needs x and y coordinates.")
        return self
