import random

import pytest
from pydantic import ValidationError

from tennisvision.models import Event, InitialScore
from tennisvision.scoring import TennisScore, timeline


def test_deuce_advantage_and_game():
    score = TennisScore(InitialScore(points=(3, 3)))
    assert score.award(0)["points"] == ["AD", "40"]
    assert score.award(1)["points"] == ["40", "40"]
    score.award(1)
    assert score.award(1)["games"] == [0, 1]
    assert score.snapshot()["points"] == ["0", "0"]


def test_set_and_match_boundary():
    score = TennisScore(InitialScore(points=(3, 0), games=(5, 0), sets=(1, 0)))
    final = score.award(0)
    assert final["sets"] == [2, 0]
    assert final["completed_sets"] == [[6, 0]]
    assert final["winner"] == 0
    with pytest.raises(ValueError, match="finished"):
        score.award(1)


def test_tiebreak_win_by_two_and_set_7_6():
    score = TennisScore(InitialScore(games=(6, 6), points=(6, 6), tiebreak=True))
    assert score.award(0)["points"] == ["7", "6"]
    final = score.award(0)
    assert final["sets"] == [1, 0]
    assert final["games"] == [0, 0]
    assert final["completed_sets"] == [[7, 6]]
    assert not final["tiebreak"]


def test_ten_point_tiebreak():
    score = TennisScore(
        InitialScore(games=(6, 6), points=(9, 9), tiebreak=True, tiebreak_target=10)
    )
    score.award(1)
    assert score.award(1)["completed_sets"] == [[6, 7]]


@pytest.mark.parametrize(
    "values",
    [
        dict(points=(4, 1)),
        dict(games=(6, 6)),
        dict(games=(6, 2)),
        dict(games=(6, 6), points=(7, 3), tiebreak=True),
        dict(sets=(2, 0)),
        dict(points=(-1, 0)),
    ],
)
def test_invalid_initial_scores(values):
    with pytest.raises(ValidationError):
        InitialScore(**values)


def test_timeline_separates_estimated_confirmed_rejected_and_unresolved():
    events = [
        Event(id="pending", time=1, winner=0),
        Event(id="reviewed", time=2, winner=1, status="confirmed"),
        Event(id="ignored", time=3, winner=0, status="rejected"),
        Event(id="unknown", time=4),
    ]
    rows, estimated = timeline(InitialScore(), events)
    _, confirmed = timeline(InitialScore(), events, confirmed_only=True)
    assert estimated["total_points"] == [1, 1]
    assert confirmed["total_points"] == [0, 1]
    assert len(rows) == 2


def test_random_match_never_emits_invalid_display_state():
    rng = random.Random(12)
    score = TennisScore()
    for _ in range(10000):
        snapshot = score.award(rng.randrange(2))
        if not snapshot["tiebreak"]:
            assert all(p in {"0", "15", "30", "40", "AD"} for p in snapshot["points"])
            if "AD" in snapshot["points"]:
                assert sorted(snapshot["points"]) == ["40", "AD"]
        if snapshot["winner"] is not None:
            break
    assert snapshot["winner"] is not None
