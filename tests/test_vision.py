import csv
import json

import numpy as np
import pytest

from tennisvision.ball import Association, ClassicalBall
from tennisvision.evaluate import ball_metrics, event_metrics
from tennisvision.geometry import Court
from tennisvision.models import AnalysisConfig, Point
from tennisvision.rallies import RallyAnalyzer


def court():
    return Court(
        [Point(x=0.2, y=0.1), Point(x=0.8, y=0.1), Point(x=0.9, y=0.9), Point(x=0.1, y=0.9)]
    )


def test_homography_roundtrip_and_metric_baselines():
    c = court()
    assert c.ground(0.2, 0.1) == pytest.approx([0, 0], abs=1e-4)
    assert c.ground(0.9, 0.9) == pytest.approx([8.23, 23.77], abs=1e-4)
    assert c.image(*c.ground(0.4, 0.6)) == pytest.approx([0.4, 0.6], abs=1e-5)


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError):
        Court([Point(x=0.1, y=0.1)] * 4)


def test_association_rejects_jump_but_recovers_after_gap():
    a = Association()
    assert a.select([(0.2, 0.2, 0.9)], 0)
    assert a.select([(0.9, 0.9, 0.99)], 0.1) is None
    assert a.select([(0.9, 0.9, 0.99)], 1)


def test_classical_ball_requires_motion():
    import cv2

    a = ClassicalBall()
    first = np.zeros((400, 600, 3), np.uint8)
    second = first.copy()
    cv2.circle(second, (200, 150), 4, (20, 240, 220), -1)
    assert a.infer(first, 0) is None
    assert a.infer(second, 0.1) is not None


def make_pose(x, y):
    return {"landmarks": [[x, y, 1] for _ in range(33)]}


def exchange():
    config = AnalysisConfig(lost_seconds=1)
    analyzer = RallyAnalyzer(config, court())
    for time, y in [(0, 0.6), (0.2, 0.8), (0.4, 0.6), (0.6, 0.2), (0.8, 0.1), (1, 0.4), (1.2, 0.7)]:
        analyzer.update(
            time,
            dict(x=0.5, y=y, quality=0.9),
            {"near": make_pose(0.5, y), "far": make_pose(0.5, y)},
        )
    return analyzer


def test_rally_loss_is_pending_and_clip_end_is_not_a_point():
    analyzer = exchange()
    analyzer.cut()
    assert analyzer.events == []
    analyzer = exchange()
    analyzer.update(2.3, None, {})
    assert len(analyzer.events) == 1
    event = analyzer.events[0]
    assert event.status == "pending"
    assert event.winner == 1
    assert event.evidence_score < 0.5


def test_single_contact_or_serve_does_not_prove_point():
    analyzer = RallyAnalyzer(AnalysisConfig(lost_seconds=1), court())
    analyzer.update(0, dict(x=0.5, y=0.8, quality=0.9), {"near": make_pose(0.5, 0.8)})
    analyzer.update(2, None, {})
    assert analyzer.events == []


def test_event_matching_is_one_to_one():
    result = event_metrics(
        [{"time": 2, "winner": 0}, {"time": 2.1, "winner": 0}], [{"time": 2, "winner": 0}], 0.5
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert result["winner_accuracy"] == 1


def test_event_assignment_does_not_lose_valid_adjacent_matches():
    result = event_metrics(
        [{"time": 0, "winner": 0}, {"time": 0.8, "winner": 1}],
        [{"time": 0.7, "winner": 0}, {"time": 1.5, "winner": 1}],
        0.8,
    )
    assert result["tp"] == 2


def test_ball_metrics_localization_and_visibility(tmp_path):
    pred = tmp_path / "ball.csv"
    labels = tmp_path / "labels.json"
    with pred.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=["frame", "visible", "x", "y"])
        out.writeheader()
        out.writerows(
            [
                dict(frame=0, visible=True, x=0.5, y=0.5),
                dict(frame=1, visible=True, x=0.9, y=0.9),
                dict(frame=2, visible=False, x="", y=""),
            ]
        )
    labels.write_text(
        json.dumps(
            {
                "labels": [
                    dict(frame=0, visible=True, x=0.5, y=0.5),
                    dict(frame=1, visible=True, x=0.1, y=0.1),
                    dict(frame=2, visible=False),
                    dict(frame=3, visible=True, x=0.4, y=0.4),
                ]
            }
        )
    )
    r = ball_metrics(pred, labels, 100, 100, 5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["unscored_labels"] == 1
    assert r["precision"] == 0.5
