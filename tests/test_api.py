import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tennisvision.models import AnalysisConfig
from tennisvision.pipeline import write_json
from tennisvision.server import create_app


@pytest.fixture
def app(tmp_path):
    with TestClient(create_app(tmp_path / "jobs")) as client:
        yield client


def test_invalid_video_and_private_path_access(app):
    response = app.post("/api/matches", files={"video": ("bad.mp4", b"not a video", "video/mp4")})
    assert response.status_code == 422
    assert app.get("/api/matches/no-such-job").status_code == 404
    assert app.get("/api/matches/" + "0" * 32 + "/files/job.json").status_code == 404


def test_upload_range_download_and_labels(app, tmp_path):
    path = tmp_path / "short.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 90))
    for _ in range(10):
        out.write(np.zeros((90, 160, 3), np.uint8))
    out.release()
    response = app.post(
        "/api/matches", files={"video": ("match.mp4", path.read_bytes(), "video/mp4")}
    )
    assert response.status_code == 201
    assert response.json()["info"]["fps"] == 30
    job = response.json()["id"]
    assert (
        app.get(f"/api/matches/{job}/files/source.mp4", headers={"Range": "bytes=0-20"}).status_code
        == 206
    )
    assert (
        app.put(
            f"/api/matches/{job}/labels", json=[dict(frame=2, visible=True, x=0.2, y=0.3)]
        ).status_code
        == 200
    )
    assert app.get(f"/api/matches/{job}/files/labels.json").json()["labels"][0]["frame"] == 2
    fork = app.post(f"/api/matches/{job}/fork")
    assert fork.status_code == 201 and fork.json()["id"] != job
    new_id = fork.json()["id"]
    assert (
        app.get(f"/api/matches/{new_id}/files/source.mp4").content
        == app.get(f"/api/matches/{job}/files/source.mp4").content
    )
    assert app.get(f"/api/matches/{new_id}/files/labels.json").json()["labels"][0]["frame"] == 2
    assert (
        app.put(f"/api/matches/{job}/labels", json=[dict(frame=100, visible=False)]).status_code
        == 422
    )
    assert (
        app.put(f"/api/matches/{job}/labels", json=[dict(frame=2, visible=True)]).status_code == 422
    )


def test_review_validation_and_original_predictions(app):
    jobs = app.app.state.jobs
    job = "1" * 32
    root = jobs.root / job
    root.mkdir()
    jobs.save(root, dict(id=job, state="complete", info={"duration": 10}, result={}))
    write_json(root / "config.json", AnalysisConfig().model_dump())
    write_json(root / "predictions.json", [{"id": "original"}])
    assert (
        app.post(f"/api/matches/{job}/analyze", json=AnalysisConfig().model_dump()).status_code
        == 409
    )
    invalid = [dict(id="duplicate", time=2, winner=0), dict(id="duplicate", time=3, winner=1)]
    assert app.put(f"/api/matches/{job}/review", json={"events": invalid}).status_code == 422
    assert (
        app.put(
            f"/api/matches/{job}/review", json={"events": [dict(id="bad", time=11, winner=0)]}
        ).status_code
        == 422
    )
    response = app.post(
        f"/api/matches/{job}/preview-score",
        json={"events": [dict(id="one", time=2, winner=0, status="confirmed")]},
    )
    assert response.json()["confirmed"]["total_points"] == [1, 0]
    assert json.loads((root / "predictions.json").read_text()) == [{"id": "original"}]
