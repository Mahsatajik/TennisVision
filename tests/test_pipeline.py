import json
from zipfile import ZipFile

import cv2
import numpy as np

from tennisvision.models import AnalysisConfig
from tennisvision.pipeline import analyze


def test_video_export_and_raw_review_artifact_separation(tmp_path, monkeypatch):
    class EmptyPose:
        def __init__(self, court):
            pass

        def infer(self, frame):
            return {}

        def reset(self):
            pass

        def close(self):
            pass

    class EmptyBall:
        name = "No detections fixture"

        def observe(self, frame):
            pass

        def infer(self, frame, time):
            return None

        def reset(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("tennisvision.pipeline.PoseEstimator", EmptyPose)
    monkeypatch.setattr("tennisvision.pipeline.create_detector", lambda name: EmptyBall())
    source = tmp_path / "source.mp4"
    out = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for _ in range(12):
        out.write(np.full((180, 320, 3), 30, np.uint8))
    out.release()
    result = analyze(source, tmp_path, AnalysisConfig())
    assert result["analysis"]["processed_frames"] == 12
    assert result["events"] == []
    assert result["scores"]["confirmed"]["total_points"] == [0, 0]
    assert json.loads((tmp_path / "predictions.json").read_text()) == []
    video = cv2.VideoCapture(str(tmp_path / "annotated.mp4"))
    assert video.isOpened() and int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    video.release()
    with ZipFile(tmp_path / "report.zip") as archive:
        assert archive.testzip() is None
        assert {"predictions.json", "events.json", "poses.jsonl", "scores.json"} <= set(
            archive.namelist()
        )
        assert "source.mp4" not in archive.namelist()


def test_export_preserves_video_when_source_audio_ends_early(tmp_path):
    import subprocess

    import imageio_ffmpeg

    from tennisvision.video import encode

    silent = tmp_path / "silent.mp4"
    out = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
    for _ in range(30):
        out.write(np.zeros((90, 160, 3), np.uint8))
    out.release()
    audio = tmp_path / "short.wav"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.2",
            str(audio),
        ],
        check=True,
    )
    destination = tmp_path / "result.mp4"
    encode(silent, audio, destination)
    cap = cv2.VideoCapture(str(destination))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    assert decoded == 30
