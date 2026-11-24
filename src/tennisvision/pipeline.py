"""Decode -> pose/ball fusion -> hypotheses -> review -> reproducible exports."""

import csv
import hashlib
import json
import platform
import time as clock
from collections import deque
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

from . import __version__
from .ball import create_detector
from .geometry import Court
from .models import Event
from .pose import PoseEstimator
from .rallies import RallyAnalyzer
from .render import render_score, vision_overlay
from .video import inspect_video, writer


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def export_review(
    root, config, events, progress=lambda value, message: None, cancelled=lambda: False
):
    scores = render_score(root, config, events, progress, cancelled)
    write_json(root / "events.json", [e.model_dump() for e in events])
    write_json(root / "scores.json", scores)
    with (root / "points.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(Event.model_fields)
        out = csv.DictWriter(stream, fieldnames=fields)
        out.writeheader()
        out.writerows(e.model_dump() for e in events)
    with ZipFile(root / "report.next.zip", "w", ZIP_DEFLATED) as archive:
        for filename in [
            "config.json",
            "analysis.json",
            "predictions.json",
            "events.json",
            "scores.json",
            "points.csv",
            "ball.csv",
            "poses.jsonl",
            "motion.csv",
            "labels.json",
        ]:
            if (root / filename).exists():
                archive.write(root / filename, filename)
    (root / "report.next.zip").replace(root / "report.zip")
    return scores


def analyze(
    source: Path, root: Path, config, progress=lambda value, message: None, cancelled=lambda: False
):
    started = clock.monotonic()
    info, first = inspect_video(source)
    root.mkdir(parents=True, exist_ok=True)
    court = Court(config.court)
    write_json(root / "config.json", config.model_dump())
    progress(0.01, "Loading pose and ball models")
    pose_engine = PoseEstimator(court)
    detector = None
    try:
        detector = create_detector(config.detector)
        return _process(
            source, root, config, court, pose_engine, detector, info, progress, cancelled, started
        )
    finally:
        pose_engine.close()
        if detector:
            detector.close()


def _process(
    source, root, config, court, pose_engine, detector, info, progress, cancelled, started
):
    cap = cv2.VideoCapture(str(source))
    scale = min(1, 1280 / info["width"])
    width, height = int(info["width"] * scale) // 2 * 2, int(info["height"] * scale) // 2 * 2
    out = writer(root / "poses.mp4", info["fps"], (width, height))
    analyzer = RallyAnalyzer(config, court)
    ball_rows, motion_rows = [], []
    poses, history, previous, previous_feet = {}, deque(maxlen=20), None, {}
    cached_poses = {}
    next_pose, next_ball = 0, 0
    frame_count, pose_observations, ball_count, cuts = 0, {"near": 0, "far": 0}, 0, 0
    best_preview = -1
    try:
        with (root / "poses.jsonl").open("w", encoding="utf-8") as pose_file:
            while True:
                if cancelled():
                    raise InterruptedError("Analysis cancelled.")
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (width, height))
                time = frame_count / info["fps"]
                small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
                cut = previous is not None and float(np.mean(cv2.absdiff(small, previous))) > 42
                previous = small
                if cut:
                    analyzer.cut()
                    detector.reset()
                    pose_engine.reset()
                    history.clear()
                    poses = {}
                    cached_poses.clear()
                    previous_feet = {}
                    cuts += 1
                    next_pose, next_ball = time, time
                detector.observe(frame)
                if time + 1e-6 >= next_pose:
                    observed_poses = pose_engine.infer(frame)
                    for slot, pose in observed_poses.items():
                        cached_poses[slot] = (time, pose)
                    next_pose = time + 1 / config.pose_fps
                    pose_file.write(
                        json.dumps(
                            dict(frame=frame_count, time=time, poses=observed_poses, cut=cut)
                        )
                        + "\n"
                    )
                    for slot, pose in observed_poses.items():
                        pose_observations[slot] += 1
                        gx, gy = pose["ground"]
                        distance = 0
                        prev = previous_feet.get(slot)
                        if prev and 0 < time - prev[0] < 0.5:
                            d = float(np.linalg.norm(np.array([gx, gy]) - prev[1:]))
                            if d / (time - prev[0]) <= 12:
                                distance = d
                        previous_feet[slot] = [time, gx, gy]
                        motion_rows.append(
                            dict(
                                time=round(time, 4),
                                slot=slot,
                                x_m=round(gx, 4),
                                y_m=round(gy, 4),
                                step_m=round(distance, 4),
                                quality=pose["quality"],
                            )
                        )
                # A held pose is only a short association aid. Exports contain actual observations.
                poses = {
                    slot: pose for slot, (seen, pose) in cached_poses.items() if time - seen <= 0.3
                }
                if time + 1e-6 >= next_ball:
                    ball = detector.infer(frame, time)
                    next_ball = time + 1 / config.inference_fps
                    # Court gating is deliberately loose: keep airborne and wide shots.
                    if ball and not (-0.5 <= court.ground(ball["x"], ball["y"])[0] / 8.23 <= 1.5):
                        ball = None
                    analyzer.update(time, ball, poses)
                    row = dict(
                        frame=frame_count,
                        time=round(time, 4),
                        visible=bool(ball),
                        x=ball["x"] if ball else "",
                        y=ball["y"] if ball else "",
                        quality=ball["quality"] if ball else 0,
                    )
                    ball_rows.append(row)
                    if ball:
                        history.append((time, ball))
                        ball_count += 1
                vision_overlay(frame, poses, history, court, time)
                out.write(frame)
                rating = len(poses) * 2 + int(bool(history))
                if rating > best_preview:
                    cv2.imwrite(str(root / "preview.jpg"), frame)
                    best_preview = rating
                if frame_count % 12 == 0:
                    progress(
                        0.03 + 0.77 * frame_count / info["frames"],
                        f"Tracking frame {frame_count + 1} / {info['frames']}",
                    )
                frame_count += 1
    finally:
        cap.release()
        out.release()
    if frame_count == 0:
        raise ValueError("The video contains no decodable frames.")
    analyzer.cut()
    for filename, rows, fields in [
        ("ball.csv", ball_rows, ["frame", "time", "visible", "x", "y", "quality"]),
        ("motion.csv", motion_rows, ["time", "slot", "x_m", "y_m", "step_m", "quality"]),
    ]:
        with (root / filename).open("w", newline="", encoding="utf-8") as stream:
            csv_out = csv.DictWriter(stream, fieldnames=fields)
            csv_out.writeheader()
            csv_out.writerows(rows)
    write_json(root / "predictions.json", [e.model_dump() for e in analyzer.events])
    analysis = dict(
        version=__version__,
        python=platform.python_version(),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        input=info,
        processed_frames=frame_count,
        ball_samples=len(ball_rows),
        ball_observations=ball_count,
        pose_observations=pose_observations,
        camera_cuts=cuts,
        detector=detector.name,
        hits=analyzer.hits,
        seconds=round(clock.monotonic() - started, 2),
        distances={
            slot: round(sum(r["step_m"] for r in motion_rows if r["slot"] == slot), 2)
            for slot in ("near", "far")
        },
        limitations=[
            "Uncalibrated event evidence scores are not probabilities.",
            "Ball disappearance is not proof of a point. Review every proposed winner.",
            "Ground distances are approximate and depend on accurate court calibration.",
        ],
    )
    write_json(root / "analysis.json", analysis)
    scores = export_review(root, config, analyzer.events, progress, cancelled)
    progress(1, "Analysis ready for review")
    return dict(analysis=analysis, events=[e.model_dump() for e in analyzer.events], scores=scores)
