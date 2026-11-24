"""Ball localization and one-to-one rally event evaluation on independent human labels."""

import csv
import json
from pathlib import Path

import numpy as np


def ball_metrics(
    prediction_path: Path, labels_path: Path, width: int, height: int, tolerance_px=12
):
    if width <= 0 or height <= 0 or tolerance_px <= 0:
        raise ValueError("Image dimensions and pixel tolerance must be positive.")
    with prediction_path.open(encoding="utf-8") as stream:
        predictions = {int(row["frame"]): row for row in csv.DictReader(stream)}
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    if document.get("width", width) != width or document.get("height", height) != height:
        raise ValueError("Evaluation dimensions do not match the label coordinate system.")
    manifest = prediction_path.parent / "analysis.json"
    if document.get("input_sha256") and manifest.is_file():
        prediction_hash = json.loads(manifest.read_text(encoding="utf-8")).get("input_sha256")
        if prediction_hash and prediction_hash != document["input_sha256"]:
            raise ValueError("Prediction and label input hashes do not match.")
    labels = document["labels"]
    tp, fp, fn, tn, unscored = 0, 0, 0, 0, 0
    errors = []
    for label in labels:
        row = predictions.get(label["frame"])
        if row is None:
            unscored += 1
            continue
        detected = str(row["visible"]).lower() == "true"
        if label["visible"] and detected:
            error = float(
                np.hypot(
                    (float(row["x"]) - label["x"]) * width, (float(row["y"]) - label["y"]) * height
                )
            )
            errors.append(error)
            if error <= tolerance_px:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif label["visible"]:
            fn += 1
        elif detected:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        mean_error_px=float(np.mean(errors)) if errors else None,
        median_error_px=float(np.median(errors)) if errors else None,
        tolerance_px=tolerance_px,
        labels=len(labels),
        unscored_labels=unscored,
        label_coverage=(len(labels) - unscored) / len(labels) if labels else None,
        note="Only frames sampled by the detector are scored. Lower inference FPS reduces label coverage.",
    )


def event_metrics(predictions, truth, tolerance_seconds=1):
    if tolerance_seconds <= 0:
        raise ValueError("Event tolerance must be positive.")
    predictions = sorted(predictions, key=lambda p: p["time"])
    truth = sorted(truth, key=lambda p: p["time"])
    n, m = len(predictions), len(truth)
    if max(n, m) > 2000:
        raise ValueError("Evaluate at most 2000 events per match.")
    counts = np.zeros((n + 1, m + 1), dtype=np.int32)
    costs = np.zeros((n + 1, m + 1), dtype=np.float64)
    parent = np.zeros((n + 1, m + 1), dtype=np.uint8)
    # Order-preserving assignment maximizes matches, then minimizes timestamp error.
    # Greedy nearest-neighbor matching can undercount adjacent valid events.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [
                (int(counts[i - 1, j]), -costs[i - 1, j], 1),
                (int(counts[i, j - 1]), -costs[i, j - 1], 2),
            ]
            delta = abs(predictions[i - 1]["time"] - truth[j - 1]["time"])
            if delta <= tolerance_seconds:
                options.append((int(counts[i - 1, j - 1]) + 1, -costs[i - 1, j - 1] - delta, 3))
            count, negative_cost, action = max(options)
            counts[i, j] = count
            costs[i, j] = -negative_cost
            parent[i, j] = action
    matches = []
    i, j = n, m
    while i and j:
        action = parent[i, j]
        if action == 3:
            matches.append((abs(predictions[i - 1]["time"] - truth[j - 1]["time"]), i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == 1:
            i -= 1
        else:
            j -= 1
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(truth) - tp
    known = [(i, j) for _, i, j in matches if truth[j].get("winner") is not None]
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        mean_time_error_seconds=float(np.mean([d for d, _, _ in matches])) if matches else None,
        winner_accuracy=sum(predictions[i].get("winner") == truth[j]["winner"] for i, j in known)
        / len(known)
        if known
        else None,
        matched_winner_labels=len(known),
        tolerance_seconds=tolerance_seconds,
    )
