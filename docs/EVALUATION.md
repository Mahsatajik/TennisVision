# Evaluation protocol

## Splits and independence

Split by match and camera, not random frames. Adjacent clips from the same match belong in one split. Tune thresholds on development matches and evaluate untouched held-out matches. TrackNet's broadcast domain is different from amateur phone footage.

Use raw `predictions.json`, never corrected `events.json`, as benchmark predictions. To obtain independent labels, annotate before viewing model output or use a separate reviewer. Record agreement and ambiguous cases when making research claims.

## Ball labels

Vision lab exports this schema (the values below are examples, not ground truth):

```json
{"schema_version":1,"coordinate_system":"normalized_image","fps":30,"labels":[
  {"frame":8,"visible":true,"x":0.418,"y":0.679},
  {"frame":16,"visible":false,"x":null,"y":null}
]}
```

Actual workspace exports also include normalized width/height and the input SHA-256. The evaluator rejects mismatched dimensions and mismatched input hashes when the prediction manifest is available.

Coordinates originate at the top-left of the **normalized input**. Frames are zero-based. Hidden balls have `visible:false`; ambiguous human labels should be excluded and reported separately.

The browser selects `round(currentTime × fps)`. Native seeking is approximate; visually verify the frame. For publication-grade exact frame labels, use a decoded-frame annotation tool. The evaluator reports only labels present in the detector's sampling schedule, with unscored count and label coverage. Compare detectors on the same sampled frames. Use 30-fps inference for all-frame evaluation of normalized 30-fps clips.

A visible detection within tolerance is TP. A badly localized visible detection contributes one FP and one FN. A missed visible ball is FN; a detection on a labeled hidden frame is FP. Mean/median errors use all co-visible labels, including poor localizations. Empty-denominator metrics are null.

```bash
tennisvision evaluate --predictions outputs/run/ball.csv \
  --labels data/held-out-labels.json --width 1280 --height 720 --tolerance 12
```

Report resolution, tolerance, visibility distribution, label coverage, sampling FPS, and runtime alongside localization metrics. High detection coverage does not imply correct detections.

## Rally labels

Use a JSON array of independently labeled point-ending times and configured player IDs:

```json
[{"time":12.4,"winner":0},{"time":27.1,"winner":1}]
```

Choose a consistent end-time convention, such as the decisive bounce/fault, in normalized input seconds. Use `winner:null` for unresolved human labels. IDs 0/1 refer to players, not near/far court.

An order-preserving assignment maximizes matched events within a time tolerance, then minimizes total timestamp error. Each prediction and truth event can match once. Unmatched predictions are FP; unmatched truth events are FN. Winner accuracy uses matched events with known human winners and counts model abstentions as incorrect. Report it together with event recall so missed points are visible.

```bash
tennisvision evaluate-events --predictions outputs/run/predictions.json \
  --truth data/held-out-points.json --tolerance 1
```

## Experiments

| Comparison | Keep fixed | Measure |
| --- | --- | --- |
| Classical vs TrackNet | Clips, calibration, resolution, sample frames | Localization F1, error, coverage, runtime |
| 4 vs 8 vs 30 samples/s | Model and held-out matches | Event F1, missed contacts, runtime |
| Calibration perturbation | Observed player poses | Ground-trace error against an independent reference |
| Fusion thresholds | Tune only on development matches | Held-out event F1 and winner accuracy |

No independently labeled point-scoring benchmark is bundled in this release. Tests and smoke runs validate execution and contracts, not officiating performance.
