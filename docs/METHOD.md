# Vision method

## Scope

One continuous fixed-camera singles segment, full court visible, no change of ends, and a supplied starting score. This is an inspectable engineering baseline, not a line judge or validated autonomous scorer.

## Ground plane and poses

Four singles baseline corners map to an 8.23 × 23.77 m ground-plane rectangle using an OpenCV homography. This supports player-foot association and approximate ground traces. Transforming an airborne ball cannot reveal its height or landing position. The code uses the near/far sign as an image-space side cue and makes no in/out decisions.

MediaPipe Pose Landmarker Full predicts up to three poses in separate overlapping court crops. Cropping enlarges the far athlete relative to whole-frame detection. After a detection, a padded body region is searched first; the court crop is the fallback. The body-region cache resets on cuts and after repeated misses. Ankles require visibility ≥0.3, selected mean body visibility must exceed 0.35, and ground-plane/side gates reject many spectators. A pose may be held for up to 0.3 s for display and contact association; exports contain actual observations only.

Slots are `near` and `far`, not biometric identities. Calibration and ankle errors directly affect traces. Step distances above 12 m/s or across gaps ≥0.5 s are omitted. This removes some jumps; it does not establish physical measurement accuracy.

## Ball detection

The classical baseline intersects yellow-green HSV regions with frame differences, then filters components by size and compactness. It is deliberately a weak, fast baseline, particularly on distant or blurred balls.

TrackNet takes three consecutive 640 × 360 RGB frames ordered current, previous, previous-previous, normalized to [0,1]. Its 256-class output produces an intensity heatmap via class argmax. Intensity ≥128 components of 1–200 pixels become candidates. `quality` is a heatmap intensity index, **not a calibrated probability**. Broadcast pretraining does not establish amateur-camera performance.

Both detectors use a short constant-velocity association gate: at most 0.22 normalized image units from the predicted position while the last observation is younger than 0.35 s. A loose horizontal court gate rejects remote candidates. Long gaps remain missing. Trails last 0.55 s and do not connect observations separated by ≥0.3 s. TrackNet buffers every decoded frame even when inference is sampled at a lower FPS.

## Pose–trajectory fusion

A possible contact requires a temporal ball-depth reversal across three observations: opposite-sign successive y changes, each larger than 0.015. Near-court maxima and far-court minima are considered. The middle ball observation must be within 0.13 normalized units of a visible wrist/elbow. The three samples must span less than 0.85 s. Consecutive contacts alternate court sides and are separated by at least 0.35 s.

After at least two contact cues, a tracking loss of `lost_seconds` (default 1.8 s) may end the exchange. If the last flight crossed toward the opponent, the last hitter is proposed as a possible winner with evidence index 0.45. Otherwise the winner is unresolved. These constants are heuristic gates, not learned decision boundaries or probabilities.

Missing detections can mean occlusion, a fault, an out ball, or tracking failure. All model events therefore remain pending. Camera cuts reset state. End-of-file never awards a point. Serves/aces with fewer than two detected contacts are not scored automatically.

## Review and extensibility

`predictions.json` preserves model hypotheses; `events.json` contains human decisions. A separate scoring state machine receives timestamps and winners. Re-rendering reuses the pose/ball video rather than rerunning inference.

The separate detector, pose, fusion, geometry, scoring, and evaluation modules support measurable extensions: supervised contact/bounce models, identity association through changes of ends, domain adaptation, calibrated event confidence, and labeled court-line evaluation. No novel method or new model training is claimed in this release.
