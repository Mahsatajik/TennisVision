# Validation notes

## Automated checks

26 tests exercise standard/deuce scoring, set/match endings, seven- and ten-point tiebreaks, invalid initial states, review separation, calibration round trips, association gaps, contact/rally uncertainty, one-to-one event assignment, uploads, video normalization, range requests, labels, isolated analysis runs, and end-to-end video/ZIP export without model downloads. These tests validate software behavior, not officiating accuracy.

Hosted CI has not run: the publishing credential lacks workflow scope. A workflow template is included as `docs/ci-checks.yml` for installation by the repository owner.

Real neural execution is a separate smoke test. Model files and footage are excluded from Git and CI.

## Real-video smoke comparison

Tested locally on Windows with Python 3.12.14, MediaPipe 0.10.35, OpenCV 4.14.0, and PyTorch 2.15.0+cpu. Input: the attributed 7.13-second clip in [DATA.md](DATA.md), normalized to 1280 × 720, 30 fps, 214 frames. Both runs use `configs/sample.json` geometry, a target of four ball samples/s, and six pose samples/s. This produces 27 ball samples; the realized sampling rate is limited by the decoded-frame grid.

| Observation | Classical baseline | TrackNet |
| --- | ---: | ---: |
| Ball detections / sampled frames | 1 / 27 | 23 / 27 |
| Near / far pose observations | 40 / 42 | 40 / 41 |
| Pose–trajectory contact cues | 0 | 2 |
| Completed rally hypotheses | 0 | 0 |
| Vision-stage elapsed time, seconds | 14.28 | 152.84 |

Normalized input SHA-256 for both runs: `92e66e5818eab60710f2f4ab334e9892ef5fea6a9d98a1f4574365cc4aec06d4`. Timings exclude upload normalization and final scoreboard/audio encoding. CPU speed and neural pose observations can vary between runs.

These are **execution and coverage observations, not accuracy results**. There is no independent ball/contact/point ground truth for this clip. The segment remains in play near its end, so the system does not turn the file ending into an awarded point. The two detected contact cues and missing contacts need human inspection. No autonomous scoring accuracy can be inferred from this test.

## UI checks

The real clip was uploaded through the in-app browser. Court coordinates were edited through accessible numeric fields; TrackNet analysis completed and displayed the generated video and traces. A temporary manual point was added to test a 15–0 confirmed score, re-rendering, CSV download, and ZIP contents. The raw model prediction file remained unchanged. That test point was then removed and the zero-event result restored; it is not ground truth for the clip.

Frame navigation was checked at frame 8 (0.2667 s). A temporary coordinate label exported with frame index 8; the test label was removed afterwards. This checks annotation plumbing, not labeling accuracy. Output MP4 and ZIP files were downloaded through the browser and decoded/inspected locally.

Desktop (1440 px), mobile (375 px), and landscape (812 px) layouts were checked in the browser without horizontal overflow. A short-audio export regression checks that audio ending early does not truncate the video.

## Limits before stronger claims

Evaluate independently labeled fixed-camera amateur matches, including serves, faults, out balls, occlusions, cuts, and changes of ends. Report ball localization, event precision/recall, winner accuracy, abstention/missed-event rates, and runtime. Review accuracy and model accuracy must remain separate. The current release is an experimental vision-assisted scoring workspace.
