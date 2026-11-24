# Contributing

Keep observations, event hypotheses, human labels, and scoring state separate. Missing detections stay missing; uncertain events never become confirmed automatically.

Use new output directories for experiments. Compare raw predictions on held-out match/camera splits. Include measured localization/event metrics with improvement claims; detection counts and corrected scores are not accuracy measurements.

Run `ruff check src tests`, `pytest -q`, and `python -m build`. Keep code, comments, docs, and UI in English. Do not commit footage, weights, job directories, credentials, or machine-specific paths. Preserve attribution.
