"""Application, asset setup, headless analysis, and benchmark commands."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="tennisvision")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Open the local video workspace")
    serve.add_argument("--port", type=int, default=7862)
    models = commands.add_parser("models", help="Download versioned model assets")
    models.add_argument("--tracknet", action="store_true")
    run = commands.add_parser("analyze", help="Analyze a continuous fixed-camera clip")
    run.add_argument("video", type=Path)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate ball predictions against human labels"
    )
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--width", type=int, required=True)
    evaluate.add_argument("--height", type=int, required=True)
    evaluate.add_argument("--tolerance", type=float, default=12)
    events = commands.add_parser(
        "evaluate-events", help="Evaluate unedited rally hypotheses against independent labels"
    )
    events.add_argument("--predictions", type=Path, required=True)
    events.add_argument("--truth", type=Path, required=True)
    events.add_argument("--tolerance", type=float, default=1)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "tennisvision.server:create_app", factory=True, host="127.0.0.1", port=args.port
        )
    elif args.command == "models":
        from .assets import fetch_models

        fetch_models(args.tracknet)
    elif args.command == "analyze":
        from .models import AnalysisConfig
        from .pipeline import analyze
        from .video import inspect_video, normalize

        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Use a new or empty output directory to preserve previous runs.")
        args.output.mkdir(parents=True, exist_ok=True)
        source = args.output / "source.mp4"
        inspect_video(args.video)
        normalize(args.video, source)
        config = AnalysisConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        result = analyze(
            source,
            args.output,
            config,
            lambda value, message: print(f"{value:.0%} {message}", flush=True),
        )
        print(json.dumps(result["analysis"], indent=2))
    elif args.command == "evaluate":
        from .evaluate import ball_metrics

        print(
            json.dumps(
                ball_metrics(
                    args.predictions, args.labels, args.width, args.height, args.tolerance
                ),
                indent=2,
            )
        )
    else:
        from .evaluate import event_metrics

        predicted = json.loads(args.predictions.read_text(encoding="utf-8"))
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        print(json.dumps(event_metrics(predicted, truth, args.tolerance), indent=2))
