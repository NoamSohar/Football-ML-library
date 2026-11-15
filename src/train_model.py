import argparse
from pathlib import Path

from src.data.dataset import build_dataset
from src.model.data import load_training_data, split_data
from src.model.evaluation import evaluate_model
from src.model.storage import load_model, save_model
from src.model.train import run_training


def print_metrics(title, metrics):
    print(f"{title} metrics")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Confusion matrix: {metrics['confusion_matrix']}")


def train(args):
    result = run_training(
        args.dataset,
        args.train_ratio,
        args.calibration_ratio,
        objective=args.selection_metric,
    )
    save_model(
        result["model"],
        result["feature_columns"],
        result["data"],
        result["draw_threshold"],
        args.model_path,
        draw_weight=result["draw_weight"],
        selection_metric=result["selection_metric"],
        split_date_ranges=result["split_date_ranges"],
    )
    print("Selection objective:", result["selection_metric"])
    if result["selection_metric"] == "macro_f1_recall":
        print("Threshold rule: maximize macro F1 with at least 30% calibration draw recall.")
    print("Model parameters:", result["model_params"])
    print(f"Draw weight: {result['draw_weight']:.2f}")
    print(f"Calibrated draw threshold: {result['draw_threshold']:.2f}")
    print_metrics("Final model", result["test_metrics"])


def evaluate(args):
    artifact = load_model(args.model_path)
    data = load_training_data(args.dataset)
    _, _, _, _, x_test, y_test = split_data(
        data, args.train_ratio, args.calibration_ratio
    )
    metrics = evaluate_model(
        artifact["model"], x_test[artifact["feature_columns"]], y_test, artifact["draw_threshold"]
    )
    print_metrics("Final model", metrics)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train and evaluate the football outcome model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-dataset", help="Create processed and training datasets.")

    for name, help_text in [
        ("train", "Train, calibrate, test, and save a model."),
        ("evaluate", "Evaluate a saved model on the historical holdout period."),
    ]:
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument(
            "--dataset", type=Path, default=root / "data" / "dataset" / "training_dataset.csv"
        )
        command.add_argument(
            "--model-path", type=Path, default=root / "data" / "models" / ("football_candidate.joblib" if name == "train" else "football_model.joblib")
        )
        command.add_argument("--train-ratio", type=float, default=0.6)
        command.add_argument("--calibration-ratio", type=float, default=0.2)
        if name == "train":
            command.add_argument("--selection-metric", choices=["draw_f1", "macro_f1", "macro_f1_recall"], default="macro_f1_recall")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "build-dataset":
        build_dataset()
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
