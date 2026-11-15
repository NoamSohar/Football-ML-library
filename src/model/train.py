from .data import load_training_data, split_data
from .evaluation import evaluate_model
from .thresholds import find_draw_threshold
from .training import DRAW_WEIGHT, MODEL_PARAMS, train_model


def run_training(
    dataset_path,
    train_ratio=0.6,
    calibration_ratio=0.2,
    objective="macro_f1_recall",
):
    """Loads data, trains with fixed settings, calibrates, and evaluates the model."""
    data = load_training_data(dataset_path)
    x_train, y_train, x_calibration, y_calibration, x_test, y_test = split_data(
        data, train_ratio, calibration_ratio
    )

    model = train_model(x_train, y_train)

    # Choose the threshold on calibration matches, then score the holdout.
    probabilities = model.predict_proba(x_calibration)
    threshold = find_draw_threshold(probabilities, y_calibration, objective=objective)
    metrics = evaluate_model(model, x_test, y_test, threshold)

    # Record which dates belong to each period.
    split_date_ranges = {}
    for name, results in [("train", y_train), ("calibration", y_calibration), ("test", y_test)]:
        dates = data.loc[results.index, "date"]
        split_date_ranges[name] = [str(dates.min()), str(dates.max())]

    return {
        "model": model,
        "data": data,
        "feature_columns": list(x_train.columns),
        "model_params": MODEL_PARAMS.copy(),
        "draw_weight": DRAW_WEIGHT,
        "draw_threshold": threshold,
        "selection_metric": objective,
        "test_metrics": metrics,
        "split_date_ranges": split_date_ranges,
    }
