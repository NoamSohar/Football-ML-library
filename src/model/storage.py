from pathlib import Path
import joblib


def save_model(
    model,
    feature_columns,
    training_data,
    draw_threshold,
    path,
    *,
    draw_weight=1.0,
    selection_metric="macro_f1",
    split_date_ranges=None,
):
    """
    Saves the trained model and its settings to a file.

    Creates the directory if needed and overwrites an existing file.

    Args:
        model (XGBClassifier): Trained model to save.
        feature_columns (iterable): Feature names in training order.
        training_data (pd.DataFrame): Data used to record the supplied date range.
        draw_threshold (float or None): Selected draw cutoff.
        path (str or Path): Output file path.
        draw_weight (float, optional): Draw weight used during training. Defaults to 1.0.
        selection_metric (str, optional): Threshold scoring rule. Defaults to "macro_f1".
        split_date_ranges (dict, optional): Dates for each data split. Defaults to an empty dictionary.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": list(feature_columns),
        "model_params": model.get_params(),
        "train_date_range": [str(training_data["date"].min()), str(training_data["date"].max())],
        "draw_threshold": draw_threshold,
        "draw_weight": draw_weight,
        "selection_metric": selection_metric,
        "split_date_ranges": split_date_ranges or {},
    }, path)


def load_model(path):
    """
    Loads a saved model and its settings.

    Args:
        path (str or Path): Path to the saved joblib file.

    Returns:
        dict: Model and saved settings.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """

    return joblib.load(path)
