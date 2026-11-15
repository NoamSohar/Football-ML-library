from sklearn.metrics import accuracy_score, confusion_matrix

from .thresholds import predict_results


def evaluate_probabilities(probabilities, results, draw_threshold):
    """
    Evaluates the given probabilities.

    Args:
        probabilities (np.ndarray): Probabilities in draw, home win, and away win order.
        results (array-like): Actual outcomes: 0 for draw, 1 for home win, 2 for away win.
        draw_threshold (float or None): Draw cutoff. None uses the most likely outcome.

    Returns:
        dict: Accuracy and a confusion matrix, ordered as draw, home win, and away win.
    """
    predictions = predict_results(probabilities, draw_threshold)
    return {
        "accuracy": float(accuracy_score(results, predictions)),
        "confusion_matrix": confusion_matrix(results, predictions, labels=[0, 1, 2]).tolist(),
    }


def evaluate_model(model, features, results, draw_threshold):
    """
    Evaluates the model using the selected draw threshold.

    Args:
        model (XGBClassifier): Trained model.
        features (pd.DataFrame): Match features to evaluate.
        results (array-like): Actual outcomes: 0 for draw, 1 for home win, 2 for away win.
        draw_threshold (float or None): Draw cutoff. None uses the most likely outcome.

    Returns:
        dict: Accuracy and a confusion matrix, ordered as draw, home win, and away win.
    """
    probabilities = model.predict_proba(features)
    return evaluate_probabilities(probabilities, results, draw_threshold)
