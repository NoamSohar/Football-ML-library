from sklearn.metrics import accuracy_score, confusion_matrix

from .thresholds import predict_results


def evaluate_probabilities(probabilities, results, draw_threshold):
    """Returns accuracy and the confusion matrix for the given predictions."""
    predictions = predict_results(probabilities, draw_threshold)
    return {
        "accuracy": float(accuracy_score(results, predictions)),
        "confusion_matrix": confusion_matrix(results, predictions, labels=[0, 1, 2]).tolist(),
    }


def evaluate_model(model, features, results, draw_threshold):
    """Evaluates the model using the selected draw threshold."""
    probabilities = model.predict_proba(features)
    return evaluate_probabilities(probabilities, results, draw_threshold)
