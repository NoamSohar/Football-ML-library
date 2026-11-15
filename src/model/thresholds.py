import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def predict_results(probabilities, draw_threshold=None):
    """
    Converts outcome probabilities into predicted results.

    Args:
        probabilities (np.ndarray): Probabilities in draw, home win, and away win order.
        draw_threshold (float, optional): Draw cutoff. None uses the most likely outcome.

    Returns:
        np.ndarray: Predicted outcomes: 0 for draw, 1 for home win, 2 for away win.
    """

    if draw_threshold is None:
        return probabilities.argmax(axis=1)

    return np.where(
        probabilities[:, 0] >= draw_threshold,
        0,
        np.where(probabilities[:, 1] >= probabilities[:, 2], 1, 2),
    )


SELECTION_OBJECTIVES = ("draw_f1", "macro_f1", "macro_f1_recall")


def selection_score(results, predictions, objective="macro_f1"):
    """
    Calculates the score for the requested objective.

    Args:
        results (array-like): Actual match outcomes.
        predictions (array-like): Predicted match outcomes.
        objective (str, optional): "draw_f1", "macro_f1", or "macro_f1_recall". Defaults to "macro_f1".

    Returns:
        float: Draw F1 or macro F1. The recall constraint is applied during threshold search.

    Raises:
        ValueError: If the objective is unknown.
    """

    if objective == "draw_f1":
        return draw_class_metrics(results, predictions)["draw_f1"]
    if objective in ("macro_f1", "macro_f1_recall"):
        return macro_f1_score(results, predictions)
    raise ValueError(f"Unknown selection objective: {objective}")


def find_draw_threshold(probabilities, results, minimum=0.1, maximum=0.5, step=0.01,
                        objective="macro_f1_recall", minimum_draw_recall=0.3):
    """
    Finds the draw cutoff with the best score on calibration data.

    The default maximizes macro F1 while catching at least 30% of calibration draws.
If there are no actual draws, the recall constraint is skipped.

    Args:
        probabilities (np.ndarray): Probabilities in draw, home win, and away win order.
        results (array-like): Actual calibration outcomes.
        minimum (float, optional): Lowest cutoff to test. Defaults to 0.1.
        maximum (float, optional): Upper limit of the cutoff search. Defaults to 0.5.
        step (float, optional): Gap between tested cutoffs. Defaults to 0.01.
        objective (str, optional): Scoring rule. Defaults to "macro_f1_recall".
        minimum_draw_recall (float, optional): Required recall for "macro_f1_recall". Defaults to 0.3.

    Returns:
        float: Selected draw threshold. Score ties prefer the first eligible cutoff.

    Raises:
        ValueError: If settings are invalid or no cutoff meets the recall requirement.
    """

    if not 0 < minimum <= maximum <= 1 or step <= 0:
        raise ValueError("Thresholds must be between 0 and 1, with a positive step.")
    if objective not in SELECTION_OBJECTIVES:
        raise ValueError(f"Unknown selection objective: {objective}")
    if not 0 <= minimum_draw_recall <= 1:
        raise ValueError("minimum_draw_recall must be between 0 and 1.")
    probabilities, results = np.asarray(probabilities), np.asarray(results)
    thresholds = np.arange(minimum, maximum + step / 2, step)
    predictions = np.where(probabilities[None, :, 0] >= thresholds[:, None], 0,
                           np.where(probabilities[None, :, 1] >= probabilities[None, :, 2], 1, 2))
    f1s = []
    for label in [0, 1, 2]:
        tp = ((predictions == label) & (results == label)).sum(axis=1)
        denominator = (predictions == label).sum(axis=1) + (results == label).sum()
        f1s.append(np.divide(2 * tp, denominator, out=np.zeros(len(thresholds)), where=denominator != 0))
    macro = np.mean(f1s, axis=0)
    eligible = np.ones(len(thresholds), dtype=bool)
    if objective == "macro_f1_recall" and (results == 0).any():
        recall = ((predictions == 0) & (results == 0)).sum(axis=1) / (results == 0).sum()
        eligible = recall >= minimum_draw_recall
        if not eligible.any():
            raise ValueError("No threshold satisfies minimum draw recall; widen the search range.")
    primary = f1s[0] if objective == "draw_f1" else macro
    # Preserve the original first-threshold tie behavior for macro F1.
    return float(max(np.flatnonzero(eligible), key=lambda i: (primary[i], macro[i], -i)) * step + minimum)


def draw_class_metrics(results, predictions):
    """
    Calculates precision, recall, and F1 for draws.

    Args:
        results (array-like): Actual outcomes. Draws are labeled 0.
        predictions (array-like): Predicted outcomes.

    Returns:
        dict: Draw precision, recall, and F1. Undefined scores are set to 0.
    """

    precision, recall, f1, _ = precision_recall_fscore_support(
        results, predictions, labels=[0], zero_division=0
    )
    return {
        "draw_precision": float(precision[0]),
        "draw_recall": float(recall[0]),
        "draw_f1": float(f1[0]),
    }


def macro_f1_score(results, predictions):
    """
    Calculates the average F1 score across the three outcomes.

    Args:
        results (array-like): Actual match outcomes.
        predictions (array-like): Predicted match outcomes.

    Returns:
        float: Equally weighted F1 for draw, home win, and away win. Undefined scores are set to 0.
    """

    return f1_score(
        results, predictions, labels=[0, 1, 2], average="macro", zero_division=0
    )
