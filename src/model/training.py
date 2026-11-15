import numpy as np

from xgboost import XGBClassifier


MODEL_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
}
DRAW_WEIGHT = 1.0


def train_model(
    x_train,
    y_train,
    model_params=None,
    draw_weight=DRAW_WEIGHT,
):
    if draw_weight <= 0:
        raise ValueError("draw_weight must be positive.")

    model = XGBClassifier(**(MODEL_PARAMS if model_params is None else model_params))
    sample_weight = np.where(y_train == 0, draw_weight, 1.0)
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
    )
    return model
