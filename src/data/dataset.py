from pathlib import Path
import numpy as np

from src.data.processing import (
    RAW_DATA_DIR,
    process_matches,
    save_processed_data,
)
from src.features.Goals import GoalCalculator
from src.features.elo import EloCalculator
from src.features.form import FormCalculator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DATA_DIR = PROJECT_ROOT / "data" / "dataset"

COLUMNS_TO_DROP = [
    "fixture_id",
    "round",
    "home_goals",
    "away_goals",
    "home_team",
    "away_team",
    "status",
]

def build_dataset(
        raw_data_dir: Path = RAW_DATA_DIR,
        dataset_data_dir: Path = DATASET_DATA_DIR
) -> Path:
    """
    Builds and saves a chronological training dataset.

    Loads finished matches, adds pre-match features, creates the match-outcome
    target, removes post-match data that would leak the result, drops incomplete
    feature rows, and saves the training dataset. Also saves processed match data.
    Target values are 0 for a draw, 1 for a home win, and 2 for an away win.

    Args:
        raw_data_dir (Path, optional): Directory containing raw match CSV files.
            Defaults to data/raw.
        dataset_data_dir (Path, optional): Output directory for the training dataset.
            Defaults to data/dataset.

    Returns:
        The path of the saved training-dataset CSV file.
    """

    dataset_data_dir.mkdir(parents=True, exist_ok=True)

    processed_matches = process_matches(raw_data_dir)
    save_processed_data(processed_matches)

    df = EloCalculator(processed_matches).add_features()
    df = FormCalculator(df).add_features()
    df = GoalCalculator(df).add_features()

    # Adds the target result column using numpy logic.
    df["result"] = np.select(
        [
            df["home_goals"] > df["away_goals"],
            df["home_goals"] < df["away_goals"],
        ],
        [1, 2],
        default=0,
    )

    df = df.drop(columns=COLUMNS_TO_DROP)

    matches_before_dropna = len(df)
    df = df.dropna()
    matches_after_dropna = len(df)

    print(f"Matches before dropping incomplete feature rows: {matches_before_dropna}")
    print(f"Matches after dropping incomplete feature rows: {matches_after_dropna}")

    output_path = dataset_data_dir / "training_dataset.csv"
    df.to_csv(output_path, index=False)

    return output_path

