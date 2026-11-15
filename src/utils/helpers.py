from pathlib import Path
import pandas as pd

def delete_processed_data():
    """
    Deletes the processed CSV file if it exists.

    Used for debugging and testing. Removes data/processed/processed_data.csv.
    """
    project_root = Path(__file__).resolve().parents[2]
    processed_data_path = project_root / "data" / "processed" / "processed_data.csv"

    if processed_data_path.exists():
        processed_data_path.unlink()

def get_start_of_season(match_date: pd.Timestamp, matches: pd.DataFrame) -> pd.Timestamp:
    """
    Finds the first opening-round kickoff in the latest available season.

    Args:
        match_date (pd.Timestamp): Latest date to consider.
        matches (pd.DataFrame): Match data sorted by date, including round names.

    Returns:
        pd.Timestamp: Earliest "Regular Season - 1" kickoff in the latest opening-round year.

    Raises:
        IndexError: If no opening-round match exists on or before match_date.
    """

    previous_matches = matches[matches["date"] <= match_date]

    round1_matches = previous_matches[previous_matches["round"] == "Regular Season - 1"]
    season_year = round1_matches.iloc[-1]["date"].year

    round1_matches_seasonal = round1_matches[round1_matches["date"].dt.year == season_year]

    return round1_matches_seasonal.iloc[0]["date"]


