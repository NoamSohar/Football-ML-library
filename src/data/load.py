from pathlib import Path
import pandas as pd

# Path used to locate the project's raw match-data files.
project_root = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = project_root / "data" / "raw"

def load_season(file_name: str, raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Loads raw match data for one season from a CSV file.

    Args:
        file_name (str): Name of the CSV file to read from.
        raw_data_dir (Path, optional): Directory containing raw data. Defaults to data/raw.

    Returns:
        pd.DataFrame: Raw match data.

    Raises:
        FileNotFoundError: If the requested CSV file does not exist.
    """

    path = raw_data_dir / file_name

    if not path.is_file():
        raise FileNotFoundError(f"Match file not found: {path}")

    return pd.read_csv(path)

def load_all_seasons(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Loads and combines every season's raw data into a single DataFrame.

    Args:
        raw_data_dir (Path, optional): Directory containing raw data. Defaults to data/raw.

    Returns:
        pd.DataFrame: Combined raw match data from all seasons.

    Raises:
        FileNotFoundError: If the directory is missing or contains no CSV files.
    """

    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw-data directory not found: {raw_data_dir}")

    file_paths = sorted(raw_data_dir.glob("*.csv"))

    if not file_paths:
        raise FileNotFoundError("No match files found in data/raw")

    data_frames = [load_season(file_path.name, raw_data_dir) for file_path in file_paths]

    return pd.concat(data_frames, ignore_index=True)

