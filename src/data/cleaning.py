import pandas as pd

def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned matches DataFrame sorted chronologically,
    using only finished matches (status: FT).

    Returns a copy with UTC dates and a reset index.

    Args:
        df (pd.DataFrame): Raw matches DataFrame.

    Returns:
        pd.DataFrame: Cleaned matches DataFrame.
    """
    cleaned_df = df.copy()

    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], utc=True)
    cleaned_df = cleaned_df.loc[cleaned_df["status"] == "FT"]

    cleaned_df = cleaned_df.sort_values("date").reset_index(drop=True)

    return cleaned_df