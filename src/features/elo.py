import pandas as pd

class EloCalculator:
    def __init__(self,
        matches: pd.DataFrame,
        starting_elo: float = 1500.0,
        k_factor: float = 20.0
    ):
        """
        Initializes the Elo Calculator.

        Args:
            matches (pd.DataFrame): Historical processed data. (containing dates, team IDs and final scores)
            starting_elo (float, optional): Rating assigned to a team before its first known match.
            k_factor (float): Power of each rating update.
        """

        self.matches = matches
        self.starting_elo = starting_elo
        self.k_factor = k_factor

        self.ratings = {}

    def get_elo(self, team_id: int) -> float:
        """
        Returns the team's current Elo rating.

        Args:
            team_id (int): ID of the team.

        Returns:
            float: Current rating. Unknown teams are assigned the starting rating.
        """

        if team_id not in self.ratings:
            self.ratings[team_id] = self.starting_elo

        return self.ratings[team_id]

    @staticmethod
    def calculate_chance(rating_a: float, rating_b: float) -> float:
        """
        Calculates the expected Elo score between two ratings.

        Args:
            rating_a (float): Rating of the first team.
            rating_b (float): Rating of the second team.

        Returns:
            float: Expected score for the first team, between 0 and 1.
        """

        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, home_team_id: int, away_team_id: int, home_goals: int, away_goals: int) -> None:

        """
        Updates both teams' Elo ratings after a finished match.

        Args:
            home_team_id (int): ID of the home team.
            away_team_id (int): ID of the away team.
            home_goals (int): Goals scored by the home team.
            away_goals (int): Goals scored by the away team.
        """

        home_rating = self.get_elo(home_team_id)
        away_rating = self.get_elo(away_team_id)

        home_chance = self.calculate_chance(home_rating, away_rating)
        away_chance = self.calculate_chance(away_rating, home_rating)

        if home_goals > away_goals:
            home_score = 1
            away_score = 0

        elif home_goals < away_goals:
            home_score = 0
            away_score = 1

        else:
            home_score = 0.5
            away_score = 0.5

        self.ratings[home_team_id] = (
                home_rating
                + self.k_factor
                * (home_score - home_chance)
        )

        self.ratings[away_team_id] = (
                away_rating
                + self.k_factor
                * (away_score - away_chance)
        )

    def add_features(self) -> pd.DataFrame:
        """
        Adds the home Elo, away Elo, and rating difference before each match.

        Matches with the same kickoff use ratings from before any of those matches.
        The calculator keeps the updated ratings after processing the matches.

        Returns:
            pd.DataFrame: A chronologically sorted copy with the Elo features.
        """

        df = self.matches.copy()
        df = df.sort_values("date").reset_index(drop=True)

        home_elos = []
        away_elos = []

        for _, fixtures_at_kickoff in df.groupby("date", sort=True):
            for _, match in fixtures_at_kickoff.iterrows():
                home_elos.append(self.get_elo(match["home_team_id"]))
                away_elos.append(self.get_elo(match["away_team_id"]))

            for _, match in fixtures_at_kickoff.iterrows():
                self.update_ratings(
                    match["home_team_id"],
                    match["away_team_id"],
                    match["home_goals"],
                    match["away_goals"],
                )

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        df["elo_diff"] = (df["home_elo"] - df["away_elo"])

        return df
