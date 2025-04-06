import os
import pandas as pd
import joblib

def get_mid_season_team_stats(team_name: str, season: str, games_threshold: int = 50):
    """
    Loads the CSV file (season.csv) containing team stats and returns the stats
    for a given team and season. The CSV file should have a 'Season' column and 
    follow the structure of the LeagueDashTeamStats output.
    """
    try:
        csv_file = "season.csv"
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} not found.")
            return None
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Filter for the specified season if the 'Season' column exists
        if "Season" in df.columns:
            df = df[df["Season"] == season]
        
        # Filter for the specified team (case-insensitive)
        team_stats = df[df["TEAM_NAME"].str.lower() == team_name.lower()]
        if team_stats.empty:
            return None
        
        stats = team_stats.iloc[0].to_dict()
        if stats.get("GP", 0) < games_threshold:
            # Not enough games played for a mid-season snapshot
            return None
        return stats
    except Exception as e:
        print(f"Error fetching team stats from CSV: {e}")
        return None

def load_prediction_model(model_path: str = "model.pkl"):
    """
    Loads a pre-trained scikit-learn model from disk.
    The model should be trained to predict remaining wins based on mid-season features.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_remaining_wins(team_stats: dict, model) -> float:
    """
    Uses the pre-trained model to predict the number of wins in the remaining 32 games.
    Expected features might include points (PTS), rebounds (REB), assists (AST), and win percentage (W_PCT).
    """
    try:
        features = [
            team_stats.get("PTS", 0),
            team_stats.get("REB", 0),
            team_stats.get("AST", 0),
            team_stats.get("W_PCT", 0)
        ]
        # The model expects a 2D array
        prediction = model.predict([features])
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0.0

if __name__ == "__main__":
    # Example usage: Predict remaining wins for the Los Angeles Lakers in the 2024-25 season
    team = "Los Angeles Lakers"
    season = "2024-25"
    stats = get_mid_season_team_stats(team, season)
    if stats:
        model = load_prediction_model()
        if model:
            remaining_wins = predict_remaining_wins(stats, model)
            print(f"{team} (after {stats.get('GP', 0)} games): Predicted wins in remaining 32 games: {remaining_wins:.1f}")
        else:
            print("Prediction model not loaded.")
    else:
        print("Team stats not found or not enough games played.")
