import os
import re
import openai
from nba_stats import get_mid_season_team_stats, load_prediction_model, predict_remaining_wins

openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your OpenAI API key is set as an environment variable

def extract_season(user_query: str) -> str:
    """
    Attempts to extract a season string from the user query.
    
    1. First, look for a full season pattern, e.g., "2024-25" or "2024–25".
    2. If not found, look for a 4-digit year (e.g., "2025") and assume season is "2025-26".
    3. If not found, look for a 2-digit number (e.g., "22") and assume season is "2022-23".
    4. Otherwise, default to "2024-25".
    """
    # 1. Full season pattern
    full_season_match = re.search(r'(\d{4}\s*[-–]\s*\d{2,4})', user_query)
    if full_season_match:
        return full_season_match.group(1).replace(" ", "")
    
    # 2. 4-digit year
    year_match = re.search(r'\b(\d{4})\b', user_query)
    if year_match:
        year = int(year_match.group(1))
        return f"{year}-{str(year+1)[-2:]}"
    
    # 3. 2-digit year (e.g., "22")
    two_digit_match = re.search(r'\b(\d{2})\b', user_query)
    if two_digit_match:
        year = int(two_digit_match.group(1))
        # Assumes 20XX, e.g., "22" becomes "2022-23"
        return f"20{year}-{year+1:02d}"
    
    # 4. Default season
    return "2024-25"

def parse_team_query(user_query: str):
    """
    Extracts the team name and season from the user query.
    Uses a hard-coded list of teams for matching and the extract_season helper.
    """
    teams = [
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers",
        "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors", "Houston Rockets",
        "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
        "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder",
        "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",
        "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"
    ]
    team_name = None
    for team in teams:
        if team.lower() in user_query.lower():
            team_name = team
            break
    if not team_name:
        team_name = "Los Angeles Lakers"  # default team

    season = extract_season(user_query)
    return team_name, season

def generate_response(user_query: str) -> str:
    """
    Processes the user query by:
      - Parsing the team and season.
      - Fetching mid-season stats from the CSV via nba_stats.
      - Loading the pre-trained prediction model.
      - Predicting the wins in the remaining 32 games.
      - Constructing a prompt for the OpenAI API.
    """
    team_name, season = parse_team_query(user_query)
    team_stats = get_mid_season_team_stats(team_name, season)
    if not team_stats:
        data_snippet = f"Could not fetch mid-season stats for {team_name} in the {season} season (or not enough games played)."
    else:
        model = load_prediction_model()
        if model:
            predicted_wins = predict_remaining_wins(team_stats, model)
            data_snippet = (
                f"For the {season} season, {team_name} have played {team_stats.get('GP', 0)} games with a win percentage of {team_stats.get('W_PCT', 0):.3f}. "
                f"Predicted wins in the remaining 32 games: {predicted_wins:.1f}. "
                f"Key stats: PTS: {team_stats.get('PTS', 0):.1f}, REB: {team_stats.get('REB', 0):.1f}, AST: {team_stats.get('AST', 0):.1f}."
            )
        else:
            data_snippet = "Prediction model could not be loaded."
    
    system_prompt = (
        "You are an NBA performance prediction assistant. Based on the team stats below, generate a concise prediction for the team's performance "
        "in the remaining 32 games of the season. Use the following acronyms: GP (Games Played), W_PCT (Win Percentage), PTS (Points), REB (Rebounds), AST (Assists).\n"
        f"Team stats:\n{data_snippet}\n\n"
        "User query:"
    )
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        max_tokens=200,
        temperature=0.3
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test with ambiguous queries
    queries = [
        "22 atlanta hawks season prediction",
        "2025 boston celtics season prediction"
    ]
    for query in queries:
        print(f"Query: {query}")
        print(generate_response(query))
        print("="*50)
