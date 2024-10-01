import json
import os
from typing import Dict, List

class TeamInfoProcessor:
    def __init__(self, team_schedule_path: str, output_dir: str):
        self.team_schedule_path = team_schedule_path
        self.output_dir = output_dir
        self.team_schedule: Dict[str, List[Dict]] = {}
        self.sos_data: Dict[str, float] = {}

    def load_team_schedule(self):
        with open(self.team_schedule_path, 'r') as f:
            self.team_schedule = json.load(f)

    def calculate_sos(self):
        for year, games in self.team_schedule.items():
            total_opponent_win_percentage = 0
            total_scoring_differential = 0
            home_games = 0
            away_games = 0

            for game in games:
                # Opponent win percentage (simplified as we don't have full opponent records)
                if game['Result'] == 'Win':
                    total_opponent_win_percentage += 0
                else:
                    total_opponent_win_percentage += 1

                # Scoring differential
                total_scoring_differential += game['Opponent-Score'] - game['KSU-Score']

                # Home/Away factor
                if game['Location'] == 'Kennesaw, GA':
                    home_games += 1
                else:
                    away_games += 1

            num_games = len(games)
            avg_opponent_win_percentage = total_opponent_win_percentage / num_games
            avg_scoring_differential = total_scoring_differential / num_games
            home_away_factor = away_games / num_games  # Higher factor for more away games

            # Normalize factors
            normalized_win_percentage = avg_opponent_win_percentage  # Already between 0 and 1
            normalized_scoring_diff = min(max((avg_scoring_differential + 30) / 60, 0), 1)  # Assume max diff of ±30

            # Calculate SOS
            sos = (normalized_win_percentage * 0.5) + (normalized_scoring_diff * 0.3) + (home_away_factor * 0.2)

            self.sos_data[year] = round(sos, 3)

    def save_sos_data(self):
        output_path = os.path.join(self.output_dir, 'strength_of_schedule.json')
        with open(output_path, 'w') as f:
            json.dump(self.sos_data, f, indent=2)

    def process(self):
        print("Loading team schedule...")
        self.load_team_schedule()

        print("Calculating strength of schedule...")
        self.calculate_sos()

        print("Saving strength of schedule data...")
        self.save_sos_data()

        print("Team info processing completed.")

