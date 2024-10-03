import json
import pandas as pd
import os

class WREngineering:
    def __init__(self, seasons, data_dir, save_dir, base_features):
        self.seasons = seasons
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.base_features = base_features
        self.receivers_features = [
            "receiving_stats.longest",
            "receiving_stats.per_game",
            "receiving_stats.recep",
            "receiving_stats.touchdowns",
            "receiving_stats.yards",
            "receiving_stats.yards_comp_pct",
            "receiving_stats.yards_game_pct",
            "scoring_stats.point_avg",
            "scoring_stats.points",
            "scoring_stats.rcpt_td",
            "scoring_stats.rush_td",
            "scoring_stats.touchdowns"
        ]
        self.all_features = self.base_features + self.receivers_features

    def engineer_wrs(self):
        for season in self.seasons:
            roster_file = f"{self.data_dir}/{season}/ksu_football_roster_{season}.json"
            
            try:
                with open(roster_file, "r") as f:
                    roster = json.load(f)
            except FileNotFoundError:
                print(f"Roster file for season {season} not found: {roster_file}")
                continue
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {roster_file}")
                continue

            if not roster:
                print(f"Roster for season {season} is empty.")
                continue

            players_data = []

            for player in roster:
                if any(key.startswith("receiving_stats") for key in player.keys()):
                    player_data = {feature: player.get(feature) for feature in self.all_features if feature in player}
                    player_data['is_primary_receiver'] = player['position'] == 'WR'
                    players_data.append(player_data)

            players_df = pd.DataFrame(players_data)

            save_path = f"{self.save_dir}/{season}"
            os.makedirs(save_path, exist_ok=True)
            players_df.to_csv(f"{save_path}/wrs.csv", index=False)

            print(f"Saved WR data for season {season}")

        print("WR engineering completed for all seasons.")