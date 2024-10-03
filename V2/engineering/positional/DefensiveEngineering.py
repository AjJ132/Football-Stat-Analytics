import json
import pandas as pd
import re
import os

class DefensiveEngineering:
    def __init__(self, seasons, data_dir, save_dir):
        self.seasons = seasons
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.defense_features = {
            "name": "name",
            "major": "major",
            "height": "height",
            "weight": "weight",
            "hometown": "hometown",
            "position": "position",
            "gp": "games_played",
            "gs": "games_started",
            "defense_stats.solo": "solo_tackles",
            "defense_stats.assist": "assisted_tackles",
            "defense_stats.total": "total_tackles",
            "defense_stats.tfl_yards": "tackles_for_loss_yards",
            "defense_stats.sacks_yards": "sacks_yards",
            "defense_stats.interceptions": "interceptions",
            "defense_stats.pass_defl": "pass_deflections",
            "defense_stats.forced_fumble": "forced_fumbles",
            "defense_stats.fumb_rec": "fumble_recoveries",
            "defense_stats.blocked": "blocked_kicks"
        }
        self.defense_features_for_prev_season = [
            'games_played',
            'games_started',
            'solo_tackles',
            'assisted_tackles',
            'total_tackles',
            'tackles_for_loss',
            'tackles_for_loss_yards',
            'sacks',
            'sacks_yards',
            'interceptions',
            'pass_deflections',
            'forced_fumbles',
            'fumble_recoveries',
            'blocked_kicks'
        ]

    def convert_height_to_inches(self, height):
        if pd.isna(height) or height == '':
            return None
        feet, inches = map(int, re.findall(r'\d+', height))
        return feet * 12 + inches

    def convert_weight_to_numeric(self, weight):
        if pd.isna(weight) or weight == '':
            return None
        return int(re.findall(r'\d+', weight)[0])
    
    def combine_and_group_data(self):
        all_dfs = []
        for season in self.seasons:
            file_path = f"./data/temp/ksu_football_def_{season}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_dfs.append(df)
            else:
                print(f"Warning: File not found for season {season}")

        if not all_dfs:
            raise ValueError("No data found for any season")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)

        if 'name' not in combined_df.columns or 'season' not in combined_df.columns:
            raise ValueError("Required columns 'name' and 'season' not found in the data")

        grouped_df = combined_df.groupby(['name', 'season']).first().reset_index()
        grouped_df = grouped_df.sort_values(['name'])

        return grouped_df
    
    def split_hyphenated_stat(self, value):
        if pd.isna(value) or value == '':
            return 0, 0
        parts = value.split('-')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        else:
            return float(value), 0  # If it's not hyphenated, assume it's just the count

    def generate_yoy_features(self, grouped_df):
        yoy_df = grouped_df.copy()
    
        required_columns = ['name', 'season']
        for col in required_columns:
            if col not in yoy_df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
    
        yoy_df = yoy_df.sort_values(['name', 'season'])
    
        for index, row in yoy_df.iterrows():
            current_season = row['season']
    
            if int(current_season) == 2021:
                previous_season = 2019
            else:
                previous_season = int(current_season) - 1
    
            previous_season_stats = yoy_df[(yoy_df['name'] == row['name']) & (yoy_df['season'] == previous_season)]
    
            if previous_season_stats.empty:
                continue
    
            previous_season_stats = previous_season_stats.iloc[0]
    
            for col in self.defense_features_for_prev_season:
                yoy_df.at[index, f"prev_{col}"] = previous_season_stats[col]
    
        current_year_columns = list(grouped_df.columns)
        prev_year_columns = [f"prev_{col}" for col in self.defense_features_for_prev_season]
        required_columns = current_year_columns + prev_year_columns
        yoy_df = yoy_df[required_columns]
    
        return yoy_df

    def engineer_defensive_players(self, min_games=0):
        all_dfs = []

        for season in self.seasons:
            with open(f"{self.data_dir}/{season}/ksu_football_roster_{season}.json", "r") as f:
                player_data = json.load(f)

            player_df = pd.DataFrame(player_data)
            player_df.columns = player_df.columns.str.lower()
            player_df = player_df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
            
            player_df = player_df[player_df["defense_stats.total"].notna() & (player_df["defense_stats.total"] != 0)]

            available_features = [col for col in self.defense_features.keys() if col in player_df.columns]
            player_df = player_df[available_features].rename(columns={col: self.defense_features[col] for col in available_features})

            if 'height' in player_df.columns:
                player_df['height'] = player_df['height'].apply(self.convert_height_to_inches)

            if 'weight' in player_df.columns:
                player_df['weight'] = player_df['weight'].apply(self.convert_weight_to_numeric)

            # Split hyphenated stats
            player_df['tackles_for_loss'], player_df['tackles_for_loss_yards'] = zip(*player_df['tackles_for_loss_yards'].apply(self.split_hyphenated_stat))
            player_df['sacks'], player_df['sacks_yards'] = zip(*player_df['sacks_yards'].apply(self.split_hyphenated_stat))

            player_df.insert(0, "season", season)

            player_df['games_played'] = pd.to_numeric(player_df['games_played'], errors='coerce')

            player_df = player_df[player_df["games_played"].notna() & (player_df["games_played"] >= min_games)]

            all_dfs.append(player_df)

            if not os.path.exists("./data/temp"):
                os.makedirs("./data/temp")

            player_df.to_csv(f"./data/temp/ksu_football_def_{season}.csv", index=False)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        all_columns = list(set(col for df in all_dfs for col in df.columns))

        for i, season in enumerate(self.seasons):
            all_dfs[i] = all_dfs[i].reindex(columns=all_columns, fill_value=pd.NA)

        grouped_df = self.combine_and_group_data()
        prepped_def_df = self.generate_yoy_features(grouped_df)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        output_path = f"{self.save_dir}/prepped_defense_data.csv"
        prepped_def_df.to_csv(output_path, index=False)

        print(f"Combined and grouped data saved to {output_path}")