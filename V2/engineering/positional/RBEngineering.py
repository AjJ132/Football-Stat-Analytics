import json
from matplotlib import pyplot as plt
import pandas as pd
import re
import os
import seaborn as sns
from scipy import stats
import numpy as np

class RBEngineering:
    def __init__(self, seasons, data_dir, save_dir):
        """
        Constructor for the RBEngineering class
        """
        self.seasons = seasons
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.rb_features = {
            "name": "name",
            "major": "major",
            "height": "height",
            "weight": "weight",
            "hometown": "hometown",
            "position": "position",
            "gp": "games_played",
            "gs": "games_started",
            "rush_stats.attempts": "rushing_attempts",
            "rush_stats.yards": "rushing_yards",
            "rush_stats.touchdowns": "rushing_touchdowns",
            "rush_stats.longest": "longest_rush",
            "rush_stats.rush_attempt_yards_pct": "average_yards_per_rush",
            "rush_stats.yards_per_game_avg": "average_rushing_yards_per_game"
        }
        self.rb_features_for_prev_season = [
            'games_played',
            'games_started',
            'rushing_attempts',
            'rushing_yards',
            'rushing_touchdowns',
            'longest_rush',
            'average_yards_per_rush',
            'average_rushing_yards_per_game'
        ]

    def convert_height_to_inches(self, height):
        """
        Convert height from string format (e.g., "6'1\"") to total inches
        """
        if pd.isna(height) or height == '':
            return None
        feet, inches = map(int, re.findall(r'\d+', height))
        return feet * 12 + inches

    def convert_weight_to_numeric(self, weight):
        """
        Convert weight from string format (e.g., "190 lbs") to numeric
        """
        if pd.isna(weight) or weight == '':
            return None
        return int(re.findall(r'\d+', weight)[0])
    
    def combine_and_group_data(self):
        """
        Combine all RB data from different seasons into a single dataframe
        and group by RB and season.
        """
        # List to store all dataframes
        all_dfs = []

        # Read CSV files for each season
        for season in self.seasons:
            file_path = f"./data/temp/ksu_football_rb_{season}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_dfs.append(df)
            else:
                print(f"Warning: File not found for season {season}")

        # Combine all dataframes
        if not all_dfs:
            raise ValueError("No data found for any season")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Ensure 'name' and 'season' columns exist
        if 'name' not in combined_df.columns or 'season' not in combined_df.columns:
            raise ValueError("Required columns 'name' and 'season' not found in the data")

        # Group by RB name and season
        grouped_df = combined_df.groupby(['name', 'season']).first().reset_index()

        # Sort the dataframe by name
        grouped_df = grouped_df.sort_values(['name'])

        return grouped_df
    
    def generate_yoy_features(self, grouped_df):
        """
        Using all the rows we are going to create and add the previous years stats to the current year in the row.
        This will be used to indicate improvements in yoy performance.
        """
        # Make a copy of the dataframe
        yoy_df = grouped_df.copy()
    
        # Check if the required columns are present
        required_columns = ['name', 'season']
        for col in required_columns:
            if col not in yoy_df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
    
        # Sort the dataframe by name and season
        yoy_df = yoy_df.sort_values(['name', 'season'])
    
        # For each row in the dataframe, get the season, subtract 1, and get the previous year's stats
        for index, row in yoy_df.iterrows():
            # Get the current season
            current_season = row['season']
    
            # Get the previous season
            if int(current_season) == 2021:
                previous_season = 2019
            else:
                previous_season = int(current_season) - 1
    
            # Get the previous season's stats
            # Filter by player name and previous season
            previous_season_stats = yoy_df[(yoy_df['name'] == row['name']) & (yoy_df['season'] == previous_season)]
    
            if previous_season_stats.empty:
                continue
    
            # Get the previous season's stats
            previous_season_stats = previous_season_stats.iloc[0]
    
            # Add the previous season's stats to the current row
            for col in self.rb_features_for_prev_season:
                yoy_df.at[index, f"prev_{col}"] = previous_season_stats[col]
    
        # Filter the DataFrame to keep all current year's stats and only the previous season's stats specified in rb_features_for_prev_season
        current_year_columns = list(grouped_df.columns)
        prev_year_columns = [f"prev_{col}" for col in self.rb_features_for_prev_season]
        required_columns = current_year_columns + prev_year_columns
        yoy_df = yoy_df[required_columns]
    
        return yoy_df
        
    def engineer_touchdowns(self, rb_df):
        """
        Engineer a new feature for total touchdowns
        """
        rb_df['total_touchdowns'] = rb_df['rushing_touchdowns']
        rb_df['prev_total_touchdowns'] = rb_df['prev_rushing_touchdowns']
        return rb_df

    def engineer_rbs(self):
        """
        Method to engineer the RB data, including combining and grouping
        """
        all_dfs = []

        for season in self.seasons:
            with open(f"{self.data_dir}/{season}/ksu_football_roster_{season}.json", "r") as f:
                player_data = json.load(f)

            # Convert to a pandas dataframe
            player_df = pd.DataFrame(player_data)

            # Convert all columns to lowercase
            player_df.columns = player_df.columns.str.lower()

            # Convert all string values to lowercase
            player_df = player_df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

            # Filter for players with rushing stats
            player_df = player_df[player_df["rush_stats.attempts"].notna() & (player_df["rush_stats.attempts"] != 0)]

            # Select only the rb features that are available in the dataframe
            available_features = [col for col in self.rb_features.keys() if col in player_df.columns]
            player_df = player_df[available_features].rename(columns={col: self.rb_features[col] for col in available_features})

            # Convert height to inches
            if 'height' in player_df.columns:
                player_df['height'] = player_df['height'].apply(self.convert_height_to_inches)

            # Convert weight to numeric
            if 'weight' in player_df.columns:
                player_df['weight'] = player_df['weight'].apply(self.convert_weight_to_numeric)

            # Insert new season column at beginning of dataframe
            player_df.insert(0, "season", season)

            all_dfs.append(player_df)

            # Ensure temp folder exists
            if not os.path.exists("./data/temp"):
                os.makedirs("./data/temp")

            # Save the dataframe to a csv file in the temp folder
            player_df.to_csv(f"./data/temp/ksu_football_rb_{season}.csv", index=False)

        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Get the union of all columns
        all_columns = list(set(col for df in all_dfs for col in df.columns))

        # Ensure each dataframe has all columns
        for i, season in enumerate(self.seasons):
            all_dfs[i] = all_dfs[i].reindex(columns=all_columns, fill_value=pd.NA)

        # Call the combine_and_group_data method
        grouped_df = self.combine_and_group_data()

        # Call the generate_yoy_features method
        prepped_rb_df = self.generate_yoy_features(grouped_df)

        # Engineer touchdowns
        prepped_rb_df = self.engineer_touchdowns(prepped_rb_df)

        # Ensure save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Save the combined and grouped data to a new CSV file
        output_path = f"{self.save_dir}/prepped_rb_data.csv"
        
        prepped_rb_df.to_csv(output_path, index=False)

        print(f"Combined and grouped data saved to {output_path}")