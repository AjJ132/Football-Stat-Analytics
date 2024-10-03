import json
from matplotlib import pyplot as plt
import pandas as pd
import re
import os
import seaborn as sns
from scipy import stats
import numpy as np

class QuarterbackEngineering:
    def __init__(self, seasons, data_dir, save_dir):
        """
        Constructor for the QuarterbackEngineering class
        """
        self.seasons = seasons
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.qb_features = {
            "name": "name",
            "major": "major",
            "height": "height",
            "weight": "weight",
            "hometown": "hometown",
            "position": "position",
            "gp": "games_played",
            "gs": "games_started",
            "pass_stats.pass_yards": "passing_yards",
            "pass_stats.completions": "pass_completions",
            "pass_stats.pass_attempts": "pass_attempts",
            "pass_stats.interceptions": "interceptions_thrown",
            "pass_stats.pass_td": "passing_touchdowns",
            "pass_stats.long_comp": "longest_completion",
            "pass_stats.pass_pct": "pass_completion_percentage",
            "pass_stats.pass_yrd_avg": "average_yards_per_pass",
            "pass_stats.average_yards_game": "average_passing_yards_per_game",
            "rush_stats.attempts": "rushing_attempts",
            "rush_stats.yards": "rushing_yards",
            "rush_stats.touchdowns": "rushing_touchdowns",
            "rush_stats.longest": "longest_rush",
            "rush_stats.rush_attempt_yards_pct": "average_yards_per_rush",
            "rush_stats.yards_per_game_avg": "average_rushing_yards_per_game"
        }
        self.qb_features_for_prev_season = [
            'games_played',
            'games_started',
            'passing_yards',
            'pass_completions',
            'pass_attempts',
            'interceptions_thrown',
            'passing_touchdowns',
            'longest_completion',
            'pass_completion_percentage',
            'average_yards_per_pass',
            'average_passing_yards_per_game',
            'rushing_attempts',
            'rushing_yards',
            'rushing_touchdowns',
            'longest_rush',
            'average_yards_per_rush',
            'average_rushing_yards_per_game'
        ]

    def calculate_passer_rating(self, comp, att, yards, td, int):
        try:
            # Convert inputs to float, replacing NaN or None with 0
            comp = float(comp) if pd.notnull(comp) else 0
            att = float(att) if pd.notnull(att) else 0
            yards = float(yards) if pd.notnull(yards) else 0
            td = float(td) if pd.notnull(td) else 0
            int = float(int) if pd.notnull(int) else 0

            # If attempts are 0 or all stats are 0, return NaN
            if att == 0 or (comp == 0 and yards == 0 and td == 0 and int == 0):
                return np.nan

            a = ((comp/att) - 0.3) * 5
            b = ((yards/att) - 3) * 0.25
            c = (td/att) * 20
            d = 2.375 - ((int/att) * 25)
            
            return ((max(min(a, 2.375), 0) + max(min(b, 2.375), 0) + max(min(c, 2.375), 0) + max(min(d, 2.375), 0)) / 6) * 100
        except (ValueError, TypeError):
            return np.nan

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
    
    def generate_advanced_stats_for_season(self, season, qb_df):
        # Ensure advanced stats folder exists
        if not os.path.exists(f"./data/stats/{season}"):
            os.makedirs(f"./data/stats/{season}")
    
        # Calculate passer rating
        qb_df['passer_rating'] = qb_df.apply(lambda row: self.calculate_passer_rating(
            row.get('pass_completions'), row.get('pass_attempts'), 
            row.get('passing_yards'), row.get('passing_touchdowns'), 
            row.get('interceptions_thrown')), axis=1)
    
        # Calculate Adjusted Yards per Attempt
        qb_df['ay/a'] = qb_df.apply(lambda row: 
            (float(row.get('passing_yards', 0)) + 20 * float(row.get('passing_touchdowns', 0)) - 45 * float(row.get('interceptions_thrown', 0))) / 
            float(row.get('pass_attempts', 1)) if pd.notnull(row.get('pass_attempts')) and float(row.get('pass_attempts', 0)) != 0 else np.nan, axis=1)
    
        # Select only numeric columns for correlation and boxplot
        numeric_columns = qb_df.select_dtypes(include=[np.number]).columns
    
        # Ensure 'interceptions_thrown' is included in numeric columns
        if 'interceptions_thrown' not in numeric_columns:
            numeric_columns = numeric_columns.append(pd.Index(['interceptions_thrown']))
    
        if len(numeric_columns) > 0:
            # Generate heatmap of correlations
            plt.figure(figsize=(12, 10))
            sns.heatmap(qb_df[numeric_columns].corr(), annot=True, cmap='coolwarm')
            plt.title(f"Correlation Heatmap of QB Stats for {season}")
            plt.savefig(f"./data/stats/{season}/correlation_heatmap.png")
            plt.close()
    
            # Generate box plots
            plt.figure(figsize=(12, 6))
            qb_df[numeric_columns].boxplot()
            plt.title(f"Distribution of Key QB Stats for {season}")
            plt.savefig(f"./data/stats/{season}/key_stats_boxplot.png")
            plt.close()
        else:
            print(f"Warning: No numeric columns found for season {season}")

    def combine_and_group_data(self):
        """
        Combine all quarterback data from different seasons into a single dataframe
        and group by quarterback and season.
        """
        # List to store all dataframes
        all_dfs = []

        # Read CSV files for each season
        for season in self.seasons:
            file_path = f"./data/temp/ksu_football_qb_{season}.csv"
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

        # Group by quarterback name and season
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
            for col in self.qb_features_for_prev_season:
                yoy_df.at[index, f"prev_{col}"] = previous_season_stats[col]
    
        # Filter the DataFrame to keep all current year's stats and only the previous season's stats specified in qb_features_for_prev_season
        current_year_columns = list(grouped_df.columns)
        prev_year_columns = [f"prev_{col}" for col in self.qb_features_for_prev_season]
        required_columns = current_year_columns + prev_year_columns
        yoy_df = yoy_df[required_columns]

         # TEMP: save to file
        # with open('temp.csv', 'w') as f:
        #     yoy_df.to_csv(f, index=False)

        # exit()
    
        return yoy_df
        
    def engineer_touchdowns(self, qb_df):
        """
        Engineer a new feature for total touchdowns
        """
        qb_df['total_touchdowns'] = qb_df['passing_touchdowns'] + qb_df['rushing_touchdowns']
        qb_df['prev_total_touchdowns'] = qb_df['prev_passing_touchdowns'] + qb_df['prev_rushing_touchdowns']
        return qb_df

    def engineer_quarterbacks(self):
        """
        Method to engineer the quarterback data, including combining and grouping
        """
        all_dfs = []

        for season in self.seasons:
            with open(f"{self.data_dir}/{season}/ksu_football_roster_{season}.json", "r") as f:
                qb_data = json.load(f)

            # Convert to a pandas dataframe
            qb_df = pd.DataFrame(qb_data)

            # Convert all columns to lowercase
            qb_df.columns = qb_df.columns.str.lower()

            # Convert all string values to lowercase
            qb_df = qb_df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

            # Filter for quarterbacks where position is qb
            qb_df = qb_df[qb_df["position"] == "qb"]

            # Select only the QB features that are available in the dataframe
            available_features = [col for col in self.qb_features.keys() if col in qb_df.columns]
            qb_df = qb_df[available_features].rename(columns={col: self.qb_features[col] for col in available_features})

            # Convert height to inches
            if 'height' in qb_df.columns:
                qb_df['height'] = qb_df['height'].apply(self.convert_height_to_inches)

            # Convert weight to numeric
            if 'weight' in qb_df.columns:
                qb_df['weight'] = qb_df['weight'].apply(self.convert_weight_to_numeric)

            # Insert new season column at beginning of dataframe
            qb_df.insert(0, "season", season)

            all_dfs.append(qb_df)

            # Ensure temp folder exists
            if not os.path.exists("./data/temp"):
                os.makedirs("./data/temp")

            # Save the dataframe to a csv file in the temp folder
            qb_df.to_csv(f"./data/temp/ksu_football_qb_{season}.csv", index=False)

            # Generate some stats
            # self.generate_advanced_stats_for_season(season, qb_df)

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
        prepped_qb_df = self.generate_yoy_features(grouped_df)

        #engineer touchdowns
        prepped_qb_df = self.engineer_touchdowns(prepped_qb_df)

        #ensure save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Save the combined and grouped data to a new CSV file
        output_path = f"{self.save_dir}/prepped_qb_data.csv"
        
        prepped_qb_df.to_csv(output_path, index=False)

        print(f"Combined and grouped data saved to {output_path}")

            

            


    