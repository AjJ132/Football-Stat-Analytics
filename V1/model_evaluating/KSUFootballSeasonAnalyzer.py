import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from joblib import load
import json
import os
import warnings

class KSUFootballSeasonAnalyzer:
    def __init__(self, data_dir, models_dir, output_dir, sos_file):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.sos_file = sos_file
        self.seasons = [2021, 2022, 2023]
        self.positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P', 'LS']
        self.stats_to_predict = [
            'Games Played', 'Games Started',
            'Rush Attempts', 'Rush Yards', 'Rush Touchdowns', 'Rush Longest',
            'Receiving Recep', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving Longest',
            'Defense Solo', 'Defense Assist', 'Defense Total', 'Defense Interceptions',
            'Defense Pass defl', 'Defense Forced fumble', 'Defense Fumb rec'
        ]
        self.position_stats = self.define_position_stats()
        self.models = {}
        self.season_data = {}
        self.predictions = {}
        self.sos_data = {}

    def define_position_stats(self):
        default_stats = ['Games Played', 'Games Started']
        return {
            'QB': default_stats + ['Rush Attempts', 'Rush Yards', 'Rush Touchdowns', 'Rush Longest',
                   'Receiving Recep', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving Longest'],
            'RB': default_stats + ['Rush Attempts', 'Rush Yards', 'Rush Touchdowns', 'Rush Longest',
                   'Receiving Recep', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving Longest'],
            'WR': default_stats + ['Receiving Recep', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving Longest',
                   'Rush Attempts', 'Rush Yards', 'Rush Touchdowns', 'Rush Longest'],
            'TE': default_stats + ['Receiving Recep', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving Longest'],
            'OL': default_stats,
            'DL': default_stats + ['Defense Solo', 'Defense Assist', 'Defense Total', 'Defense Forced fumble', 'Defense Fumb rec'],
            'LB': default_stats + ['Defense Solo', 'Defense Assist', 'Defense Total', 'Defense Interceptions',
                   'Defense Pass defl', 'Defense Forced fumble', 'Defense Fumb rec'],
            'DB': default_stats + ['Defense Solo', 'Defense Assist', 'Defense Total', 'Defense Interceptions',
                   'Defense Pass defl', 'Defense Forced fumble', 'Defense Fumb rec'],
            'K': default_stats,
            'P': default_stats,
            'LS': default_stats  # Added Long Snapper position
        }

    def load_data(self):
        for season in self.seasons:
            file_path = os.path.join(self.data_dir, f'ksu_football_data_{season}.json')
            with open(file_path, 'r') as file:
                self.season_data[season] = pd.DataFrame(json.load(file))

        # Load SOS data
        with open(self.sos_file, 'r') as file:
            self.sos_data = json.load(file)

    def load_models(self):
        for stat in self.stats_to_predict:
            model_path = os.path.join(self.models_dir, f"{stat.replace(' ', '_').lower()}_model.joblib")
            if os.path.exists(model_path):
                self.models[stat] = load(model_path)

    def predict_season_performance(self):
        for season in self.seasons:
            self.predictions[season] = {}
            season_sos = self.sos_data.get(str(season), 0)  # Default to 0 if SOS not available
            for _, player in self.season_data[season].iterrows():
                position = player['Position']
                player_predictions = {'Name': player['Name'], 'Position': position}
                position_stats = self.position_stats.get(position, ['Games Played', 'Games Started'])
                for stat in position_stats:
                    if stat in self.models:
                        model = self.models[stat]
                        X_pred = pd.DataFrame([[season, season_sos]], columns=['Year', 'SOS'])
                        prediction = model.predict(X_pred)[0]
                        player_predictions[stat] = max(0, prediction)  # Ensure non-negative predictions
                self.predictions[season][player['Name']] = player_predictions

    def generate_charts(self):
        for season in self.seasons:
            for position in set(player['Position'] for player in self.season_data[season].to_dict('records')):
                self.generate_position_charts(season, position)


    def plot_actual_vs_predicted(self, actual, predicted, stat, output_dir):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted {stat}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{stat.replace(" ", "_").lower()}_actual_vs_predicted.png'))
        plt.close()

    def plot_percentiles(self, actual, predicted, stat, output_dir):
        plt.figure(figsize=(10, 6))
        percentiles = range(0, 101, 10)
        actual_percentiles = np.percentile(actual, percentiles)
        predicted_percentiles = np.percentile(predicted, percentiles)
        plt.plot(percentiles, actual_percentiles, label='Actual')
        plt.plot(percentiles, predicted_percentiles, label='Predicted')
        plt.xlabel('Percentile')
        plt.ylabel(stat)
        plt.title(f'Percentile Distribution of {stat}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{stat.replace(" ", "_").lower()}_percentiles.png'))
        plt.close()

    def plot_bell_curve(self, actual, predicted, stat, output_dir):
        plt.figure(figsize=(10, 6))
        
        # Check for zero variance in actual and predicted data
        actual_var = np.var(actual)
        predicted_var = np.var(predicted)
        
        if actual_var > 0:
            sns.kdeplot(actual, fill=True, label='Actual')
        else:
            plt.axvline(x=actual[0], color='blue', linestyle='--', label='Actual')
        
        if predicted_var > 0:
            sns.kdeplot(predicted, fill=True, label='Predicted')
        else:
            plt.axvline(x=predicted[0], color='orange', linestyle='--', label='Predicted')
        
        plt.xlabel(stat)
        plt.ylabel('Density')
        plt.title(f'Distribution of {stat}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{stat.replace(" ", "_").lower()}_distribution.png'))
        plt.close()

    def generate_position_charts(self, season, position):
        position_data = [player for player in self.predictions[season].values() if player['Position'] == position]
        if not position_data:
            return  # Skip if no players for this position

        position_stats = self.position_stats.get(position, ['Games Played', 'Games Started'])
        for stat in position_stats:
            if stat not in self.models:
                continue  # Skip if no model for this stat

            actual_data = self.season_data[season][
                (self.season_data[season]['Position'] == position) &
                (self.season_data[season][stat] > 0)
            ]

            # Create a dictionary of actual values keyed by player name
            actual_dict = dict(zip(actual_data['Name'], actual_data[stat]))

            # Only include players who have both actual and predicted values
            valid_data = [(actual_dict[player['Name']], player[stat]) 
                          for player in position_data 
                          if stat in player and player['Name'] in actual_dict]

            if not valid_data:
                continue  # Skip if no valid data points

            actual_values, predicted_values = zip(*valid_data)

            if len(actual_values) < 2 or len(predicted_values) < 2:
                print(f"Skipping {stat} for {position} in season {season} due to insufficient data points.")
                continue  # Skip if not enough data points for meaningful analysis

            # Create output directory for this season and position
            output_dir = os.path.join(self.output_dir, str(season), position)
            os.makedirs(output_dir, exist_ok=True)

            # Generate charts
            self.plot_actual_vs_predicted(actual_values, predicted_values, stat, output_dir)
            self.plot_percentiles(actual_values, predicted_values, stat, output_dir)
            self.plot_bell_curve(actual_values, predicted_values, stat, output_dir)

    def run_analysis(self):
        print("Loading data...")
        self.load_data()

        print("Loading models...")
        self.load_models()

        print("Predicting season performance...")
        self.predict_season_performance()

        print("Generating charts...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.generate_charts()

        print(f"Analysis complete. Charts saved in {self.output_dir}")

