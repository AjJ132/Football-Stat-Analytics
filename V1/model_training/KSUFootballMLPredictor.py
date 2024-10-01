import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import json
import os
from tqdm import tqdm
from joblib import dump


class KSUFootballMLPredictor:
    def __init__(self, data_dir, season_data_dir):
        self.data_dir = data_dir
        self.season_data_dir = season_data_dir
        self.df = None
        self.years = [2021, 2022]
        self.prediction_year = 2023
        self.stats_to_predict = [
            'Games Played', 'Games Started',
            'Rush Attempts', 'Rush Yards', 'Rush Touchdowns', 'Rush Longest',
            'Receiving Recep', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving Longest',
            'Defense Solo', 'Defense Assist', 'Defense Total', 'Defense Interceptions',
            'Defense Pass defl', 'Defense Forced fumble', 'Defense Fumb rec'
        ]
        self.sos_data = None
        self.confidence_level = 0.95  # 95% confidence interval
        self.models = {}  # Add this line to store trained models

    def load_data(self):
        all_data = []
        for year in self.years:
            file_path = os.path.join(self.data_dir, f'ksu_football_data_{year}.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    for player in data:
                        player['Year'] = year
                    all_data.extend(data)
            else:
                print(f"Warning: File not found: {file_path}")
        
        if not all_data:
            raise ValueError("No data was loaded. Check the file paths and data directory structure.")
        
        self.df = pd.DataFrame(all_data)

        # Load strength of schedule data
        sos_file_path = os.path.join(os.path.dirname(self.season_data_dir), 'season_data', 'strength_of_schedule.json')
        with open(sos_file_path, 'r') as file:
            self.sos_data = json.load(file)

    def preprocess_data(self):
        # Convert all stats to numeric, replacing non-numeric values with NaN
        for stat in self.stats_to_predict:
            self.df[stat] = pd.to_numeric(self.df[stat], errors='coerce')

        # Create a SimpleImputer to fill NaN values with 0
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.df[self.stats_to_predict] = imputer.fit_transform(self.df[self.stats_to_predict])

        # Add strength of schedule data
        self.df['SOS'] = self.df['Year'].astype(str).map(self.sos_data)

        # Convert Year to categorical
        self.df['Year'] = self.df['Year'].astype('category')

    def train_and_predict(self):
        predictions = []
    

        for _, player_data in tqdm(self.df.groupby('Name'), desc="Processing players"):
            player_predictions = {'Name': player_data['Name'].iloc[0]}

            for stat in self.stats_to_predict:
                X = self.df[['Year', 'SOS']]
                y = self.df[stat]

                if len(np.unique(y)) == 1:  # If all values are the same, predict the same value
                    player_predictions[stat] = {
                        'lower': y.iloc[0],
                        'upper': y.iloc[0],
                        'mean': y.iloc[0]
                    }
                else:
                    model = RandomForestRegressor(n_estimators=1000, random_state=42)
                    model.fit(X, y)

                    # Save the trained model
                    self.models[stat] = model
                    
                    # Predict using the SOS for the prediction year
                    prediction_sos = self.sos_data.get(str(self.prediction_year), np.mean(list(self.sos_data.values())))
                    X_pred = pd.DataFrame([[self.prediction_year, prediction_sos]], columns=['Year', 'SOS'])
                    
                    # Get predictions from all trees
                    tree_predictions = []
                    for tree in model.estimators_:
                        tree_predictions.append(tree.predict(X_pred.values)[0])  # Use .values to avoid feature names warning
                    
                    # Calculate confidence interval
                    lower = np.percentile(tree_predictions, (1 - self.confidence_level) / 2 * 100)
                    upper = np.percentile(tree_predictions, (1 + self.confidence_level) / 2 * 100)
                    mean = np.mean(tree_predictions)

                    player_predictions[stat] = {
                        'lower': max(0, round(lower, 2)),
                        'upper': max(0, round(upper, 2)),
                        'mean': max(0, round(mean, 2))
                    }

            predictions.append(player_predictions)

        return predictions

    def export_predictions(self, predictions, output_path):
        # Format predictions for better readability
        formatted_predictions = []
        for player in predictions:
            formatted_player = {'Name': player['Name']}
            for stat, values in player.items():
                if stat != 'Name':
                    formatted_player[stat] = f"{values['lower']} - {values['upper']} (avg: {values['mean']})"
            formatted_predictions.append(formatted_player)

        with open(output_path, 'w') as f:
            json.dump(formatted_predictions, f, indent=2)

    def save_models(self, models_dir):
        os.makedirs(models_dir, exist_ok=True)
        for stat, model in self.models.items():
            model_path = os.path.join(models_dir, f"{stat.replace(' ', '_').lower()}_model.joblib")
            dump(model, model_path)

    def run_prediction_pipeline(self, output_path, models_dir):
        print("Loading data...")
        self.load_data()

        print("Preprocessing data...")
        self.preprocess_data()

        print("Training models and generating predictions...")
        predictions = self.train_and_predict()

        print("Saving trained models...")
        self.save_models(models_dir)

        print("Exporting predictions...")
        self.export_predictions(predictions, output_path)

        print(f"Predictions for {self.prediction_year} season exported to {output_path}")
        print(f"Trained models saved in {models_dir}")

