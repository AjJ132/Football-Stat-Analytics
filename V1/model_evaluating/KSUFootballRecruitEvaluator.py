import pandas as pd
import numpy as np
import json
from joblib import load
import os

class KSUFootballRecruitEvaluator:
    def __init__(self, models_dir, recruits_file, sos_file):
        self.models_dir = models_dir
        self.recruits_file = recruits_file
        self.sos_file = sos_file
        self.models = {}
        self.recruits_data = None
        self.sos_data = None
        self.prediction_year = 2023  # Set this to the year you're predicting for
        self.confidence_level = 0.95

    def load_models(self):
        for model_file in os.listdir(self.models_dir):
            if model_file.endswith('_model.joblib'):
                stat = model_file.replace('_model.joblib', '').replace('_', ' ')
                model_path = os.path.join(self.models_dir, model_file)
                self.models[stat] = load(model_path)

    def load_recruits_data(self):
        with open(self.recruits_file, 'r') as f:
            self.recruits_data = json.load(f)

    def load_sos_data(self):
        with open(self.sos_file, 'r') as f:
            self.sos_data = json.load(f)

    def preprocess_recruit_data(self, recruit):
        processed_data = {'Year': self.prediction_year}
        processed_data['SOS'] = self.sos_data.get(str(self.prediction_year), np.mean(list(self.sos_data.values())))
        return pd.DataFrame([processed_data])

    def predict_recruit_stats(self):
        predictions = []
        for recruit in self.recruits_data:
            recruit_predictions = {'Name': recruit['Name']}
            X_pred = self.preprocess_recruit_data(recruit)

            for stat, model in self.models.items():
                tree_predictions = []
                for tree in model.estimators_:
                    tree_predictions.append(tree.predict(X_pred)[0])

                lower = np.percentile(tree_predictions, (1 - self.confidence_level) / 2 * 100)
                upper = np.percentile(tree_predictions, (1 + self.confidence_level) / 2 * 100)
                mean = np.mean(tree_predictions)

                recruit_predictions[stat] = {
                    'lower': max(0, round(lower, 2)),
                    'upper': max(0, round(upper, 2)),
                    'mean': max(0, round(mean, 2))
                }

            predictions.append(recruit_predictions)

        return predictions

    def export_predictions(self, predictions, output_path):
        formatted_predictions = []
        for player in predictions:
            formatted_player = {'Name': player['Name']}
            for stat, values in player.items():
                if stat != 'Name':
                    formatted_player[stat] = f"{values['lower']} - {values['upper']} (avg: {values['mean']})"
            formatted_predictions.append(formatted_player)

        with open(output_path, 'w') as f:
            json.dump(formatted_predictions, f, indent=2)

    def run_evaluation_pipeline(self, output_path):
        print("Loading models...")
        self.load_models()

        print("Loading recruits data...")
        self.load_recruits_data()

        print("Loading strength of schedule data...")
        self.load_sos_data()

        print("Generating predictions for recruits...")
        predictions = self.predict_recruit_stats()

        print("Exporting predictions...")
        self.export_predictions(predictions, output_path)

        print(f"Recruit predictions exported to {output_path}")