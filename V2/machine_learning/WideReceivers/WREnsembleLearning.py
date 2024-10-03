import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

class WREnsembleLearning:
    def __init__(self, data_path, predictions_path, training_seasons, test_season, training_data_dir):
        self.data_path = data_path
        self.predictions_path = predictions_path
        self.training_seasons = [int(season) for season in training_seasons]
        self.test_season = int(test_season)
        self.training_data_dir = training_data_dir
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()

    def prepare_data(self):
        data = pd.read_csv(self.data_path)
        print(f"Total rows in data: {len(data)}")

        features = [
            'prev_games_played', 'prev_games_started', 'prev_receptions',
            'prev_receiving_yards', 'prev_receiving_touchdowns', 'prev_longest_reception',
            'prev_receptions_per_game', 'prev_average_yards_per_reception',
            'prev_average_receiving_yards_per_game', 'prev_total_touchdowns', 'prev_total_points'
        ]
        target = 'total_touchdowns'

        # Debug: Check null values for each feature and target
        print("\nNull values in each column:")
        print(data[features + [target]].isnull().sum())

        # Debug: Print a few rows with null values
        print("\nSample rows with null values:")
        print(data[data[features + [target]].isnull().any(axis=1)].head())

        model_data = data.dropna(subset=features + [target])
        print(f"\nRows after dropping null values: {len(model_data)}")

        # Debug: Check which features caused the most data loss
        dropped_rows = len(data) - len(model_data)
        if dropped_rows > 0:
            print("\nFeatures causing data loss:")
            for feature in features + [target]:
                feature_null_count = data[feature].isnull().sum()
                if feature_null_count > 0:
                    print(f"{feature}: {feature_null_count} null values")

        train_data = model_data[model_data['season'].isin(self.training_seasons)]
        test_data = model_data[model_data['season'] == self.test_season]

        print(f"\nRows in training data: {len(train_data)}")
        print(f"Rows in test data: {len(test_data)}")

        if not os.path.exists(self.training_data_dir):
            os.makedirs(self.training_data_dir)

        train_data.to_csv(os.path.join(self.training_data_dir, 'wr_train.csv'), index=False)
        test_data.to_csv(os.path.join(self.training_data_dir, 'wr_test.csv'), index=False)

        X_train = self.scaler.fit_transform(train_data[features])
        y_train = train_data[target]
        X_test = self.scaler.transform(test_data[features])
        y_test = test_data[target]

        return X_train, y_train, X_test, y_test, test_data

    def train_and_evaluate(self):
        X_train, y_train, X_test, y_test, test_data = self.prepare_data()
        results = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results[name] = {
                'predictions': predictions,
                'mse': mse,
                'r2': r2
            }
            print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

        ensemble_predictions = np.mean([results[name]['predictions'] for name in self.models], axis=0)
        ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
        ensemble_r2 = r2_score(y_test, ensemble_predictions)
        results['Ensemble'] = {
            'predictions': ensemble_predictions,
            'mse': ensemble_mse,
            'r2': ensemble_r2
        }
        print(f"Ensemble - MSE: {ensemble_mse:.4f}, R2: {ensemble_r2:.4f}")

        self.export_predictions(test_data, results)

        return results

    def export_predictions(self, test_data, results):
        predictions_df = test_data[['name', 'season', 'games_played', 'games_started', 'total_touchdowns']].copy()
        
        for model_name, model_results in results.items():
            predictions_df[f'{model_name}_predictions'] = model_results['predictions']

        predictions_file = os.path.join(self.predictions_path, f'wr_ensemble_predictions_{self.test_season}.csv')
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions exported to {predictions_file}")

    def run_ensemble_pipeline(self):
        return self.train_and_evaluate()