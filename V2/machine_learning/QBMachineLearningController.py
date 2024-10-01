from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

class QBMachineLearningController:
    def __init__(self, data_path, predictions_path, training_seasons, test_season):
        self.data_path = data_path
        self.predictions_path = predictions_path
        self.training_seasons = [int(season) for season in training_seasons]
        self.test_season = int(test_season)

    def run_ml_pipeline(self):
        """
        Run the entire machine learning pipeline with multi-season training and save predictions.
        """
        features = [
            'prev_games_played', 'prev_games_started', 'prev_passing_yards',
            'prev_pass_completions', 'prev_pass_attempts', 'prev_interceptions_thrown',
            'prev_passing_touchdowns', 'prev_longest_completion',
            'prev_pass_completion_percentage', 'prev_average_yards_per_pass',
            'prev_average_passing_yards_per_game', 'prev_rushing_attempts',
            'prev_rushing_yards', 'prev_rushing_touchdowns', 'prev_longest_rush',
            'prev_average_yards_per_rush', 'prev_average_rushing_yards_per_game',
            'prev_total_touchdowns',
        ]

        target = 'total_touchdowns'

        # Load CSV into dataframe
        data = pd.read_csv(self.data_path)
        print(f"Total rows in data: {len(data)}")

        # Subset of data that has no null values
        model_data = data.dropna(subset=features + [target])
        print(f"Rows after dropping null values: {len(model_data)}")

        # Select training seasons and test season
        train_data = model_data[model_data['season'].isin(self.training_seasons)]
        test_data = model_data[model_data['season'] == self.test_season]
        
        print(f"Rows in training data: {len(train_data)}")
        print(f"Rows in test data: {len(test_data)}")
        print(f"Unique seasons in data: {model_data['season'].unique()}")
        print(f"Training seasons: {self.training_seasons}")
        print(f"Test season: {self.test_season}")

        if len(train_data) == 0:
            print("Error: No data available for the specified training seasons.")
            return

        model = LinearRegression()

        # Fit or train the model on the training data
        model.fit(train_data[features], train_data[target])
        
        # Predict on the test data
        preds = model.predict(test_data[features])

        # Add predictions to test data
        test_data.loc[:, 'predicted_touchdowns'] = preds

        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(test_data[target], test_data['predicted_touchdowns']))
        r2 = r2_score(test_data[target], test_data['predicted_touchdowns'])
        
        print(f"RMSE: {rmse}")
        print(f"R2: {r2}")

        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.coef_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Important Features:")
        print(feature_importance.head())

        # Save predictions to CSV
        predictions_file = os.path.join(self.predictions_path, f'qb_predictions_{self.test_season}.csv')

        # Copy df
        predictions_df = test_data.copy()

        # Remove all columns except for name, season, games_played, games_started, total_touchdowns, predicted_touchdowns
        columns_to_keep = ['name', 'season', 'games_played', 'games_started', 'total_touchdowns', 'predicted_touchdowns']
        predictions_df = predictions_df[columns_to_keep]

        # Save to csv
        predictions_df.to_csv(predictions_file, index=False)

        return model, test_data, feature_importance