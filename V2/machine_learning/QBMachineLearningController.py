from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

class QBMachineLearningController:
    def __init__(self, data_path):
        self.data_path = data_path

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

        # Subset of data that has no null values
        model_data = data.dropna(subset=features + [target])

        # Select training seasons (2019, 2021, 2022) and test season (2023)
        train_data = model_data[model_data['season'].isin([ 2022,2023])]
        test_data = model_data[model_data['season'] == 2024]

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
        output_columns = ['name', 'season', target, 'predicted_touchdowns'] + features
        output_data = test_data[output_columns]
        output_path = os.path.join(os.path.dirname(self.data_path), 'qb_predictions_2023.csv')
        output_data.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

        return model, test_data, feature_importance