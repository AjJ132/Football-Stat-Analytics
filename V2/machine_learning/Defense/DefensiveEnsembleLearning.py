import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import os

class DefenseEnsembleLearning:
    def __init__(self, data_path, predictions_path, training_seasons, test_season, training_data_dir):
        self.data_path = data_path
        self.predictions_path = predictions_path
        self.training_seasons = [int(season) for season in training_seasons]
        self.test_season = int(test_season)
        self.training_data_dir = training_data_dir
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, solver='adam')
        }
        self.scaler = StandardScaler()
        
    def engineer_features(self, X):
        X = X.copy()  # Create a copy to avoid SettingWithCopyWarning
        X.loc[:, 'games_played_squared'] = X['games_played'] ** 2
        X.loc[:, 'games_played_log'] = np.log1p(X['games_played'])

        for feature in ['prev_solo_tackles', 'prev_assisted_tackles', 'prev_tackles_for_loss', 'prev_sacks']:
            X.loc[:, f'{feature}_per_game'] = X[feature] / X['prev_games_played']

        return X

    def prepare_data(self):
        data = pd.read_csv(self.data_path)
        print(f"Total rows in data: {len(data)}")

        features = [
            'prev_solo_tackles', 'prev_assisted_tackles', 'prev_total_tackles',
            'prev_tackles_for_loss', 'prev_tackles_for_loss_yards', 
            'prev_sacks', 'prev_sacks_yards', 'prev_interceptions',
            'prev_pass_deflections', 'prev_forced_fumbles', 'prev_fumble_recoveries', 'prev_blocked_kicks',
            'games_played', 'prev_games_played'
        ]
        target = 'total_tackles'

        model_data = data.dropna(subset=features + [target])
        print(f"\nRows after dropping null values: {len(model_data)}")

        train_data = model_data[model_data['season'].isin(self.training_seasons)]
        test_data = model_data[model_data['season'] == self.test_season]

        print(f"\nRows in training data: {len(train_data)}")
        print(f"Rows in test data: {len(test_data)}")

        X_train = self.engineer_features(train_data[features])
        y_train = train_data[target]
        X_test = self.engineer_features(test_data[features])
        y_test = test_data[target]

        # Feature selection
        selector = RFECV(estimator=RandomForestRegressor(n_estimators=100, random_state=42), step=1, cv=5)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)

        return X_train_scaled, y_train, X_test_scaled, y_test, test_data

    def train_and_evaluate(self):
        X_train, y_train, X_test, y_test, test_data = self.prepare_data()
        
        if X_train is None or y_train is None or X_test is None or y_test is None:
            print("Error: Unable to prepare data. Exiting train_and_evaluate.")
            return None

        results = {}

        for name, model in self.models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results[name] = {
                'predictions': predictions,
                'mse': mse,
                'r2': r2,
                'cv_rmse': cv_rmse
            }
            print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}, CV RMSE: {cv_rmse:.4f}")

        # Weighted ensemble
        weights = {name: 1/result['cv_rmse'] for name, result in results.items()}
        total_weight = sum(weights.values())
        normalized_weights = {name: weight/total_weight for name, weight in weights.items()}

        ensemble_predictions = np.zeros_like(y_test, dtype=float)
        for name, weight in normalized_weights.items():
            ensemble_predictions += weight * results[name]['predictions']

        ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
        ensemble_r2 = r2_score(y_test, ensemble_predictions)
        results['Weighted Ensemble'] = {
            'predictions': ensemble_predictions,
            'mse': ensemble_mse,
            'r2': ensemble_r2
        }
        print(f"Weighted Ensemble - MSE: {ensemble_mse:.4f}, R2: {ensemble_r2:.4f}")

        self.export_predictions(test_data, results)

        return results

    def export_predictions(self, test_data, results):
        predictions_df = test_data[['name', 'season', 'games_played', 'games_started', 'total_tackles']].copy()
        
        for model_name, model_results in results.items():
            predictions_df[f'{model_name}_predictions'] = model_results['predictions']

        predictions_file = os.path.join(self.predictions_path, f'defense_ensemble_predictions_{self.test_season}.csv')
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions exported to {predictions_file}")

    def run_ensemble_pipeline(self):
        return self.train_and_evaluate()