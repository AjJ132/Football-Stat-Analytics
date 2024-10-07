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
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from machine_learning.MultiPositionStatsAnalysis import ResultsAnalyzer

class EnsembleLearning:
    def __init__(self, data_path, predictions_path, analytics_path, training_seasons, test_season, training_data_dir, features, target, position, prune_models=True, prune_threshold=2.0):
        self.data_path = data_path
        self.predictions_path = predictions_path
        self.analytics_path = analytics_path
        self.training_seasons = [int(season) for season in training_seasons]
        self.test_season = int(test_season)
        self.training_data_dir = training_data_dir
        self.features = features
        self.target = target
        self.position = position
        self.prune_models = prune_models
        self.prune_threshold = prune_threshold
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
        X = X.copy()
        if 'games_played' in X.columns:
            X['games_played_squared'] = X['games_played'] ** 2
            X['games_played_log'] = np.log1p(X['games_played'])

        for feature in X.columns:
            if feature.startswith('prev_') and 'prev_games_played' in X.columns:
                X[f'{feature}_per_game'] = X[feature] / X['prev_games_played']

        return X

    def prepare_data(self):
        data = pd.read_csv(self.data_path)
        print(f"Total rows in data: {len(data)}")

        model_data = data.dropna(subset=self.features + [self.target])
        print(f"\nRows after dropping null values: {len(model_data)}")

        train_data = model_data[model_data['season'].isin(self.training_seasons)]
        test_data = model_data[model_data['season'] == self.test_season]

        print(f"\nRows in training data: {len(train_data)}")
        print(f"Rows in test data: {len(test_data)}")

        X_train = self.engineer_features(train_data[self.features])
        y_train = train_data[self.target]
        X_test = self.engineer_features(test_data[self.features])
        y_test = test_data[self.target]

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

        results = {
            'test_data': test_data,
            'y_test': y_test,
            'models': {}
        }

        for name, model in self.models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            accuracy = self.calculate_accuracy_percentage(y_test, predictions)
            results['models'][name] = {
                'predictions': predictions,
                'mse': mse,
                'r2': r2,
                'cv_rmse': cv_rmse,
                'accuracy': accuracy
            }
            print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}, CV RMSE: {cv_rmse:.4f}, Accuracy: {accuracy:.2f}%")

        if self.prune_models:
            results = self.prune_poor_models(results)

        # Weighted ensemble
        weights = {name: 1/result['cv_rmse'] for name, result in results['models'].items()}
        total_weight = sum(weights.values())
        normalized_weights = {name: weight/total_weight for name, weight in weights.items()}

        ensemble_predictions = np.zeros_like(y_test, dtype=float)
        for name, weight in normalized_weights.items():
            ensemble_predictions += weight * results['models'][name]['predictions']

        ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
        ensemble_r2 = r2_score(y_test, ensemble_predictions)
        ensemble_accuracy = self.calculate_accuracy_percentage(y_test, ensemble_predictions)
        results['models']['Weighted Ensemble'] = {
            'predictions': ensemble_predictions,
            'mse': ensemble_mse,
            'r2': ensemble_r2,
            'accuracy': ensemble_accuracy
        }
        print(f"Weighted Ensemble - MSE: {ensemble_mse:.4f}, R2: {ensemble_r2:.4f}, Accuracy: {ensemble_accuracy:.2f}%")

        return results
    
    def prune_poor_models(self, results):
        if not self.prune_models:
            return results

        mean_mse = np.mean([model['mse'] for model in results['models'].values()])
        pruned_results = {
            'test_data': results['test_data'],
            'y_test': results['y_test'],
            'models': {}
        }

        for name, model_result in results['models'].items():
            if model_result['mse'] <= self.prune_threshold * mean_mse:
                pruned_results['models'][name] = model_result
            else:
                print(f"Pruned {name} due to poor performance (MSE: {model_result['mse']:.4f})")

        return pruned_results

    def export_predictions(self, test_data, model_results):
        predictions_df = test_data[['name', 'season', 'games_played', 'games_started', self.target]].copy()
        
        for model_name, model_result in model_results.items():
            predictions_df[f'{model_name}_predictions'] = model_result['predictions']
            predictions_df[f'{model_name}_accuracy'] = model_result['accuracy']

        # Move the Weighted Ensemble columns to right after the target column
        cols = list(predictions_df.columns)
        target_index = cols.index(self.target)
        weighted_ensemble_pred_col = 'Weighted Ensemble_predictions'
        weighted_ensemble_acc_col = 'Weighted Ensemble_accuracy'
        if weighted_ensemble_pred_col in cols and weighted_ensemble_acc_col in cols:
            cols.remove(weighted_ensemble_pred_col)
            cols.remove(weighted_ensemble_acc_col)
            cols.insert(target_index + 1, weighted_ensemble_pred_col)
            cols.insert(target_index + 2, weighted_ensemble_acc_col)
        predictions_df = predictions_df[cols]

        predictions_file = os.path.join(self.predictions_path, f'{self.position}_ensemble_predictions_{self.test_season}.csv')
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions exported to {predictions_file}")
        return predictions_df

    def run_ensemble_pipeline(self):
        results = self.train_and_evaluate()
        if results:
            predictions_df = self.export_predictions(results['test_data'], results['models'])
            self.analyze_results(predictions_df)
        else:
            print("Error: No results to analyze. Ensemble pipeline failed.")
        return results

    def analyze_results(self, df):
        analyzer = ResultsAnalyzer(df, self.analytics_path, self.position, self.target)
        analyzer.run_analysis()

    def calculate_accuracy_percentage(self, y_true, y_pred, error_margin=10):
        """
        Calculate the percentage of predictions within the error margin of the true values.
        
        :param y_true: Array-like of true values
        :param y_pred: Array-like of predicted values
        :param error_margin: The acceptable error margin (default 10 yards)
        :return: Accuracy percentage
        """
        within_margin = np.abs(y_true - y_pred) <= error_margin
        return np.mean(within_margin) * 100
