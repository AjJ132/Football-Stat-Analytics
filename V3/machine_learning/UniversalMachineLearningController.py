import numpy as np# type: ignore
import pandas as pd # type: ignore
import os
import json
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor# type: ignore
from sklearn.neural_network import MLPRegressor# type: ignore
from sklearn.metrics import mean_squared_error, r2_score# type: ignore
from sklearn.preprocessing import StandardScaler# type: ignore
from sklearn.impute import SimpleImputer# type: ignore
from sklearn.ensemble import GradientBoostingRegressor# type: ignore
from sklearn.linear_model import Ridge, Lasso# type: ignore
from sklearn.svm import SVR# type: ignore

from util.MachineLearningTypes import MachineLearningType

class UniversalMachineLearningController:
    def __init__(self, data_path, save_predictions_path, features, target, training_seasons, test_season, models_to_use):
        self.data_path = data_path
        self.save_predictions_path = save_predictions_path
        self.features = features
        self.target = target
        self.training_seasons = training_seasons
        self.test_season = test_season
        self.models_to_use = models_to_use
        
        self.models = {
            MachineLearningType.LINEAR_REGRESSION: LinearRegression(),
            MachineLearningType.RANDOM_FOREST: RandomForestRegressor(n_estimators=100, random_state=42),
            MachineLearningType.NEURAL_NETWORKS: MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, solver='adam'),
            MachineLearningType.GRADIENT_BOOSTING: GradientBoostingRegressor(n_estimators=100, random_state=42),
            MachineLearningType.RIDGE_REGRESSION: Ridge(alpha=1.0),
            MachineLearningType.LASSO_REGRESSION: Lasso(alpha=1.0),
            MachineLearningType.SVR: SVR(),
        }
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
    
    def prepare_data(self):
        data = pd.read_csv(self.data_path)
        print(f"Total rows in data: {len(data)}")

        # Convert seasons to strings for comparison
        data['season'] = data['season'].astype(str)
        self.training_seasons = [str(season) for season in self.training_seasons]
        self.test_season = str(self.test_season)

        print(f"Training seasons: {self.training_seasons}")
        print(f"Test season: {self.test_season}")

        # Remove NaN values
        model_data = data.dropna(subset=self.features + [self.target])
        print(f"\nRows after dropping null values: {len(model_data)}")

        train_data = model_data[model_data['season'].isin(self.training_seasons)]
        test_data = model_data[model_data['season'] == self.test_season]

        print(f"\nUnique seasons in train_data: {train_data['season'].unique()}")
        print(f"Unique seasons in test_data: {test_data['season'].unique()}")

        print(f"\nRows in training data: {len(train_data)}")
        print(f"Rows in test data: {len(test_data)}")

        X_train = train_data[self.features]
        y_train = train_data[self.target]
        X_test = test_data[self.features]
        y_test = test_data[self.target]

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

        print("\ny_train description:")
        print(y_train.describe())
        print("\ny_test description:")
        print(y_test.describe())

        # Check for data leakage
        train_indices = set(train_data.index)
        test_indices = set(test_data.index)
        overlap = train_indices.intersection(test_indices)
        if overlap:
            print(f"WARNING: Data leakage detected. {len(overlap)} rows appear in both train and test sets.")

        # Scale the features
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)

        print("\nAfter scaling:")
        print("X_train_scaled shape:", X_train_scaled.shape)
        print("X_test_scaled shape:", X_test_scaled.shape)

        #save training and testing data to csv
        X_train_scaled.to_csv('data/X_train_scaled.csv', index=False)
        y_train.to_csv('data/y_train.csv', index=False)

        return X_train_scaled, y_train, X_test_scaled, y_test, test_data

    def _print_nan_columns(self, data, data_type):
        nan_columns = data.columns[data.isna().any()].tolist()
        print(f"\nColumns with NaN values in {data_type} data:")
        for col in nan_columns:
            nan_count = data[col].isna().sum()
            print(f"{col}: {nan_count} NaN values")
    
    def train_and_evaluate(self):
        try:
            X_train, y_train, X_test, y_test, test_data = self.prepare_data()
            
            if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
                print("Error: One or more datasets are empty. Unable to train and evaluate models.")
                return None

            results = {
                'test_data': test_data,
                'y_test': y_test,
                'models': {}
            }
            
            for ml_type in self.models_to_use:
                if ml_type in self.models:
                    model = self.models[ml_type]
                    try:
                        print(f"Training {ml_type.value}...")
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        
                        if np.isnan(predictions).any() or np.isinf(predictions).any():
                            print(f"Warning: {ml_type.value} produced NaN or infinite predictions.")
                            continue
                        
                        mse = mean_squared_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        results['models'][ml_type] = {
                            'predictions': predictions,
                            'mse': mse,
                            'r2': r2
                        }
                        print(f"{ml_type.value} - MSE: {mse:.4f}, R2: {r2:.4f}")
                    except Exception as e:
                        print(f"Error training {ml_type.value}: {str(e)}")
            
            if not results['models']:
                print("Error: No models were successfully trained and evaluated.")
                return None
            
            return results
        except Exception as e:
            print(f"An unexpected error occurred in train_and_evaluate: {str(e)}")
            return None
    
    def run_ensemble(self):
        results = self.train_and_evaluate()
        
        if results is None:
            print("Error: Unable to run ensemble due to issues in train_and_evaluate.")
            return None
        
        valid_predictions = []
        for name, model_results in results['models'].items():
            predictions = model_results['predictions']
            if not np.isnan(predictions).any() and not np.isinf(predictions).any():
                valid_predictions.append(predictions)
            else:
                print(f"Warning: {name} produced NaN or infinite predictions and will be excluded from the ensemble.")
        
        if not valid_predictions:
            print("Error: No valid predictions available for ensemble.")
            return results
        
        # Simple average ensemble
        ensemble_predictions = np.mean(valid_predictions, axis=0)
        
        if np.isnan(ensemble_predictions).any() or np.isinf(ensemble_predictions).any():
            print("Error: Ensemble predictions contain NaN or infinite values.")
            return results
        
        ensemble_mse = mean_squared_error(results['y_test'], ensemble_predictions)
        ensemble_r2 = r2_score(results['y_test'], ensemble_predictions)
        
        results['models']['Ensemble'] = {
            'predictions': ensemble_predictions,
            'mse': ensemble_mse,
            'r2': ensemble_r2
        }
        print(f"Ensemble - MSE: {ensemble_mse:.4f}, R2: {ensemble_r2:.4f}")
        
        self.save_results(results)
        return results


    def save_results(self, results):
        # Save predictions to CSV
        predictions_df = results['test_data'].copy()
        
        # Ensure the actual target value is in the dataframe
        if self.target not in predictions_df.columns:
            predictions_df[self.target] = results['y_test']
        
        # Add model predictions
        for model_name, model_results in results['models'].items():
            if isinstance(model_name, MachineLearningType):
                model_name = model_name.value
            predictions_df[f"{model_name}_prediction"] = model_results['predictions']
        
        # Reorder columns
        front_columns = ['name', 'season', 'opponent', 'date', self.target]
        
        # Place Ensemble prediction immediately after the target variable
        if 'Ensemble_prediction' in predictions_df.columns:
            front_columns.append('Ensemble_prediction')
        
        # Get other model prediction columns, excluding Ensemble
        model_prediction_columns = [col for col in predictions_df.columns 
                                    if col.endswith('_prediction') and col != 'Ensemble_prediction']
        
        # Include all other columns that are not in front_columns or model_prediction_columns
        other_columns = [col for col in predictions_df.columns 
                         if col not in front_columns and col not in model_prediction_columns]
        
        # Combine all columns in the desired order
        all_columns = front_columns + model_prediction_columns + other_columns
        
        # Ensure all columns exist before reordering
        existing_columns = [col for col in all_columns if col in predictions_df.columns]
        
        predictions_df = predictions_df[existing_columns]

        predictions_df.to_csv(self.save_predictions_path, index=False)
        print(f"Predictions saved to {self.save_predictions_path}")

        # Save results to JSON
        results_dict = {
            'features': self.features,
            'target': self.target,
            'training_seasons': self.training_seasons,
            'test_season': self.test_season,
            'models': {}
        }

        for model_name, model_results in results['models'].items():
            # Convert MachineLearningType to string if necessary
            if isinstance(model_name, MachineLearningType):
                model_name = model_name.value
            results_dict['models'][model_name] = {
                'mse': float(model_results['mse']),
                'r2': float(model_results['r2'])
            }

        json_path = os.path.splitext(self.save_predictions_path)[0] + '_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {json_path}")