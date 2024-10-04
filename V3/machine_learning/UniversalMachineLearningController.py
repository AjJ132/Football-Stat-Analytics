import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class UniversalMachineLearningController:
    def __init__(self, data_path, features, target, training_seasons, test_season, models_to_use):
        self.data_path = data_path
        self.features = features
        self.target = target
        self.training_seasons = training_seasons
        self.test_season = test_season
        self.models_to_use = models_to_use
        
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
        self.imputer = SimpleImputer(strategy='median')
    
    def prepare_data(self):
        data = pd.read_csv(self.data_path)
        print("Original data shape:", data.shape)
        print("Unique seasons in data:", data['season'].unique())
        
        # Convert seasons to strings for comparison
        data['season'] = data['season'].astype(str)
        self.training_seasons = [str(season) for season in self.training_seasons]
        self.test_season = str(self.test_season)
        
        train_data = data[data['season'].isin(self.training_seasons)]
        test_data = data[data['season'] == self.test_season]
        
        # Debug: Print columns with NaN values in training data
        nan_columns_train = train_data.columns[train_data.isna().any()].tolist()
        print("\nColumns with NaN values in training data:")
        for col in nan_columns_train:
            nan_count = train_data[col].isna().sum()
            print(f"{col}: {nan_count} NaN values")
        
        # Debug: Print columns with NaN values in test data
        nan_columns_test = test_data.columns[test_data.isna().any()].tolist()
        print("\nColumns with NaN values in test data:")
        for col in nan_columns_test:
            nan_count = test_data[col].isna().sum()
            print(f"{col}: {nan_count} NaN values")
        
        X_train = train_data[self.features]
        y_train = train_data[self.target]
        X_test = test_data[self.features]
        y_test = test_data[self.target]
        
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)
        
        print("y_train description:")
        print(y_train.describe())
        print("y_test description:")
        print(y_test.describe())
        
        # Check for NaN or infinite values
        print("NaN values in y_train:", np.isnan(y_train).sum())
        print("Infinite values in y_train:", np.isinf(y_train).sum())
        print("NaN values in y_test:", np.isnan(y_test).sum())
        print("Infinite values in y_test:", np.isinf(y_test).sum())
        
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            raise ValueError("Training or test data is empty. Please check your season selections.")
        
        # Impute NaN values in the target variable
        y_train_imputed = self.imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_imputed = self.imputer.transform(y_test.values.reshape(-1, 1)).ravel()
        
        print("After imputing targets:")
        print("y_train shape:", y_train_imputed.shape)
        print("y_test shape:", y_test_imputed.shape)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, y_train_imputed, X_test_scaled, y_test_imputed, test_data
    
    def train_and_evaluate(self):
        X_train, y_train, X_test, y_test, test_data = self.prepare_data()
        
        results = {
            'test_data': test_data,
            'y_test': y_test,
            'models': {}
        }
        
        for name in self.models_to_use:
            if name in self.models:
                model = self.models[name]
                try:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    
                    if np.isnan(predictions).any() or np.isinf(predictions).any():
                        print(f"Warning: {name} produced NaN or infinite predictions.")
                        continue
                    
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    results['models'][name] = {
                        'predictions': predictions,
                        'mse': mse,
                        'r2': r2
                    }
                    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
        
        return results
    
    def run_ensemble(self):
        results = self.train_and_evaluate()
        
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
        
        return results