import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

class KSUFootballMLEvaluator:
    def __init__(self, data_dir, predictions_path, models_dir, sos_file_path):
        self.data_dir = data_dir
        self.predictions_path = predictions_path
        self.models_dir = models_dir
        self.sos_file_path = sos_file_path
        self.df = None
        self.years = [2021, 2022, 2023]
        self.prediction_year = 2023
        self.stats_to_predict = [
            'Games Played', 'Games Started',
            'Rush Attempts', 'Rush Yards', 'Rush Touchdowns', 'Rush Longest',
            'Receiving Recep', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving Longest',
            'Defense Solo', 'Defense Assist', 'Defense Total', 'Defense Interceptions',
            'Defense Pass defl', 'Defense Forced fumble', 'Defense Fumb rec'
        ]
        self.evaluation_metrics = {}
        self.prediction_accuracy = {}
        self.accuracy_summary = None
        self.models = {}
        self.ml_predictions = {}
        self.ml_accuracy = {}
        self.sos_data = None

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
        with open(self.sos_file_path, 'r') as file:
            self.sos_data = json.load(file)

    def load_predictions(self):
        with open(self.predictions_path, 'r') as file:
            return json.load(file)

    def preprocess_data(self):
        # Convert all stats to numeric, replacing non-numeric values with NaN
        for stat in self.stats_to_predict:
            self.df[stat] = pd.to_numeric(self.df[stat], errors='coerce')

        # Fill NaN values with 0
        self.df[self.stats_to_predict] = self.df[self.stats_to_predict].fillna(0)

        # Add strength of schedule data
        self.df['SOS'] = self.df['Year'].astype(str).map(self.sos_data)

        # Convert Year to categorical
        self.df['Year'] = self.df['Year'].astype('category')

    def evaluate_model(self):
        for stat in self.stats_to_predict:
            X = self.df[['Year']]
            y = self.df[stat]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.evaluation_metrics[stat] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }

    def compare_predictions_to_actual(self):
        predictions = self.load_predictions()
        actual_2023 = self.df[self.df['Year'] == self.prediction_year]

        for player in predictions:
            player_name = player['Name']
            actual_player_data = actual_2023[actual_2023['Name'] == player_name]

            if not actual_player_data.empty:
                self.prediction_accuracy[player_name] = {}
                for stat in self.stats_to_predict:
                    if stat in player:
                        predicted_range = player[stat].split(' - ')
                        predicted_lower = float(predicted_range[0])
                        predicted_upper = float(predicted_range[1].split(' ')[0])
                        actual_value = actual_player_data[stat].values[0]

                        is_accurate = predicted_lower <= actual_value <= predicted_upper
                        self.prediction_accuracy[player_name][stat] = {
                            'predicted_range': f"{predicted_lower} - {predicted_upper}",
                            'actual_value': float(actual_value),
                            'is_accurate': str(is_accurate)  # Convert bool to string
                        }

    def generate_accuracy_report(self, output_path):
        self.accuracy_summary = {
            'overall_accuracy': {},
            'player_accuracy': self.prediction_accuracy
        }

        for stat in self.stats_to_predict:
            accurate_predictions = sum(1 for player in self.prediction_accuracy.values() 
                                       if stat in player and player[stat]['is_accurate'] == 'True')
            total_predictions = sum(1 for player in self.prediction_accuracy.values() if stat in player)
            accuracy_rate = accurate_predictions / total_predictions if total_predictions > 0 else 0
            self.accuracy_summary['overall_accuracy'][stat] = f"{accuracy_rate:.2%}"

        with open(output_path, 'w') as f:
            json.dump(self.accuracy_summary, f, indent=2)

    def generate_visualizations(self, output_dir):
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Bar plot of R2 scores
        plt.figure(figsize=(12, 6))
        r2_scores = [self.evaluation_metrics[stat]['R2'] for stat in self.stats_to_predict]
        sns.barplot(x=self.stats_to_predict, y=r2_scores)
        plt.title('R2 Scores for Each Statistic')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'r2_scores.png'))
        plt.close()

        # Heatmap of all metrics
        plt.figure(figsize=(12, 8))
        metrics_df = pd.DataFrame(self.evaluation_metrics).T
        sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('Evaluation Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'))
        plt.close()

        # Scatter plot of actual vs predicted for a selected statistic (e.g., 'Rush Yards')
        selected_stat = 'Rush Yards'
        X = self.df[['Year']]
        y = self.df[selected_stat]
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted {selected_stat}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{selected_stat.lower().replace(" ", "_")}_actual_vs_predicted.png'))
        plt.close()

        # Add new visualization for prediction accuracy
        plt.figure(figsize=(12, 6))
        accuracy_rates = [float(rate[:-1]) / 100 for rate in self.accuracy_summary['overall_accuracy'].values()]
        sns.barplot(x=list(self.accuracy_summary['overall_accuracy'].keys()), y=accuracy_rates)
        plt.title('Prediction Accuracy by Statistic')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_accuracy.png'))
        plt.close()

    def export_evaluation_metrics(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=2)

    def load_models(self):
        for stat in self.stats_to_predict:
            model_path = os.path.join(self.models_dir, f"{stat.replace(' ', '_').lower()}_model.joblib")
            self.models[stat] = load(model_path)

    def generate_ml_predictions(self):
        X_2023 = self.df[self.df['Year'] == 2023][['Year', 'SOS']]
        for stat in self.stats_to_predict:
            self.ml_predictions[stat] = self.models[stat].predict(X_2023)

    def compare_ml_predictions(self):
        actual_2023 = self.df[self.df['Year'] == 2023]
        for stat in self.stats_to_predict:
            correct = 0
            incorrect = 0
            outside_range = 0
            below_range = 0
            above_range = 0
            total = len(actual_2023)

            for i, (_, row) in enumerate(actual_2023.iterrows()):
                actual = row[stat]
                predicted = self.ml_predictions[stat][i]
                lower, upper = self.get_prediction_range(stat, predicted)

                if lower <= actual <= upper:
                    correct += 1
                else:
                    incorrect += 1
                    if actual < lower:
                        below_range += 1
                        outside_range += 1
                    elif actual > upper:
                        above_range += 1
                        outside_range += 1

            accuracy = correct / total
            percent_outside = outside_range / total * 100

            self.ml_accuracy[stat] = {
                'average_accuracy': accuracy,
                'correct': correct,
                'incorrect': incorrect,
                'outside_range': outside_range,
                'percent_outside': percent_outside,
                'below_range': below_range,
                'above_range': above_range
            }


    def get_prediction_range(self, stat, predicted):
        # This is a simplified method. You might want to adjust this based on your specific prediction model
        lower = predicted * 0.9
        upper = predicted * 1.1
        return lower, upper

    def generate_ml_accuracy_report(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.ml_accuracy, f, indent=2)

    def generate_ml_visualizations(self, output_dir):
        # Bar plot of average accuracy for each statistic
        plt.figure(figsize=(12, 6))
        accuracies = [self.ml_accuracy[stat]['average_accuracy'] for stat in self.stats_to_predict]
        sns.barplot(x=self.stats_to_predict, y=accuracies)
        plt.title('Average Accuracy for Each Statistic')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ml_average_accuracy.png'))
        plt.close()

        # Pie chart of correct vs incorrect predictions for a selected statistic
        selected_stat = 'Rush Yards'  # You can change this to any statistic you're interested in
        plt.figure(figsize=(10, 10))
        sizes = [self.ml_accuracy[selected_stat]['correct'], self.ml_accuracy[selected_stat]['incorrect']]
        labels = ['Correct', 'Incorrect']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'Correct vs Incorrect Predictions for {selected_stat}')
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, f'{selected_stat.lower().replace(" ", "_")}_predictions_pie.png'))
        plt.close()

        # Add new visualization for below/above range predictions
        plt.figure(figsize=(12, 6))
        stats = list(self.ml_accuracy.keys())
        below_range = [self.ml_accuracy[stat]['below_range'] for stat in stats]
        above_range = [self.ml_accuracy[stat]['above_range'] for stat in stats]
        
        x = range(len(stats))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], below_range, width, label='Below Range', color='blue')
        plt.bar([i + width/2 for i in x], above_range, width, label='Above Range', color='red')
        
        plt.xlabel('Statistics')
        plt.ylabel('Count')
        plt.title('Predictions Below and Above Range')
        plt.xticks(x, stats, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'below_above_range_predictions.png'))
        plt.close()

    def run_evaluation_pipeline(self, metrics_output_path, visualizations_output_dir, accuracy_report_path, ml_accuracy_report_path):
        print("Loading data...")
        self.load_data()

        print("Preprocessing data...")
        self.preprocess_data()

        print("Evaluating model...")
        self.evaluate_model()

        print("Comparing predictions to actual data...")
        self.compare_predictions_to_actual()

        print("Loading trained models...")
        self.load_models()

        print("Generating ML predictions...")
        self.generate_ml_predictions()

        print("Comparing ML predictions to actual data...")
        self.compare_ml_predictions()

        print("Generating accuracy report...")
        self.generate_accuracy_report(accuracy_report_path)

        print("Generating ML accuracy report...")
        self.generate_ml_accuracy_report(ml_accuracy_report_path)

        print("Exporting evaluation metrics...")
        self.export_evaluation_metrics(metrics_output_path)

        print("Generating visualizations...")
        self.generate_visualizations(visualizations_output_dir)
        self.generate_ml_visualizations(visualizations_output_dir)

        print(f"Evaluation metrics exported to {metrics_output_path}")
        print(f"Accuracy report exported to {accuracy_report_path}")
        print(f"ML accuracy report exported to {ml_accuracy_report_path}")
        print(f"Visualizations saved in {visualizations_output_dir}")

