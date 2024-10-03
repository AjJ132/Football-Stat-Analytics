import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class AnalyzeDefResults:
    def __init__(self, csv_file, export_dir):
        self.df = pd.read_csv(csv_file)
        self.export_dir = export_dir
        self.ensemble_column = 'Weighted Ensemble_predictions'
        if self.ensemble_column not in self.df.columns:
            raise ValueError(f"Column '{self.ensemble_column}' not found in the CSV file.")
        self.error = self.df[self.ensemble_column] - self.df['total_tackles']
        self.models = [col.replace('_predictions', '') for col in self.df.columns if col.endswith('_predictions')]

    def plot_analysis(self):
        # 1. Actual vs Predicted Total Tackles
        fig, ax1 = plt.subplots(figsize=(9, 9))
        ax1.scatter(self.df['total_tackles'], self.df[self.ensemble_column], alpha=0.6)
        ax1.plot([0, self.df['total_tackles'].max()], [0, self.df['total_tackles'].max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Total Tackles')
        ax1.set_ylabel('Predicted Total Tackles (Ensemble)')
        ax1.set_title('Actual vs Predicted Total Tackles')
        for i, txt in enumerate(self.df['name']):
            ax1.annotate(txt, (self.df['total_tackles'].iloc[i], self.df[self.ensemble_column].iloc[i]), fontsize=8)
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
        plt.savefig(os.path.join(self.export_dir, 'actual_vs_predicted_total_tackles.png'))
        plt.close()

        # 2. Prediction Error vs Games Played
        fig, ax2 = plt.subplots(figsize=(9, 9))
        ax2.scatter(self.df['games_played'], self.error, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Games Played')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Prediction Error vs Games Played')
        for i, txt in enumerate(self.df['name']):
            ax2.annotate(txt, (self.df['games_played'].iloc[i], self.error.iloc[i]), fontsize=8)
        plt.savefig(os.path.join(self.export_dir, 'prediction_error_vs_games_played.png'))
        plt.close()

        # 3. Model Comparison
        fig, ax3 = plt.subplots(figsize=(9, 9))
        mse = [(self.df[f'{model}_predictions'] - self.df['total_tackles'])**2 for model in self.models]
        ax3.boxplot(mse, labels=self.models)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Squared Error')
        ax3.set_title('Model Performance Comparison')
        plt.savefig(os.path.join(self.export_dir, 'model_performance_comparison.png'))
        plt.close()

        # 4. Over-predictions
        fig, ax4 = plt.subplots(figsize=(9, 9))
        over_predictions = self.df[self.df[self.ensemble_column] > self.df['total_tackles']]
        ax4.scatter(over_predictions['total_tackles'], over_predictions[self.ensemble_column], alpha=0.6, color='orange')
        ax4.plot([0, over_predictions['total_tackles'].max()], [0, over_predictions['total_tackles'].max()], 'r--', lw=2)
        ax4.set_xlabel('Actual Total Tackles')
        ax4.set_ylabel('Predicted Total Tackles (Ensemble)')
        ax4.set_title('Over-predictions')
        for i, row in over_predictions.iterrows():
            ax4.annotate(row['name'], (row['total_tackles'], row[self.ensemble_column]), fontsize=8)
        plt.savefig(os.path.join(self.export_dir, 'over_predictions.png'))
        plt.close()

    def print_statistics(self):
        stats = {
            "Average Games Played": float(self.df['games_played'].mean()),
            "Correlation between Games Played and Prediction Error": float(self.df['games_played'].corr(self.error)),
            "Model Performance (Mean Squared Error)": {},
            "Over-prediction Statistics": {}
        }

        for model in self.models:
            mse = float(((self.df[f'{model}_predictions'] - self.df['total_tackles'])**2).mean())
            stats["Model Performance (Mean Squared Error)"][model] = mse

        over_predictions = self.df[self.df[self.ensemble_column] > self.df['total_tackles']]
        if not over_predictions.empty:
            stats["Over-prediction Statistics"] = {
                "Average Over-prediction": float((over_predictions[self.ensemble_column] - over_predictions['total_tackles']).mean()),
                "Maximum Over-prediction": float((over_predictions[self.ensemble_column] - over_predictions['total_tackles']).max()),
                "Minimum Over-prediction": float((over_predictions[self.ensemble_column] - over_predictions['total_tackles']).min())
            }

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        with open(os.path.join(self.export_dir, 'def_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=4)

    def analyze_predictions(self):
        overestimation = int((self.df[self.ensemble_column] > self.df['total_tackles']).sum())
        underestimation = int((self.df[self.ensemble_column] < self.df['total_tackles']).sum())
        analysis = {
            "Overestimation count": overestimation,
            "Underestimation count": underestimation
        }

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        with open(os.path.join(self.export_dir, 'def_prediction_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=4)

    def identify_accuracy(self):
        self.df['absolute_error'] = abs(self.df[self.ensemble_column] - self.df['total_tackles'])
        most_accurate = self.df.loc[self.df['absolute_error'].idxmin()]
        least_accurate = self.df.loc[self.df['absolute_error'].idxmax()]

        accuracy = {
            "Most accurate prediction": {
                "Player": most_accurate['name'],
                "Actual": float(most_accurate['total_tackles']),
                "Predicted": float(most_accurate[self.ensemble_column])
            },
            "Least accurate prediction": {
                "Player": least_accurate['name'],
                "Actual": float(least_accurate['total_tackles']),
                "Predicted": float(least_accurate[self.ensemble_column])
            }
        }

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
            
        with open(os.path.join(self.export_dir, 'def_accuracy_analysis.json'), 'w') as f:
            json.dump(accuracy, f, indent=4)

    def run_analysis(self):
        self.plot_analysis()
        self.print_statistics()
        self.analyze_predictions()
        self.identify_accuracy()
