import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class WRPredictionAnalysis:
    def __init__(self, csv_file, export_dir):
        self.df = pd.read_csv(csv_file)
        self.export_dir = export_dir
        self.error = self.df['Ensemble_predictions'] - self.df['total_touchdowns']
        self.models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest', 'Gradient Boosting', 'Ensemble']

    def plot_analysis(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
        fig.suptitle('Wide Receiver Prediction Analysis for 2023 Season', fontsize=16)

        # 1. Actual vs Predicted Touchdowns
        ax1.scatter(self.df['total_touchdowns'], self.df['Ensemble_predictions'], alpha=0.6)
        ax1.plot([0, self.df['total_touchdowns'].max()], [0, self.df['total_touchdowns'].max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Total Touchdowns')
        ax1.set_ylabel('Predicted Total Touchdowns (Ensemble)')
        ax1.set_title('Actual vs Predicted Total Touchdowns')
        for i, txt in enumerate(self.df['name']):
            ax1.annotate(txt, (self.df['total_touchdowns'][i], self.df['Ensemble_predictions'][i]), fontsize=8)

        # 2. Prediction Error vs Games Played
        ax2.scatter(self.df['games_played'], self.error, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Games Played')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Prediction Error vs Games Played')
        for i, txt in enumerate(self.df['name']):
            ax2.annotate(txt, (self.df['games_played'][i], self.error[i]), fontsize=8)

        # 3. Model Comparison
        mse = [(self.df[f'{model}_predictions'] - self.df['total_touchdowns'])**2 for model in self.models]
        ax3.boxplot(mse, labels=[m.split('_')[0] for m in self.models])
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Squared Error')
        ax3.set_title('Model Performance Comparison')

        plt.tight_layout()

        #ensure the export directory exists
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        plt.savefig(os.path.join(self.export_dir, 'wr_prediction_analysis.png'))
        plt.close()

    def print_statistics(self):
        stats = {
            "Average Games Played": float(self.df['games_played'].mean()),
            "Correlation between Games Played and Prediction Error": float(self.df['games_played'].corr(self.error)),
            "Model Performance (Mean Squared Error)": {}
        }
        for model in self.models:
            mse = float(((self.df[f'{model}_predictions'] - self.df['total_touchdowns'])**2).mean())
            stats["Model Performance (Mean Squared Error)"][model] = mse

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        with open(os.path.join(self.export_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=4)

    def analyze_predictions(self):
        overestimation = int((self.df['Ensemble_predictions'] > self.df['total_touchdowns']).sum())
        underestimation = int((self.df['Ensemble_predictions'] < self.df['total_touchdowns']).sum())
        analysis = {
            "Overestimation count": overestimation,
            "Underestimation count": underestimation
        }

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        with open(os.path.join(self.export_dir, 'prediction_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=4)

    def identify_accuracy(self):
        self.df['absolute_error'] = abs(self.df['Ensemble_predictions'] - self.df['total_touchdowns'])
        most_accurate = self.df.loc[self.df['absolute_error'].idxmin()]
        least_accurate = self.df.loc[self.df['absolute_error'].idxmax()]

        accuracy = {
            "Most accurate prediction": {
                "Player": most_accurate['name'],
                "Actual": float(most_accurate['total_touchdowns']),
                "Predicted": float(most_accurate['Ensemble_predictions'])
            },
            "Least accurate prediction": {
                "Player": least_accurate['name'],
                "Actual": float(least_accurate['total_touchdowns']),
                "Predicted": float(least_accurate['Ensemble_predictions'])
            }
        }

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
            
        with open(os.path.join(self.export_dir, 'accuracy_analysis.json'), 'w') as f:
            json.dump(accuracy, f, indent=4)

    def run_analysis(self):
        self.plot_analysis()
        self.print_statistics()
        self.analyze_predictions()
        self.identify_accuracy()

# Example usage:
# analysis = WRPredictionAnalysis('wr_ensemble_predictions_2023.csv', 'export_directory')
# analysis.run_analysis()