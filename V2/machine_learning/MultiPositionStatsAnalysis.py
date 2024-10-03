import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd



class ResultsAnalyzer:
    def __init__(self, df, export_dir, position, target):
        self.df = df
        self.export_dir = export_dir
        self.position = position
        self.target = target
        self.ensemble_column = 'Weighted Ensemble_predictions'
        if self.ensemble_column not in self.df.columns:
            raise ValueError(f"Column '{self.ensemble_column}' not found in the DataFrame.")
        self.error = self.df[self.ensemble_column] - self.df[self.target]
        self.abs_error = np.abs(self.error)
        self.models = [col.replace('_predictions', '') for col in self.df.columns if col.endswith('_predictions')]

    def plot_analysis(self):

        #ensure export directory exists
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        # 1. Actual vs Predicted Target
        fig, ax1 = plt.subplots(figsize=(9, 9))
        ax1.scatter(self.df[self.target], self.df[self.ensemble_column], alpha=0.6)
        ax1.plot([0, self.df[self.target].max()], [0, self.df[self.target].max()], 'r--', lw=2)
        ax1.set_xlabel(f'Actual {self.target.replace("_", " ").title()}')
        ax1.set_ylabel(f'Predicted {self.target.replace("_", " ").title()} (Ensemble)')
        ax1.set_title(f'Actual vs Predicted {self.target.replace("_", " ").title()}')
        for i, txt in enumerate(self.df['name']):
            ax1.annotate(txt, (self.df[self.target].iloc[i], self.df[self.ensemble_column].iloc[i]), fontsize=8)
        plt.savefig(os.path.join(self.export_dir, f'{self.position}_actual_vs_predicted_{self.target}.png'))
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
        plt.savefig(os.path.join(self.export_dir, f'{self.position}_prediction_error_vs_games_played.png'))
        plt.close()

        # 3. Model Comparison
        fig, ax3 = plt.subplots(figsize=(9, 9))
        mse = [(self.df[f'{model}_predictions'] - self.df[self.target])**2 for model in self.models]
        ax3.boxplot(mse, labels=self.models)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Squared Error')
        ax3.set_title('Model Performance Comparison')
        plt.savefig(os.path.join(self.export_dir, f'{self.position}_model_performance_comparison.png'))
        plt.close()

    def print_statistics(self):
        stats = {
            "Average Games Played": float(self.df['games_played'].mean()),
            "Correlation between Games Played and Prediction Error": float(self.df['games_played'].corr(self.error)),
            "Model Performance (Mean Squared Error)": {}
        }

        for model in self.models:
            mse = float(((self.df[f'{model}_predictions'] - self.df[self.target])**2).mean())
            stats["Model Performance (Mean Squared Error)"][model] = mse

        with open(os.path.join(self.export_dir, f'{self.position}_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=4)

    def analyze_predictions(self):
        overestimation = int((self.df[self.ensemble_column] > self.df[self.target]).sum())
        underestimation = int((self.df[self.ensemble_column] < self.df[self.target]).sum())
        analysis = {
            "Overestimation count": overestimation,
            "Underestimation count": underestimation
        }

        with open(os.path.join(self.export_dir, f'{self.position}_prediction_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=4)

    def identify_accuracy(self):
        self.df['absolute_error'] = abs(self.df[self.ensemble_column] - self.df[self.target])
        most_accurate = self.df.loc[self.df['absolute_error'].idxmin()]
        least_accurate = self.df.loc[self.df['absolute_error'].idxmax()]

        accuracy = {
            "Most accurate prediction": {
                "Player": most_accurate['name'],
                "Actual": float(most_accurate[self.target]),
                "Predicted": float(most_accurate[self.ensemble_column])
            },
            "Least accurate prediction": {
                "Player": least_accurate['name'],
                "Actual": float(least_accurate[self.target]),
                "Predicted": float(least_accurate[self.ensemble_column])
            }
        }

        with open(os.path.join(self.export_dir, f'{self.position}_accuracy_analysis.json'), 'w') as f:
            json.dump(accuracy, f, indent=4)

    def plot_error_distribution(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Box plot
        sns.boxplot(x=self.abs_error, ax=ax1)
        ax1.set_title('Distribution of Absolute Prediction Errors')
        ax1.set_xlabel('Absolute Error')
        
        # Add statistical annotations
        stats_text = f"Mean: {self.abs_error.mean():.2f}\n"
        stats_text += f"Median: {self.abs_error.median():.2f}\n"
        stats_text += f"Min: {self.abs_error.min():.2f}\n"
        stats_text += f"Max: {self.abs_error.max():.2f}\n"
        stats_text += f"Q1: {self.abs_error.quantile(0.25):.2f}\n"
        stats_text += f"Q3: {self.abs_error.quantile(0.75):.2f}"
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Histogram with KDE
        sns.histplot(self.abs_error, kde=True, ax=ax2)
        ax2.set_title('Histogram and KDE of Absolute Prediction Errors')
        ax2.set_xlabel('Absolute Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, f'{self.position}_error_distribution.png'))
        plt.close()

    def calculate_heat_map_stat(self):
        # Calculate normalized error (error divided by actual value)
        self.df['normalized_error'] = self.error / self.df[self.target]
        
        # Group by relevant features (e.g., position and games_played ranges)
        self.df['games_played_bin'] = pd.cut(self.df['games_played'], bins=5)
        heat_map_data = self.df.groupby(['name', 'games_played_bin'])['normalized_error'].mean().unstack()
        
        # Save heat map data
        heat_map_data.to_csv(os.path.join(self.export_dir, f'{self.position}_heat_map_data.csv'))
        
        return heat_map_data

    def run_analysis(self):
        self.plot_analysis()
        self.print_statistics()
        self.analyze_predictions()
        self.identify_accuracy()
        self.plot_error_distribution()
        heat_map_data = self.calculate_heat_map_stat()
        
        # You can use this heat_map_data to create a heat map visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(heat_map_data, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title(f'Normalized Error by Position and Games Played ({self.position})')
        plt.savefig(os.path.join(self.export_dir, f'{self.position}_error_heatmap.png'))
        plt.close()

