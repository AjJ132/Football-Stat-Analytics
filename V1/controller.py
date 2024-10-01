import os
import json
from scraping.KSUFootballScraper import KSUFootballScraper
from engineering.EngineerData import EngineerData
from model_training.KSUFootballMLPredictor import KSUFootballMLPredictor
from model_evaluating.KSUFootballMLEvaluator import KSUFootballMLEvaluator
from team_info_processor.TeamInfoProcessor import TeamInfoProcessor
from model_evaluating.KSUFootballRecruitEvaluator import KSUFootballRecruitEvaluator
from model_evaluating.KSUFootballSeasonAnalyzer import KSUFootballSeasonAnalyzer

def main():
    """
    Main function of the program
    """
    print("Starting KSU Football Data Processing Pipeline")

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_data_path = os.path.join(base_dir, 'data', 'fetched', 'ksu_football')
    output_csv_path = os.path.join(base_dir, 'data', 'engineered', 'ksu_football_ml_data.csv')
    output_json_dir = os.path.join(base_dir, 'data', 'engineered', 'ksu_football')
    predictions_path = os.path.join(base_dir, 'data', 'predictions', 'ksu_football_2023_predictions.json')
    metrics_output_path = os.path.join(base_dir, 'data', 'evaluation', 'ksu_football_model_metrics.json')
    visualizations_output_dir = os.path.join(base_dir, 'data', 'evaluation', 'visualizations')
    season_data_dir = os.path.join(base_dir, 'data', 'season_data')
    team_schedule_path = os.path.join(base_dir, 'data', 'team_info', 'team_schedule.json')
    accuracy_report_path = os.path.join(base_dir, 'data', 'evaluation', 'ksu_football_prediction_accuracy.json')
    models_dir = os.path.join(base_dir, 'data', 'models')
    recruits_file = os.path.join(base_dir, 'data', 'recruiting', '2023_recruits.json')
    recruit_predictions_path = os.path.join(base_dir, 'data', 'predictions', 'ksu_football_2023_recruit_predictions.json')
    sos_file = os.path.join(season_data_dir, 'strength_of_schedule.json')
    ml_accuracy_report_path = os.path.join(base_dir, 'data', 'evaluation', 'ksu_football_ml_accuracy.json')

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    os.makedirs(visualizations_output_dir, exist_ok=True)
    os.makedirs(season_data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Process team info and calculate strength of schedule
    print("Processing team info and calculating strength of schedule...")
    team_info_processor = TeamInfoProcessor(team_schedule_path, season_data_dir)
    team_info_processor.process()

    # Initiate KSUFootballScraper object
    ksu_scraper = KSUFootballScraper()

    # Run scrape method (uncomment when ready to scrape fresh data)
    print("Scraping KSU Football data...")
    # ksu_scraper.scrape()
    print("Scraping completed.")

    # Initiate EngineerData object
    print("Starting data engineering process...")
    engineer = EngineerData(input_data_path)

    # Run process_data method
    # engineer.process_data(output_csv_path, output_json_dir)
    print("Data engineering process completed.")

    # Initiate KSUFootballMLPredictor object
    print("Starting prediction process...")
    predictor = KSUFootballMLPredictor(output_json_dir, season_data_dir)

    # Run prediction pipeline
    # predictor.run_prediction_pipeline(predictions_path, models_dir)
    print("Prediction process completed.")

    # Initiate KSUFootballMLEvaluator object
    print("Starting model evaluation process...")
    evaluator = KSUFootballMLEvaluator(output_json_dir, predictions_path, models_dir, './data/season_data/strength_of_schedule.json')   

    # Run evaluation pipeline
    evaluator.run_evaluation_pipeline(metrics_output_path, visualizations_output_dir, accuracy_report_path, ml_accuracy_report_path)
    print("Model evaluation process completed.")

    # Initiate KSUFootballRecruitEvaluator object
    print("Starting recruit evaluation process...")
    recruit_evaluator = KSUFootballRecruitEvaluator(models_dir, recruits_file, sos_file)

    # Run recruit evaluation pipeline
    recruit_evaluator.run_evaluation_pipeline(recruit_predictions_path)
    print("Recruit evaluation process completed.")

    # Initialize and run the season analyzer
    print("Starting season analysis process...")
    season_analyzer = KSUFootballSeasonAnalyzer(
        data_dir=os.path.join(base_dir, 'data', 'engineered', 'ksu_football'),
        models_dir=models_dir,
        output_dir=os.path.join(base_dir, 'data', 'analysis'),
        sos_file='./data/season_data/strength_of_schedule.json'
    )
    # season_analyzer.run_analysis()
    print("Season analysis process completed.")

if __name__ == '__main__':
    main()