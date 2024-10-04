import os
import json
import pandas

from scrapers.FootballRosterScraper import FootballRosterScraper
from scrapers.FootballGameStatsScraper import FootballGameStatsScraper
from engineering.FootballDataEngineeringCoordinatorV2 import FootballDataEngineeringCoordinatorV2

from machine_learning.Cleaning.GeneralCleaning import GeneralCleaning
from machine_learning.UniversalMachineLearningController import UniversalMachineLearningController


def main():
    """
    Main function of the program
    """

    #ensure the directories are created
    directories = [
        'data',
        'data/temp',
        'data/raw',
        'data/raw/roster',
        'data/engineered',
        'data/prepped',
        'data/ml_ready',
        'machine_learning',

    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    #clean the temp directory
    for file in os.listdir('./data/temp'):
        os.remove(f'./data/temp/{file}')


    #define the seasons to scrape, train, and test upon 
    all_seasons = ["2017", "2018", "2019","2021", "2022", "2023", "2024"]
    training_seasons = ["2017", "2018", "2019", "2021", "2022"]
    test_season = "2023"

    #scrape roster data from the web
    roster_scraper = FootballRosterScraper(
        seasons=all_seasons,
    )

    # roster_scraper.scrape()

    game_stats_scraper = FootballGameStatsScraper(
        seasons=all_seasons,
    )

    # game_stats_scraper.scrape()

    coordinator = FootballDataEngineeringCoordinatorV2(
        seasons=all_seasons,
        data_dir="data/raw/roster",
        stats_dir="data/raw/stats",
        save_dir="data/engineered",
        machine_learning_dir="data/prepped",
    )

    # coordinator.engineer_all_positions()

    #clean the data
    cleaner = GeneralCleaning()

    #load data from file
    wr_data = pandas.read_csv('data/prepped/wrs.csv')
    wr_data_cleaned = cleaner.clean_data(wr_data)
    
    
    # print(wr_data['temperature'].unique())

    #save the cleaned data
    wr_data_cleaned.to_csv('data/ml_ready/wrs.csv', index=False)

    #load WR features from file
    wr_features = {}
    with open('features/wr_features.json', 'r') as f:
        wr_features = json.load(f)

    #run machine learning predictions
    machine_learning_controller = UniversalMachineLearningController(
        data_path='data/ml_ready/wrs.csv',
        target='receiving_offensive_yds',
        features=wr_features,
        training_seasons=training_seasons,
        test_season=test_season,
        models_to_use=['LinearRegression', 'RandomForestRegressor', 'GradientBoostingRegressor']
    )

    machine_learning_controller.run_ensemble()



    
if __name__ == '__main__':
    main()
