import json
from scraping.KSUFootballRosterScraper import KSUFootballRosterScraper
from scraping.KSUFootballStatsScraper import KSUFootballStatsScraper
from engineering.EngineerPlayerProfiles import EngineerPlayerProfiles
from engineering.EngineerPlayerStats import EngineerPlayerStats
from engineering.positional.QuarterbackEngineering import QuarterbackEngineering
from machine_learning.QBMachineLearningController import QBMachineLearningController

from sklearn.ensemble import RandomForestRegressor

import time


def main():
    """
    Main function of the program
    """

    start_time = time.time()

    seasons = ["2019","2021", "2022", "2023", "2024"]

    # Initiate the scraper
    roster_scraper = KSUFootballRosterScraper(seasons=seasons)

    #run the scraper
    print("Scraping KSU Football Roster")
    # roster_scraper.scrape()

    # Initiate the stats scraper
    stats_scraper = KSUFootballStatsScraper(seasons=seasons)

    # Run the stats scraper
    print("Scraping KSU Football Stats")
    # stats_scraper.scrape()

    # Initiate the player profiles
    player_profiles = EngineerPlayerProfiles(
        roster_data_dir="data/fetched/roster",
        stats_data_dir="data/fetched/stats",
        seasons=seasons
    )

    # Run the player profiles
    print("Creating Player Profiles")
    # player_profiles.create_profiles()

    # Initiate the player stats
    player_stats = EngineerPlayerStats(
        roster_data_dir="data/fetched/roster",
        stats_data_dir="data/fetched/stats",
        engineered_data_dir="data/engineered",
        seasons=seasons
    )

    # Run the player stats
    print("Creating Player Stats")
    # player_stats.get_player_stats()

    # Initiate the positional engineering
    qb_engineering = QuarterbackEngineering(
        seasons=seasons,
        seasons_to_aggregate=['2020', '2021', '2022'],
        data_dir="data/fetched/roster",
        save_dir="data/engineered/positions/quarterback"
    )

    # qb_engineering.engineer_quarterbacks()

    ml_controller = QBMachineLearningController(
        data_path="data/engineered/positions/quarterback/prepped_qb_data.csv",
    )
    
    ml_controller.run_ml_pipeline()
    

    end_time = time.time()

    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Program ran in {int(minutes)} minutes and {seconds:.2f} seconds")




if __name__ == '__main__':
    main()
