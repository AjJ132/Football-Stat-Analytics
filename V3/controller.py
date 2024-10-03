import os
import json

from scrapers.FootballRosterScraper import FootballRosterScraper
from scrapers.FootballGameStatsScraper import FootballGameStatsScraper
from engineering.FootballDataEngineeringCoordinatorV2 import FootballDataEngineeringCoordinatorV2


def main():
    """
    Main function of the program
    """

    #ensure the directories are created
    directories = [
        'data',
        'data/raw',
        'data/raw/roster',
        'data/engineered',
        'temp',
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    #clean the temp directory
    for file in os.listdir("temp"):
        os.remove(f"temp/{file}")


    #define the seasons to scrape, train, and test upon 
    all_seasons = ["2017", "2018", "2019","2021", "2022", "2023", "2024"]
    training_seasons = ["2019", "2021", "2022"]
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
        save_dir="data/engineered",
    )

    coordinator.engineer_all_positions()





    
if __name__ == '__main__':
    main()
