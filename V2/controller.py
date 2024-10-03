import json
from scraping.KSUFootballRosterScraper import KSUFootballRosterScraper
from scraping.KSUFootballStatsScraper import KSUFootballStatsScraper
from engineering.EngineerPlayerProfiles import EngineerPlayerProfiles
from engineering.EngineerPlayerStats import EngineerPlayerStats
from engineering.positional.QuarterbackEngineering import QuarterbackEngineering
from engineering.positional.RBEngineering import RBEngineering
from engineering.positional.WREngineering import WREngineering
from engineering.positional.DefensiveEngineering import DefensiveEngineering
from machine_learning.Quarterbacks.QBEnsembleLearning import QBEnsembleLearning
from machine_learning.Runningbacks.RBEnsembleLearning import RBEnsembleLearning
from machine_learning.WideReceivers.WREnsembleLearning import WREnsembleLearning
from machine_learning.WideReceivers.AnalyzeWRResults import WRPredictionAnalysis
from machine_learning.Defense.DefensiveEnsembleLearning import DefenseEnsembleLearning
from machine_learning.Defense.AnalyzeDEFResults import AnalyzeDefResults

from machine_learning.MultiPositionEnsembleLearning import EnsembleLearning

from sklearn.ensemble import RandomForestRegressor

import time


def main():
    """
    Main function of the program
    """

    start_time = time.time()

    seasons = ["2017", "2018", "2019","2021", "2022", "2023", "2024"]
    training_seasons = ["2019", "2021", "2022"]
    test_season = "2023"

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
        data_dir="data/fetched/roster",
        save_dir="data/engineered/positions/quarterback"
    )

    # qb_engineering.engineer_quarterbacks()

    rb_engineering = RBEngineering(
        seasons=seasons,
        data_dir="data/fetched/roster",
        save_dir="data/engineered/positions/runningback"
    )

    # rb_engineering.engineer_rbs()

    wr_engineering = WREngineering(
        seasons=seasons,
        data_dir="data/fetched/roster",
        save_dir="data/engineered/positions/widereceiver"
    )

    # wr_engineering.engineer_wrs()

    # Initiate the defensive engineering
    defense_engineering = DefensiveEngineering(
        seasons=seasons,
        data_dir="data/fetched/roster",
        save_dir="data/engineered/positions/defense"
    )

    # defense_engineering.engineer_defensive_players(
    #     min_games=0
    # )
    
    # qb_ensemble_learner = QBEnsembleLearning(
    #     data_path="data/engineered/positions/quarterback/prepped_qb_data.csv",
    #     predictions_path="data/predictions",
    #     training_seasons=training_seasons,
    #     test_season=test_season,
    #     training_data_dir="data/model_training/quarterback"
    # )

    # rb_ensemble_learner = RBEnsembleLearning(
    #     data_path="data/engineered/positions/runningback/prepped_rb_data.csv",
    #     predictions_path="data/predictions",
    #     training_seasons=training_seasons,
    #     test_season=test_season,
    #     training_data_dir="data/model_training/runningback"
    # )
    # wr_ensemble_learner = WREnsembleLearning(
    #     data_path="data/engineered/positions/widereceiver/prepped_wr_data.csv",
    #     predictions_path="data/predictions",
    #     training_seasons=training_seasons,
    #     test_season=test_season,
    #     training_data_dir="data/model_training/widereceiver"
    # )

    # defense_ensemble_learner = DefenseEnsembleLearning(
    #     data_path="data/engineered/positions/defense/prepped_defense_data.csv",
    #     predictions_path="data/predictions",
    #     training_seasons=training_seasons,
    #     test_season=test_season,
    #     training_data_dir="data/model_training/defense"
    # )

    defensive_features = [
        'prev_solo_tackles', 'prev_assisted_tackles', 'prev_total_tackles',
        'prev_tackles_for_loss', 'prev_tackles_for_loss_yards', 
        'prev_sacks', 'prev_sacks_yards', 'prev_interceptions',
        'prev_pass_deflections', 'prev_forced_fumbles', 'prev_fumble_recoveries', 'prev_blocked_kicks',
        'games_played', 'prev_games_played'
    ]

    
    defensive_model = EnsembleLearning(
        data_path="data/engineered/positions/defense/prepped_defense_data.csv",
        predictions_path="data/predictions",
        analytics_path="data/model_training/analytics/defense",
        training_seasons=training_seasons,
        test_season=test_season,
        training_data_dir="data/model_training/defense",
        features=defensive_features,
        target="total_tackles",
        position="DEF"
    )

    # defensive_model.run_ensemble_pipeline()

    qb_features = [
        'prev_games_played', 'prev_games_started', 'prev_passing_yards',
        'prev_pass_completions', 'prev_pass_attempts', 'prev_interceptions_thrown',
        'prev_passing_touchdowns', 'prev_longest_completion',
        'prev_pass_completion_percentage', 'prev_average_yards_per_pass',
        'prev_average_passing_yards_per_game', 'prev_rushing_attempts',
        'prev_rushing_yards', 'prev_rushing_touchdowns', 'prev_longest_rush',
        'prev_average_yards_per_rush', 'prev_average_rushing_yards_per_game',
        'prev_total_touchdowns'
    ]

    qb_model = EnsembleLearning(
        data_path="data/engineered/positions/quarterback/prepped_qb_data.csv",
        predictions_path="data/predictions",
        analytics_path="data/model_training/analytics/quarterback",
        training_seasons=training_seasons,
        test_season=test_season,
        training_data_dir="data/model_training/quarterback",
        features=qb_features,
        target="total_touchdowns",
        position="QB"
    )

    qb_model.run_ensemble_pipeline()

    rb_features = [
        'prev_games_played', 'prev_games_started', 'prev_rushing_attempts',
        'prev_rushing_yards', 'prev_rushing_touchdowns', 'prev_longest_rush',
        'prev_average_yards_per_rush', 'prev_average_rushing_yards_per_game',
        'prev_total_touchdowns'
    ]

    rb_model = EnsembleLearning(
        data_path="data/engineered/positions/runningback/prepped_rb_data.csv",
        predictions_path="data/predictions",
        analytics_path="data/model_training/analytics/runningback",
        training_seasons=training_seasons,
        test_season=test_season,
        training_data_dir="data/model_training/runningback",
        features=rb_features,
        target="total_touchdowns",
        position="RB"
    )

    # rb_model.run_ensemble_pipeline()

    features = [
        'prev_games_played', 'prev_games_started', 'prev_receptions',
        'prev_receiving_yards', 'prev_receiving_touchdowns', 'prev_longest_reception',
        'prev_receptions_per_game', 'prev_average_yards_per_reception',
        'prev_average_receiving_yards_per_game', 'prev_total_touchdowns', 'prev_total_points'
    ]

    wr_model = EnsembleLearning(
        data_path="data/engineered/positions/widereceiver/prepped_wr_data.csv",
        predictions_path="data/predictions",
        analytics_path="data/model_training/analytics/widereceiver",
        training_seasons=training_seasons,
        test_season=test_season,
        training_data_dir="data/model_training/widereceiver",
        features=features,
        target="total_touchdowns",
        position="WR"
    )

    # wr_model.run_ensemble_pipeline()


    end_time = time.time()

    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Program ran in {int(minutes)} minutes and {seconds:.2f} seconds")




if __name__ == '__main__':
    main()
