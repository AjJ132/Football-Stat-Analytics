from engineering.positions.WREngineering import WREngineering
import os
import pandas

class FootballDataEngineeringCoordinatorV2:
    def __init__(self, seasons, data_dir, stats_dir, save_dir, machine_learning_dir):
        self.seasons = seasons
        self.data_dir = data_dir # data/raw/roster
        self.stats_dir = stats_dir # data/raw/stats
        self.save_dir = save_dir # data/engineered
        self.machine_learning_dir = machine_learning_dir # data/prepped
        self.base_features = [
            "name",
            "season",
            "year",
            "position",
            "height",
            "weight",
            "major",
            "hometown",
            "gp",
            "gs",
        ]
        
        # Ensure save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def engineer_all_positions(self):
        """
        Coordinate the engineering process for all positions.
        """
        self.engineer_wide_receivers()


    def engineer_wide_receivers(self):
        """
        Engineer wide receiver data.
        """
        print("Engineering wide receivers...")
        wr_engineering = WREngineering(self.seasons, 
                                       data_dir=self.data_dir, 
                                       stats_dir=self.stats_dir,
                                       save_dir=self.save_dir, 
                                       base_features=self.base_features)
        wr_engineering.engineer_wrs()

        #foreach season, combbine the data into one dataframe

        combined_df = pandas.DataFrame()

        #loop over each season and go to the engineered directory and read the wrs.csv file
        for season in self.seasons:
            season_df = pandas.read_csv(f"{self.save_dir}/{season}/wrs.csv")
            combined_df = combined_df._append(season_df)

        #sort by name and season
        combined_df = combined_df.sort_values(by=['name', 'season'])

        #save the combined dataframe to the machine learning directory
        combined_df.to_csv(f"{self.machine_learning_dir}/wrs.csv", index=False)
        



