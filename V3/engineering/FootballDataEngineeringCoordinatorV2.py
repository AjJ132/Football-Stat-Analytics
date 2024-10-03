from engineering.positions.WREngineering import WREngineering
import os

class FootballDataEngineeringCoordinatorV2:
    def __init__(self, seasons, data_dir, save_dir):
        self.seasons = seasons
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.base_features = [
            "position",
            "major",
            "height",
            "weight",
            "year",
            "hometown",
            "season",
            "name",
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
        wr_engineering = WREngineering(self.seasons, self.data_dir, self.save_dir, base_features=self.base_features)
        wr_engineering.engineer_wrs()



