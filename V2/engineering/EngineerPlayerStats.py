import json
import os
import hashlib
from collections import defaultdict

class EngineerPlayerStats:
    def __init__(self, roster_data_dir, stats_data_dir, engineered_data_dir, seasons):
        self.roster_data_dir = roster_data_dir
        self.stats_data_dir = stats_data_dir
        self.engineered_data_dir = engineered_data_dir
        self.seasons = seasons
        self.player_profiles = None
        self.game_id_pairs = {}

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def normalize_name(self, name):
        name = name.replace('_', ' ').upper()
        parts = name.split(',') if ',' in name else name.split()
        return ', '.join(parts[::-1] if len(parts) == 2 and ',' not in name else parts)

    def parse_game_data(self, game_data, opponent):
        game_info = game_data["box_score"]["gameinfo"]
        date = game_info["date"]
        game_id = hashlib.sha256(f"{date}_{opponent}".encode('utf-8')).hexdigest()
        
        self.game_id_pairs[game_id] = opponent

        participation = game_data["participation"]
        starters = set(self.normalize_name(name) for name in participation["starters"]["hometeam"])
        player_participation = set(self.normalize_name(name) for name in participation["playerparticipation"]["hometeam"])
        
        individual_stats = game_data["individual_stats"]

        for player in self.player_profiles:
            normalized_name = self.normalize_name(player["player_info"]["Name"])
            
            if normalized_name in starters or normalized_name in player_participation:
                performance_object = {
                    'game_id': game_id,
                    'opponent': opponent,
                    'starter': normalized_name in starters,
                    'stats': self.handle_parse_game_participation(normalized_name, individual_stats)
                }
                player["performances"].append(performance_object)

    def handle_parse_game_participation(self, normalized_name, individual_stats):
        for player_name, stats in individual_stats.items():
            if self.normalize_name(player_name) == normalized_name:
                return {stat_name.lower().replace(' ', '_').replace('/', '_').replace('-', '_'): stat_value 
                        for stat_name, stat_value in stats.items()}
        return {}

    def get_player_stats(self):
        self.player_profiles = self.load_data(os.path.join(self.engineered_data_dir, "player_profiles.json"))
        for player in self.player_profiles:
            player["performances"] = []

        for season in self.seasons:
            path = os.path.join(self.stats_data_dir, season, "game_by_game")
            team_folders = os.listdir(path)

            for team_folder in team_folders:
                opponent = team_folder.split("_vs" if "_vs" in team_folder else "_at")[1].strip()
                game_data = self.load_data(os.path.join(path, team_folder, "game_data.json"))
                self.parse_game_data(game_data, opponent)

        #save to player_profiles.json
        #overwrite the file
        with open(os.path.join(self.engineered_data_dir, "player_profiles.json"), "w") as f:
            json.dump(self.player_profiles, f, indent=4)