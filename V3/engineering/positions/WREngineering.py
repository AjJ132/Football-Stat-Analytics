import json
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class WREngineering:
    def __init__(self, seasons, data_dir, stats_dir, save_dir, base_features):
        self.seasons = seasons
        self.data_dir = data_dir
        self.stats_dir = stats_dir
        self.save_dir = save_dir
        self.base_features = base_features
        self.receivers_features = [
            "receiving_stats.longest",
            "receiving_stats.per_game",
            "receiving_stats.recep",
            "receiving_stats.touchdowns",
            "receiving_stats.yards",
            "receiving_stats.yards_comp_pct",
            "receiving_stats.yards_game_pct",
            "scoring_stats.point_avg",
            "scoring_stats.points",
            "scoring_stats.rcpt_td",
            "scoring_stats.rush_td",
            "scoring_stats.touchdowns"
        ]
        self.game_data_features = [
            "is_home_game",
            "date",
            "opponent",
            "weather",
            "temperature",
            "wind",
            "duration",
            "kickoff_time",
            "attendance",
            "total_points_scored",
            "average_points_scored",
            "total_first_downs",
            "passing_first_downs",
            "team_qb_passing_completions",
            "team_qb_passing_attempts",
            "team_qb_passing_interceptions",
            "team_qb_passing_avg_per_attempt",
            "team_qb_passing_avg_per_completion",
            "team_qb_passing_touchdowns",
            "team_total_offense_yards",
            "team_total_offense_plays"
        ]
        self.all_features = self.base_features + self.receivers_features + self.game_data_features

    def parse_player_name(self, player_name):
        name_parts = player_name.replace(" ", "").replace(",", "_").split("_")
        
        if len(name_parts) < 2:
            return player_name
        
        name_parts = [part.capitalize() for part in name_parts]
        return f"{name_parts[0]}, {' '.join(name_parts[1:])}"

    def get_game_data(self, season):
        game_data = []
        game_by_game_dir = f"{self.stats_dir}/{season}/game_by_game"

        for folder in os.listdir(game_by_game_dir):
            game_data_file = f"{game_by_game_dir}/{folder}/game_data.json"
            try:
                with open(game_data_file, "r") as f:
                    game_data.append({folder: json.load(f)})
            except FileNotFoundError:
                print(f"Game data file not found: {game_data_file}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {game_data_file}")

        return game_data

    def process_game(self, game, players_df):
        game_name = list(game.keys())[0]
        game_name_split = game_name.split("_vs") if "_vs" in game_name else game_name.split("_at")
        date_split = game_name_split[0].split("-")

        is_home_game = "_vs" in game_name
        opponent_name = game_name_split[1].strip().replace("_", " ")

        game_info = game[game_name]['box_score']['gameinfo']

        weather = game_info.get('weather')
        temperature = game_info.get('temperature')
        wind = game_info.get('wind')
        duration = game_info.get('duration')
        kickoff_time = game_info.get('kickoff_time')
        attendance = game_info.get('attendance')

        opponent_history = game_info.get('opponenthistory', {})
        total_points_scored = opponent_history.get('total_points')
        average_points_scored = opponent_history.get('average_points')

        home_team_stats_section = game[game_name]['team_stats']['hometeam']
        home_team_stats = self.extract_team_stats(home_team_stats_section)

        individual_stats = game[game_name].get('individual_stats', {})

        players_game_data = []
        for player_name, player_stats in individual_stats.items():
            if player_name == 'team':
                continue

            player_name_formatted = self.parse_player_name(player_name)

            player_name_lower = player_name_formatted.lower().replace(" ", "")
            if player_name_lower not in players_df['name'].str.lower().str.replace(" ", "").values:
                continue

            player_game_stats = {
                'name': player_name_formatted,
                'game': game_name,
                'is_home_game': is_home_game,
                'date': f"{date_split[1]}/{date_split[0]}/{date_split[2]}",
                'opponent': opponent_name,
                'weather': weather,
                'temperature': temperature,
                'wind': wind,
                'duration': duration,
                'kickoff_time': kickoff_time,
                'attendance': attendance,
                'total_points_scored': total_points_scored,
                'average_points_scored': average_points_scored,
            }

            player_game_stats.update(home_team_stats)

            for stat_type, stat_data in player_stats.items():
                if isinstance(stat_data, dict):
                    for sub_type, sub_data in stat_data.items():
                        for key, value in sub_data.items():
                            player_game_stats[f"{stat_type}_{sub_type}_{key}"] = value
                else:
                    player_game_stats[stat_type] = stat_data

            players_game_data.append(player_game_stats)

        return players_game_data

    def get_players_game_by_game_data(self, game_data, players_df):
        players_game_data = []
    
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.process_game, game, players_df) for game in game_data]
    
            for future in as_completed(futures):
                players_game_data.extend(future.result())
    
        return pd.DataFrame(players_game_data)

    def extract_team_stats(self, team_stats):
        extracted_stats = {}
        
        if 'first_downs' in team_stats:
            extracted_stats['total_first_downs'] = team_stats['first_downs'].get('total')
            extracted_stats['passing_first_downs'] = team_stats['first_downs'].get('passing')
        
        if 'passing' in team_stats:
            passing_stats = team_stats['passing']
            comp_att_int = passing_stats.get('comp.-att.-int.', '').split('-')
            extracted_stats['team_qb_passing_completions'] = comp_att_int[0] if len(comp_att_int) > 0 else None
            extracted_stats['team_qb_passing_attempts'] = comp_att_int[1] if len(comp_att_int) > 1 else None
            extracted_stats['team_qb_passing_interceptions'] = comp_att_int[2] if len(comp_att_int) > 2 else None
            extracted_stats['team_qb_passing_avg_per_attempt'] = passing_stats.get('avg._/_att.')
            extracted_stats['team_qb_passing_avg_per_completion'] = passing_stats.get('avg._/_comp.')
            extracted_stats['team_qb_passing_touchdowns'] = passing_stats.get('tds')

        if 'total_offense' in team_stats:
            extracted_stats['team_total_offense_yards'] = team_stats['total_offense'].get('yards')
            extracted_stats['team_total_offense_plays'] = team_stats['total_offense'].get('plays')

        return extracted_stats

    def filter_players_data(self, players_df):
        columns_to_keep = [col for col in players_df.columns 
                           if col in self.all_features or 'receiving' in col]
        filtered_df = players_df.loc[:, columns_to_keep]
        return filtered_df.reset_index(drop=True)

    def engineer_wrs(self):
        for season in tqdm(self.seasons, desc="Processing seasons"):
            game_data = self.get_game_data(season)
            roster_file = f"{self.data_dir}/{season}/ksu_football_roster_{season}.json"
            
            try:
                with open(roster_file, "r") as f:
                    roster = json.load(f)
            except FileNotFoundError:
                print(f"Roster file for season {season} not found: {roster_file}")
                continue
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {roster_file}")
                continue

            if not roster:
                print(f"Roster for season {season} is empty.")
                continue

            players_data = [
                {**{feature: player.get(feature) for feature in self.base_features if feature in player},
                 'is_primary_receiver': player.get('position', 'WR') == 'WR'}
                for player in roster
                if any(key.startswith("receiving_stats") for key in player.keys())
            ]

            players_df = pd.DataFrame(players_data)
            players_game_data_df = self.get_players_game_by_game_data(game_data, players_df)
            players_df = self.filter_players_data(players_game_data_df)
            players_df.insert(1, 'season', season)
            players_df = players_df.sort_values(by=['name', 'season'])

            save_path = f"{self.save_dir}/{season}"
            os.makedirs(save_path, exist_ok=True)
            players_df.to_csv(f"{save_path}/wrs.csv", index=False)

        print("WR engineering completed for all seasons.")

