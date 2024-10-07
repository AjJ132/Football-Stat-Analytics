import json
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime

class WREngineering:
    def __init__(self, seasons, data_dir, stats_dir, save_dir, base_features):
        self.seasons = seasons
        self.data_dir = data_dir
        self.stats_dir = stats_dir
        self.save_dir = save_dir
        self.base_features = base_features
        self.historical_features = [
            #prev WR features
            "prev_game_receiving_yards",
            "prev_game_receptions",
            "prev_game_touchdowns",
            "prev_game_longest_reception",
            "prev_first_downs",
            "prev_passing_first_downs",
            #prev QB features
            "prev_qb_passing_completions",
            "prev_qb_passing_attempts",
            "prev_qb_passing_interceptions",
            "prev_qb_passing_avg_per_attempt",
            "prev_qb_passing_avg_per_completion",
            "prev_qb_passing_touchdowns",
            "prev_total_offense_yards",
            "prev_total_offense_plays",
            #begin weather features
            "prev_game_weather",
            "prev_game_temperature",
            "prev_game_wind",
            #prev game features
            "prev_game_opponent",
            "prev_game_duration",
            "prev_game_kickoff_time",
            "prev_game_attendance",
            "prev_game_total_points_scored",
            "prev_is_home_game",
            "prev_game_average_points_scored",
            "prev_game_total_points_scored",
            "prev_game_total_first_downs",
            "prev_game_passing_first_downs",
            #last 5 games features
            "last_5_games_avg_receiving_yards",
            "last_5_games_avg_receptions",
            "last_5_games_avg_touchdowns",
            "last_5_games_avg_targets",
            "last_5_games_catch_rate",
            #season total features
            "season_total_receiving_yards",
            "season_total_receptions",
            "season_total_touchdowns",
            "season_avg_yards_per_reception",
            "season_avg_yards_per_game",
        ]
        self.upcoming_game_features = [
            "is_home_game",
            "opponent",
            "kickoff_time",
            "days_since_last_game",
        ]
        self.team_features = [
            "team_season_avg_passing_yards",
            "team_season_avg_points_scored",
            "team_season_avg_first_downs",
            "team_qb_season_avg_passing_yards",
            "team_qb_season_avg_touchdowns",
        ]
        # self.opponent_features = [ I would love to have this but this involves scraping data from the web and would be very time consuming
        #     "opponent_season_avg_passing_yards_allowed",
        #     "opponent_season_avg_points_allowed",
        #     "opponent_season_avg_first_downs_allowed",
        # ]
        self.weather_forecast_features = [
            "forecast_temperature",
            "forecast_wind_speed",
            "forecast_precipitation_chance",
            "forecast_is_dome",
        ]
        self.all_features = (self.base_features + self.historical_features + 
                             self.upcoming_game_features + self.team_features  + self.weather_forecast_features)
        self.temp_median = None
        self.mlb = MultiLabelBinarizer()

    def save_features(self):
        with open('features/wr_features.json', 'w') as f:
            json.dump(self.all_features, f, indent=2)

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

    def parse_player_name(self, player_name):
        name_parts = player_name.replace(" ", "").replace(",", "_").split("_")
        
        if len(name_parts) < 2:
            return player_name
        
        name_parts = [part.capitalize() for part in name_parts]
        return f"{name_parts[0]}, {' '.join(name_parts[1:])}"


    def clean_weather(self, data):
        data['wind'] = data['wind'].fillna(0)
        
        def extract_wind_speed(wind_string):
            if isinstance(wind_string, (int, float)):
                return wind_string
            if isinstance(wind_string, str):
                match = re.search(r'\d+', wind_string)
                if match:
                    return int(match.group())
            return 0

        data['wind'] = data['wind'].apply(extract_wind_speed)
        
        data['temperature'] = data['temperature'].apply(self.parse_temperature)
        
        self.temp_median = data['temperature'].median()
        data['temp_was_missing'] = data['temperature'].isna().astype(int)
        data['temperature'] = data['temperature'].fillna(self.temp_median)

        data = data[['temperature', 'temp_was_missing'] + [col for col in data.columns if col not in ['temperature', 'temp_was_missing']]]
        
        return data

    def parse_temperature(self, temp):
        if pd.isna(temp):
            return np.nan
        
        if isinstance(temp, (int, float)):
            return temp
        
        temp = str(temp).lower()
        
        match = re.search(r'-?\d+', temp)
        if match:
            value = int(match.group())
            
            if 'f' in temp:
                return (value - 32) * 5/9
            
            if 'mid' in temp:
                return value + 5
            
            if temp.endswith('s'):
                return value + 5
            
            return value
        
        return np.nan

    def process_weather_column(self, data):
        data['weather'] = data['weather'].fillna('Unknown').str.lower()

        def extract_main_condition(weather):
            conditions = []
            if 'sun' in weather or 'clear' in weather:
                conditions.append('sunny')
            if 'cloud' in weather:
                conditions.append('cloudy')
            if 'rain' in weather:
                conditions.append('rainy')
            if 'snow' in weather:
                conditions.append('snowy')
            if 'overcast' in weather:
                conditions.append('overcast')
            if not conditions:
                conditions.append('other')
            return conditions

        data['is_windy'] = data['weather'].str.contains('wind').astype(int)
        data['is_foggy'] = data['weather'].str.contains('fog').astype(int)

        def cloud_cover_scale(weather):
            if 'clear' in weather or 'sunny' in weather:
                return 0
            elif 'partly' in weather:
                return 1
            elif 'mostly' in weather:
                return 2
            elif 'overcast' in weather or 'cloudy' in weather:
                return 3
            else:
                return np.nan

        data['cloud_cover'] = data['weather'].apply(cloud_cover_scale)
        data['cloud_cover'] = data['cloud_cover'].fillna(data['cloud_cover'].median())

        main_conditions = data['weather'].apply(extract_main_condition)
        encoded_conditions = self.mlb.fit_transform(main_conditions)
        condition_columns = self.mlb.classes_
        for i, condition in enumerate(condition_columns):
            data[f'weather_{condition}'] = encoded_conditions[:, i]

        data = data.drop('weather', axis=1)

        return data
    
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


    def sort_games_chronologically(self, game_data):
            sorted_games = []
            for game in game_data:
                game_name = list(game.keys())[0]
                date_str = game_name.split("_")[0]
                try:
                    game_date = datetime.strptime(date_str, "%m-%d-%Y")
                except ValueError:
                    print(f"Game Name: {game_name}")
                    print(f"Date String: {date_str}")
                    raise
                sorted_games.append((game_date, game))
            return [game for _, game in sorted(sorted_games)]

    def process_game_v2(self, game, players_df, prev_game=None, next_game=None):
        game_name = list(game.keys())[0]
        game_date = datetime.strptime(game_name.split("_")[0], "%m-%d-%Y")
        is_home_game = "_vs" in game_name
        opponent_name = game_name.split("_vs" if is_home_game else "_at")[1].strip().replace("_", " ")

        game_info = game[game_name]['box_score']['gameinfo']
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
                'prev_game': game_name,
                'prev_game_date': game_date,
                'upcoming_is_home_game': is_home_game,
                'upcoming_opponent': opponent_name,
            }

            # Add previous game data
            if prev_game:
                player_game_stats.update(self.get_previous_game_data(prev_game, player_name_formatted))

            # Add upcoming game data
            if next_game:
                player_game_stats.update(self.get_upcoming_game_data(next_game, game_date))

            players_game_data.append(player_game_stats)

        return players_game_data

    def get_previous_game_data(self, prev_game, player_name):
        prev_game_name = list(prev_game.keys())[0]
        prev_game_info = prev_game[prev_game_name]['box_score']['gameinfo']
        prev_game_stats = prev_game[prev_game_name]['individual_stats'].get(player_name, {})
        
        prev_data = {
            'prev_weather': prev_game_info.get('weather'),
            'prev_temperature': prev_game_info.get('temperature'),
            'prev_wind': prev_game_info.get('wind'),
            'prev_duration': prev_game_info.get('duration'),
            'prev_kickoff_time': prev_game_info.get('kickoff_time'),
            'prev_attendance': prev_game_info.get('attendance'),
            'prev_total_points_scored': prev_game_info.get('opponenthistory', {}).get('total_points'),
            'prev_average_points_scored': prev_game_info.get('opponenthistory', {}).get('average_points'),
        }
        
        receiving_stats = prev_game_stats.get('receiving', {})
        prev_data.update({
            'prev_receiving_yards': receiving_stats.get('yards', 0),
            'prev_receptions': receiving_stats.get('receptions', 0),
            'prev_touchdowns': receiving_stats.get('touchdowns', 0),
            'prev_longest_reception': receiving_stats.get('longest', 0),
            'prev_first_downs': receiving_stats.get('first_downs', 0),
        })
        
        return prev_data

    def get_upcoming_game_data(self, next_game, current_game_date):
        next_game_name = list(next_game.keys())[0]
        next_game_date = datetime.strptime(next_game_name.split("_")[0], "%m-%d-%Y")
        is_home_game = "_vs" in next_game_name
        opponent = next_game_name.split("_vs" if is_home_game else "_at")[1].strip().replace("_", " ")
        
        upcoming_data = {
            'upcoming_is_home_game': is_home_game,
            'upcoming_opponent': opponent,
            'upcoming_kickoff_time': next_game[next_game_name]['box_score']['gameinfo'].get('kickoff_time'),
            'upcoming_days_until_game': (next_game_date - current_game_date).days,
        }
        
        # Add placeholder weather forecast features
        upcoming_data.update({
            'upcoming_forecast_temperature': None,
            'upcoming_forecast_wind_speed': None,
            'upcoming_forecast_precipitation_chance': None,
            'upcoming_forecast_is_dome': None,
        })
        
        return upcoming_data

    def get_players_game_by_game_data(self, game_data, players_df):
        sorted_games = self.sort_games_chronologically(game_data)
        players_game_data = []
        
        for i, game in enumerate(sorted_games):
            prev_game = sorted_games[i-1] if i > 0 else None
            next_game = sorted_games[i+1] if i < len(sorted_games) - 1 else None
            
            game_data = self.process_game_v2(game, players_df, prev_game, next_game)
            players_game_data.extend(game_data)
        
        return pd.DataFrame(players_game_data)

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

            #combine players df and game data df
            #comine on name and season. add all columns from players_df to game_data_df
            #TODO: this is not working
            # players_game_data_df = pd.merge(players_game_data_df, players_df, on=['name', 'season'], how='left')



            
            # Apply weather cleaning and processing
            # players_game_data_df = self.clean_weather(players_game_data_df)
            # players_game_data_df = self.process_weather_column(players_game_data_df)
            
            # players_df = self.filter_players_data(players_game_data_df)
            # players_df.insert(1, 'season', season)
            players_game_data_df = players_game_data_df.sort_values(by=['name', 'prev_game_date'])

            #following name, sort by wether or not the column name has prev_ or upcoming_ in it
            #then sort by the column name
            players_game_data_df = players_game_data_df.reindex(sorted(players_game_data_df.columns, key=lambda x: (x.startswith('prev_'), x.startswith('upcoming_'), x)), axis=1)

            

            save_path = f"{self.save_dir}/{season}"
            os.makedirs(save_path, exist_ok=True)
            players_game_data_df.to_csv(f"{save_path}/wrs.csv", index=False)

            print(players_game_data_df.columns)
            exit()

        print("WR engineering completed for all seasons.")
        self.save_features()