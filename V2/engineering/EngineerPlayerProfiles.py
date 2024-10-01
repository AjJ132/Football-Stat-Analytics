import json
import os
import pandas as pd


class EngineerPlayerProfiles:
    def __init__(self, roster_data_dir, stats_data_dir, seasons):
        self.roster_data_dir = roster_data_dir
        self.stats_data_dir = stats_data_dir
        self.seasons = seasons

    def load_data(self, file_path):
        """
        Load data from a JSON file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    def gather_players(self):
        """
        Returns a dictionary of unique players.
        Combines players that have the same name and hometown over multiple years.
        """
        players = {}

        for season in self.seasons:
            print(f"Processing {season} season")
            roster_data = self.load_data(os.path.join(self.roster_data_dir, str(season), f'ksu_football_roster_{str(season)}.json'))

            for player in roster_data:
                if player is None:
                    continue

                player_name = player.get('Name', '')
                player_hometown = player.get('Hometown', '')

                if not player_name or not player_hometown:
                    print(f"Warning: Skipping player with incomplete data in season {season}: {player}")
                    continue

                player_name = player_name.lower()
                player_hometown = player_hometown.lower()
                player_id = hash(f"{player_name}_{player_hometown}")

                if player_id not in players:
                    players[player_id] = {
                        'Name': player_name,
                        'Hometown': player_hometown,
                        'Positions': set(),
                        'Years': set(),
                        'Class': set(),
                        'Major': set(),
                        'Height': {},
                        'Weight': {},
                        'Uniform': set()
                    }

                player_data = players[player_id]
                
                position = player.get('Position')
                if position:
                    player_data['Positions'].add(position.lower())
                
                player_data['Years'].add(season)

                player_class = player.get('Year')
                if player_class:
                    player_data['Class'].add(player_class.lower())
                
                major = player.get('Major')
                if major:
                    player_data['Major'].add(major.lower())
                
                height = player.get('Height')
                if height:
                    player_data['Height'][season] = height.lower()
                
                weight = player.get('Weight')
                if weight:
                    player_data['Weight'][season] = weight.lower()
                
                uniform = player.get('Uniform')
                if uniform:
                    player_data['Uniform'].add(uniform.lower())

        # Convert sets to sorted lists for consistent output
        for player_data in players.values():
            player_data['Positions'] = sorted(player_data['Positions'])
            player_data['Years'] = sorted(player_data['Years'])
            player_data['Class'] = sorted(player_data['Class'])
            player_data['Major'] = sorted(player_data['Major'])
            player_data['Uniform'] = sorted(player_data['Uniform'])

        return players

    def create_team_context_by_season(self, season):
        """
        Create a team context object for a given season
        Add team record, number of games, and ppg for team and opponents
        Get data from stats dir labelled by season
        """

        # Load season stats
        directory = os.path.join(self.stats_data_dir, season)
        path = os.path.join(directory, f'{season}_season_stats.json')
        season_stats = self.load_data(path)

        # Under scoring, find the Points Per Game and Total
        scoring = season_stats.get('Scoring', {})

        ppg = scoring.get('Points Per Game', 0)
        team_ppg = ppg.get('team', 0)
        opp_ppg = ppg.get('opponents', 0)

        #if string convert to float
        if isinstance(team_ppg, str):
            team_ppg = float(team_ppg)
        if isinstance(opp_ppg, str):
            opp_ppg = float(opp_ppg)

        # Get total score
        total = scoring.get('Total', {})
        team_total = total.get('team', 0)
        opp_total = total.get('opponents', 0)

        # convert to float
        if isinstance(team_total, str):
            team_total = float(team_total)
        if isinstance(opp_total, str):
            opp_total = float(opp_total)

        # Get number of games from {season}_schedule.json
        path = os.path.join(directory, f'{season}_schedule.json')
        schedule = self.load_data(path)

        #count number of objects in the JSON file
        num_games = len(schedule)

        #'Result' is a key in the JSON file that holds the result of the game. L or W
        wins = 0
        losses = 0

        for game in schedule:
            result = game.get('Result', '')
            if result == 'W':
                wins += 1
            elif result == 'L':
                losses += 1
            else:
                print(f"Warning: Unexpected result '{result}' in season {season}")
        
        # Create team context object
        team_context = {
            'Season': season,
            'Wins': wins,
            'Losses': losses,
            'Games': num_games,
            'PPG': team_ppg,
            'OppPPG': opp_ppg,
            'Total': team_total,
            'OppTotal': opp_total
        }

        return team_context

    def create_profiles(self):
        """
        Create player profiles with fetched data
        """
        self.seasons.sort(key=int)
        players = self.gather_players()

        print(f"Total players: {len(players)}")

        # Reorganize player data into an array of objects with nested player_info
        player_profiles = []
        for player_id, player_data in players.items():
            player_profile = {
                "id": player_id,
                "player_info": {
                    'Name': player_data['Name'],
                    'Hometown': player_data['Hometown'],
                    'Positions': player_data['Positions'],
                    'Years': player_data['Years'],
                    'Class': player_data['Class'],
                    'Major': player_data['Major'],
                    'Height': player_data['Height'],
                    'Weight': player_data['Weight'],
                    'Uniform': player_data['Uniform']
                }
            }
            player_profiles.append(player_profile)

        # Add team context to each player profile
        team_contexts = {}
        for season in self.seasons:
            team_context = self.create_team_context_by_season(season)
            team_contexts[season] = team_context

        for player_profile in player_profiles:
            player_years = player_profile['player_info']['Years']
            player_profile['SeasonContext'] = {}
            for year in player_years:
                if year in team_contexts:
                    player_profile['SeasonContext'][year] = team_contexts[year]
                else:
                    print(f"Warning: Missing team context for player {player_profile['player_info']['Name']} in season {year}")


        #TODO: Add Strength of Schedule to team context

        # Save to file
        with open('data/engineered/player_profiles.json', 'w') as f:
            json.dump(player_profiles, f, indent=4)

        return

        

        

