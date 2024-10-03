import os
import json
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from tqdm import tqdm

class FootballRosterScraper:
    def __init__(self, seasons, base_save_path='data/raw/roster'):
        self.base_url = 'https://ksuowls.com'
        self.stats_url = f'{self.base_url}/services/responsive-roster-bio.ashx'
        self.base_save_path = base_save_path
        self.valid_positions = [
            'QB', 'RB', 'FB', 'WR', 'TE', 'OL', 'C', 'G', 'T',
            'DE', 'DT', 'LB', 'CB', 'S', 'K', 'P', 'LS', 'DB', 'OLB', 'ILB', 'DL'
        ]
        self.seasons = seasons

    def normalize_name(self, name):
        # Split the name into parts
        parts = name.split(',')
        if len(parts) == 2:
            # If the name is already in "lastname, firstname" format
            return name.strip().lower()
        else:
            # If the name is in "firstname lastname" format
            parts = name.split()
            if len(parts) >= 2:
                return f"{parts[-1]}, {' '.join(parts[:-1])}".strip().lower()
            else:
                return name.strip().lower()

    def get_roster_soup(self, year):
        roster_url = f'{self.base_url}/sports/football/roster/{year}'
        print(f"Fetching roster page for {year}")
        response = requests.get(roster_url)
        return BeautifulSoup(response.content, 'html.parser'), roster_url

    def get_player_rows(self, soup):
        print("Extracting player rows from roster")
        roster_section = soup.find('section', {'aria-label': 'Men\'s Player Roster'})
        if roster_section:
            player_rows = roster_section.find_all('li', class_='sidearm-roster-player')
            print(f"Found {len(player_rows)} player rows")
            return player_rows
        else:
            print("Could not find the roster section. The page structure might have changed.")
            return []

    def extract_player_info(self, player):
        player_data = {}
        player_data['Name'] = player.find('h3').text.strip() if player.find('h3') else ''
        
        position_div = player.find('div', class_='sidearm-roster-player-position')
        if position_div:
            position_spans = position_div.find_all('span', class_='text-bold')
            raw_position = position_spans[0].text.strip() if position_spans else ''
            player_data['Position'] = self.clean_position(raw_position)
            
            major_span = position_div.find('span', class_='sidearm-roster-player-custom1')
            player_data['Major'] = major_span.text.strip() if major_span else ''
        
        player_data['Height'] = player.find('span', class_='sidearm-roster-player-height').text.strip() if player.find('span', class_='sidearm-roster-player-height') else ''
        player_data['Weight'] = player.find('span', class_='sidearm-roster-player-weight').text.strip() if player.find('span', class_='sidearm-roster-player-weight') else ''
        
        year_span = player.find('span', class_='sidearm-roster-player-academic-year')
        if year_span:
            player_data['Year'] = year_span.text.strip()
        else:
            year_span = player.find('span', class_='sidearm-roster-player-class')
            player_data['Year'] = year_span.text.strip() if year_span else ''
        
        player_data['Hometown'] = player.find('span', class_='sidearm-roster-player-hometown').text.strip() if player.find('span', class_='sidearm-roster-player-hometown') else ''
        
        return player_data
    
    def clean_position(self, position):
        # Remove any whitespace and newline characters
        cleaned = re.sub(r'\s+', ' ', position).strip()
        # Split the string in case there are multiple positions
        positions = cleaned.split()
        # Check if any of the positions match our valid list
        for pos in positions:
            if pos in self.valid_positions:
                return pos
        return None

    def get_player_id(self, player_link):
        if player_link:
            player_id = player_link['href'].split('/')[-1]
            return player_id
        return None

    def fetch_player_stats(self, player_id, roster_url):
        params = {
            'type': 'career-stats',
            'rp_id': player_id,
            'path': 'football'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': roster_url,
        }
        try:
            response = requests.get(self.stats_url, params=params, headers=headers)
            
            if response.status_code == 200:
                try:
                    stats_data = response.json()
                    return stats_data
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON for player ID {player_id}")
            else:
                print(f"Failed to fetch stats for player ID {player_id}. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed for player ID {player_id}: {str(e)}")
        
        return None

    def process_player_stats(self, stats_data, current_year):
        processed_stats = {}
        if stats_data and current_year in stats_data:
            try:
                year_data = stats_data[current_year]
                
                # Save all the top-level information
                for key, value in year_data.items():
                    if key != 'season_stats':
                        processed_stats[key] = value
                
                # Save all the season_stats information
                if 'season_stats' in year_data:
                    for category, category_stats in year_data['season_stats'].items():
                        for stat, value in category_stats.items():
                            processed_stats[f"{category}.{stat}"] = value
                
            except KeyError as e:
                print(f"KeyError while processing stats for year {current_year}: {str(e)}")
        return processed_stats

    def save_players(self, players, year):
        save_path = os.path.join(self.base_save_path, year)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        json_filepath = os.path.join(save_path, f'ksu_football_roster_{year}.json')
        with open(json_filepath, 'w') as f:
            json.dump(players, f, indent=2)
        print(f"Saved player data to {json_filepath}")

    def process_player_row(self, row, year, roster_url):
        player_data = self.extract_player_info(row)
        player_data['season'] = year
        player_link = row.find('a', href=True)
        if player_link:
            player_id = self.get_player_id(player_link)
            if player_id:
                stats_data = self.fetch_player_stats(player_id, roster_url)
                if stats_data:
                    player_stats = self.process_player_stats(stats_data, year)
                    player_data.update(player_stats)
        
        # Convert all keys to lowercase, except 'Name'
        lowercase_data = {k.lower() if k != 'Name' else k: v for k, v in player_data.items()}

        #drp
        
        # drop the 'Name' key from the data
        lowercase_data.pop('Name', None)

        # Normalize the name 
        lowercase_data['name'] = self.normalize_name(player_data['Name'])
        
        
        return lowercase_data

    def scrape(self):
        for year in self.seasons:
            print(f"Scraping data for {year} season")
            soup, roster_url = self.get_roster_soup(year)
            player_rows = self.get_player_rows(soup)

            players = []
            if player_rows:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_player = {executor.submit(self.process_player_row, player_row, year, roster_url): player_row for player_row in player_rows}
                    for future in tqdm(as_completed(future_to_player), total=len(player_rows), desc=f"Processing players for {year}"):
                        try:
                            player_data = future.result()
                            if player_data:
                                players.append(player_data)
                        except Exception as e:
                            print(f"Error processing player: {str(e)}")

            self.save_players(players, year)