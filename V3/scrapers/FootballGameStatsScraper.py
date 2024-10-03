import os
import json
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

class FootballGameStatsScraper:
    def __init__(self, seasons, base_save_path='data/raw/stats'):
        self.base_url = 'https://ksuowls.com'
        self.stats_url = f'{self.base_url}/sports/football/stats'
        self.base_save_path = base_save_path
        self.seasons = seasons
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })


    def get_soup(self, url):
        response = self.session.get(url, allow_redirects=True)
        return BeautifulSoup(response.content, 'html.parser')

    def get_stats_soup(self, year):
        url = f'{self.stats_url}/{year}'
        return self.get_soup(url)

    def extract_game_data(self, row):
        game_data = {}
        cells = row.find_all(['td', 'th'])
        
        game_data['Date'] = cells[0].text.strip()
        opponent_cell = cells[1]
        game_data['Opponent'] = opponent_cell.text.strip()
        game_data['OpponentLink'] = opponent_cell.find('a')['href'] if opponent_cell.find('a') else None
        game_data['Result'] = cells[2].text.strip()
        game_data['Score'] = cells[3].text.strip()
        game_data['Record'] = cells[4].text.strip()
        game_data['Duration'] = cells[5].text.strip()
        game_data['Attendance'] = cells[6].text.strip()

        return game_data

    def process_season_stats(self, html_content, year):
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table', {'class': 'sidearm-table'})
        
        stats = {}
        current_category = None
        
        for row in table.find_all('tr'):
            th = row.find('th')
            if th and th.get('colspan') == '3':
                current_category = th.text.strip()
                stats[current_category] = {}
            elif th and not th.get('colspan'):
                span = th.find('span')
                stat_name = span.text.strip() if span else th.text.strip()
                td_cells = row.find_all('td')
                if len(td_cells) >= 2:
                    team_value = td_cells[0].text.strip()
                    opponent_value = td_cells[1].text.strip()
                    
                    stats[current_category][stat_name] = {
                        "team": team_value,
                        "opponent": opponent_value
                    }
                else:
                    #TODO: Fix me, i think this always hit on the first row of the table not sure.
                    print(f"Warning: Unexpected row structure for stat '{stat_name}' in category '{current_category}'")
        
        # Save to JSON file
        save_path = os.path.join(self.base_save_path, year)
        os.makedirs(save_path, exist_ok=True)
        json_filename = os.path.join(save_path, f'{year}_season_stats.json')
        with open(json_filename, 'w') as f:
            json.dump(stats, f, indent=2)
        

        return stats

    def process_season(self, year):
        soup = self.get_stats_soup(year)
        team_section = soup.find('section', {'id': 'team', 'aria-label': 'Team'})
        
        if team_section:
            season_stats = self.process_season_stats(str(team_section), year)
        
        results_section = soup.find('section', {'id': 'gbg_results', 'aria-label': 'Game-By-Game - Results'})
        
        if not results_section:
            print(f"Could not find Game-By-Game Results section for {year}")
            return []


        
        table = results_section.find('table', {'class': 'sidearm-table'})
        
        if not table:
            print(f"Could not find stats table for {year}")
            return []

        
        games = []
        rows = table.find('tbody').find_all('tr')
        
        for row in rows:
            game_data = self.extract_game_data(row)
            games.append(game_data)

        return games


    def parse_box_score_page(self, box_score_soup):
        box_score = {}

        #remove everything in soup except for <article> tag
        article = box_score_soup.find('article')

        #first parse the box score summary page
        box_score_summary = self.parse_box_score_summary(article)

        #parse the game information
        game_info = self.parse_box_score_game_info(article)

        #get the scoring summary
        scoring_summary = self.parse_box_score_scoring_summary(article)
        

        box_score['Summary'] = box_score_summary
        box_score['GameInfo'] = game_info
        box_score['ScoringSummary'] = scoring_summary

        return box_score
    


    #------------------PARSE BOX SCORE TAB ------------------

    def parse_box_score_summary(self, box_score_soup):
        box_score_section = box_score_soup.find('section', {'id': 'box-score', 'aria-label': 'Box score'})
        if not box_score_section:
            print("Could not find Box Score section")
            return {}
    
        header = box_score_section.find('header')
        table = header.find('table', {'class': 'sidearm-table'})
        if not table:
            print("Could not find the table")
            return {}
    
        box_score = {}
        tbody = table.find('tbody')
        if not tbody:
            print("Could not find the table body")
            return {}
    
    
        for row in tbody.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 6:
                continue
    
            # Extract team name, handling hidden elements
            team_cell = cells[0]
            visible_span = team_cell.find('span', class_='hide-on-small-down')
            team_name = visible_span.get_text(strip=True) if visible_span else team_cell.get_text(strip=True)
            
            # Check for winner indicator
            winner_span = team_cell.find('span', class_='boxscore-winner')

            # Extract scores
            scores = [cell.get_text(strip=True) for cell in cells[1:6]]
    
            # Store the extracted data in the dictionary
            box_score[team_name] = scores

        return box_score

    def parse_box_score_game_info(self, box_score_soup):
            """
            Extract the game information and historical statistics from the box score page
            """
            header = box_score_soup.find('header')
            dl = header.find('dl', class_='text-center inline')

            if not dl:
                print("Could not find the dl tag")
                return {}

            game_info = {}
            opponent_history = {}

            view_tag = dl.find('a')
            view_tag_href = view_tag['href']
            for dt, dd in zip(dl.find_all('dt'), dl.find_all('dd')):
                if dt.text.strip() == 'VIEW':
                    continue
                game_info[dt.text.strip()] = dd.text.strip()

            opponent_history_url = f"{self.base_url}{view_tag_href}"
            opponent_history_page = self.get_soup(opponent_history_url)
            article = opponent_history_page.find('main').find('article')
            history_data_section = article.find('section', class_='sidearm-opponent-history__data')

            if history_data_section:
                # Wins and Losses
                wins_losses = history_data_section.find_all('div', class_='sidearm-opponent-history__item--number thick')
                if len(wins_losses) >= 2:
                    opponent_history['Wins'] = wins_losses[0].text.strip()
                    opponent_history['Losses'] = wins_losses[1].text.strip()

                # Streak
                streak_div = history_data_section.find('div', class_='large-4 medium-12 x-small-12 flex columns')
                if streak_div:
                    streak_number = streak_div.find('div', class_='sidearm-opponent-history__item--number')
                    if streak_number:
                        opponent_history['Streak'] = streak_number.text.strip()

                # Home, Away, and Conference Records
                record_divs = history_data_section.find_all('div', class_='sidearm-opponent-history__item')
                for div in record_divs:
                    title = div.find('h2')
                    if title:
                        title_text = title.text.strip()
                        if title_text in ['Home Record', 'Away Record', 'Conference Record']:
                            record = div.find('div', class_='sidearm-opponent-history__item--number')
                            if record:
                                opponent_history[title_text] = record.text.strip()

                # New statistics
                stat_items = history_data_section.find_all('div', class_='sidearm-opponent-history__item')
                for item in stat_items:
                    title = item.find('h2')
                    if title:
                        title_text = title.text.strip()
                        value_div = item.find('div', class_='sidearm-opponent-history__item--number')
                        if value_div:
                            value = value_div.text.strip()
                            if title_text == 'Total Points':
                                opponent_history['Total Points'] = value
                            elif title_text == 'Average Points':
                                opponent_history['Average Points'] = value
                            elif title_text == 'Largest Margin of Victory':
                                opponent_history['Largest Margin of Victory'] = value
                            elif title_text == 'Smallest Margin of Victory':
                                opponent_history['Smallest Margin of Victory'] = value
                            elif title_text == 'Last 3 Matchups':
                                opponent_history['Last 3 Matchups'] = value

            # Work on historical statistics table
            historical_table_section = article.find('section', class_='sidearm-opponent-history__table')
            table_soup = historical_table_section.find('table')
            historical_games = []

            if table_soup:
                table_body = table_soup.find('tbody')
                rows = table_body.find_all('tr')
                for row in rows:
                    date = row.find('th').text.strip()
                    location_cell = row.find('td', {'data-label': 'Location'})
                    location_text = location_cell.text.strip()
                    location = "Home" if "Home" in location_text else "Away"
                    score = row.find('td', {'data-label': 'Score'}).text.strip()
                    historical_games.append({
                        'Date': date.split('\n')[0].strip(),  # Take only the first part of the date
                        'Location': location,  # Simplified to just "Home" or "Away"
                        'Score': ' '.join(score.split())  # Remove extra whitespace
                    })

            game_info['OpponentHistory'] = opponent_history
            game_info['HistoricalGames'] = historical_games

            # Clean up the game_info dictionary
            for key, value in game_info.items():
                if isinstance(value, str):
                    game_info[key] = ' '.join(value.split())  # Remove extra whitespace
            
            return game_info
 
    def parse_box_score_scoring_summary(self, box_score_soup):
        """
        Get the scoring summary from the box score page
        """    

        scoring_summary = {}

        #find section with class 'panel' and aria-label 'Scoring Summary'
        socring_summary_section = box_score_soup.find('section', {'class': 'panel', 'aria-label': 'Scoring Summary'})

        if not socring_summary_section:
            print("Could not find Scoring Summary section")
            return {}
        
        #find table with class sidearm-table scoring-summary highlight-hover collapse-on-medium
        table = socring_summary_section.find('table', {'class': 'sidearm-table scoring-summary highlight-hover collapse-on-medium'})

        if not table:
            print("Could not find the table")
            return {}
        
        #foreach row in table body, parse the data
        # 1st column is the quarter, 2nd is the time, 3rd is the score summary, 
       
        for row in table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 3:
                continue

            quarter = cells[1].text.strip()
            time = cells[2].text.strip()
            score_summary = cells[3].text.strip()
            
            #team is score summary before the first '-' in the string
            #EXAMPLE: KSU - ROBERTSON, N 24 yd field goal 16 plays, 76 yards, TOP 7:58
            team = score_summary.split('-')[0].strip()
            

            #add new row to the scoring summary under the quarter
            if quarter not in scoring_summary:
                scoring_summary[quarter] = []

            scoring_summary[quarter].append({
                'Time': time,
                'ScoreSummary': score_summary,
                'Team': team
            })

        return scoring_summary


    #------------------PARSE TEAM TAB ------------------

    def parse_team_stats(self, team_stats_soup):
        team_stats = {}
        team_stats_section = team_stats_soup.find('section', {'id': 'team-stats', 'aria-label': 'Game Statistics By Team'})
        
        if not team_stats_section:
            print("Could not find Team Stats section")
            return {}
        
        table = team_stats_section.find('table', {'class': 'sidearm-table overall-stats highlight-hover full'})
        
        if not table:
            print("Could not find the table")
            return {}
        
        home_team = {}
        away_team = {}
        table_body = table.find('tbody')
        current_category = None

        for row in table_body.find_all('tr'):
            th = row.find('th')
            
            if th and th.get('colspan') == '3':
                current_category = th.text.strip()
                home_team[current_category] = {}
                away_team[current_category] = {}
            else:
                cells = row.find_all('td')
                if len(cells) < 3:
                    continue
                
                stat_name = cells[0].text.strip()
                away_team_value = self.clean_value(cells[1].text.strip())
                home_team_value = self.clean_value(cells[2].text.strip())
                
                home_team[current_category][stat_name] = home_team_value
                away_team[current_category][stat_name] = away_team_value

        team_stats['HomeTeam'] = home_team
        team_stats['AwayTeam'] = away_team


        return team_stats


    # ------------------PARSE INDIVIDUAL TAB ------------------
    def parse_individual_stats(self, individual_stats_soup):
        individual_stats = {}

        #get the section with id = 'individual-stats' and aria-label = 'Individual Statistics'
        individual_stats_section = individual_stats_soup.find('section', {'id': 'individual-stats', 'aria-label': 'Individual Statistics'})

        #inside this section are three sections
        #they are aria-label = 'Offensive', 'Defensive', 'Special Teams'
        
        offensive_section = individual_stats_section.find('section', {'aria-label': 'Offensive'})
        defensive_section = individual_stats_section.find('section', {'aria-label': 'Defensive'})
        special_teams_section = individual_stats_section.find('section', {'aria-label': 'Special Teams'})

        #parse the offensive section
        offensive_stats = self.handle_parse_offensive_individual_stats_section(offensive_section)

        #parse the defensive section
        defensive_stats = self.handle_parse_defensive_individual_stats_section(defensive_section)

        #parse the special teams section
        special_teams_stats = self.handle_parse_special_teams_individual_stats_section(special_teams_section)

        combined_individual_stats = self.handle_combining_individual_stats(offensive_stats, defensive_stats, special_teams_stats)

        return combined_individual_stats

    def handle_parse_offensive_individual_stats_section(self, offensive_section):
        home_team = {}
        away_team = {}

        def parse_individual_stats(section):
            row_div = section.find('div', {'class': 'row'})
            away_team_div, home_team_div = row_div.find_all('div', recursive=False)

            def parse_team_stats(team_div):
                stat_names = [th.text.strip() for th in team_div.find('thead').find_all('th')]
                stat_names.pop(0)  # Remove 'Player' from the list

                team_stats = {}
                for row in team_div.find('tbody').find_all('tr'):  # Exclude the last row (total)
                    cells = row.find_all('td')
                    player_name = cells[0].text.strip()
                    stats = [cell.text.strip() for cell in cells[1:]]
                    team_stats[player_name] = dict(zip(stat_names, stats))
                return team_stats

            return parse_team_stats(away_team_div), parse_team_stats(home_team_div)

        sections = [
            ('Passing', 'individual-passing', 'Individual Passing Statistics'),
            ('Rushing', 'individual-rushing', 'Individual Rushing Statistics'),
            ('Receiving', 'individual-receiving', 'Individual Receiving Statistics')
        ]

        for category, section_id, aria_label in sections:
            section = offensive_section.find('section', {'id': section_id, 'aria-label': aria_label})
            away_stats, home_stats = parse_individual_stats(section)
            away_team[category] = away_stats
            home_team[category] = home_stats

        return {'AwayTeam': away_team, 'HomeTeam': home_team}

    def handle_parse_defensive_individual_stats_section(self, defensive_section):
        def parse_defensive_stats(section):
            table = section.find('table')
            stats = {}
            
            for row in table.find('tbody').find_all('tr'):
                cells = row.find_all('td')
                player_name = cells[0].text.strip()
                player_stats = {cell['data-label']: cell.text.strip() for cell in cells[1:]}
                
                # Clean data
                for stat_name, stat_value in player_stats.items():
                    if stat_value in ['-', '-/-']:
                        player_stats[stat_name] = '0'
                
                stats[player_name] = player_stats
            
            return stats

        defensive_away_section = defensive_section.find('section', {'id': 'defense-away'})
        defensive_home_section = defensive_section.find('section', {'id': 'defense-home'})

        defensive_away_stats = parse_defensive_stats(defensive_away_section)
        defensive_home_stats = parse_defensive_stats(defensive_home_section)

        return {
            'AwayTeam': defensive_away_stats,
            'HomeTeam': defensive_home_stats
        }

    def handle_parse_special_teams_individual_stats_section(self, special_teams_section):
        home_team = {}
        away_team = {}

        def clean_text(text):
            return ' '.join(text.split()).strip()

        def parse_individual_stats(section):
            row_div = section.find('div', {'class': 'row'})
            away_team_div, home_team_div = row_div.find_all('div', recursive=False)

            def parse_team_stats(team_div):
                stat_names = [clean_text(th.text) for th in team_div.find('thead').find_all('th')]
                stat_names.pop(0)  # Remove 'Player' from the list

                team_stats = {}
                rows = team_div.find('tbody').find_all('tr')
                
                for row in rows:
                    cells = row.find_all('td')
                    player_name_cell = cells[0]
                    
                    # Extract player name
                    player_name = clean_text(player_name_cell.text.split('\n')[0])
                    
                    # Check if it's a "Totals" or "TEAM" row
                    if player_name.lower() in ['totals', 'team']:
                        continue
                    
                    # Check if there's a comma in the name
                    if ',' not in player_name:
                        continue
                    
                    # Extract stats
                    player_stats = [clean_text(cell.text) for cell in cells[1:]]
                    
                    # Ensure we have the correct number of stats
                    if len(player_stats) < len(stat_names):
                        # If we're missing stats, pad with empty strings
                        player_stats = player_stats + [''] * (len(stat_names) - len(player_stats))
                    elif len(player_stats) > len(stat_names):
                        # If we have too many stats, truncate
                        player_stats = player_stats[:len(stat_names)]
                    
                    team_stats[player_name] = dict(zip(stat_names, player_stats))
                
                return team_stats

            return parse_team_stats(away_team_div), parse_team_stats(home_team_div)

        sections = [
            ('Punting', 'individual-punting', 'Individual Punting Statistics'),
            ('Returns', 'individual-allreturns', 'Individual Return Statistics'),
            ('FieldGoals', 'individual-fieldgoals', 'Individual Field Goal Statistics'),
            ('Kickoffs', 'individual-kickoffs-stats', 'Individual Kickoff Statistics'),
            ('PAT', 'individual-pat-stats', 'Individual PAT Statistics')
        ]

        for category, section_id, aria_label in sections:
            section = special_teams_section.find('section', {'id': section_id, 'aria-label': aria_label})
            if section:
                away_stats, home_stats = parse_individual_stats(section)
                
                away_team[category] = away_stats
                home_team[category] = home_stats
            else:
                print(f"Warning: Section '{category}' not found")

        return {'AwayTeam': away_team, 'HomeTeam': home_team}

    def handle_combining_individual_stats(self, offensive_stats, defensive_stats, special_teams_stats):
        combined_player_stats = {}

        def add_stats(team_type, category, stats, team):
            for player, player_stats in stats.items():
                if player not in combined_player_stats:
                    combined_player_stats[player] = {}
                if category not in combined_player_stats[player]:
                    combined_player_stats[player][category] = {}
                combined_player_stats[player][category][team_type] = player_stats

        # Combine offensive stats
        for team in ['AwayTeam', 'HomeTeam']:
            if team in offensive_stats:
                for category in ['Passing', 'Rushing', 'Receiving']:
                    if category in offensive_stats[team]:
                        add_stats('Offensive', category, offensive_stats[team][category], team)

        # Combine defensive stats
        for team in ['AwayTeam', 'HomeTeam']:
            if team in defensive_stats:
                add_stats('Defensive', 'Defense', defensive_stats[team], team)

        # Combine special teams stats
        for team in ['AwayTeam', 'HomeTeam']:
            if team in special_teams_stats:
                for category in ['Punting', 'Returns', 'FieldGoals', 'Kickoffs', 'PAT']:
                    if category in special_teams_stats[team]:
                        add_stats('SpecialTeams', category, special_teams_stats[team][category], team)

        return combined_player_stats

    def clean_value(self, value):
        # Remove newlines, carriage returns, and extra spaces
        cleaned = re.sub(r'\s+', ' ', value).strip()
        # Remove "of" from values like "8 of 11"
        cleaned = re.sub(r'(\d+)\s+of\s+(\d+)', r'\1/\2', cleaned)
        return cleaned
        

    # ------------------PARSE DRIVE CHART  ------------------
    def parse_drive_chart(self, drive_chart_soup):
        #get drive chart section
        drive_chart_section = drive_chart_soup.find('section', {'id': 'drive-chart', 'aria-label': 'Drive Chart'})

        #drive chart section has three sections. 
        # We want to get the data from id = 'visitor-drives' and 'home-drives' sections
        visitor_drives_section = drive_chart_section.find('section', {'id': 'visitor-drives'})
        home_drives_section = drive_chart_section.find('section', {'id': 'home-drives'})

        #parse the drive chart for the visitor and home teams
        visitor_drives = self.parse_drive_chart_table(visitor_drives_section)
        home_drives = self.parse_drive_chart_table(home_drives_section)

        return {
            'AwayTeam': visitor_drives,
            'HomeTeam': home_drives
        }


    def parse_drive_chart_table(self, section):
        table = section.find('table')
        
        if not table:
            return []

        headers = [th.text.strip() for th in table.find_all('th') if th.text.strip()]
        headers = [
            'Team', 'Qtr', 'Drive Started Spot', 'Drive Started Time', 'Drive Started Obtained',
            'Drive Ended Spot', 'Drive Ended Time', 'Drive Ended How Lost',
            'Consumed Plays-Yds', 'Consumed TOP'
        ]

        drives = []
        for row in table.find('tbody').find_all('tr'):
            drive = {}
            cells = row.find_all('td')
            
            for i, cell in enumerate(cells[1:]):  # Skip the first hidden cell
                if i < len(headers):
                    drive[headers[i]] = cell.text.strip()
            
            drives.append(drive)

        return drives


    #------------------PARSE PLAY BY PLAY ------------------

    def parse_play_by_play(self, play_by_play_soup):
        #get play by play section
        play_by_play_section = play_by_play_soup.find('section', {'id': 'play-by-play'})


        #insdie are sections for all 4 quarters, get each section
        first_quarter_section = play_by_play_section.find('section', {'id': '1st'})
        second_quarter_section = play_by_play_section.find('section', {'id': '2nd'})
        third_quarter_section = play_by_play_section.find('section', {'id': '3rd'})
        fourth_quarter_section = play_by_play_section.find('section', {'id': '4th'})

        return {}
    
    # ------------------PARSE PARTICIPATION ------------------
    def parse_participation(self, participation_soup):
        #there are two section, aira-label = 'Starters' and 'Player Participation'
        starters_section = participation_soup.find('section', {'aria-label': 'Starters'})
        player_participation_section = participation_soup.find('section', {'aria-label': 'Player Participation'})

        starters = self.handle_parse_starters_section(starters_section)

        player_participation = self.handle_parse_player_participation_section(player_participation_section)

        return {
            'Starters': starters,
            'PlayerParticipation': player_participation
        }

        
    def handle_parse_starters_section(self, starters_section):
        row_header_div, row_div = starters_section.find_all('div', recursive=False)
        away_team_div, home_team_div = row_div.find_all('div', recursive=False)

        away_team_starters = {}
        home_team_starters = {}

        # Parse away team
        away_tables = away_team_div.find_all('table')
        if len(away_tables) >= 2:
            away_offensive_table = away_tables[0]
            away_defensive_table = away_tables[1]

            for row in away_offensive_table.find('tbody').find_all('tr'):
                cells = row.find_all('td')
                player_name = cells[1].text.strip()
                away_team_starters[player_name] = 'Offense'

            for row in away_defensive_table.find('tbody').find_all('tr'):
                cells = row.find_all('td')
                player_name = cells[1].text.strip()
                away_team_starters[player_name] = 'Defense'

        # Parse home team
        home_tables = home_team_div.find_all('table')
        if len(home_tables) >= 2:
            home_offensive_table = home_tables[0]
            home_defensive_table = home_tables[1]

            for row in home_offensive_table.find('tbody').find_all('tr'):
                cells = row.find_all('td')
                player_name = cells[1].text.strip()
                home_team_starters[player_name] = 'Offense'

            for row in home_defensive_table.find('tbody').find_all('tr'):
                cells = row.find_all('td')
                player_name = cells[1].text.strip()
                home_team_starters[player_name] = 'Defense'

        return {
            'AwayTeam': away_team_starters,
            'HomeTeam': home_team_starters
        }

    def handle_parse_player_participation_section(self, player_participation_section):
        away_team_div, home_team_div = player_participation_section.find_all('div', recursive=False)[1].find_all('div', recursive=False)

        away_team_participants = []
        home_team_participants = []

        # Parse away team
        away_table = away_team_div.find('table')
        for row in away_table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            player_name = cells[1].text.strip()
            away_team_participants.append(player_name)

        # Parse home team
        home_table = home_team_div.find('table')
        for row in home_table.find('tbody').find_all('tr'):
            cells = row.find_all('td')
            player_name = cells[1].text.strip()
            home_team_participants.append(player_name)

        return {
            'AwayTeam': away_team_participants,
            'HomeTeam': home_team_participants
        }

    def convert_keys_to_lowercase_and_underscore(self, data):
        if isinstance(data, dict):
            return {k.lower().replace(' ', '_').replace(':', ''): self.convert_keys_to_lowercase_and_underscore(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_keys_to_lowercase_and_underscore(item) for item in data]
        else:
            return data

    def process_game(self, game, year):
        opponent = re.sub(r'[^\w\-_\. ]', '_', game['opponent'])
        game_date = game['date'].replace('/', '-')
        folder_name = f"{game_date}_{opponent}"
        
        save_folder = os.path.join(self.base_save_path, year, 'game_by_game', folder_name)
        os.makedirs(save_folder, exist_ok=True)

        if game['opponentlink']:
            opponent_url = f"{self.base_url}{game['opponentlink']}"

            box_score_page_soup = self.get_soup(opponent_url)
            box_score_stats = self.parse_box_score_page(box_score_page_soup)
            team_stats = self.parse_team_stats(box_score_page_soup)
            individual_stats = self.parse_individual_stats(box_score_page_soup)
            drive_chart = self.parse_drive_chart(box_score_page_soup)
            participation = self.parse_participation(box_score_page_soup)

            game_data = {
                'box_score': box_score_stats,
                'team_stats': team_stats,
                'individual_stats': individual_stats,
                'drive_chart': drive_chart,
                'participation': participation
            }

            # Convert all keys to lowercase and replace spaces with underscores
            game_data_formatted = self.convert_keys_to_lowercase_and_underscore(game_data)

            # Save to JSON in the correct folder
            json_filename = os.path.join(save_folder, 'game_data.json')
            with open(json_filename, 'w') as f:
                json.dump(game_data_formatted, f, indent=2)

            return game_data_formatted

    def scrape(self):
        for year in self.seasons:
            print(f"Scraping data for {year} season")
            games = self.process_season(year)
            if games:
                # Convert games list keys to lowercase and replace spaces with underscores
                games_formatted = self.convert_keys_to_lowercase_and_underscore(games)
                self.save_games(games_formatted, year)
                for game in tqdm(games_formatted, desc=f"Processing games for {year}"):
                    game_data = self.process_game(game, year)

                    if game_data:
                        pass
                    else: 
                        print(f"Error processing game data for {game['date']} vs {game['opponent']}")
            else:
                print(f"No game data found for {year}")

    def save_games(self, games, year):
        save_path = os.path.join(self.base_save_path, year)
        os.makedirs(save_path, exist_ok=True)

        json_filepath = os.path.join(save_path, f'{year}_schedule.json')
        with open(json_filepath, 'w') as f:
            json.dump(games, f, indent=2)


