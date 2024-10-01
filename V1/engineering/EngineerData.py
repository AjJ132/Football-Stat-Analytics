import pandas as pd
import json
import os
import numpy as np
import pandas as pd

class EngineerData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.years = ['2021', '2022', '2023']

    def load_data(self):
        all_data = []
        for year in self.years:
            year_dir = os.path.join(self.data_path, year)
            if os.path.exists(year_dir):
                file_path = os.path.join(year_dir, f'ksu_football_roster_{year}.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        for player in data:
                            player['Year_Scraped'] = year
                        all_data.extend(data)
                else:
                    print(f"Warning: File not found: {file_path}")
            else:
                print(f"Warning: Directory not found: {year_dir}")
        
        if not all_data:
            raise ValueError("No data was loaded. Check the file paths and data directory structure.")
        
        # Create DataFrame
        self.df = pd.DataFrame(all_data)
        
        # Get all unique columns across all players
        all_columns = set()
        for player in all_data:
            all_columns.update(player.keys())
        
        # Fill missing values with 0 for numeric columns, 'Unknown' for non-numeric
        for col in all_columns:
            if col not in self.df.columns:
                if col in ['Rush Attempts', 'Rush Yards', 'Rush Touchdowns', 'Rush Longest', 
                           'Rush Rush attempt yards pct', 'Rush Yards per game avg', 
                           'Games Played', 'Games Started', 'Sacks', 'Tackles', 'Interceptions',
                           'Passing Yards', 'Passing Touchdowns', 'Receptions', 'Receiving Yards']:
                    self.df[col] = 0
                else:
                    self.df[col] = 'Unknown'

    def clean_data(self):
        # Convert height to inches
        if 'Height' in self.df.columns:
            self.df['Height_inches'] = self.df['Height'].apply(lambda x: int(x.split("'")[0]) * 12 + int(x.split("'")[1].replace('"', '')) if isinstance(x, str) else np.nan)
        
        # Convert weight to numeric
        if 'Weight' in self.df.columns:
            self.df['Weight'] = pd.to_numeric(self.df['Weight'].astype(str).str.replace(' lbs', ''), errors='coerce')
        
        # List of all columns that should be numeric
        numeric_columns = [
            'Rush Attempts', 'Rush Yards', 'Rush Touchdowns', 'Rush Longest', 
            'Rush Rush attempt yards pct', 'Rush Yards per game avg', 
            'Games Played', 'Games Started', 'Sacks', 'Tackles', 'Interceptions',
            'Passing Yards', 'Passing Touchdowns', 'Receptions', 'Receiving Yards',
            'Receiving Recep', 'Receiving Touchdowns', 'Receiving Longest', 'Receiving Per game',
            'Receiving Yards comp pct', 'Receiving Yards game pct',
            'Defense Solo', 'Defense Assist', 'Defense Total', 'Defense Interceptions',
            'Defense Pass defl', 'Defense Forced fumble', 'Defense Fumb rec', 'Defense Blocked'
        ]
        
        # Filter numeric_columns to only include columns that exist in the DataFrame
        existing_numeric_columns = [col for col in numeric_columns if col in self.df.columns]
        
        # Convert all existing numeric columns to numeric, replacing non-numeric values with NaN
        for col in existing_numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Handle special cases like 'Defense Tfl yards' and 'Defense Sacks yards'
        special_columns = ['Defense Tfl yards', 'Defense Sacks yards']
        existing_special_columns = [col for col in special_columns if col in self.df.columns]
        for col in existing_special_columns:
            self.df[col] = self.df[col].apply(lambda x: float(x.split('-')[0]) if isinstance(x, str) and '-' in x else np.nan)

        # Fill NaN values with 0 for numeric columns
        self.df[existing_numeric_columns + existing_special_columns] = self.df[existing_numeric_columns + existing_special_columns].fillna(0)

    def export_to_csv(self, output_path):
        self.df.to_csv(output_path, index=False)

    def export_to_json(self, output_dir):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Export a JSON file for each year
        for year in self.years:
            year_data = self.df[self.df['Year_Scraped'] == year]
            if not year_data.empty:
                output_path = os.path.join(output_dir, f'ksu_football_data_{year}.json')
                year_data.to_json(output_path, orient='records', indent=2)
                print(f"Exported data for {year} to {output_path}")
            else:
                print(f"No data available for {year}")

    def process_data(self, output_csv_path, output_json_dir):
        self.load_data()
        if self.df is not None and not self.df.empty:
            self.clean_data()
            self.export_to_csv(output_csv_path)
            self.export_to_json(output_json_dir)
        else:
            print("No data to process. Check if data was loaded correctly.")