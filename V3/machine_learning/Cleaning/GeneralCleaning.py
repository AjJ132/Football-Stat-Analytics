import re
import numpy as np
import pandas as pd

class GeneralCleaning:

    def __init__(self):
        self.temp_median = None

    def convert_time_to_minutes(self, time):
        if pd.isna(time):
            return np.nan
        
        if isinstance(time, (int, float)):
            return time
        
        if isinstance(time, str):
            match = re.match(r'(\d+):(\d+)', time)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                return hours * 60 + minutes
        
        return np.nan
    
    def normalize_kickoff_time(self, time):
        if pd.isna(time):
            return np.nan
        
        time = str(time).lower().strip()
        
        if time == 'noon':
            return 12 * 60
        
        match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', time)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2) or 0)
            period = match.group(3)

            if period == 'pm' and hours != 12:
                hours += 12
            elif period == 'am' and hours == 12:
                hours = 0

            return hours * 60 + minutes
        
        return np.nan

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

    def clean_data(self, data):
        data = self.clean_weather(data)

        data['duration'] = data['duration'].apply(self.convert_time_to_minutes)
        data['kickoff_time'] = data['kickoff_time'].apply(self.normalize_kickoff_time)
        data['is_home_game'] = data['is_home_game'].astype(int)

        data.columns = data.columns.str.rstrip('.')
        data = data[['name', 'season'] + [col for col in data.columns if col not in ['name', 'season']]]
        data = data.sort_values(by=['name', 'season'])
        data = data.dropna()

        return data