import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class GeneralCleaning:

    def __init__(self):
        self.temp_median = None
        self.mlb = MultiLabelBinarizer()

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
    
    def generate_game_count(self, data):
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
        
        # Sort by date and kickoff_time
        data = data.sort_values(['date', 'kickoff_time'])
        
        # Generate game count for each date
        data['game_number'] = data.groupby('date').cumcount() + 1
        
        return data
    
    def normalize_kickoff_time(self, time):
        if pd.isna(time):
            return np.nan
        
        time = str(time).lower().strip()
        
        # Handle 'noon'
        if time == 'noon':
            return 12 * 60
        
        # Extract hours and minutes
        match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)?', time)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2) or 0)
            period = match.group(3)

            # Adjust for PM
            if period == 'pm' and hours != 12:
                hours += 12
            # Adjust for AM
            elif period == 'am' and hours == 12:
                hours = 0

            return hours * 60 + minutes
        
        return np.nan

    def clean_weather(self, data):
        # Clean wind data (as before)
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
        
        # Clean temperature data
        data['temperature'] = data['temperature'].apply(self.parse_temperature)
        
        # Impute NaN temperatures and add flag
        self.temp_median = data['temperature'].median()
        
        # Explicitly create the temp_was_missing column
        data['temp_was_missing'] = data['temperature'].isna().astype(int)
        
        # Fill NaN values in temperature
        data['temperature'] = data['temperature'].fillna(self.temp_median)

        #place temp was missing column after temperature
        data = data[['temperature', 'temp_was_missing'] + [col for col in data.columns if col not in ['temperature', 'temp_was_missing']]]
        
        # Process weather column
        data = self.process_weather_column(data)

        return data

    def parse_temperature(self, temp):
        if pd.isna(temp):
            return np.nan
        
        if isinstance(temp, (int, float)):
            return temp
        
        temp = str(temp).lower()
        
        # Extract the first number from the string
        match = re.search(r'-?\d+', temp)
        if match:
            value = int(match.group())
            
            # Convert Fahrenheit to Celsius if 'f' is in the string
            if 'f' in temp:
                return (value - 32) * 5/9
            
            # Handle 'mid' cases (e.g., 'Mid 80s')
            if 'mid' in temp:
                return value + 5  # Assuming 'Mid 80s' means around 85
            
            # Handle decade cases (e.g., '40s')
            if temp.endswith('s'):
                return value + 5  # Assuming '40s' means around 45
            
            return value
        
        return np.nan

    def process_weather_column(self, data):
        # Fill NaN values with 'Unknown'
        data['weather'] = data['weather'].fillna('Unknown')

        # Convert to lowercase
        data['weather'] = data['weather'].str.lower()

        # Extract main weather conditions
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

        # Create binary flags
        data['is_windy'] = data['weather'].str.contains('wind').astype(int)
        data['is_foggy'] = data['weather'].str.contains('fog').astype(int)

        # Create cloud cover scale
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

        #if cloud cover is nan, fill with median
        data['cloud_cover'] = data['cloud_cover'].fillna(data['cloud_cover'].median())

        # One-hot encode main weather conditions
        main_conditions = data['weather'].apply(extract_main_condition)
        encoded_conditions = self.mlb.fit_transform(main_conditions)
        condition_columns = self.mlb.classes_
        for i, condition in enumerate(condition_columns):
            data[f'weather_{condition}'] = encoded_conditions[:, i]

        # Drop original weather column
        data = data.drop('weather', axis=1)

        return data

    def clean_data(self, data):
        data = self.clean_weather(data)

        #covert duration into toal minutes
        #Example: 2:59 -> 179
        data['duration'] = data['duration'].apply(self.convert_time_to_minutes)

        # Normalize kickoff time
        data['kickoff_time'] = data['kickoff_time'].apply(self.normalize_kickoff_time)

        #normalize is_home_game to 1 and 0
        data['is_home_game'] = data['is_home_game'].astype(int)

        # Generate game count
        data = self.generate_game_count(data)

        #foreach column name remove periods on end
        data.columns = data.columns.str.rstrip('.')

        #move name and season to front of dataframe
        data = data[['name', 'season'] + [col for col in data.columns if col not in ['name', 'season']]]

        #sort by name and season
        data = data.sort_values(by=['name', 'season'])

        #rows with NaN values remove
        data = data.dropna()


        return data