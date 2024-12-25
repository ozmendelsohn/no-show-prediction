import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path
import os
import json
import logging

logger = logging.getLogger(__name__)

class WeatherLoader:
    """Class to load weather data from OpenWeatherMap API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the weather loader.
        
        Parameters
        ----------
        api_key : Optional[str]
            OpenWeatherMap API key. If not provided, will look for OPENWEATHER_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key must be provided or set in OPENWEATHER_API_KEY environment variable")
        
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.cache_dir = Path("data/weather_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, lat: float, lon: float, date: datetime) -> Path:
        """Get the cache file path for a specific location and date."""
        date_str = date.strftime("%Y%m%d")
        return self.cache_dir / f"weather_{lat}_{lon}_{date_str}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load weather data from cache if available."""
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache file {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save weather data to cache."""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Error saving to cache file {cache_path}: {e}")
    
    def get_weather(self, lat: float, lon: float, date: datetime) -> Dict[str, Any]:
        """
        Get weather data for a specific location and date.
        
        Parameters
        ----------
        lat : float
            Latitude
        lon : float
            Longitude
        date : datetime
            Date to get weather for
        
        Returns
        -------
        Dict[str, Any]
            Weather data including temperature, rain, and conditions
        """
        cache_path = self._get_cache_path(lat, lon, date)
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_path)
        if cached_data:
            return cached_data
        
        # If date is in the past, use historical data endpoint
        if date.date() < datetime.now().date():
            endpoint = f"{self.base_url}/onecall/timemachine"
            timestamp = int(date.timestamp())
            params = {
                "lat": lat,
                "lon": lon,
                "dt": timestamp,
                "appid": self.api_key,
                "units": "metric"
            }
        else:
            # For current/future dates, use forecast endpoint
            endpoint = f"{self.base_url}/forecast"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process and format the weather data
            weather_data = self._process_weather_data(data, date)
            
            # Cache the processed data
            self._save_to_cache(cache_path, weather_data)
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_default_weather_data()
    
    def _process_weather_data(self, raw_data: Dict, target_date: datetime) -> Dict[str, Any]:
        """Process raw weather API response into standardized format."""
        try:
            if 'daily' in raw_data:  # Historical data
                day_data = raw_data['daily'][0]
            else:  # Forecast data
                # Find the forecast closest to our target date
                forecasts = raw_data['list']
                target_timestamp = target_date.timestamp()
                day_data = min(forecasts, key=lambda x: abs(x['dt'] - target_timestamp))
            
            # Extract temperature data
            temp_data = day_data.get('temp', {})
            if isinstance(temp_data, dict):
                avg_temp = (temp_data.get('min', 0) + temp_data.get('max', 0)) / 2
                max_temp = temp_data.get('max', 0)
            else:
                avg_temp = temp_data
                max_temp = temp_data
            
            # Extract rain data
            rain_data = day_data.get('rain', {})
            if isinstance(rain_data, dict):
                rain_amount = rain_data.get('1h', 0)
            else:
                rain_amount = rain_data or 0
            
            # Determine rain intensity
            if rain_amount == 0:
                rain_intensity = 'no_rain'
            elif rain_amount < 2.5:
                rain_intensity = 'light'
            elif rain_amount < 7.6:
                rain_intensity = 'moderate'
            else:
                rain_intensity = 'heavy'
            
            # Determine heat intensity
            if avg_temp < 20:
                heat_intensity = 'mild'
            elif avg_temp < 25:
                heat_intensity = 'moderate'
            elif avg_temp < 30:
                heat_intensity = 'high'
            else:
                heat_intensity = 'extreme'
            
            return {
                'average_temp_day': avg_temp,
                'max_temp_day': max_temp,
                'average_rain_day': rain_amount,
                'max_rain_day': rain_amount,  # Using same value since we only have daily data
                'rain_intensity': rain_intensity,
                'heat_intensity': heat_intensity,
                'weather_condition': day_data.get('weather', [{}])[0].get('main', 'unknown'),
                'weather_description': day_data.get('weather', [{}])[0].get('description', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error processing weather data: {e}")
            return self._get_default_weather_data()
    
    def _get_default_weather_data(self) -> Dict[str, Any]:
        """Return default weather data when API call fails."""
        return {
            'average_temp_day': 20,
            'max_temp_day': 25,
            'average_rain_day': 0,
            'max_rain_day': 0,
            'rain_intensity': 'unknown',
            'heat_intensity': 'unknown',
            'weather_condition': 'unknown',
            'weather_description': 'unknown'
        }
    
    def add_weather_features(self, df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
        """
        Add weather features to a DataFrame containing appointment data.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing appointment data with 'AppointmentDay' column
        lat : float
            Latitude of the appointment location
        lon : float
            Longitude of the appointment location
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added weather features
        """
        if 'AppointmentDay' not in df.columns:
            raise ValueError("DataFrame must contain 'AppointmentDay' column")
        
        # Convert AppointmentDay to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['AppointmentDay']):
            df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
        
        # Create new columns for weather features
        weather_features = []
        
        for idx, row in df.iterrows():
            appointment_date = row['AppointmentDay']
            weather_data = self.get_weather(lat, lon, appointment_date)
            
            # Get weather from previous day for storm/rain indicators
            prev_day = appointment_date - timedelta(days=1)
            prev_weather = self.get_weather(lat, lon, prev_day)
            
            # Add previous day indicators
            weather_data['rainy_day_before'] = prev_weather['rain_intensity'] != 'no_rain'
            weather_data['storm_day_before'] = prev_weather['weather_condition'] == 'Thunderstorm'
            
            weather_features.append(weather_data)
        
        # Convert weather features to DataFrame
        weather_df = pd.DataFrame(weather_features)
        
        # Add weather features to original DataFrame
        for col in weather_df.columns:
            df[col] = weather_df[col]
        
        return df 