import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class InteractiveCityDatabase:
    """Database of cities with station coordinates loaded from unique_stations.csv"""
    def __init__(self):
        self.stations_df = pd.read_csv("data/unique_stations.csv")
        self.stations_df['lat'] = pd.to_numeric(self.stations_df['lat'], errors='coerce')
        self.stations_df['lon'] = pd.to_numeric(self.stations_df['lon'], errors='coerce')
        # Optionally, add city assignment logic here if needed

    def get_available_cities(self) -> List[str]:
        # If you have a city column, return unique cities; else return ['All']
        if 'city' in self.stations_df.columns:
            return sorted(self.stations_df['city'].dropna().unique().tolist())
        return ['All']

    def get_city_stations(self, city: str) -> Dict[str, Tuple[float, float]]:
        if city == 'All':
            df = self.stations_df
        elif 'city' in self.stations_df.columns:
            df = self.stations_df[self.stations_df['city'] == city]
        else:
            df = self.stations_df
        return {str(row['station_id']): (row['lat'], row['lon']) for _, row in df.iterrows()}

    def get_city_metadata(self, city: str) -> Dict:
        # Not used, but could return stats
        return {}

    def get_city_info(self, city: str) -> Dict:
        stations = self.get_city_stations(city)
        return {
            'name': city,
            'stations': len(stations),
            'center': (np.mean([lat for lat, lon in stations.values()]), np.mean([lon for lat, lon in stations.values()])) if stations else (0, 0)
        }
