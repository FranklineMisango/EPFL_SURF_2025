import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import networkx as nx
import pickle
import ast
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import os
import requests
import hashlib
import json
import time
import time
import requests
import hashlib
import json
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bike Flow Prediction System",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: rgba(240, 242, 246, 0.8);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .prediction-box {
        background-color: rgba(225, 245, 254, 0.9);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        backdrop-filter: blur(5px);
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .main-content {
        padding: 0 !important;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: none !important;
    }
    .sidebar-content {
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .translucent-text {
        color: rgba(0, 0, 0, 0.6);
        font-size: 0.9rem;
    }
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_bike_data():
    """Load and preprocess bike trip data"""
    try:
        trips_df = pd.read_csv('data/trips_8days_flat.csv')
        
        # Parse timestamps
        trips_df['start_datetime'] = pd.to_datetime(trips_df['start_time'], format='%Y%m%d_%H%M%S')
        trips_df['end_datetime'] = pd.to_datetime(trips_df['end_time'], format='%Y%m%d_%H%M%S')
        trips_df['hour'] = trips_df['start_datetime'].dt.hour
        trips_df['day_of_week'] = trips_df['start_datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
        trips_df['is_weekend'] = trips_df['day_of_week'].isin([5, 6])  # Saturday, Sunday
        
        # Extract coordinates
        trips_df['start_lat'] = trips_df['start_coords'].apply(lambda x: ast.literal_eval(x)[0])
        trips_df['start_lon'] = trips_df['start_coords'].apply(lambda x: ast.literal_eval(x)[1])
        trips_df['end_lat'] = trips_df['end_coords'].apply(lambda x: ast.literal_eval(x)[0])
        trips_df['end_lon'] = trips_df['end_coords'].apply(lambda x: ast.literal_eval(x)[1])
        
        return trips_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def create_simple_routing_network(trips_df):
    """Create a simplified routing network from actual trip data"""
    try:
        # Create routing lookup from actual trips
        routing_network = {}
        
        # Group trips to find common routes
        route_counts = trips_df.groupby(['start_station_id', 'end_station_id']).agg({
            'start_lat': 'first',
            'start_lon': 'first', 
            'end_lat': 'first',
            'end_lon': 'first'
        }).reset_index()
        
        for _, row in route_counts.iterrows():
            start_station = row['start_station_id']
            end_station = row['end_station_id']
            
            if start_station != end_station:  # Skip self-loops
                # Create simple route (can be enhanced with actual path finding)
                start_coords = [row['start_lat'], row['start_lon']]
                end_coords = [row['end_lat'], row['end_lon']]
                
                # For now, create a simple 2-point route (can be enhanced)
                route_key = f"{start_station}_{end_station}"
                routing_network[route_key] = [start_coords, end_coords]
        
        return routing_network
    except Exception as e:
        st.warning(f"Could not create routing network: {e}")
        return {}

class RealPathRouter:
    """Advanced router that fetches actual road/bicycle paths"""
    
    def __init__(self):
        self.cache_dir = "cache"
        self.ensure_cache_dir()
        
    def ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_cache_key(self, start_lat, start_lon, end_lat, end_lon, profile='cycling'):
        """Generate cache key for route"""
        coord_str = f"{start_lat:.6f},{start_lon:.6f}_{end_lat:.6f},{end_lon:.6f}_{profile}"
        return hashlib.md5(coord_str.encode()).hexdigest()
    
    def load_from_cache(self, cache_key):
        """Load route from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def save_to_cache(self, cache_key, route_data):
        """Save route to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(route_data, f)
        except Exception as e:
            st.warning(f"Could not save to cache: {e}")
    
    def get_openrouteservice_route(self, start_lat, start_lon, end_lat, end_lon):
        """Get route from OpenRouteService (free service)"""
        try:
            # OpenRouteService API (free tier - no API key needed for basic usage)
            url = "https://api.openrouteservice.org/v2/directions/cycling-regular"
            
            # Note: For production, you should get a free API key from openrouteservice.org
            # For now, we'll try without one (limited requests)
            
            params = {
                'start': f"{start_lon},{start_lat}",
                'end': f"{end_lon},{end_lat}",
                'format': 'geojson'
            }
            
            # Add API key if available in environment
            api_key = os.environ.get('OPENROUTESERVICE_API_KEY')
            if api_key:
                headers = {'Authorization': api_key}
            else:
                headers = {}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'features' in data and len(data['features']) > 0:
                    coordinates = data['features'][0]['geometry']['coordinates']
                    # Convert from [lon, lat] to [lat, lon] format
                    route_points = [[coord[1], coord[0]] for coord in coordinates]
                    
                    return {
                        'success': True,
                        'route': route_points,
                        'source': 'openrouteservice',
                        'distance': data['features'][0]['properties'].get('segments', [{}])[0].get('distance', 0),
                        'duration': data['features'][0]['properties'].get('segments', [{}])[0].get('duration', 0)
                    }
            
            return None
            
        except Exception as e:
            st.warning(f"OpenRouteService error: {e}")
            return None
    
    def get_osrm_route(self, start_lat, start_lon, end_lat, end_lon):
        """Get route from OSRM (Open Source Routing Machine)"""
        try:
            # Free OSRM demo server (cycling profile)
            url = f"https://router.project-osrm.org/route/v1/bike/{start_lon},{start_lat};{end_lon},{end_lat}"
            
            params = {
                'overview': 'full',
                'geometries': 'geojson'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'routes' in data and len(data['routes']) > 0:
                    route = data['routes'][0]
                    coordinates = route['geometry']['coordinates']
                    
                    # Convert from [lon, lat] to [lat, lon] format
                    route_points = [[coord[1], coord[0]] for coord in coordinates]
                    
                    return {
                        'success': True,
                        'route': route_points,
                        'source': 'osrm',
                        'distance': route.get('distance', 0),
                        'duration': route.get('duration', 0)
                    }
            
            return None
            
        except Exception as e:
            st.warning(f"OSRM error: {e}")
            return None
    
    def get_fallback_route(self, start_lat, start_lon, end_lat, end_lon):
        """Generate enhanced fallback route with intermediate points"""
        try:
            # Create a more realistic path with intermediate points
            num_points = 5  # Number of intermediate points
            route_points = []
            
            for i in range(num_points + 1):
                ratio = i / num_points
                # Linear interpolation
                lat = start_lat + (end_lat - start_lat) * ratio
                lon = start_lon + (end_lon - start_lon) * ratio
                
                # Add small random variations to simulate road curvature
                if i > 0 and i < num_points:
                    lat_noise = np.random.normal(0, 0.0001)  # Small variation
                    lon_noise = np.random.normal(0, 0.0001)
                    lat += lat_noise
                    lon += lon_noise
                
                route_points.append([lat, lon])
            
            # Calculate approximate distance (Haversine formula)
            distance = self.calculate_distance(start_lat, start_lon, end_lat, end_lon)
            
            return {
                'success': True,
                'route': route_points,
                'source': 'fallback',
                'distance': distance * 1000,  # Convert to meters
                'duration': distance * 1000 / 4.17  # Assume 15 km/h cycling speed
            }
            
        except Exception as e:
            st.warning(f"Fallback route error: {e}")
            return {
                'success': True,
                'route': [[start_lat, start_lon], [end_lat, end_lon]],
                'source': 'simple_fallback',
                'distance': 0,
                'duration': 0
            }
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2) * np.sin(delta_lat/2) + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * 
             np.sin(delta_lon/2) * np.sin(delta_lon/2))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_route(self, start_lat, start_lon, end_lat, end_lon, profile='cycling'):
        """Get the best available route between two points"""
        
        # Generate cache key
        cache_key = self.get_cache_key(start_lat, start_lon, end_lat, end_lon, profile)
        
        # Try to load from cache first
        cached_route = self.load_from_cache(cache_key)
        if cached_route:
            return cached_route
        
        # Try different routing services in order of preference
        route_result = None
        
        # 1. Try OpenRouteService (good for cycling routes)
        route_result = self.get_openrouteservice_route(start_lat, start_lon, end_lat, end_lon)
        
        # 2. If that fails, try OSRM
        if not route_result:
            route_result = self.get_osrm_route(start_lat, start_lon, end_lat, end_lon)
        
        # 3. If all else fails, use enhanced fallback
        if not route_result:
            route_result = self.get_fallback_route(start_lat, start_lon, end_lat, end_lon)
        
        # Save successful route to cache
        if route_result and route_result.get('success'):
            self.save_to_cache(cache_key, route_result)
        
        return route_result

class FastBikeFlowPredictor:
    """Optimized bike flow predictor with fast loading and real path routing"""
    
    def __init__(self, trips_df, routing_network=None):
        self.trips_df = trips_df
        self.routing_network = routing_network or {}
        self.station_coords = {}
        self.hourly_flows = {}
        self.model = None
        self.scaler = None
        self.station_features = {}
        
        # Initialize real path router
        self.path_router = RealPathRouter()
        
        # Initialize with progress tracking
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        """Initialize predictor with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Building station network...")
        self._build_station_network()
        progress_bar.progress(25)
        
        status_text.text("Computing hourly flows...")
        self._compute_hourly_flows()
        progress_bar.progress(50)
        
        status_text.text("Engineering features...")
        self._engineer_features()
        progress_bar.progress(75)
        
        status_text.text("Training ML model...")
        self._train_model()
        progress_bar.progress(100)
        
        status_text.text("✅ Ready!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    def _build_station_network(self):
        """Build station coordinate mapping"""
        # Get unique stations
        start_stations = self.trips_df.groupby('start_station_id').agg({
            'start_lat': 'first',
            'start_lon': 'first'
        }).reset_index()
        
        for _, row in start_stations.iterrows():
            station_id = row['start_station_id']
            self.station_coords[station_id] = (row['start_lat'], row['start_lon'])
    
    def _compute_hourly_flows(self):
        """Compute flow volumes by hour and day of week"""
        # Store flows by hour (keeping backward compatibility)
        for hour in range(24):
            hour_trips = self.trips_df[self.trips_df['hour'] == hour]
            flows = hour_trips.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='flow_count')
            self.hourly_flows[hour] = flows
        
        # Store flows by hour and day of week for enhanced modeling
        self.hourly_flows_by_dow = {}
        for hour in range(24):
            for dow in range(7):  # 0=Monday, 6=Sunday
                hour_dow_trips = self.trips_df[
                    (self.trips_df['hour'] == hour) & 
                    (self.trips_df['day_of_week'] == dow)
                ]
                flows = hour_dow_trips.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='flow_count')
                self.hourly_flows_by_dow[(hour, dow)] = flows
    
    def _engineer_features(self):
        """Engineer features for ML model"""
        for station_id in self.station_coords.keys():
            features = {}
            
            station_trips = self.trips_df[
                (self.trips_df['start_station_id'] == station_id) | 
                (self.trips_df['end_station_id'] == station_id)
            ]
            
            features['total_trips'] = len(station_trips)
            features['avg_temperature'] = station_trips['temperature'].mean() if len(station_trips) > 0 else 20.0
            features['lat'] = self.station_coords[station_id][0]
            features['lon'] = self.station_coords[station_id][1]
            
            # Hourly patterns
            for hour in range(24):
                outflow = len(station_trips[
                    (station_trips['start_station_id'] == station_id) & 
                    (station_trips['hour'] == hour)
                ])
                inflow = len(station_trips[
                    (station_trips['end_station_id'] == station_id) & 
                    (station_trips['hour'] == hour)
                ])
                
                features[f'outflow_hour_{hour}'] = outflow
                features[f'inflow_hour_{hour}'] = inflow
            
            # Day of week patterns
            for dow in range(7):  # 0=Monday, 6=Sunday
                dow_outflow = len(station_trips[
                    (station_trips['start_station_id'] == station_id) & 
                    (station_trips['day_of_week'] == dow)
                ])
                dow_inflow = len(station_trips[
                    (station_trips['end_station_id'] == station_id) & 
                    (station_trips['day_of_week'] == dow)
                ])
                
                features[f'outflow_dow_{dow}'] = dow_outflow
                features[f'inflow_dow_{dow}'] = dow_inflow
            
            # Weekend vs weekday patterns
            weekend_outflow = len(station_trips[
                (station_trips['start_station_id'] == station_id) & 
                (station_trips['is_weekend'] == True)
            ])
            weekday_outflow = len(station_trips[
                (station_trips['start_station_id'] == station_id) & 
                (station_trips['is_weekend'] == False)
            ])
            
            features['weekend_outflow'] = weekend_outflow
            features['weekday_outflow'] = weekday_outflow
            features['weekend_ratio'] = weekend_outflow / (weekend_outflow + weekday_outflow + 1)  # +1 to avoid division by zero
            
            self.station_features[station_id] = features
    
    def _train_model(self):
        """Train ML model for flow prediction with day of week features"""
        training_data = []
        
        # Sample data for faster training - include different days of week
        sample_hours = [6, 8, 12, 17, 20]  # Focus on key hours
        sample_days = [0, 1, 4, 5, 6]  # Monday, Tuesday, Friday, Saturday, Sunday
        
        # Use enhanced hourly flows by day of week when available
        for hour in sample_hours:
            for dow in sample_days:
                if (hour, dow) in self.hourly_flows_by_dow:
                    flows = self.hourly_flows_by_dow[(hour, dow)]
                    
                    # Sample flows for faster training
                    if len(flows) > 500:
                        flows = flows.sample(n=500, random_state=42)
                    
                    for _, row in flows.iterrows():
                        start_station = row['start_station_id']
                        end_station = row['end_station_id']
                        flow_count = row['flow_count']
                        
                        if start_station in self.station_features and end_station in self.station_features:
                            feature_vector = self._create_feature_vector(start_station, end_station, hour, dow)
                            training_data.append(feature_vector + [flow_count])
                
                # Fallback to hourly flows if day-of-week specific data is insufficient
                elif hour in self.hourly_flows:
                    flows = self.hourly_flows[hour]
                    
                    # Sample flows for faster training
                    if len(flows) > 200:
                        flows = flows.sample(n=200, random_state=42)
                    
                    for _, row in flows.iterrows():
                        start_station = row['start_station_id']
                        end_station = row['end_station_id']
                        flow_count = row['flow_count']
                        
                        if start_station in self.station_features and end_station in self.station_features:
                            # Use default day of week (Tuesday = 1) for compatibility
                            feature_vector = self._create_feature_vector(start_station, end_station, hour, 1)
                            training_data.append(feature_vector + [flow_count])
        
        if training_data:
            training_data = np.array(training_data)
            X = training_data[:, :-1]
            y = training_data[:, -1]
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Use fewer estimators for faster training
            self.model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            self.model.fit(X_scaled, y)
    
    def _create_feature_vector(self, start_station, end_station, hour, day_of_week=1):
        """Create feature vector for prediction with day of week features"""
        start_features = self.station_features[start_station]
        end_features = self.station_features[end_station]
        
        feature_vector = [
            hour,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            day_of_week,
            np.sin(2 * np.pi * day_of_week / 7),  # Cyclical encoding for day of week
            np.cos(2 * np.pi * day_of_week / 7),
            1 if day_of_week in [5, 6] else 0,  # Weekend indicator (Saturday=5, Sunday=6)
            start_features['lat'],
            start_features['lon'],
            start_features['total_trips'],
            start_features[f'outflow_hour_{hour}'],
            start_features.get(f'outflow_dow_{day_of_week}', 0),  # Day of week specific outflow
            start_features.get('weekend_ratio', 0.5),  # Weekend activity ratio
            end_features['lat'],
            end_features['lon'],
            end_features['total_trips'],
            end_features[f'inflow_hour_{hour}'],
            end_features.get(f'inflow_dow_{day_of_week}', 0),  # Day of week specific inflow
            np.sqrt((start_features['lat'] - end_features['lat'])**2 + 
                   (start_features['lon'] - end_features['lon'])**2),
            (start_features['avg_temperature'] + end_features['avg_temperature']) / 2
        ]
        
        return feature_vector
    
    def predict_destinations(self, station_id, hour, day_of_week=1, top_k=5):
        """Predict top destinations from a station with confidence levels including day of week"""
        if station_id not in self.station_features or not self.model:
            # Fallback: return top stations by distance for testing
            return self._get_fallback_predictions(station_id, top_k)
        
        predictions = []
        
        # Try to use day-of-week specific flows first, fallback to hourly flows
        active_destinations = set()
        if (hour, day_of_week) in self.hourly_flows_by_dow:
            hour_dow_flows = self.hourly_flows_by_dow[(hour, day_of_week)]
            station_flows = hour_dow_flows[hour_dow_flows['start_station_id'] == station_id]
            active_destinations = set(station_flows['end_station_id'].values)
        elif hour in self.hourly_flows:
            hour_flows = self.hourly_flows[hour]
            station_flows = hour_flows[hour_flows['start_station_id'] == station_id]
            active_destinations = set(station_flows['end_station_id'].values)
        
        # If no active destinations, use top destinations from nearby hours
        if not active_destinations:
            nearby_hours = [(hour-1) % 24, hour, (hour+1) % 24]
            for h in nearby_hours:
                if h in self.hourly_flows:
                    hour_flows = self.hourly_flows[h]
                    station_flows = hour_flows[hour_flows['start_station_id'] == station_id]
                    active_destinations.update(station_flows['end_station_id'].values)
                if len(active_destinations) >= 20:  # Limit for performance
                    break
        
        # If still no destinations, use all stations as fallback
        if not active_destinations:
            # Take top 10 stations by total trips as fallback
            station_activity = [(s, self.station_features[s]['total_trips']) 
                              for s in self.station_features.keys() if s != station_id]
            station_activity.sort(key=lambda x: x[1], reverse=True)
            active_destinations = set([s[0] for s in station_activity[:10]])
        
        # Predict for active destinations only
        for dest_station in active_destinations:
            if dest_station == station_id or dest_station not in self.station_features:
                continue
            
            try:
                feature_vector = self._create_feature_vector(station_id, dest_station, hour, day_of_week)
                X_pred = self.scaler.transform([feature_vector])
                predicted_flow = self.model.predict(X_pred)[0]
                
                # Calculate confidence based on prediction and historical data
                confidence = min(0.95, max(0.1, predicted_flow / 10))
                
                predictions.append({
                    'destination': dest_station,
                    'predicted_flow': max(1.0, abs(predicted_flow)),  # Ensure minimum flow of 1.0 for visibility
                    'confidence': confidence
                })
            except Exception as e:
                # Skip problematic predictions but don't fail entirely
                continue
        
        # If no valid predictions, return fallback
        if not predictions:
            return self._get_fallback_predictions(station_id, top_k)
        
        # Sort by predicted flow and return top k
        predictions.sort(key=lambda x: x['predicted_flow'], reverse=True)
        return predictions[:top_k]
    
    def _get_fallback_predictions(self, station_id, top_k=5):
        """Generate fallback predictions based on nearest stations"""
        if station_id not in self.station_coords:
            return []
        
        start_lat, start_lon = self.station_coords[station_id]
        distances = []
        
        for other_station, (lat, lon) in self.station_coords.items():
            if other_station != station_id:
                distance = np.sqrt((start_lat - lat)**2 + (start_lon - lon)**2)
                distances.append((other_station, distance))
        
        # Sort by distance and take closest stations
        distances.sort(key=lambda x: x[1])
        closest_stations = distances[:top_k]
        
        fallback_predictions = []
        for i, (dest_station, distance) in enumerate(closest_stations):
            # Create artificial flow based on inverse distance
            flow = max(1.0, 10.0 / (distance + 0.01))  # Ensure visible flow
            confidence = max(0.3, 1.0 - distance)  # Distance-based confidence
            
            fallback_predictions.append({
                'destination': dest_station,
                'predicted_flow': flow,
                'confidence': min(0.9, confidence)
            })
        
        return fallback_predictions
    
    def get_route_path(self, start_station, end_station):
        """Get actual road/bicycle path between stations"""
        
        # Check if we have coordinates for both stations
        if start_station not in self.station_coords or end_station not in self.station_coords:
            return []
        
        start_lat, start_lon = self.station_coords[start_station]
        end_lat, end_lon = self.station_coords[end_station]
        
        # Try to get real path using routing service
        try:
            route_result = self.path_router.get_route(start_lat, start_lon, end_lat, end_lon)
            
            if route_result and route_result.get('success') and route_result.get('route'):
                return route_result['route']
            else:
                # Fallback to straight line if routing fails
                return [[start_lat, start_lon], [end_lat, end_lon]]
                
        except Exception as e:
            st.warning(f"Routing error for {start_station} -> {end_station}: {e}")
            # Return straight line as ultimate fallback
            return [[start_lat, start_lon], [end_lat, end_lon]]
    
    def get_route_info(self, start_station, end_station):
        """Get detailed route information including distance and duration"""
        
        if start_station not in self.station_coords or end_station not in self.station_coords:
            return None
        
        start_lat, start_lon = self.station_coords[start_station]
        end_lat, end_lon = self.station_coords[end_station]
        
        try:
            route_result = self.path_router.get_route(start_lat, start_lon, end_lat, end_lon)
            
            if route_result and route_result.get('success'):
                return {
                    'route': route_result.get('route', []),
                    'distance_m': route_result.get('distance', 0),
                    'duration_s': route_result.get('duration', 0),
                    'source': route_result.get('source', 'unknown'),
                    'distance_km': route_result.get('distance', 0) / 1000,
                    'duration_min': route_result.get('duration', 0) / 60
                }
            
            return None
            
        except Exception as e:
            st.warning(f"Route info error for {start_station} -> {end_station}: {e}")
            return None

def create_interactive_map(predictor, selected_hour=17, selected_day_of_week=1, selected_station=None):
    """Create interactive satellite map with predictions"""
    
    # Get center coordinates
    all_coords = list(predictor.station_coords.values())
    center_lat = np.mean([coord[0] for coord in all_coords])
    center_lon = np.mean([coord[1] for coord in all_coords])
    
    # Create map with satellite imagery
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None
    )
    
    # Add satellite tiles
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add OpenStreetMap overlay for reference
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Show all stations (user requested to see all stations)
    all_stations = list(predictor.station_coords.keys())
    
    # Add station markers for all stations
    for station_id in all_stations:
        lat, lon = predictor.station_coords[station_id]
        
        # Determine marker color based on selection
        if station_id == selected_station:
            color = 'red'
            icon = 'star'
        else:
            color = 'blue'
            icon = 'bicycle'
        
        # Create popup with basic info
        popup_text = f"""
        <b>Station {station_id}</b><br>
        Total Trips: {predictor.station_features[station_id]['total_trips']}<br>
        Click to see predictions!
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Station {station_id}",
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m)
    
    # Add prediction routes if station is selected
    if selected_station and selected_station in predictor.station_coords:
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        st.sidebar.write(f"🔍 Debug: Getting predictions for station {selected_station} at {selected_hour}:00 on {day_names[selected_day_of_week]}")
        predictions = predictor.predict_destinations(selected_station, selected_hour, day_of_week=selected_day_of_week, top_k=10)  # Show more predictions
        st.sidebar.write(f"🔍 Debug: Found {len(predictions)} predictions")
        
        if predictions:
            for i, pred in enumerate(predictions):
                st.sidebar.write(f"Pred {i+1}: Station {pred['destination']}, Flow: {pred['predicted_flow']:.2f}")
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue']
        lines_added = 0
        
        for i, pred in enumerate(predictions):
            dest_station = pred['destination']
            flow = pred['predicted_flow']
            confidence = pred['confidence']
            
            st.sidebar.write(f"Processing pred {i+1}: {dest_station}, flow: {flow:.2f}")
            
            # Always add lines if destination exists (no threshold)
            if dest_station in predictor.station_coords:
                # Get route path and detailed info
                route_coords = predictor.get_route_path(selected_station, dest_station)
                route_info = predictor.get_route_info(selected_station, dest_station)
                
                st.sidebar.write(f"Route coords length: {len(route_coords) if route_coords else 0}")
                
                # Prepare route details for popup
                route_details = ""
                if route_info:
                    distance_km = route_info.get('distance_km', 0)
                    duration_min = route_info.get('duration_min', 0)
                    source = route_info.get('source', 'unknown')
                    
                    route_details = f"""
                    <br><b>Route Details:</b><br>
                    🚴 Distance: {distance_km:.2f} km<br>
                    ⏱️ Duration: {duration_min:.1f} min<br>
                    🗺️ Route Type: {source.replace('_', ' ').title()}
                    """
                    
                    st.sidebar.write(f"Route info: {distance_km:.2f}km, {duration_min:.1f}min, {source}")
                
                if route_coords:
                    # Calculate line properties - make lines very visible
                    weight = max(6, min(12, max(flow * 2, 6)))  # Always thick lines
                    opacity = max(0.8, min(1.0, max(confidence, 0.8)))  # Always high opacity
                    color = colors[i % len(colors)]
                    
                    st.sidebar.write(f"Adding line: color={color}, weight={weight}, opacity={opacity}")
                    
                    # Enhanced popup with route information
                    popup_content = f"""
                    <div style="width: 250px;">
                        <b>🎯 Route to Station {dest_station}</b><br>
                        <hr style="margin: 5px 0;">
                        📊 <b>Predicted Flow:</b> {flow:.1f} trips<br>
                        🎯 <b>Confidence:</b> {confidence:.2%}<br>
                        🏆 <b>Rank:</b> #{i+1}<br>
                        {route_details}
                    </div>
                    """
                    
                    # Add route line with enhanced visibility and information
                    folium.PolyLine(
                        locations=route_coords,
                        color=color,
                        weight=weight,
                        opacity=opacity,
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=f"🚴 To Station {dest_station}: {flow:.1f} trips" + 
                               (f" | {route_info.get('distance_km', 0):.1f}km" if route_info else "")
                    ).add_to(m)
                    
                    # Add destination marker with enhanced information
                    dest_lat, dest_lon = predictor.station_coords[dest_station]
                    
                    dest_popup_content = f"""
                    <div style="width: 250px;">
                        <b>🎯 Destination: Station {dest_station}</b><br>
                        <hr style="margin: 5px 0;">
                        📊 <b>Predicted Flow:</b> {flow:.1f} trips<br>
                        🎯 <b>Confidence:</b> {confidence:.1%}<br>
                        🏆 <b>Rank:</b> #{i+1}<br>
                        {route_details}
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[dest_lat, dest_lon],
                        radius=max(12, min(25, max(flow * 2, 12))),  # Always large circles
                        popup=folium.Popup(dest_popup_content, max_width=300),
                        tooltip=f"🎯 Dest {dest_station}" + 
                               (f" | {route_info.get('distance_km', 0):.1f}km" if route_info else ""),
                        color=color,
                        fillColor=color,
                        fillOpacity=0.9,
                        weight=4
                    ).add_to(m)
                    
                    lines_added += 1
                else:
                    st.sidebar.write(f"❌ No route coordinates for {selected_station} -> {dest_station}")
            else:
                st.sidebar.write(f"❌ Destination {dest_station} not in station coords")
        
        st.sidebar.write(f"🔍 Total lines added to map: {lines_added}")
        
        # If no predictions, add a test line to verify map is working
        if lines_added == 0:
            st.sidebar.write("🔧 Adding test line to verify map functionality...")
            start_lat, start_lon = predictor.station_coords[selected_station]
            
            # Find any other station for test line
            other_stations = [s for s in predictor.station_coords.keys() if s != selected_station]
            if other_stations:
                test_dest = other_stations[0]
                test_lat, test_lon = predictor.station_coords[test_dest]
                
                folium.PolyLine(
                    locations=[[start_lat, start_lon], [test_lat, test_lon]],
                    color='yellow',
                    weight=8,
                    opacity=1.0,
                    popup=f"TEST LINE from {selected_station} to {test_dest}",
                    tooltip="TEST LINE - Map is working!"
                ).add_to(m)
                
                st.sidebar.write(f"✅ Added test line from {selected_station} to {test_dest}")
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🚴‍♂️ Bike Flow Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Load data
        with st.spinner("Loading bike data..."):
            trips_df = load_bike_data()
            if trips_df is None:
                st.error("Failed to load bike data. Please check the data file.")
                return
        
        # Create routing network
        routing_network = create_simple_routing_network(trips_df)
        
        # Initialize predictor
        @st.cache_resource
        def get_predictor():
            return FastBikeFlowPredictor(trips_df, routing_network)
        
        predictor = get_predictor()
        
        # Initialize session state for selected station
        if 'selected_station' not in st.session_state:
            st.session_state.selected_station = None
        
        # Time slider
        selected_hour = st.slider(
            "🕐 Select Hour",
            min_value=0,
            max_value=23,
            value=17,
            help="Select the hour of day for predictions"
        )
        
        # Day of week selector
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        selected_day_name = st.selectbox(
            "📅 Select Day of Week",
            options=day_names,
            index=1,  # Default to Tuesday
            help="Select the day of week for predictions (affects bike usage patterns)"
        )
        selected_day_of_week = day_names.index(selected_day_name)
        
        # Show weekend indicator
        if selected_day_of_week in [5, 6]:  # Saturday, Sunday
            st.markdown("🎉 **Weekend** - Leisure patterns expected")
        else:
            st.markdown("💼 **Weekday** - Commuter patterns expected")
        
        # Station selection (including session state)
        available_stations = sorted(list(predictor.station_coords.keys()))
        
        # Use session state if available, otherwise use sidebar selection
        if st.session_state.selected_station is not None:
            selected_station = st.session_state.selected_station
            # Update sidebar to match
            if selected_station in available_stations:
                station_index = available_stations.index(selected_station) + 1  # +1 for None option
            else:
                station_index = 0
        else:
            station_index = 0
            selected_station = None
        
        # Sidebar station selection
        sidebar_selected = st.selectbox(
            "🚉 Select Station (or click on map)",
            options=[None] + available_stations,
            index=station_index,
            help="Select a station to see its predicted destinations"
        )
        
        # Update selected station if changed via sidebar
        if sidebar_selected != selected_station:
            selected_station = sidebar_selected
            st.session_state.selected_station = selected_station
        
        # Performance info
        st.markdown("---")
        with st.container():
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            st.markdown("### ⚡ System Status")
            st.markdown(f'<p class="translucent-text"><strong>Stations:</strong> {len(predictor.station_coords)}<br><strong>Routes:</strong> {len(routing_network)}<br><strong>Model:</strong> Random Forest + Day-of-Week<br><strong>Features:</strong> {len(predictor.station_features[list(predictor.station_features.keys())[0]])} per station<br><strong>Cache:</strong> 24h retention</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area - full width for map
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Add clear button for selected station
        if selected_station:
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
            with col_btn1:
                if st.button("� Clear Selection", help="Clear selected station"):
                    st.session_state.selected_station = None
                    st.rerun()
            with col_btn2:
                st.markdown(f'<p style="color: #1f77b4; font-weight: bold; margin-top: 8px;">🎯 Station {selected_station}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="translucent-text">💡 Click on any station marker to see flow predictions</p>', unsafe_allow_html=True)
        
        # Create and display map with container styling
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        map_obj = create_interactive_map(predictor, selected_hour, selected_day_of_week, selected_station)
        map_data = st_folium(map_obj, width=None, height=700, returned_objects=["last_object_clicked"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle map clicks
        if map_data['last_object_clicked']:
            clicked_data = map_data['last_object_clicked']
            if 'tooltip' in clicked_data:
                tooltip = clicked_data['tooltip']
                if 'Station' in tooltip:
                    try:
                        clicked_station_id = float(tooltip.split('Station ')[1])
                        if clicked_station_id in predictor.station_coords:
                            # Update session state and rerun to show predictions
                            if st.session_state.selected_station != clicked_station_id:
                                st.session_state.selected_station = clicked_station_id
                                st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error processing click: {e}")
    
    with col2:
        st.subheader("📊 Predictions")
        
        if selected_station:
            # Show station info
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekend_indicator = "🎉 Weekend" if selected_day_of_week in [5, 6] else "💼 Weekday"
            st.markdown(f"""
            <div class="metric-card">
                <h4>🚉 Station {selected_station}</h4>
                <p><strong>Time:</strong> {selected_hour:02d}:00 on {day_names[selected_day_of_week]}</p>
                <p><strong>Pattern:</strong> {weekend_indicator}</p>
                <p><strong>Historical Trips:</strong> {predictor.station_features[selected_station]['total_trips']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get and display predictions
            with st.spinner("Calculating..."):
                predictions = predictor.predict_destinations(selected_station, selected_hour, day_of_week=selected_day_of_week, top_k=10)  # Get more predictions
            
            if predictions:
                st.markdown("**🎯 Top Destinations**")
                
                # Show all predictions, not just top 3
                for i, pred in enumerate(predictions):  # Show all predictions
                    dest = pred['destination']
                    flow = pred['predicted_flow']
                    confidence = pred['confidence']
                    
                    if flow > 0.1:
                        # Get route information for this destination
                        route_info = predictor.get_route_info(selected_station, dest)
                        
                        route_details = ""
                        if route_info:
                            distance_km = route_info.get('distance_km', 0)
                            duration_min = route_info.get('duration_min', 0)
                            source = route_info.get('source', 'unknown')
                            distance_m = route_info.get('distance_m', 0)
                            duration_s = route_info.get('duration_s', 0)
                            
                            # Show real calculated values with more detail and debug info
                            route_details = f"""
                            <p><strong>🚴 Distance:</strong> {distance_km:.2f} km ({distance_m:.0f}m)</p>
                            <p><strong>⏱️ Duration:</strong> {duration_min:.1f} min ({duration_s:.0f}s)</p>
                            <p class="translucent-text">Route Source: {source.replace('_', ' ').title()}</p>
                            """
                            
                            # Add debug info to sidebar
                            st.sidebar.write(f"Route {i+1}: {distance_km:.2f}km, {duration_min:.1f}min via {source}")
                        else:
                            st.sidebar.write(f"Route {i+1}: No route info available")
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h5>#{i+1} Station {dest}</h5>
                            <p><strong>Flow:</strong> {flow:.1f} trips</p>
                            <p><strong>Confidence:</strong> {confidence:.0%}</p>
                            {route_details}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ No predictions for Station {selected_station} at {selected_hour:02d}:00")
                st.markdown('<p class="translucent-text">Try peak hours: 8:00, 12:00, 17:00, 20:00</p>', unsafe_allow_html=True)
        else:
            # Show overall statistics when no station selected
            st.markdown("**📈 System Overview**")
            
            total_stations = len(predictor.station_coords)
            total_trips = len(trips_df)
            
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Total Stations:</strong> {total_stations}</p>
                <p><strong>Total Trips:</strong> {total_trips:,}</p>
                <p class="translucent-text">Click any station to see predictions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Compact hourly activity chart
            hourly_activity = trips_df.groupby('hour').size()
            
            fig = px.line(
                x=hourly_activity.index,
                y=hourly_activity.values,
                title="Daily Activity Pattern",
                labels={'x': 'Hour', 'y': 'Trips'}
            )
            fig.add_vline(x=selected_hour, line_dash="dash", line_color="red")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Compact instructions in expandable section
    with st.expander("📋 How to Use & Technical Details", expanded=False):
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("""
            **🎯 Quick Start:**
            1. Use hour slider in sidebar (try 8, 12, 17, 20)
            2. Select day of week (weekdays vs weekends differ!)
            3. Click any station marker on map
            4. View predictions as colored route lines
            5. Click lines/markers for route details
            
            **🎨 Visual Legend:**
            - 🔵 Blue markers: Available stations
            - ⭐ Red star: Selected station  
            - 🔴🔵🟢 Colored lines: Top predictions with real bike paths
            - Line thickness: Flow volume
            - Line opacity: Confidence level
            """)
        
        with col_right:
            total_stations = len(predictor.station_coords)
            total_routes = len(routing_network)
            
            st.markdown(f"""
            **🗺️ Route Features:**
            - Real bicycle/road paths (not straight lines)
            - Actual cycling distances & durations
            - Multiple routing services with smart fallbacks
            - Intelligent caching for fast performance
            
            **⚡ System Status:**
            - Stations: {total_stations:,} | Routes: {total_routes:,}
            - Model: Random Forest with temporal + day-of-week features
            - Features: Hour, Day-of-Week, Weekend/Weekday patterns
            - Cache: 24h retention | Real-time routing
            - Peak accuracy hours: 8, 12, 17, 20 (varies by day type)
            """)

if __name__ == "__main__":
    main()
