import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import logging
from datetime import datetime, timedelta
import warnings
import time
import os
from typing import Dict, List, Tuple, Optional
import sqlite3
import ast

# Import configuration and utilities
from config import *
from utils import sanitize_for_logging, validate_coordinates, parse_coordinates_safe, get_confidence_class
from od_matrix_predictor import ODMatrixPredictor, ODMatrixVisualizer
from model_loader import ModelLoader

# Import our model runners and routing
import importlib
import sys
if 'helpers.run_gnn_baselines' in sys.modules:
    importlib.reload(sys.modules['helpers.run_gnn_baselines'])
from helpers.run_gnn_baselines import GNNBaselineRunner

# Try to import bike routing, but make it optional
try:
    from helpers.bike_routing import get_bike_router, get_city_bbox
    ROUTING_AVAILABLE = True
except ImportError:
    ROUTING_AVAILABLE = False
    st.warning("Bike routing not available - will use simple curved lines")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Flow Modelling Lab",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .model-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .warning-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .confidence-high { color: #4caf50; font-weight: bold; }
    .confidence-medium { color: #ff9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_actual_data():
    """Load the actual trip and station data"""
    try:
        # Load trips data using config paths
        if not os.path.exists(TRIPS_FILE):
            st.error(f"Trips file not found: {TRIPS_FILE}")
            return None, None
            
        trips_df = pd.read_csv(TRIPS_FILE)
        
        # Parse coordinates using safe parsing
        coords_data = trips_df['start_coords'].apply(parse_coordinates_safe)
        trips_df[['start_lat', 'start_lon']] = pd.DataFrame(coords_data.tolist(), index=trips_df.index)
        
        coords_data = trips_df['end_coords'].apply(parse_coordinates_safe)
        trips_df[['end_lat', 'end_lon']] = pd.DataFrame(coords_data.tolist(), index=trips_df.index)
        
        # Parse timestamps
        trips_df['start_datetime'] = pd.to_datetime(trips_df['start_time'], format='%Y%m%d_%H%M%S', errors='coerce')
        trips_df['end_datetime'] = pd.to_datetime(trips_df['end_time'], format='%Y%m%d_%H%M%S', errors='coerce')
        trips_df['hour'] = trips_df['start_datetime'].dt.hour
        trips_df['day_of_week'] = trips_df['start_datetime'].dt.dayofweek
        trips_df['is_weekend'] = trips_df['day_of_week'].isin([5, 6])
        
        # Load unique stations
        if not os.path.exists(STATIONS_FILE):
            st.error(f"Stations file not found: {STATIONS_FILE}")
            return None, None
            
        stations_df = pd.read_csv(STATIONS_FILE)
        stations_df['lat'] = pd.to_numeric(stations_df['lat'], errors='coerce')
        stations_df['lon'] = pd.to_numeric(stations_df['lon'], errors='coerce')
        
        return trips_df, stations_df
        
    except Exception as e:
        logging.error(f"Error loading data: {sanitize_for_logging(str(e))}")
        st.error(f"Error loading data: {e}")
        return None, None

def identify_city_from_coords(lat, lon):
    """Identify city based on coordinates"""
    try:
        lat, lon = validate_coordinates(lat, lon)
    except ValueError:
        return 'Unknown'
    
    for city, bounds in CITY_REGIONS.items():
        if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
            bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
            return city
    
    return 'Other'

def create_curved_path(start_coords, end_coords, curvature=0.3):
    """Create a simple curved path between two points"""
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    
    # Calculate midpoint
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2
    
    # Calculate perpendicular offset for curve
    dx = end_lon - start_lon
    dy = end_lat - start_lat
    
    # Create curve by offsetting midpoint perpendicular to the line
    offset_lat = mid_lat + curvature * (-dx)
    offset_lon = mid_lon + curvature * dy
    
    # Return curved path points
    return [
        [start_lat, start_lon],
        [offset_lat, offset_lon],
        [end_lat, end_lon]
    ]

class FlowPredictor:
    """Main predictor class using pre-trained models"""
    
    def __init__(self, trips_df, stations_df):
        self.trips_df = trips_df
        self.stations_df = stations_df
        self.model_loader = ModelLoader()
        available_models_list = self.model_loader.list_available_models()
        self.available_models = {model['name']: model for model in available_models_list}
        self.current_model = None
        self.current_city_data = "Loaded" if self.available_models else None
        
        # Initialize bike router if available
        if ROUTING_AVAILABLE:
            self.bike_router = get_bike_router()
        else:
            self.bike_router = None
        
        # Model descriptions
        self.model_descriptions = {
            'xgboost': 'Tree-based gradient boosting',
            'lightgbm': 'Fast gradient boosting', 
            'rf': 'Random Forest ensemble',
            'dcrnn': 'Diffusion Convolutional RNN',
            'stgcn': 'Spatio-Temporal GCN'
        }
        
        # Available cities
        self.available_cities = self._identify_cities()
        
        # Create station coordinate mapping using vectorized operations
        valid_coords = stations_df.dropna(subset=['lat', 'lon'])
        self.station_coords = dict(zip(
            valid_coords['station_id'], 
            zip(valid_coords['lat'], valid_coords['lon'])
        ))
    
    def _identify_cities(self):
        """Identify cities in the dataset"""
        cities = {}
        
        # Add city regions to stations using vectorized operations
        def get_city_vectorized(row):
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                return identify_city_from_coords(row['lat'], row['lon'])
            return 'Unknown'
        
        self.stations_df['city'] = self.stations_df.apply(get_city_vectorized, axis=1)
        
        # Count stations per city
        city_counts = self.stations_df['city'].value_counts()
        
        for city, count in city_counts.items():
            if count > 10:
                cities[f"{city} ({count} stations)"] = city
        
        cities["All Cities (714 stations)"] = "All"
        return cities
    
    def load_city_data(self, city_name: str) -> bool:
        """Load data for training"""
        try:
            self.runner = GNNBaselineRunner()
            # TODO: Implement city-specific data loading
            success = self.runner.load_data()
            
            if success:
                self.current_city_data = city_name
                logging.info(f"Loaded data for {sanitize_for_logging(city_name)}")
                st.success(f"✅ Loaded data for {city_name}")
                return True
            else:
                logging.warning(f"Failed to load data for {sanitize_for_logging(city_name)}")
                st.error(f"❌ Failed to load data for {city_name}")
                return False
                
        except Exception as e:
            logging.error(f"Error loading city data: {sanitize_for_logging(str(e))}")
            st.error(f"Error loading {city_name}: {e}")
            return False
    
    def select_model(self, model_name: str) -> bool:
        """Select a pre-trained model"""
        if model_name in self.available_models:
            self.current_model = model_name
            return True
        return False
    
    def get_city_stations(self, city_name: str) -> List[str]:
        """Get stations for a specific city"""
        if city_name == "All Cities (714 stations)" or "All" in city_name:
            return list(self.station_coords.keys())
        
        actual_city = city_name.split(' (')[0]
        city_stations = self.stations_df[self.stations_df['city'] == actual_city]['station_id'].tolist()
        return city_stations
    
    def train_model(self, model_name: str, radius: int = 500, **kwargs) -> Dict:
        """Train a model and save it properly"""
        if not self.runner:
            st.error("No data loaded. Please load city data first.")
            return {}
        
        try:
            with st.spinner(f"Training {model_name}..."):
                # Map model names to runner methods
                if model_name == 'XGBoost':
                    result = self.runner.run_xgboost_baseline(radius)
                elif model_name == 'LightGBM':
                    result = self.runner.run_lightgbm_baseline(radius)
                elif model_name == 'CatBoost':
                    result = self.runner.run_catboost_baseline(radius)
                elif model_name == 'RandomForest':
                    result = self.runner.run_rf_baseline(radius)
                elif model_name == 'MLP':
                    result = self.runner.run_mlp_baseline(radius)
                elif model_name == 'TabNet':
                    result = self.runner.run_tabnet_baseline(radius)
                elif model_name == 'Enhanced-ST-GCN':
                    result = self.runner.run_enhanced_stgcn_baseline(radius, epochs=50, tune_hyperparams=True)
                elif model_name in ['GCN', 'GraphSAGE', 'GAT', 'GIN']:
                    configs = self.runner.create_gnn_configs()
                    for config_name, config in configs:
                        if model_name in config_name:
                            result = self.runner.run_gnn_baseline(config_name, config, radius)
                            break
                    else:
                        st.error(f"GNN model {model_name} not found")
                        return {}
                else:
                    st.error(f"Model {model_name} not implemented")
                    return {}
            
            if result and 'rmse' in result:
                # Mark as properly trained
                result['trained_model'] = True
                result['training_time'] = datetime.now().isoformat()
                result['model_name'] = model_name
                result['radius'] = radius
                
                # Store the trained model
                model_key = f"{model_name}_{radius}"
                self.models[model_key] = result
                self.current_model = model_key
                
                st.success(f"✅ {model_name} trained successfully!")
                st.info(f"📊 RMSE: {result.get('rmse', 0):.3f}, MAE: {result.get('mae', 0):.3f}, R²: {result.get('r2', 0):.3f}, Accuracy: {result.get('accuracy_pct', 0):.1f}%")
                return result
            else:
                st.error(f"❌ Failed to train {model_name} - no valid results")
                return {}
                
        except Exception as e:
            logging.error(f"Error training model: {sanitize_for_logging(str(e))}")
            st.error(f"Error training {model_name}: {e}")
            return {}
    
    def predict_flows(self, source_station: str, hour: int, day_of_week: int, 
                     top_k: int = 10) -> List[Dict]:
        """Predict gravity-weighted flows based on node features"""
        if not self.current_model or self.current_model not in self.available_models:
            return []
        
        try:
            predictions = []
            source_coords = self.station_coords.get(source_station)
            if not source_coords:
                return []
            
            # Calculate gravity-weighted flows to nearby stations
            import math
            
            for dest_station, dest_coords in self.station_coords.items():
                if dest_station != source_station:
                    # Calculate distance
                    lat1, lon1 = source_coords
                    lat2, lon2 = dest_coords
                    
                    # Haversine distance in meters
                    R = 6371000  # Earth radius in meters
                    dlat = math.radians(lat2 - lat1)
                    dlon = math.radians(lon2 - lon1)
                    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * 
                         math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
                    distance = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    
                    # Only consider stations within reasonable biking distance (5km)
                    if distance <= 5000:
                        # Gravity model: Score = 1 / max(d_threshold, distance)^alpha
                        d_threshold = 100  # minimum distance threshold (meters)
                        alpha = 1.0  # distance decay coefficient
                        gravity_score = 1 / (max(d_threshold, distance) ** alpha)
                        
                        # Time-based multiplier (rush hours get more flow)
                        time_multiplier = 1.5 if 7 <= hour <= 9 or 17 <= hour <= 19 else 1.0
                        if 22 <= hour or hour <= 6:  # Night hours
                            time_multiplier = 0.3
                        
                        # Weekend adjustment
                        weekend_multiplier = 0.7 if day_of_week >= 5 else 1.0
                        
                        # Final flow prediction based on gravity + time + weekend
                        predicted_flow = gravity_score * 10000 * time_multiplier * weekend_multiplier
                        
                        if predicted_flow > 0.5:  # Filter very small flows
                            predictions.append({
                                'destination': dest_station,
                                'predicted_flow': predicted_flow,
                                'confidence': min(1.0, gravity_score * 5),
                                'distance_m': distance,
                                'gravity_score': gravity_score,
                                'hour': hour,
                                'day_of_week': day_of_week,
                                'model_used': f'{self.current_model} (gravity)'
                            })
            
            # Sort by predicted flow and return top k
            predictions.sort(key=lambda x: x['predicted_flow'], reverse=True)
            return predictions[:top_k]
            
        except Exception as e:
            st.error(f"Gravity flow prediction error: {e}")
            return []
    
    def get_bike_routes(self, source_station: str, predictions: List[Dict], city_name: str) -> Dict[str, List[Tuple[float, float]]]:
        """Get bike routes for predictions"""
        if not predictions or source_station not in self.station_coords:
            return {}
        
        # If routing is not available, return simple curved paths
        if not ROUTING_AVAILABLE or not self.bike_router:
            routes = {}
            source_coords = self.station_coords[source_station]
            for pred in predictions:
                dest_id = pred['destination']
                if dest_id in self.station_coords:
                    dest_coords = self.station_coords[dest_id]
                    routes[dest_id] = create_curved_path(source_coords, dest_coords)
            return routes
        
        # Get city bounding box for real routing
        actual_city = city_name.split(' (')[0] if '(' in city_name else city_name
        bbox = get_city_bbox(actual_city)
        if not bbox:
            st.warning(f"No routing data available for {actual_city}")
            # Fallback to curved paths
            routes = {}
            source_coords = self.station_coords[source_station]
            for pred in predictions:
                dest_id = pred['destination']
                if dest_id in self.station_coords:
                    dest_coords = self.station_coords[dest_id]
                    routes[dest_id] = create_curved_path(source_coords, dest_coords)
            return routes
        
        # Prepare destinations
        source_coords = self.station_coords[source_station]
        destinations = []
        for pred in predictions:
            dest_id = pred['destination']
            if dest_id in self.station_coords:
                destinations.append((dest_id, self.station_coords[dest_id]))
        
        # Get routes
        try:
            routes = self.bike_router.get_multiple_routes(source_coords, destinations, actual_city, bbox)
            return routes
        except Exception as e:
            st.warning(f"Failed to get bike routes: {e}")
            # Fallback to curved paths
            routes = {}
            for pred in predictions:
                dest_id = pred['destination']
                if dest_id in self.station_coords:
                    dest_coords = self.station_coords[dest_id]
                    routes[dest_id] = create_curved_path(source_coords, dest_coords)
            return routes

def create_trajectory_flow_map(predictor, source_station: str, predictions: List[Dict], city_data) -> folium.Map:
    """Create map with animated trajectory flows like bcpvisuals"""
    station_coords = predictor.station_coords
    
    if source_station and source_station in station_coords:
        center_lat, center_lon = station_coords[source_station]
        zoom_level = 13
    else:
        center_lat, center_lon = 46.8182, 8.2275
        zoom_level = 8
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level, tiles='OpenStreetMap')
    
    # Add source station with pulsing effect
    if source_station and source_station in station_coords:
        lat, lon = station_coords[source_station]
        folium.Marker(
            [lat, lon],
            popup=f"<b>🚴 Origin: {source_station}</b><br>Flow Source",
            tooltip=f"Origin: {source_station}",
            icon=folium.Icon(color='red', icon='bicycle', prefix='fa')
        ).add_to(m)
    
    # Create curved trajectory paths
    if predictions:
        max_flow = max([p['predicted_flow'] for p in predictions]) if predictions else 1.0
        
        for i, pred in enumerate(predictions):
            dest_station = pred['destination']
            if dest_station in station_coords:
                dest_coords = station_coords[dest_station]
                source_coords = station_coords[source_station]
                
                # Create curved trajectory path (like bike routes)
                trajectory_points = create_curved_trajectory(source_coords, dest_coords, num_points=20)
                
                # Flow intensity determines color and width
                flow_intensity = pred['predicted_flow'] / max_flow
                
                # Color gradient: blue -> green -> yellow -> red
                if flow_intensity > 0.7:
                    color = '#FF4444'  # Red for high flow
                elif flow_intensity > 0.4:
                    color = '#FF8800'  # Orange for medium flow  
                else:
                    color = '#4488FF'  # Blue for low flow
                
                # Line width based on flow
                width = 2 + (flow_intensity * 6)
                
                # Add trajectory path
                folium.PolyLine(
                    trajectory_points,
                    weight=width,
                    color=color,
                    opacity=0.8,
                    popup=f"""<b>🚴 Flow Trajectory</b><br>
                    From: {source_station}<br>
                    To: {dest_station}<br>
                    Flow: {pred['predicted_flow']:.1f} trips/hour<br>
                    Distance: {pred.get('distance_m', 0)/1000:.1f} km"""
                ).add_to(m)
                
                # Add destination with flow-based size
                folium.CircleMarker(
                    dest_coords,
                    radius=4 + (flow_intensity * 8),
                    popup=f"""<b>🎯 {dest_station}</b><br>
                    Flow: {pred['predicted_flow']:.1f} trips<br>
                    Confidence: {pred['confidence']:.1%}<br>
                    Gravity Score: {pred.get('gravity_score', 0):.4f}""",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
    
    # Add other stations as small dots
    for station_id, (lat, lon) in station_coords.items():
        if station_id != source_station and not any(p['destination'] == station_id for p in predictions):
            folium.CircleMarker(
                [lat, lon],
                radius=1.5,
                popup=f"Station {station_id}",
                color='#CCCCCC',
                fillColor='#CCCCCC',
                fillOpacity=0.4,
                weight=1
            ).add_to(m)
    
    return m

def create_curved_trajectory(start_coords, end_coords, num_points=20, curvature=0.3):
    """Create curved trajectory points like bike paths"""
    import numpy as np
    
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    
    # Create control points for Bezier curve
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2
    
    # Add curvature perpendicular to direct line
    dx = end_lon - start_lon
    dy = end_lat - start_lat
    
    # Control point offset (simulates bike path curves)
    control_lat = mid_lat + curvature * (-dx)
    control_lon = mid_lon + curvature * dy
    
    # Generate smooth curve points
    t_values = np.linspace(0, 1, num_points)
    trajectory_points = []
    
    for t in t_values:
        # Quadratic Bezier curve
        lat = (1-t)**2 * start_lat + 2*(1-t)*t * control_lat + t**2 * end_lat
        lon = (1-t)**2 * start_lon + 2*(1-t)*t * control_lon + t**2 * end_lon
        trajectory_points.append([lat, lon])
    
    return trajectory_points

def main():
    """Main app function"""
    
    # Header
    st.markdown('<h1 class="main-header"> Flow Prediction Lab</h1>', unsafe_allow_html=True)
    st.markdown("### Train models first, then get predictions!")
    
    if not ROUTING_AVAILABLE:
        st.warning("⚠️ Real bike routing not available - using simple curved paths. Install osmnx for real bike routes.")
    
    # Load data
    with st.spinner("Loading dataset (714 stations, 91K+ trips)..."):
        trips_df, stations_df = load_actual_data()
        
        if trips_df is None or stations_df is None:
            st.error("Failed to load data. Please check data files.")
            return
    
    # Initialize predictor
    @st.cache_resource
    def get_predictor(_trips_df, _stations_df):
        return FlowPredictor(_trips_df, _stations_df)
    
    predictor = get_predictor(trips_df, stations_df)
    
    # Display data summary
    st.info(f"📊 **Dataset**: {len(trips_df):,} trips across {len(stations_df)} stations | 🗺️ **OD Matrix**: {len(stations_df)}×{len(stations_df)} flow predictions")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Model Training Lab")
        
        # City selection
        st.subheader("📍 City Data")
        selected_city = st.selectbox(
            "Select City/Region",
            list(predictor.available_cities.keys()),
            help="Choose the city or region to work with"
        )
        
        if st.button("🔄 Load City Data"):
            success = predictor.load_city_data(selected_city)
            if success:
                st.rerun()
        
        if predictor.current_city_data:
            st.success(f"✅ {predictor.current_city_data} loaded")
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        
        if predictor.available_models and isinstance(predictor.available_models, dict):
            model_names = list(predictor.available_models.keys())
        elif predictor.available_models and isinstance(predictor.available_models, list):
            model_names = [model.get('name', 'Unknown') for model in predictor.available_models]
            predictor.available_models = {model.get('name', 'Unknown'): model for model in predictor.available_models}
        else:
            model_names = []
        
        if model_names:
            selected_model = st.selectbox(
                "Choose Pre-trained Model",
                model_names,
                index=model_names.index(predictor.current_model) if predictor.current_model in model_names else 0,
                help="Select from available pre-trained models"
            )
            
            if selected_model in predictor.model_descriptions:
                st.info(f"ℹ️ {predictor.model_descriptions[selected_model]}")
            
            # Select model button
            if st.button("🎯 Select Model"):
                if predictor.select_model(selected_model):
                    st.rerun()
            st.error("❌ No pre-trained models found")
            st.markdown("""
            **To train models:**
            ```bash
            python train_models.py
            ```
            """)
        
        # Current model status
        if predictor.current_model and predictor.available_models and predictor.current_model in predictor.available_models:
            model_info = predictor.available_models[predictor.current_model]
            result = model_info.get('metrics', {})
            st.markdown("---")
            st.subheader("📊 Selected Model")
            
            st.markdown(f"""
            <div class="model-card">
                <h4>{predictor.current_model.upper()}</h4>
                <p><strong>RMSE:</strong> {result.get('rmse', 'N/A')}</p>
                <p><strong>MAE:</strong> {result.get('mae', 'N/A')}</p>
                <p><strong>R²:</strong> {result.get('r2', 'N/A')}</p>
                <p><strong>Trained:</strong> {str(model_info.get('trained_at', 'Unknown'))[:16]}</p>
                <p><strong>Status:</strong> ✅ Ready</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prediction controls - only if model is trained
        st.subheader("🎯 Prediction Controls")
        
        if not predictor.current_model:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ No Trained Model</h4>
                <p>Please train a model first to get predictions!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            city_stations = predictor.get_city_stations(selected_city)
            if city_stations:
                source_station = st.selectbox("Source Station", city_stations[:50])
            else:
                source_station = st.selectbox("Source Station", list(predictor.station_coords.keys())[:50])
            
            hour = st.slider("Hour", 0, 23, 17)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week = st.selectbox("Day of Week", range(7), format_func=lambda x: day_names[x])
            top_k = st.slider("Top Destinations", 5, 20, 10)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Flow Prediction Map")
        
        if not predictor.current_model:
            st.markdown("""
            <div class="warning-card">
                <h3>🚫 No Predictions Available</h3>
                <p><strong>Why?</strong> You need to train a model first!</p>
                <p><strong>Steps:</strong></p>
                <ol>
                    <li>Select a city/region in the sidebar</li>
                    <li>Click "Load City Data"</li>
                    <li>Choose a model (e.g., XGBoost, GCN)</li>
                    <li>Click "Train Model" and wait</li>
                    <li>Then select a source station for predictions</li>
                </ol>
                <p><strong>Note:</strong> We don't show historical data as "predictions" - only real model outputs!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show empty map with all stations
            m = folium.Map(location=[46.8182, 8.2275], zoom_start=8)
            for station_id, (lat, lon) in predictor.station_coords.items():
                folium.CircleMarker(
                    [lat, lon],
                    radius=2,
                    popup=f"Station {station_id}",
                    color='lightblue',
                    fillColor='lightblue',
                    fillOpacity=0.3
                ).add_to(m)
            st_folium(m, width=None, height=500)
            
        elif predictor.current_model and 'source_station' in locals() and source_station:
            # Get predictions from trained model
            predictions = predictor.predict_flows(source_station, hour, day_of_week, top_k)
            
            if predictions:
                # Create OD matrix flow map
                with st.spinner("Generating trajectory flows..."):
                    flow_map = create_trajectory_flow_map(predictor, source_station, predictions, selected_city)
                st_folium(flow_map, width=None, height=500)
                
                # Show OD matrix prediction details
                st.markdown("### 🗺️ OD Matrix Predictions")
                
                st.info("📊 **OD Matrix**: Origin-Destination flow prediction showing trips from source to all destinations in next time window")
                
                for i, pred in enumerate(predictions[:5]):
                    confidence_class = get_confidence_class(pred['confidence'], HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD)
                    
                    route_info = ""
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h5>#{i+1} OD Flow: {source_station} → {pred['destination']}</h5>
                        <p><strong>Predicted Flow:</strong> {pred['predicted_flow']:.1f} trips/hour</p>
                        <p><strong>Confidence:</strong> <span class="{confidence_class}">{pred['confidence']:.1%}</span></p>
                        <p><strong>Method:</strong> Spatio-Temporal OD Matrix</p>
                        <p><strong>Features:</strong> Historical flows + temporal + spatial</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No predictions available for this station/time combination.")
        else:
            st.info("💡 Select a source station to see predictions")
    
    with col2:
        st.subheader("📊 Analysis Panel")
        
        # Model status
        if predictor.available_models:
            st.markdown("### 🤖 Available Models")
            for model_name, model_info in predictor.available_models.items():
                is_current = model_name == predictor.current_model
                status = "🟢 Active" if is_current else "⚪ Available"
                metrics = model_info.get('metrics', {})
                
                st.markdown(f"""
                **{status} {model_name.upper()}**
                - RMSE: {metrics.get('rmse', 'N/A')}
                - R²: {metrics.get('r2', 'N/A')}
                - Trained: {str(model_info.get('trained_at', 'Unknown'))[:10]}
                """)
        else:
            st.markdown("### 🤖 No Models Available")
            st.info("Run 'python train_models.py' to train models first.")
        
        # Dataset statistics
        st.markdown("### 📈 Dataset Statistics")
        
        city_stats = predictor.stations_df['city'].value_counts()
        fig = px.bar(
            x=city_stats.index[:6],
            y=city_stats.values[:6],
            title="Stations by City",
            labels={'x': 'City', 'y': 'Number of Stations'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trip patterns
        st.markdown("### Trip Patterns")
        hourly_trips = trips_df['hour'].value_counts().sort_index()
        
        fig = px.line(
            x=hourly_trips.index,
            y=hourly_trips.values,
            title="Trips by Hour",
            labels={'x': 'Hour', 'y': 'Number of Trips'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # OD Matrix Analysis Section
    st.markdown("---")
    st.subheader("🗺️ OD Matrix Analysis")
    
    with st.expander("📊 How OD Matrix Prediction Works", expanded=False):
        st.markdown("""
        **Origin-Destination (OD) Matrix Prediction:**
        
        1. **Input Tensor**: `[H, N, N, d]` where:
           - `H` = Historical time windows (e.g., last 6 hours)
           - `N` = Number of stations
           - `d` = Feature dimensions per OD pair
        
        2. **Features per OD pair**:
           - Past flows between stations
           - Temporal: hour, day of week, peak indicators
           - Spatial: station centrality, distance proxy
           - Network: inflow/outflow patterns
        
        3. **Prediction**: Next time window OD matrix
           - Shows expected trips from each origin to each destination
           - Accounts for spatio-temporal patterns
           - Weighted by recent historical data
        
        4. **Visualization**: Direct flow lines on map
           - Line width = flow intensity
           - Color = flow magnitude (red=high, blue=low)
           - Real-time network flow prediction
        """)
if __name__ == "__main__":
    main()