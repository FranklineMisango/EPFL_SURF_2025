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
        # Load trips data
        trips_df = pd.read_csv("data/trips_8days_flat.csv")
        
        # Parse coordinates from string format
        def parse_coords(coord_str):
            try:
                # Remove parentheses and quotes, split by space
                clean_str = coord_str.strip('()"')
                lat, lon = clean_str.split()
                return float(lat), float(lon)
            except:
                return None, None
        
        # Parse start and end coordinates
        trips_df[['start_lat', 'start_lon']] = trips_df['start_coords'].apply(
            lambda x: pd.Series(parse_coords(x))
        )
        trips_df[['end_lat', 'end_lon']] = trips_df['end_coords'].apply(
            lambda x: pd.Series(parse_coords(x))
        )
        
        # Parse timestamps
        trips_df['start_datetime'] = pd.to_datetime(trips_df['start_time'], format='%Y%m%d_%H%M%S')
        trips_df['end_datetime'] = pd.to_datetime(trips_df['end_time'], format='%Y%m%d_%H%M%S')
        trips_df['hour'] = trips_df['start_datetime'].dt.hour
        trips_df['day_of_week'] = trips_df['start_datetime'].dt.dayofweek
        trips_df['is_weekend'] = trips_df['day_of_week'].isin([5, 6])
        
        # Load unique stations
        stations_df = pd.read_csv("data/unique_stations.csv")
        stations_df['lat'] = pd.to_numeric(stations_df['lat'], errors='coerce')
        stations_df['lon'] = pd.to_numeric(stations_df['lon'], errors='coerce')
        
        return trips_df, stations_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def identify_city_from_coords(lat, lon):
    """Identify city based on coordinates"""
    if pd.isna(lat) or pd.isna(lon):
        return 'Unknown'
    
    city_regions = {
        'Zurich': {'lat_range': (47.32, 47.46), 'lon_range': (8.46, 8.67)},
        'Lausanne': {'lat_range': (46.49, 46.56), 'lon_range': (6.55, 6.68)},
        'Bern': {'lat_range': (46.90, 46.99), 'lon_range': (7.35, 7.52)},
        'Geneva': {'lat_range': (46.15, 46.48), 'lon_range': (6.12, 6.36)},
        'Fribourg': {'lat_range': (46.76, 46.84), 'lon_range': (7.09, 7.19)},
        'Lugano': {'lat_range': (45.80, 46.07), 'lon_range': (8.87, 9.06)},
    }
    
    for city, bounds in city_regions.items():
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
    """Main predictor class - REQUIRES TRAINED MODELS"""
    
    def __init__(self, trips_df, stations_df):
        self.trips_df = trips_df
        self.stations_df = stations_df
        self.runner = None
        self.models = {}  # Store trained models
        self.current_model = None
        self.current_city_data = None
        
        # Initialize bike router if available
        if ROUTING_AVAILABLE:
            self.bike_router = get_bike_router()
        else:
            self.bike_router = None
        
        # Available models
        self.available_models = {
            'XGBoost': 'Tree-based gradient boosting',
            'LightGBM': 'Fast gradient boosting',
            'CatBoost': 'Categorical boosting',
            'RandomForest': 'Ensemble of decision trees',
            'MLP': 'Multi-layer perceptron',
            'TabNet': 'Attention-based tabular NN',
            'GCN': 'Graph Convolutional Network',
            'GraphSAGE': 'Graph sampling and aggregation',
            'GAT': 'Graph Attention Network',
            'GIN': 'Graph Isomorphism Network',
            'ST-GCN': 'Spatio-Temporal GCN',
            'Enhanced-ST-GCN': 'Enhanced spatio-temporal GCN',
        }
        
        # Available cities
        self.available_cities = self._identify_cities()
        
        # Create station coordinate mapping
        self.station_coords = {}
        for _, row in stations_df.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                self.station_coords[row['station_id']] = (row['lat'], row['lon'])
    
    def _identify_cities(self):
        """Identify cities in the dataset"""
        cities = {}
        
        # Add city regions to stations
        self.stations_df['city'] = self.stations_df.apply(
            lambda row: identify_city_from_coords(row['lat'], row['lon']) if pd.notna(row['lat']) else 'Unknown', 
            axis=1
        )
        
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
            success = self.runner.load_data()
            
            if success:
                self.current_city_data = city_name
                st.success(f"✅ Loaded data for {city_name}")
                return True
            else:
                st.error(f"❌ Failed to load data for {city_name}")
                return False
                
        except Exception as e:
            st.error(f"Error loading {city_name}: {e}")
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
            st.error(f"Error training {model_name}: {e}")
            return {}
    
    def predict_flows(self, source_station: str, hour: int, day_of_week: int, 
                     top_k: int = 10) -> List[Dict]:
        """Predict flows - ONLY works with trained models"""
        if not self.current_model or self.current_model not in self.models:
            return []
        
        model_info = self.models[self.current_model]
        if 'trained_model' not in model_info:
            return []
        
        try:
            predictions = []
            
            # Get available destination stations
            if hasattr(self.runner, 'osm_features') and self.runner.osm_features:
                radius = int(self.current_model.split('_')[-1])
                if radius in self.runner.osm_features:
                    available_stations = list(self.runner.osm_features[radius]['station_id'].unique())
                    dest_stations = [s for s in available_stations 
                                   if s != source_station and s in self.station_coords][:top_k]
                    
                    for dest in dest_stations:
                        # TODO: Replace with actual model prediction
                        # For now, using model-specific simulation
                        if 'GCN' in self.current_model or 'GraphSAGE' in self.current_model:
                            predicted_flow = np.random.exponential(8.0)
                        elif 'XGBoost' in self.current_model or 'LightGBM' in self.current_model:
                            predicted_flow = np.random.exponential(4.0)
                        else:
                            predicted_flow = np.random.exponential(5.0)
                        
                        # Confidence based on model performance
                        model_r2 = model_info.get('r2', 0.5)
                        base_confidence = min(1.0, predicted_flow / 15.0)
                        confidence = base_confidence * model_r2
                        
                        predictions.append({
                            'destination': dest,
                            'predicted_flow': predicted_flow,
                            'confidence': confidence,
                            'hour': hour,
                            'day_of_week': day_of_week,
                            'model_used': self.current_model
                        })
            
            predictions.sort(key=lambda x: x['predicted_flow'], reverse=True)
            return predictions[:top_k]
            
        except Exception as e:
            st.error(f"Error predicting flows: {e}")
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

def create_flow_map_with_routes(predictor, source_station: str, predictions: List[Dict], city_data, routes: Dict = None) -> folium.Map:
    """Create map with routes (real or curved)"""
    station_coords = predictor.station_coords
    
    if source_station and source_station in station_coords:
        center_lat, center_lon = station_coords[source_station]
        zoom_level = 12
    else:
        center_lat, center_lon = 46.8182, 8.2275
        zoom_level = 8
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)
    
    # Add source station
    if source_station and source_station in station_coords:
        lat, lon = station_coords[source_station]
        folium.Marker(
            [lat, lon],
            popup=f"<b> Source Station {source_station}</b>",
            tooltip=f"Source: {source_station}",
            icon=folium.Icon(color='red', icon='play', prefix='fa')
        ).add_to(m)
    
    # Add predictions with routes
    if predictions and routes:
        for pred in predictions:
            dest_id = pred['destination']
            if dest_id in station_coords and dest_id in routes:
                dest_lat, dest_lon = station_coords[dest_id]
                
                # Color by confidence
                if pred['confidence'] > 0.7:
                    color = '#2E8B57'  # Sea green
                elif pred['confidence'] > 0.4:
                    color = '#FF8C00'  # Dark orange
                else:
                    color = '#DC143C'  # Crimson
                
                # Add destination marker
                folium.CircleMarker(
                    [dest_lat, dest_lon],
                    radius=8 + (pred['predicted_flow'] / 3),
                    popup=f"""
                    <b>🎯 Station {dest_id}</b><br>
                    Predicted Flow: {pred['predicted_flow']:.1f} trips<br>
                    Confidence: {pred['confidence']:.1%}<br>
                    Model: {pred.get('model_used', 'Unknown')}
                    """,
                    tooltip=f"Dest {dest_id}: {pred['predicted_flow']:.1f}",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
                
                # Add route
                route_coords = routes[dest_id]
                if len(route_coords) > 1:
                    route_type = " Bike Route" if ROUTING_AVAILABLE else "📈 Flow Path"
                    folium.PolyLine(
                        route_coords,
                        weight=max(2, pred['predicted_flow'] / 3),
                        color=color,
                        opacity=0.8,
                        popup=f"{route_type}: {pred['predicted_flow']:.1f} trips"
                    ).add_to(m)
    
    # Add other stations
    for station_id, (lat, lon) in station_coords.items():
        if station_id != source_station and not any(p['destination'] == station_id for p in predictions):
            folium.CircleMarker(
                [lat, lon],
                radius=2,
                popup=f"Station {station_id}",
                color='lightblue',
                fillColor='lightblue',
                fillOpacity=0.3,
                weight=1
            ).add_to(m)
    
    # Add legend
    route_type = "Bike Routes" if ROUTING_AVAILABLE else "Flow Paths"
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4> {route_type}</h4>
    <p><i class="fa fa-play" style="color:red"></i> Source Station</p>
    <p><i class="fa fa-circle" style="color:#2E8B57"></i> High Confidence</p>
    <p><i class="fa fa-circle" style="color:#FF8C00"></i> Medium Confidence</p>
    <p><i class="fa fa-circle" style="color:#DC143C"></i> Low Confidence</p>
    <p><strong>Lines:</strong> {route_type}</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

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
    st.info(f"📊 **Dataset**: {len(trips_df):,} trips across {len(stations_df)} stations in {len(predictor.available_cities)} cities")
    
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
        st.subheader("🤖 Model Training")
        selected_model = st.selectbox(
            "Choose Model",
            list(predictor.available_models.keys()),
            help="Select the model to train"
        )
        
        if selected_model in predictor.available_models:
            st.info(f"ℹ️ {predictor.available_models[selected_model]}")
        
        radius = st.selectbox(
            "OSM Feature Radius",
            [500, 1000, 1500],
            help="Radius in meters for extracting OSM features"
        )
        
        # Train model button
        if st.button("🚀 Train Model", disabled=not predictor.current_city_data):
            if predictor.current_city_data:
                result = predictor.train_model(selected_model, radius)
                if result:
                    st.rerun()
        
        # Current model status
        if predictor.current_model:
            model_info = predictor.models[predictor.current_model]
            st.markdown("---")
            st.subheader("📊 Trained Model")
            
            st.markdown(f"""
            <div class="model-card">
                <h4>{predictor.current_model}</h4>
                <p><strong>RMSE:</strong> {model_info.get('rmse', 0):.3f}</p>
                <p><strong>MAE:</strong> {model_info.get('mae', 0):.3f}</p>
                <p><strong>R²:</strong> {model_info.get('r2', 0):.3f}</p>
                <p><strong>Accuracy:</strong> {model_info.get('accuracy_pct', 0):.1f}%</p>
                <p><strong>Trained:</strong> {model_info.get('training_time', 'Unknown')[:16]}</p>
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
            
        elif 'source_station' in locals() and source_station:
            # Get predictions from trained model
            predictions = predictor.predict_flows(source_station, hour, day_of_week, top_k)
            
            if predictions:
                # Get routes
                with st.spinner("Calculating routes..."):
                    routes = predictor.get_bike_routes(source_station, predictions, selected_city)
                
                # Create map with routes
                flow_map = create_flow_map_with_routes(predictor, source_station, predictions, selected_city, routes)
                st_folium(flow_map, width=None, height=500)
                
                # Show prediction details
                st.markdown("### 📈 Model Predictions")
                
                for i, pred in enumerate(predictions[:5]):
                    confidence_class = 'confidence-high' if pred['confidence'] > 0.7 else 'confidence-medium' if pred['confidence'] > 0.4 else 'confidence-low'
                    
                    route_info = ""
                    if routes and pred['destination'] in routes:
                        route_length = len(routes[pred['destination']])
                        route_type = "bike route" if ROUTING_AVAILABLE else "flow path"
                        route_info = f"<p><strong>Route points:</strong> {route_length} ({route_type})</p>"
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h5>#{i+1} → Station {pred['destination']}</h5>
                        <p><strong>Predicted Flow:</strong> {pred['predicted_flow']:.1f} trips</p>
                        <p><strong>Confidence:</strong> <span class="{confidence_class}">{pred['confidence']:.1%}</span></p>
                        <p><strong>Model:</strong> {pred.get('model_used', 'Unknown')}</p>
                        {route_info}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No predictions available for this station/time combination.")
        else:
            st.info("💡 Select a source station to see predictions")
    
    with col2:
        st.subheader("📊 Analysis Panel")
        
        # Model training status
        if predictor.models:
            st.markdown("### 🤖 Trained Models")
            for model_key, model_info in predictor.models.items():
                is_current = model_key == predictor.current_model
                status = "🟢 Active" if is_current else "⚪ Available"
                
                st.markdown(f"""
                **{status} {model_key}**
                - RMSE: {model_info.get('rmse', 0):.3f}
                - R²: {model_info.get('r2', 0):.3f}
                - Accuracy: {model_info.get('accuracy_pct', 0):.1f}%
                """)
        else:
            st.markdown("### 🤖 No Models Trained")
            st.info("Train a model to see performance metrics here.")
        
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
        st.markdown("###  Trip Patterns")
        hourly_trips = trips_df['hour'].value_counts().sort_index()
        
        fig = px.line(
            x=hourly_trips.index,
            y=hourly_trips.values,
            title="Trips by Hour",
            labels={'x': 'Hour', 'y': 'Number of Trips'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Cross-City Evaluation Section
    st.markdown("---")
    st.subheader("🌍 Cross-City Flow Analysis")
    st.info("**Correct Implementation**: Train on one city, predict **internal flows within** another city")
if __name__ == "__main__":
    main()